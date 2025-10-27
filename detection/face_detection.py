# detection/face_detector.py
import cv2
import numpy as np

class FaceDetector:
    """Face detection using OpenCV DNN with Caffe model."""
    
    def __init__(self, prototxt_path="detection/models/deploy.prototxt",
                 model_path="detection/models/res10_300x300_ssd_iter_140000.caffemodel",
                 conf_threshold=0.5):
        """
        Args:
            prototxt_path: Path to deploy.prototxt
            model_path: Path to caffemodel
            conf_threshold: Confidence threshold
        """
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.conf_threshold = conf_threshold
        
    def detect(self, frame):
        """
        Detect faces in frame.
        
        Args:
            frame: Input image (BGR)
            
        Returns:
            List of face bounding boxes [(x, y, w, h), ...]
        """
        (h, w) = frame.shape[:2]
        
        # Prepare input blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Convert to (x, y, w, h) format
                x, y = max(0, x1), max(0, y1)
                w_box = min(x2 - x1, w - x)
                h_box = min(y2 - y1, h - y)
                
                if w_box > 0 and h_box > 0:
                    boxes.append((x, y, w_box, h_box))
        
        return boxes