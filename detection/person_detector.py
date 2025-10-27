# detection/person_detector.py
from ultralytics import YOLO
import cv2

class PersonDetector:
    """Detect persons using YOLOv8."""
    
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        """
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
    def detect_people(self, frame):
        """
        Detect all persons in frame.
        
        Args:
            frame: Input image (BGR)
            
        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
        results = self.model(frame, classes=[0], conf=self.conf_threshold, verbose=False)
        people = []
        
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # person class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                    people.append((x, y, w, h))
        
        return people