# blurring/blurring_module.py
import cv2
import numpy as np
from typing import List, Tuple, Optional


class FaceBlurrer:
    """
    Handles selective face blurring with multiple blur techniques.
    """
    
    def __init__(self, blur_type='gaussian', blur_intensity='medium'):
        """
        Initialize the face blurrer.
        
        Args:
            blur_type: 'gaussian', 'pixelate', 'black_box', or 'mosaic'
            blur_intensity: 'low', 'medium', 'high'
        """
        self.blur_type = blur_type
        self.blur_intensity = blur_intensity
        
        # Intensity settings
        self.intensity_map = {
            'gaussian': {'low': 15, 'medium': 35, 'high': 55},
            'pixelate': {'low': 20, 'medium': 15, 'high': 10}
        }
    
    def apply_gaussian_blur(self, face_roi: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply Gaussian blur to face region."""
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(face_roi, (kernel_size, kernel_size), 0)
    
    def apply_pixelation(self, face_roi: np.ndarray, pixel_size: int) -> np.ndarray:
        """Apply pixelation effect to face region."""
        h, w = face_roi.shape[:2]
        
        # Ensure pixel_size is valid
        if pixel_size <= 0:
            pixel_size = 10
        
        # Downsample
        temp = cv2.resize(face_roi, (w // pixel_size, h // pixel_size), 
                         interpolation=cv2.INTER_LINEAR)
        
        # Upsample back
        pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        return pixelated
    
    def apply_black_box(self, face_roi: np.ndarray) -> np.ndarray:
        """Replace face with black box."""
        return np.zeros_like(face_roi)
    
    def apply_mosaic_blur(self, face_roi: np.ndarray) -> np.ndarray:
        """Apply mosaic-style blur (combination of pixelation and blur)."""
        # First pixelate
        pixelated = self.apply_pixelation(face_roi, 12)
        # Then add slight blur
        mosaic = cv2.GaussianBlur(pixelated, (5, 5), 0)
        return mosaic
    
    def blur_face(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int], 
                  padding: float = 0.1) -> np.ndarray:
        """
        Blur a single face in the frame.
        
        Args:
            frame: Input frame (BGR format)
            face_bbox: (x, y, w, h) of face bounding box
            padding: Extra padding around face (0.1 = 10% of bbox size)
            
        Returns:
            Frame with blurred face
        """
        x, y, w, h = face_bbox
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)
        
        # Extract face ROI
        face_roi = frame[y1:y2, x1:x2].copy()
        
        if face_roi.size == 0:
            return frame
        
        # Apply blur based on type
        if self.blur_type == 'gaussian':
            kernel_size = self.intensity_map['gaussian'][self.blur_intensity]
            blurred = self.apply_gaussian_blur(face_roi, kernel_size)
        
        elif self.blur_type == 'pixelate':
            pixel_size = self.intensity_map['pixelate'][self.blur_intensity]
            blurred = self.apply_pixelation(face_roi, pixel_size)
        
        elif self.blur_type == 'black_box':
            blurred = self.apply_black_box(face_roi)
        
        elif self.blur_type == 'mosaic':
            blurred = self.apply_mosaic_blur(face_roi)
        
        else:
            # Default to gaussian
            blurred = self.apply_gaussian_blur(face_roi, 35)
        
        # Replace face region in original frame
        frame[y1:y2, x1:x2] = blurred
        return frame
    
    def blur_faces(self, frame: np.ndarray, faces_to_blur: List[Tuple[int, int, int, int]],
                   show_boxes: bool = False) -> np.ndarray:
        """
        Blur multiple faces in a frame.
        
        Args:
            frame: Input frame (BGR format)
            faces_to_blur: List of (x, y, w, h) bounding boxes
            show_boxes: Draw green boxes around blurred faces
            
        Returns:
            Frame with all faces blurred
        """
        result = frame.copy()
        
        for face_bbox in faces_to_blur:
            result = self.blur_face(result, face_bbox)
            
            # Optionally draw bounding box
            if show_boxes:
                x, y, w, h = face_bbox
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result, "BLURRED", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result
    
    def set_blur_type(self, blur_type: str):
        """Change blur type on the fly."""
        valid_types = ['gaussian', 'pixelate', 'black_box', 'mosaic']
        if blur_type in valid_types:
            self.blur_type = blur_type
    
    def set_intensity(self, intensity: str):
        """Change blur intensity on the fly."""
        valid_intensities = ['low', 'medium', 'high']
        if intensity in valid_intensities:
            self.blur_intensity = intensity


class SelectiveBlurPipeline:
    """
    Complete pipeline integrating detection, decision, and blurring.
    """
    
    def __init__(self, detector, decision_module, blurrer: Optional[FaceBlurrer] = None):
        """
        Initialize the complete pipeline.
        
        Args:
            detector: IntegratedDetector instance from detection module
            decision_module: DecisionModule instance from decision module
            blurrer: FaceBlurrer instance (creates default if None)
        """
        self.detector = detector
        self.decision_module = decision_module
        self.blurrer = blurrer or FaceBlurrer(blur_type='gaussian', blur_intensity='medium')
    
    def process_frame(self, frame: np.ndarray, user_text: Optional[str] = None,
                     show_debug: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Process a single frame: detect, decide, blur.
        
        Args:
            frame: Input frame (BGR format)
            user_text: Optional user instruction (e.g., "blur all students")
            show_debug: Show debug information on frame
            
        Returns:
            Tuple of (blurred_frame, detection_results)
        """
        # Step 1: Detection
        results = self.detector.process_frame(frame)
        
        # Step 2: Extract face bounding boxes
        all_faces = []
        face_to_person = {}  # Map face index to person data
        
        for det in results['detections']:
            if det['bbox_face']:
                face_bbox = det['bbox_face']
                all_faces.append(face_bbox)
                face_to_person[len(all_faces) - 1] = det
        
        if not all_faces:
            return frame, results
        
        # Step 3: Decision logic
        # Parse user text if provided
        from decision.nlp.rule_parser import parse_user_text
        user_intent = parse_user_text(user_text) if user_text else None
        
        # Get scene context
        scene = results.get('scene', '')
        
        # Decide which faces to blur
        if user_intent and user_intent.get('mode') == 'all':
            faces_to_blur = all_faces
        else:
            # Use decision module
            faces_to_blur = self.decision_module.analyze(all_faces, frame)
        
        # Step 4: Blur selected faces
        blurred_frame = self.blurrer.blur_faces(frame, faces_to_blur, show_boxes=show_debug)
        
        # Step 5: Add debug info if requested
        if show_debug:
            # Show scene and statistics
            stats_text = [
                f"Scene: {scene}",
                f"Total Faces: {len(all_faces)}",
                f"Blurred: {len(faces_to_blur)}",
                f"Blur: {self.blurrer.blur_type} ({self.blurrer.blur_intensity})"
            ]
            
            y_offset = 30
            for text in stats_text:
                cv2.putText(blurred_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25
        
        return blurred_frame, results
    
    def process_video(self, video_path: str, output_path: str,
                     user_text: Optional[str] = None,
                     show_debug: bool = False,
                     frame_skip: int = 1) -> bool:
        """
        Process entire video file.
        
        Args:
            video_path: Input video path
            output_path: Output video path
            user_text: Optional user instruction
            show_debug: Show debug overlay
            frame_skip: Process every Nth frame (1 = all frames)
            
        Returns:
            True if successful
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process or skip frame
                if frame_count % frame_skip == 0:
                    blurred_frame, _ = self.process_frame(frame, user_text, show_debug)
                    out.write(blurred_frame)
                    processed_count += 1
                else:
                    out.write(frame)
                
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames...")
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        print(f"Video processing complete! Saved to: {output_path}")
        print(f"Total frames: {frame_count}, Processed: {processed_count}")
        return True
    
    def process_webcam(self, user_text: Optional[str] = None, show_debug: bool = True):
        """
        Real-time webcam processing with selective blurring.
        
        Args:
            user_text: Optional user instruction
            show_debug: Show debug overlay
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        print("Webcam blur started. Press 'q' to quit.")
        print("Press 'g' for Gaussian, 'p' for Pixelate, 'b' for Black box, 'm' for Mosaic")
        print("Press '1', '2', '3' for Low, Medium, High intensity")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            blurred_frame, _ = self.process_frame(frame, user_text, show_debug)
            
            # Display
            cv2.imshow('Selective Face Blur - Press Q to quit', blurred_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('g'):
                self.blurrer.set_blur_type('gaussian')
                print("Switched to Gaussian blur")
            elif key == ord('p'):
                self.blurrer.set_blur_type('pixelate')
                print("Switched to Pixelate")
            elif key == ord('b'):
                self.blurrer.set_blur_type('black_box')
                print("Switched to Black box")
            elif key == ord('m'):
                self.blurrer.set_blur_type('mosaic')
                print("Switched to Mosaic")
            elif key == ord('1'):
                self.blurrer.set_intensity('low')
                print("Set to Low intensity")
            elif key == ord('2'):
                self.blurrer.set_intensity('medium')
                print("Set to Medium intensity")
            elif key == ord('3'):
                self.blurrer.set_intensity('high')
                print("Set to High intensity")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam processing stopped.")


# Utility function for quick testing
def quick_blur_test(image_path: str, output_path: str, blur_type: str = 'gaussian'):
    """Quick test function for single image blurring."""
    from detector import IntegratedDetector
    from decision.decision_module import DecisionModule
    
    # Initialize pipeline
    detector = IntegratedDetector()
    decision = DecisionModule(mode='all')  # Blur all faces for testing
    blurrer = FaceBlurrer(blur_type=blur_type, blur_intensity='medium')
    pipeline = SelectiveBlurPipeline(detector, decision, blurrer)
    
    # Load and process image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Cannot load image {image_path}")
        return
    
    blurred, results = pipeline.process_frame(frame, show_debug=True)
    
    # Save result
    cv2.imwrite(output_path, blurred)
    print(f"Blurred image saved to: {output_path}")
    print(f"Found {len(results['detections'])} person(s)")