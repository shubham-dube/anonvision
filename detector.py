# integrated_detector.py
import cv2
import numpy as np
import torch
from typing import List, Dict, Any
import json

# Import all detection modules
from detection.person_detector import PersonDetector
from detection.face_detection import FaceDetector
from detection.attribute_extractor import AttributeExtractor
from detection.clothing_analyzer import analyze_clothing
from detection.pose_estimator import PoseEstimator
from detection.scene_classifier import SceneClassifier


class IntegratedDetector:
    """
    Integrated pipeline for multi-person analysis in a single frame.
    Detects persons, faces, attributes, clothing, pose, and scene context.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize all detection modules."""
        print(f"Initializing detectors on device: {device}")
        
        self.person_detector = PersonDetector()
        self.face_detector = FaceDetector()
        self.attribute_extractor = AttributeExtractor()
        self.pose_estimator = PoseEstimator()
        self.scene_classifier = SceneClassifier()
        
        self.device = device
        print("All detectors initialized successfully!")
    
    def detect_face_in_person(self, frame, person_bbox):
        """
        Detect face within a person's bounding box for efficiency.
        
        Args:
            frame: Full frame image
            person_bbox: (x, y, w, h) of person
            
        Returns:
            Face bbox relative to full frame or None
        """
        x, y, w, h = person_bbox
        
        # Crop person region with some padding for face detection
        padding = 10
        y1 = max(0, y - padding)
        y2 = min(frame.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(frame.shape[1], x + w + padding)
        
        person_crop = frame[y1:y2, x1:x2]
        
        # Detect faces in the cropped region
        faces = self.face_detector.detect(person_crop)
        
        if len(faces) > 0:
            # Take the largest face (likely the person's face)
            face = max(faces, key=lambda f: f[2] * f[3])
            fx, fy, fw, fh = face
            
            # Convert back to full frame coordinates
            return [x1 + fx, y1 + fy, fw, fh]
        
        return None
    
    def classify_pose(self, keypoints, person_bbox):
        """
        Simple pose classification based on keypoint positions.
        
        Args:
            keypoints: Dictionary of pose keypoints
            person_bbox: (x, y, w, h) of person
            
        Returns:
            Pose label: 'standing', 'sitting', 'walking', 'unknown'
        """
        if not keypoints:
            return "unknown"
        
        try:
            # MediaPipe landmark indices:
            # 11, 12: shoulders, 23, 24: hips, 25, 26: knees, 27, 28: ankles
            
            left_shoulder = keypoints.get(11)
            right_shoulder = keypoints.get(12)
            left_hip = keypoints.get(23)
            right_hip = keypoints.get(24)
            left_knee = keypoints.get(25)
            right_knee = keypoints.get(26)
            
            if not all([left_shoulder, right_shoulder, left_hip, right_hip, left_knee]):
                return "unknown"
            
            # Calculate average positions
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_y = (left_hip[1] + right_hip[1]) / 2
            knee_y = (left_knee[1] + right_knee[1]) / 2
            
            # Normalized relative positions (0-1 scale)
            torso_length = hip_y - shoulder_y
            
            if torso_length <= 0:
                return "unknown"
            
            knee_hip_dist = (knee_y - hip_y) / torso_length
            
            # Simple heuristics
            if knee_hip_dist < 0.5:
                return "sitting"
            elif knee_hip_dist > 1.2:
                return "standing"
            else:
                return "walking"
                
        except Exception as e:
            print(f"Pose classification error: {e}")
            return "unknown"
    
    def process_frame(self, frame: np.ndarray, frame_id: int = 0) -> Dict[str, Any]:
        """
        Process a single frame and extract all information.
        
        Args:
            frame: Input image (BGR format)
            frame_id: Frame number/identifier
            
        Returns:
            Dictionary containing all detection results
        """
        h, w = frame.shape[:2]
        
        # Step 1: Scene classification (global context)
        print(f"Processing frame {frame_id}...")
        scene = self.scene_classifier.predict_scene(frame)
        
        # Step 2: Detect all persons in frame
        people_bboxes = self.person_detector.detect_people(frame)
        print(f"Detected {len(people_bboxes)} person(s)")
        
        detections = []
        
        print("before each persons")
        # Step 3: Process each detected person
        for idx, person_bbox in enumerate(people_bboxes):
            person_data = {
                "id": idx + 1,
                "bbox_person": list(person_bbox),
                "bbox_face": None,
                "attributes": None,
                "cloth": None,
                "dress_color": None,
                "pose": "unknown"
            }
            
            x, y, w, h = person_bbox
            print(f"before face {idx}")
            
            # 3a. Face detection within person bbox
            face_bbox = self.detect_face_in_person(frame, person_bbox)
            print(f"after face detection {idx}")
            
            if face_bbox:
                person_data["bbox_face"] = face_bbox
                
                # 3b. Extract facial attributes
                try:
                    fx, fy, fw, fh = face_bbox
                    face_crop = frame[fy:fy+fh, fx:fx+fw]
                    
                    if face_crop.size > 0:
                        attributes = self.attribute_extractor.analyze(face_crop)
                        person_data["attributes"] = attributes
                except Exception as e:
                    print(f"Attribute extraction failed for person {idx+1}: {e}")


            print(f"before clothes {idx}")
            
            # 3c. Clothing analysis
            try:
                clothing_info = analyze_clothing(frame, [x, y, x+w, y+h])
                person_data["cloth"] = clothing_info.get("clothing_type")
                person_data["dress_color"] = clothing_info.get("color")
            except Exception as e:
                print(f"Clothing analysis failed for person {idx+1}: {e}")
            
            print(f"after clothes {idx}")

            # 3d. Pose estimation
            try:
                keypoints = self.pose_estimator.estimate(frame, person_bbox)
                pose_label = self.classify_pose(keypoints, person_bbox)
                person_data["pose"] = pose_label
            except Exception as e:
                print(f"Pose estimation failed for person {idx+1}: {e}")
            
            detections.append(person_data)
            print(f"after pose {idx}")
        
        # Compile final result
        result = {
            "frame_id": frame_id,
            "scene": scene,
            "detections": detections
        }
        
        return result
    
    def visualize_results(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Draw detection results on frame for visualization.
        
        Args:
            frame: Original frame
            results: Detection results dictionary
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        # Draw scene label
        cv2.putText(vis_frame, f"Scene: {results['scene']}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        for det in results['detections']:
            # Draw person bbox
            x, y, w, h = det['bbox_person']
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw face bbox
            if det['bbox_face']:
                fx, fy, fw, fh = det['bbox_face']
                cv2.rectangle(vis_frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
            
            # Create info text
            info_lines = [f"ID: {det['id']}"]
            
            if det['attributes']:
                attr = det['attributes']
                info_lines.append(f"Age: {attr['age']}, {attr['gender']}")
                info_lines.append(f"Emotion: {attr['emotion']}")
            
            if det['cloth']:
                info_lines.append(f"{det['dress_color']} {det['cloth']}")
            
            info_lines.append(f"Pose: {det['pose']}")
            
            # Draw info box
            y_offset = y - 10
            for line in reversed(info_lines):
                cv2.putText(vis_frame, line, (x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset -= 20
        
        return vis_frame
    
    def process_video(self, video_path: str, output_path: str = None, 
                     save_json: bool = True, visualize: bool = True):
        """
        Process entire video frame by frame.
        
        Args:
            video_path: Path to input video
            output_path: Path for output video (if visualize=True)
            save_json: Save results to JSON file
            visualize: Create annotated output video
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if visualizing
        writer = None
        if visualize and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_results = []
        frame_id = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                results = self.process_frame(frame, frame_id)
                all_results.append(results)
                
                # Visualize if requested
                if visualize:
                    vis_frame = self.visualize_results(frame, results)
                    if writer:
                        writer.write(vis_frame)
                    
                    # Optional: display live
                    # cv2.imshow('Detection', vis_frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                
                frame_id += 1
                if frame_id % 10 == 0:
                    print(f"Processed {frame_id}/{total_frames} frames")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        # Save results to JSON
        if save_json:
            json_path = video_path.rsplit('.', 1)[0] + '_results.json'
            with open(json_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Results saved to: {json_path}")
        
        return all_results


def main():
    """Example usage for single frame."""
    # Initialize detector
    detector = IntegratedDetector()
    
    # Load test image
    frame = cv2.imread("test_image.jpg")
    
    if frame is None:
        print("Error: Could not load test image")
        return
    
    # Process frame
    results = detector.process_frame(frame, frame_id=0)
    
    # Print results
    print("\n" + "="*50)
    print("DETECTION RESULTS")
    print("="*50)
    print(json.dumps(results, indent=2))
    
    # Visualize
    vis_frame = detector.visualize_results(frame, results)
    cv2.imwrite("output_annotated.jpg", vis_frame)
    print("\nAnnotated image saved as 'output_annotated.jpg'")
    
    # Save JSON
    with open("detection_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved as 'detection_results.json'")


if __name__ == "__main__":
    main()