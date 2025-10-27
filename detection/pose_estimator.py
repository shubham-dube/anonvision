 #detection/pose_estimator.py
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose

class PoseEstimator:
    """Estimate human pose using MediaPipe."""
    
    def __init__(self):
        """Initialize pose estimator."""
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def estimate(self, frame, person_bbox=None):
        """
        Estimate pose keypoints.
        
        Args:
            frame: Input frame (BGR)
            person_bbox: Optional (x, y, w, h) to focus on person
            
        Returns:
            Dictionary of keypoints {landmark_id: (x, y, visibility)}
        """
        # Optionally crop to person region for better accuracy
        if person_bbox:
            x, y, w, h = person_bbox
            x, y = max(0, x), max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                crop = frame
        else:
            crop = frame
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.pose.process(img_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Convert to dictionary format
        keypoints = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints[idx] = (landmark.x, landmark.y, landmark.visibility)
        
        return keypoints
