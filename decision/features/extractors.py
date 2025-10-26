# decision/features/extractors.py
import numpy as np
import cv2

def bbox_area(b):
    x,y,w,h = b
    return w*h

def bbox_center(b):
    x,y,w,h = b
    return (x + w/2, y + h/2)

def relative_height(body_bbox, frame_h):
    # rough proxy for distance/size (teacher likely larger if close)
    _,_,_,h = body_bbox
    return h / frame_h

def face_to_body_ratio(face_bbox, body_bbox):
    fx,fy,fw,fh = face_bbox
    bx,by,bw,bh = body_bbox
    if bw*bh == 0: return 0.0
    return (fw*fh) / (bw*bh)

def position_in_frame(body_bbox, frame_w, frame_h):
    cx, cy = bbox_center(body_bbox)
    return (cx / frame_w, cy / frame_h)

def pose_features(keypoints):
    """
    keypoints: mediapipe format or list of (x,y,visibility)
    Return simple features:
     - standing (distance between hip and shoulder)
     - hands_up ratio
     - orientation proxy
    """
    # keypoint indices: use mediapipe pose landmarks if available
    # fallback: return zeros
    if keypoints is None:
        return np.zeros(5)
    # Example using normalized coordinates (assume list/dict)
    try:
        # indices (mediapipe): 11=left_shoulder,12=right_shoulder,23=left_hip,24=right_hip,0=head
        ls = keypoints.get('left_shoulder')
        rs = keypoints.get('right_shoulder')
        lh = keypoints.get('left_hip')
        rh = keypoints.get('right_hip')
        head = keypoints.get('nose') or keypoints.get('left_eye')
        if None in (ls,rs,lh,rh,head):
            return np.zeros(5)
        shoulder_y = (ls[1] + rs[1]) / 2
        hip_y = (lh[1] + rh[1]) / 2
        torso_length = abs(hip_y - shoulder_y)
        head_y = head[1]
        # hands up detection (rough)
        hands_up = 0
        # keys: left_wrist, right_wrist
        lw = keypoints.get('left_wrist')
        rw = keypoints.get('right_wrist')
        if lw and lw[1] < shoulder_y: hands_up += 1
        if rw and rw[1] < shoulder_y: hands_up += 1
        return np.array([torso_length, head_y, shoulder_y, hip_y, hands_up])
    except Exception:
        return np.zeros(5)