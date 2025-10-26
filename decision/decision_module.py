import numpy as np

class DecisionModule:
    def __init__(self, mode="largest"):
        """
        mode: str
            - "all"           → blur all faces
            - "largest"       → blur all except the largest (main subject)
            - "crowd"         → blur all if too many faces detected
            - "center_focus"  → blur faces not near the frame center
        """
        self.mode = mode

    def analyze(self, faces, frame):
        """
        faces: list of (x, y, w, h)
        frame: np.ndarray (video frame)
        return: list of faces to blur
        """
        if not faces:
            return []

        # RULE 1 — Blur All
        if self.mode == "all":
            return faces

        # RULE 2 — Blur All Except Largest
        elif self.mode == "largest":
            areas = [w * h for (_, _, w, h) in faces]
            largest_idx = np.argmax(areas)
            return [f for i, f in enumerate(faces) if i != largest_idx]

        # RULE 3 — Blur All in Crowd
        elif self.mode == "crowd":
            if len(faces) > 3:  # arbitrary threshold
                return faces
            else:
                return []

        # RULE 4 — Blur Outside Center Region
        elif self.mode == "center_focus":
            h, w, _ = frame.shape
            cx, cy = w // 2, h // 2
            blur_faces = []
            for (x, y, fw, fh) in faces:
                face_cx, face_cy = x + fw // 2, y + fh // 2
                dist = np.sqrt((cx - face_cx)**2 + (cy - face_cy)**2)
                if dist > min(w, h) / 4:
                    blur_faces.append((x, y, fw, fh))
            return blur_faces

        else:
            return faces
    # decision/decision_module.py (add)
def analyze_with_roles(self, faces, frame, roles=None, user_intent=None, scene=None):
    # faces: [(x,y,w,h)] same order as roles list
    # roles: ['teacher','student','unknown']
    faces_to_blur = []
    # if user_intent has 'target' or 'preserve', apply those first
    if user_intent:
        if user_intent.get('mode') == 'all':
            return faces
        if user_intent.get('target'):
            target = user_intent.get('target')
            for f, r in zip(faces, roles):
                if r == target:
                    faces_to_blur.append(f)
            return faces_to_blur
        if user_intent.get('preserve'):
            preserve = user_intent.get('preserve')
            for f, r in zip(faces, roles):
                if r != preserve:
                    faces_to_blur.append(f)
            return faces_to_blur
    # otherwise fallback to scene-based defaults
    if scene == 'classroom':
        for f, r in zip(faces, roles):
            if r == 'student':
                faces_to_blur.append(f)
    elif scene == 'street':
        faces_to_blur = faces
    else:
        # fallback existing logic
        faces_to_blur = self.analyze(faces, frame)
    return faces_to_blur

