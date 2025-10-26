# decision/pipeline/integrator.py
from decision.features.extractors import *
from decision.role_classifier import RoleClassifier
from decision.nlp.rule_parser import parse_user_text
from decision.decision_module import DecisionModule  # your existing module
# assume you have detection.FaceDetector and person detector (YOLOv8) accessible

class Integrator:
    def __init__(self, person_detector, face_detector, pose_estimator=None):
        self.person_detector = person_detector
        self.face_detector = face_detector
        self.pose_estimator = pose_estimator
        self.role_clf = RoleClassifier(use_trained=False)

    def process_frame(self, frame, user_text=None, scene=None):
        # 1) detect persons and faces
        persons = self.person_detector.detect_persons(frame)  # returns list of bboxes
        faces = self.face_detector.detect_faces(frame)        # list of face bboxes

        # map faces to person boxes (assign face to nearest person bbox center)
        assigned = []  # list of dicts per person: {'person_bbox':..., 'face_bbox':..., 'pose':...}
        for p in persons:
            px,py,pw,ph = p
            pcx,pcy = px+pw/2, py+ph/2
            # find face with center inside person bbox (or nearest)
            matched_face = None
            min_dist = 1e9
            for f in faces:
                fx,fy,fw,fh = f
                fcx,fcy = fx+fw/2, fy+fh/2
                if (px <= fcx <= px+pw) and (py <= fcy <= py+ph):
                    matched_face = f
                    break
                d = (fcx-pcx)**2 + (fcy-pcy)**2
                if d < min_dist:
                    min_dist = d
                    matched_face = f
            # get pose if available
            pose_kps = None
            if self.pose_estimator:
                pose_kps = self.pose_estimator.estimate(frame, p)  # define API accordingly
            assigned.append({'person':p, 'face':matched_face, 'pose':pose_kps})

        # 2) build features per person and predict role
        frame_h, frame_w = frame.shape[:2]
        persons_roles = []
        for a in assigned:
            p = a['person']; f = a['face']; pose = a['pose']
            feat = {}
            feat['body_area'] = bbox_area(p)
            feat['face_area'] = bbox_area(f) if f else 0
            feat['face_body_ratio'] = face_to_body_ratio(f,p) if f else 0
            feat['rel_height'] = relative_height(p, frame_h)
            fx,fy,fw,fh = (f if f else (0,0,0,0))
            bx,by,bw,bh = p
            cx,cy = bbox_center(p)
            feat['pos_x'], feat['pos_y'] = cx/frame_w, cy/frame_h
            feat['pose_vec'] = pose
            feat['scene'] = scene or ''
            role = self.role_clf.predict(feat)
            persons_roles.append({'person':p, 'face':f, 'role':role, 'features':feat})

        # 3) parse user instruction if exists
        user_intent = parse_user_text(user_text) if user_text else None

        # 4) decide which faces to blur
        dm = DecisionModule()  # uses your existing rules
        faces_to_blur = []
        for pr in persons_roles:
            role = pr['role']
            face = pr['face']
            # Apply user intent override rules
            blur = False
            if user_intent:
                if user_intent.get('mode') == 'all':
                    blur = True
                if user_intent.get('target') == 'student' and role == 'student':
                    blur = True
                if user_intent.get('preserve') == 'teacher' and role == 'teacher':
                    blur = False
                # add more mappings...
            else:
                # automatic context-based default
                if scene == 'classroom':
                    if role == 'student' and face: blur = True
                    if role == 'teacher' and face: blur = False
                elif scene == 'street':
                    if face: blur = True
                else:
                    # fallback to decision_module rules (e.g., largest)
                    # convert persons_roles to faces list for decision module
                    pass
            if blur and face:
                faces_to_blur.append(face)

        # fallback: if no user intent and no scene mapping used, use DecisionModule with all faces
        if not faces_to_blur:
            # call your existing decision module with all faces
            all_faces = [pr['face'] for pr in persons_roles if pr['face']]
            # choose mode by scene if available
            mode = 'largest' if scene == 'classroom' else 'all' if scene == 'street' else 'crowd'
            dm = DecisionModule(mode=mode)
            faces_to_blur = dm.analyze(all_faces, frame)

        return faces_to_blur, persons_roles, user_intent
