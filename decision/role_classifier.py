# decision/role_classifier.py
import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "role_mlp.joblib")

class RoleClassifier:
    def __init__(self, use_trained=False):
        self.model = None
        if use_trained and os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        else:
            self.model = None  # fallback to heuristics

    def heuristic_role(self, features):
        """
        features: dict containing:
         - 'face_area', 'body_area', 'face_body_ratio', 'rel_height',
         - 'pos_x', 'pos_y', 'pose_vec' (np.array)
         - 'scene' (optional)
        returns: 'teacher' / 'student' / 'unknown'
        """
        rel_h = features.get('rel_height', 0)
        pos_y = features.get('pos_y', 0.5)  # normalized
        face_body_ratio = features.get('face_body_ratio', 0)
        pose = features.get('pose_vec', np.zeros(5))
        # heuristics
        score = 0
        # teacher likely larger in height (>0.15)
        if rel_h > 0.18: score += 1
        # teacher often near front: pos_y < 0.5
        if pos_y < 0.45: score += 1
        # fewer hands up maybe
        if pose is not None and pose.shape[0] >= 5:
            hands_up = pose[-1]
            if hands_up == 0: score += 1
        # face_body_ratio: teacher face smaller proportionally (standing)
        if face_body_ratio < 0.12: score += 1
        # scene: classroom increases teacher probability
        if features.get('scene','').lower() == 'classroom':
            score += 1

        if score >= 3:
            return 'teacher'
        elif score >= 1:
            return 'unknown'
        else:
            return 'student'

    def predict(self, feature_vector, use_heuristic=True):
        """
        feature_vector: dict as in heuristic_role
        """
        if self.model is not None:
            # prepare vector in same order you trained the model
            X = np.hstack([
                feature_vector.get('rel_height',0),
                feature_vector.get('pos_x',0),
                feature_vector.get('pos_y',0),
                feature_vector.get('face_body_ratio',0),
                np.mean(feature_vector.get('pose_vec', np.zeros(5)))
            ])
            X = np.array(X).reshape(1,-1)
            pred = self.model.predict(X)[0]
            return pred
        else:
            return self.heuristic_role(feature_vector)

    def train(self, X, y):
        """
        Train a small MLP (example).
        X: numpy array shape (n, d)
        y: labels 'teacher'/'student' mapped to ints
        """
        self.model = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500)
        self.model.fit(X,y)
        joblib.dump(self.model, MODEL_PATH)
