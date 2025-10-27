"""
Detection module for integrated multi-person analysis.
"""

from .person_detector import PersonDetector
from .face_detection import FaceDetector
from .attribute_extractor import AttributeExtractor
from .clothing_analyzer import analyze_clothing
from .pose_estimator import PoseEstimator
from .scene_classifier import SceneClassifier

__all__ = [
    'PersonDetector',
    'FaceDetector',
    'AttributeExtractor',
    'analyze_clothing',
    'PoseEstimator',
    'SceneClassifier'
]