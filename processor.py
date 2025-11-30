"""
AnonVision Core Processing Engine
Optimized for real-time performance with selective computation
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time

# Lazy imports for optional modules
_face_detector = None
_person_detector = None
_attribute_extractor = None
_scene_classifier = None


class ProcessingMode(Enum):
    FACE_ONLY = "face_only"
    BODY_ONLY = "body_only"
    FACE_AND_BODY = "face_and_body"
    QUERY_BASED = "query_based"


class AnonymizationTechnique(Enum):
    GAUSSIAN_BLUR = "gaussian_blur"
    PIXELATE = "pixelate"
    MOSAIC = "mosaic"
    BLACK_BOX = "black_box"
    MEDIAN_BLUR = "median_blur"
    BILATERAL_FILTER = "bilateral_filter"
    MASK_OVERLAY = "mask_overlay"
    EDGE_PRESERVE_BLUR = "edge_preserve_blur"
    OIL_PAINTING = "oil_painting"
    CARTOON = "cartoon"
    NEGATIVE = "negative"
    GRAYSCALE = "grayscale"
    SEPIA = "sepia"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"


@dataclass
class ProcessingConfig:
    """Configuration for processing pipeline"""
    mode: ProcessingMode = ProcessingMode.FACE_ONLY
    technique: AnonymizationTechnique = AnonymizationTechnique.GAUSSIAN_BLUR
    intensity: str = "medium"  # low, medium, high
    frame_skip: int = 2  # Process every Nth frame for video
    face_padding: float = 0.15
    body_padding: float = 0.05
    require_context: bool = False  # Only extract context if needed
    require_attributes: bool = False  # Only extract attributes if needed
    min_face_size: int = 30  # Skip tiny faces
    confidence_threshold: float = 0.5
    query: Optional[str] = None
    use_gpu: bool = torch.cuda.is_available()


class LazyDetectorLoader:
    """Load detectors only when needed to save memory and startup time"""
    
    @staticmethod
    def get_face_detector():
        global _face_detector
        if _face_detector is None:
            from detection.face_detection import FaceDetector
            _face_detector = FaceDetector()
        return _face_detector
    
    @staticmethod
    def get_person_detector():
        global _person_detector
        if _person_detector is None:
            from detection.person_detector import PersonDetector
            _person_detector = PersonDetector()
        return _person_detector
    
    @staticmethod
    def get_attribute_extractor():
        global _attribute_extractor
        if _attribute_extractor is None:
            from detection.attribute_extractor import AttributeExtractor
            _attribute_extractor = AttributeExtractor()
        return _attribute_extractor
    
    @staticmethod
    def get_scene_classifier():
        global _scene_classifier
        if _scene_classifier is None:
            from detection.scene_classifier import SceneClassifier
            _scene_classifier = SceneClassifier()
        return _scene_classifier


class AnonymizationEngine:
    """Efficient anonymization techniques with intensity control"""
    
    INTENSITY_PARAMS = {
        'gaussian_blur': {'low': 11, 'medium': 25, 'high': 45},
        'pixelate': {'low': 20, 'medium': 12, 'high': 6},
        'median_blur': {'low': 9, 'medium': 15, 'high': 25},
        'bilateral': {'low': (5, 50, 50), 'medium': (9, 75, 75), 'high': (15, 100, 100)},
        'oil_painting': {'low': (3, 1), 'medium': (7, 1), 'high': (10, 2)},
    }
    
    @staticmethod
    def apply_technique(roi: np.ndarray, technique: AnonymizationTechnique, 
                       intensity: str = "medium") -> np.ndarray:
        """Apply anonymization technique to ROI"""
        
        if technique == AnonymizationTechnique.GAUSSIAN_BLUR:
            ksize = AnonymizationEngine.INTENSITY_PARAMS['gaussian_blur'][intensity]
            ksize = ksize if ksize % 2 == 1 else ksize + 1
            return cv2.GaussianBlur(roi, (ksize, ksize), 0)
        
        elif technique == AnonymizationTechnique.PIXELATE:
            pixel_size = AnonymizationEngine.INTENSITY_PARAMS['pixelate'][intensity]
            h, w = roi.shape[:2]
            temp = cv2.resize(roi, (w // pixel_size, h // pixel_size), 
                            interpolation=cv2.INTER_LINEAR)
            return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        
        elif technique == AnonymizationTechnique.MOSAIC:
            # Pixelate + slight blur
            pixelated = AnonymizationEngine.apply_technique(
                roi, AnonymizationTechnique.PIXELATE, intensity
            )
            return cv2.GaussianBlur(pixelated, (5, 5), 0)
        
        elif technique == AnonymizationTechnique.BLACK_BOX:
            return np.zeros_like(roi)
        
        elif technique == AnonymizationTechnique.MEDIAN_BLUR:
            ksize = AnonymizationEngine.INTENSITY_PARAMS['median_blur'][intensity]
            ksize = ksize if ksize % 2 == 1 else ksize + 1
            return cv2.medianBlur(roi, ksize)
        
        elif technique == AnonymizationTechnique.BILATERAL_FILTER:
            d, sc, ss = AnonymizationEngine.INTENSITY_PARAMS['bilateral'][intensity]
            return cv2.bilateralFilter(roi, d, sc, ss)
        
        elif technique == AnonymizationTechnique.MASK_OVERLAY:
            # Create semi-transparent colored mask
            mask = np.ones_like(roi) * [128, 128, 128]
            alpha = 0.7 if intensity == "low" else 0.85 if intensity == "medium" else 0.95
            return cv2.addWeighted(roi, 1 - alpha, mask, alpha, 0)
        
        elif technique == AnonymizationTechnique.EDGE_PRESERVE_BLUR:
            # Edge-preserving smoothing
            flags = cv2.RECURS_FILTER if intensity == "high" else cv2.NORMCONV_FILTER
            return cv2.edgePreservingFilter(roi, flags=flags, sigma_s=60, sigma_r=0.4)
        
        elif technique == AnonymizationTechnique.OIL_PAINTING:
            size, dyn = AnonymizationEngine.INTENSITY_PARAMS['oil_painting'][intensity]
            return cv2.xphoto.oilPainting(roi, size, dyn)
        
        elif technique == AnonymizationTechnique.CARTOON:
            # Cartoon effect using bilateral filter + edge detection
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(roi, 9, 250, 250)
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            return cartoon
        
        elif technique == AnonymizationTechnique.NEGATIVE:
            return cv2.bitwise_not(roi)
        
        elif technique == AnonymizationTechnique.GRAYSCALE:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        elif technique == AnonymizationTechnique.SEPIA:
            kernel = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
            sepia = cv2.transform(roi, kernel)
            return np.clip(sepia, 0, 255).astype(np.uint8)
        
        elif technique == AnonymizationTechnique.BRIGHTNESS:
            factor = 0.7 if intensity == "low" else 0.5 if intensity == "medium" else 0.3
            return cv2.convertScaleAbs(roi, alpha=factor, beta=0)
        
        elif technique == AnonymizationTechnique.CONTRAST:
            factor = 0.5 if intensity == "low" else 0.3 if intensity == "medium" else 0.1
            return cv2.convertScaleAbs(roi, alpha=factor, beta=128)
        
        return roi


class QueryParser:
    """Lightweight NLP query parser for anonymization rules"""
    
    # Age-related keywords
    AGE_KEYWORDS = {
        'child': (0, 12), 'children': (0, 12), 'kid': (0, 12), 'kids': (0, 12),
        'teen': (13, 19), 'teenager': (13, 19), 'teens': (13, 19),
        'adult': (20, 64), 'adults': (20, 64),
        'elderly': (65, 120), 'senior': (65, 120), 'old': (65, 120)
    }
    
    # Gender keywords
    GENDER_KEYWORDS = ['male', 'female', 'man', 'woman', 'men', 'women', 'boy', 'girl']
    
    # Emotion keywords
    EMOTION_KEYWORDS = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']
    
    # Clothing color keywords
    COLOR_KEYWORDS = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 
                     'orange', 'purple', 'pink', 'brown']
    
    @staticmethod
    def parse(query: str) -> Dict[str, Any]:
        """
        Parse natural language query into filtering rules
        
        Examples:
            "blur all children" -> {age_range: (0, 12)}
            "anonymize people wearing red" -> {clothing_color: "red"}
            "blur all except adults" -> {age_range: (20, 64), invert: True}
        """
        query = query.lower().strip()
        rules = {'mode': 'filter', 'filters': []}
        
        # Check for "all" or "everyone"
        if 'all' in query or 'everyone' in query or 'everybody' in query:
            if 'except' not in query:
                return {'mode': 'all'}
        
        # Check for "none" or "nobody"
        if 'none' in query or 'nobody' in query or 'no one' in query:
            return {'mode': 'none'}
        
        # Parse age filters
        for keyword, age_range in QueryParser.AGE_KEYWORDS.items():
            if keyword in query:
                rules['filters'].append({'type': 'age', 'range': age_range})
        
        # Parse gender filters
        for gender in QueryParser.GENDER_KEYWORDS:
            if gender in query:
                normalized = 'male' if gender in ['male', 'man', 'men', 'boy'] else 'female'
                rules['filters'].append({'type': 'gender', 'value': normalized})
        
        # Parse emotion filters
        for emotion in QueryParser.EMOTION_KEYWORDS:
            if emotion in query:
                rules['filters'].append({'type': 'emotion', 'value': emotion})
        
        # Parse clothing color
        for color in QueryParser.COLOR_KEYWORDS:
            if color in query:
                rules['filters'].append({'type': 'clothing_color', 'value': color})
        
        # Check for inversion (except, exclude)
        if 'except' in query or 'exclude' in query or 'but not' in query:
            rules['invert'] = True
        
        return rules if rules['filters'] else {'mode': 'all'}


class AnonVisionProcessor:
    """Main processing engine optimized for real-time performance"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.loader = LazyDetectorLoader()
        self.engine = AnonymizationEngine()
        self.parser = QueryParser()
        
        # Performance tracking
        self.frame_count = 0
        self.total_time = 0
        
    def process_frame(self, frame: np.ndarray, force_process: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Process single frame with selective computation
        
        Args:
            frame: Input frame (BGR)
            force_process: Force processing even if frame_skip says skip
            
        Returns:
            (processed_frame, metadata)
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Frame skipping for video optimization
        if not force_process and self.frame_count % self.config.frame_skip != 0:
            return frame, {'skipped': True, 'frame_id': self.frame_count}
        
        metadata = {
            'frame_id': self.frame_count,
            'skipped': False,
            'detections': 0,
            'anonymized': 0
        }
        
        result_frame = frame.copy()
        
        # Determine what to detect based on mode
        if self.config.mode == ProcessingMode.FACE_ONLY:
            result_frame = self._process_faces_only(result_frame, metadata)
        
        elif self.config.mode == ProcessingMode.BODY_ONLY:
            result_frame = self._process_bodies_only(result_frame, metadata)
        
        elif self.config.mode == ProcessingMode.FACE_AND_BODY:
            result_frame = self._process_faces_and_bodies(result_frame, metadata)
        
        elif self.config.mode == ProcessingMode.QUERY_BASED:
            result_frame = self._process_query_based(result_frame, metadata)
        
        # Performance metrics
        elapsed = time.time() - start_time
        self.total_time += elapsed
        metadata['processing_time_ms'] = round(elapsed * 1000, 2)
        metadata['fps'] = round(1 / elapsed, 2) if elapsed > 0 else 0
        
        return result_frame, metadata
    
    def _process_faces_only(self, frame: np.ndarray, metadata: Dict) -> np.ndarray:
        """Fast face-only processing"""
        face_detector = self.loader.get_face_detector()
        faces = face_detector.detect(frame)
        
        metadata['detections'] = len(faces)
        
        for face_bbox in faces:
            x, y, w, h = face_bbox
            
            # Skip tiny faces
            if w < self.config.min_face_size or h < self.config.min_face_size:
                continue
            
            # Apply padding
            pad_w = int(w * self.config.face_padding)
            pad_h = int(h * self.config.face_padding)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(frame.shape[1], x + w + pad_w)
            y2 = min(frame.shape[0], y + h + pad_h)
            
            roi = frame[y1:y2, x1:x2]
            
            if roi.size > 0:
                anonymized_roi = self.engine.apply_technique(
                    roi, self.config.technique, self.config.intensity
                )
                frame[y1:y2, x1:x2] = anonymized_roi
                metadata['anonymized'] += 1
        
        return frame
    
    def _process_bodies_only(self, frame: np.ndarray, metadata: Dict) -> np.ndarray:
        """Process full body regions"""
        person_detector = self.loader.get_person_detector()
        people = person_detector.detect_people(frame)
        
        metadata['detections'] = len(people)
        
        for person_bbox in people:
            x, y, w, h = person_bbox
            
            # Apply padding
            pad_w = int(w * self.config.body_padding)
            pad_h = int(h * self.config.body_padding)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(frame.shape[1], x + w + pad_w)
            y2 = min(frame.shape[0], y + h + pad_h)
            
            roi = frame[y1:y2, x1:x2]
            
            if roi.size > 0:
                anonymized_roi = self.engine.apply_technique(
                    roi, self.config.technique, self.config.intensity
                )
                frame[y1:y2, x1:x2] = anonymized_roi
                metadata['anonymized'] += 1
        
        return frame
    
    def _process_faces_and_bodies(self, frame: np.ndarray, metadata: Dict) -> np.ndarray:
        """Process both faces and bodies"""
        # First detect people
        person_detector = self.loader.get_person_detector()
        people = person_detector.detect_people(frame)
        
        # Then detect faces
        face_detector = self.loader.get_face_detector()
        faces = face_detector.detect(frame)
        
        metadata['detections'] = len(people) + len(faces)
        
        # Anonymize bodies first (larger regions)
        for person_bbox in people:
            x, y, w, h = person_bbox
            pad_w = int(w * self.config.body_padding)
            pad_h = int(h * self.config.body_padding)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(frame.shape[1], x + w + pad_w)
            y2 = min(frame.shape[0], y + h + pad_h)
            
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                anonymized_roi = self.engine.apply_technique(
                    roi, self.config.technique, self.config.intensity
                )
                frame[y1:y2, x1:x2] = anonymized_roi
                metadata['anonymized'] += 1
        
        # Then anonymize faces with potentially different technique
        for face_bbox in faces:
            x, y, w, h = face_bbox
            if w < self.config.min_face_size or h < self.config.min_face_size:
                continue
            
            pad_w = int(w * self.config.face_padding)
            pad_h = int(h * self.config.face_padding)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(frame.shape[1], x + w + pad_w)
            y2 = min(frame.shape[0], y + h + pad_h)
            
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                anonymized_roi = self.engine.apply_technique(
                    roi, self.config.technique, self.config.intensity
                )
                frame[y1:y2, x1:x2] = anonymized_roi
        
        return frame
    
    def _process_query_based(self, frame: np.ndarray, metadata: Dict) -> np.ndarray:
        """Query-based selective anonymization"""
        if not self.config.query:
            return frame
        
        # Parse query
        rules = self.parser.parse(self.config.query)
        
        # If mode is 'all' or 'none', use simpler processing
        if rules.get('mode') == 'all':
            return self._process_faces_and_bodies(frame, metadata)
        elif rules.get('mode') == 'none':
            metadata['detections'] = 0
            metadata['anonymized'] = 0
            return frame
        
        # Need context - detect people and analyze
        person_detector = self.loader.get_person_detector()
        face_detector = self.loader.get_face_detector()
        people = person_detector.detect_people(frame)
        
        metadata['detections'] = len(people)
        
        # For each person, check if they match filters
        for person_bbox in people:
            x, y, w, h = person_bbox
            
            # Detect face within person
            person_crop = frame[y:y+h, x:x+w]
            faces_in_person = face_detector.detect(person_crop)
            
            should_anonymize = False
            
            if faces_in_person:
                # Extract attributes only if needed
                fx, fy, fw, fh = faces_in_person[0]
                face_crop = person_crop[fy:fy+fh, fx:fx+fw]
                
                if face_crop.size > 0:
                    # Check which attributes we need
                    need_age = any(f['type'] == 'age' for f in rules.get('filters', []))
                    need_gender = any(f['type'] == 'gender' for f in rules.get('filters', []))
                    need_emotion = any(f['type'] == 'emotion' for f in rules.get('filters', []))
                    need_color = any(f['type'] == 'clothing_color' for f in rules.get('filters', []))
                    
                    if need_age or need_gender or need_emotion:
                        extractor = self.loader.get_attribute_extractor()
                        attrs = extractor.analyze(face_crop)
                        
                        # Check each filter
                        for filter_rule in rules.get('filters', []):
                            if filter_rule['type'] == 'age':
                                age = attrs.get('age', 0)
                                min_age, max_age = filter_rule['range']
                                if min_age <= age <= max_age:
                                    should_anonymize = True
                            
                            elif filter_rule['type'] == 'gender':
                                gender = attrs.get('gender', '').lower()
                                if filter_rule['value'] in gender:
                                    should_anonymize = True
                            
                            elif filter_rule['type'] == 'emotion':
                                emotion = attrs.get('dominant_emotion', '').lower()
                                if filter_rule['value'] in emotion:
                                    should_anonymize = True
                    
                    if need_color:
                        # Import clothing analyzer only if needed
                        from detection.clothing_analyzer import analyze_clothing
                        clothing = analyze_clothing(frame, [x, y, x+w, y+h])
                        color = clothing.get('color', '').lower()
                        
                        for filter_rule in rules.get('filters', []):
                            if filter_rule['type'] == 'clothing_color':
                                if filter_rule['value'] in color:
                                    should_anonymize = True
            
            # Handle inversion
            if rules.get('invert', False):
                should_anonymize = not should_anonymize
            
            # Anonymize if matches
            if should_anonymize:
                pad_w = int(w * self.config.body_padding)
                pad_h = int(h * self.config.body_padding)
                
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(frame.shape[1], x + w + pad_w)
                y2 = min(frame.shape[0], y + h + pad_h)
                
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    anonymized_roi = self.engine.apply_technique(
                        roi, self.config.technique, self.config.intensity
                    )
                    frame[y1:y2, x1:x2] = anonymized_roi
                    metadata['anonymized'] += 1
        
        return frame
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.frame_count = 0
        self.total_time = 0
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        avg_time = self.total_time / self.frame_count if self.frame_count > 0 else 0
        avg_fps = 1 / avg_time if avg_time > 0 else 0
        
        return {
            'frames_processed': self.frame_count,
            'total_time_seconds': round(self.total_time, 2),
            'avg_processing_time_ms': round(avg_time * 1000, 2),
            'avg_fps': round(avg_fps, 2)
        }