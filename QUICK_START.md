# Quick Start Guide

## 🚀 5-Minute Setup

```bash
# 1. Clone/setup project
mkdir detection_project && cd detection_project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install torch torchvision opencv-python ultralytics deepface mediapipe scikit-learn pillow
pip install git+https://github.com/openai/CLIP.git

# 4. Download face detection models
mkdir -p detection/models
cd detection/models
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
cd ../..

# 5. Create __init__.py
touch detection/__init__.py

# 6. Run test
python test_single_image.py your_image.jpg
```

## 📋 Complete File Checklist

```
✓ integrated_detector.py
✓ detection/__init__.py
✓ detection/person_detector.py
✓ detection/face_detector.py
✓ detection/attribute_extractor.py
✓ detection/clothing_analyzer.py
✓ detection/pose_estimator.py
✓ detection/scene_classifier.py
✓ detection/models/deploy.prototxt
✓ detection/models/res10_300x300_ssd_iter_140000.caffemodel
✓ test_single_image.py
✓ test_video.py
✓ requirements.txt
```

## 🎯 Usage Examples

### Example 1: Basic Image Processing

```python
from integrated_detector import IntegratedDetector
import cv2

detector = IntegratedDetector()
frame = cv2.imread("photo.jpg")
results = detector.process_frame(frame)

print(f"Found {len(results['detections'])} people")
for person in results['detections']:
    print(f"Person {person['id']}: {person['attributes']}")
```

### Example 2: Video with Frame Skip

```python
from integrated_detector import IntegratedDetector
import cv2

detector = IntegratedDetector()
cap = cv2.VideoCapture("video.mp4")

frame_count = 0
skip_frames = 5  # Process every 5th frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_count % skip_frames == 0:
        results = detector.process_frame(frame, frame_count)
        print(f"Frame {frame_count}: {len(results['detections'])} people")
    
    frame_count += 1

cap.release()
```

### Example 3: Real-time Webcam

```python
from integrated_detector import IntegratedDetector
import cv2

detector = IntegratedDetector()
cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = detector.process_frame(frame)
    vis_frame = detector.visualize_results(frame, results)
    
    cv2.imshow('Real-time Detection', vis_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## ⚡ Performance Optimization Strategies

### 1. **Multi-Threading for Video Processing**

```python
import concurrent.futures
import cv2
from integrated_detector import IntegratedDetector

def process_frame_wrapper(args):
    frame, frame_id, detector = args
    return detector.process_frame(frame, frame_id)

# Read all frames first
cap = cv2.VideoCapture("video.mp4")
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

# Process in parallel
detector = IntegratedDetector()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    args = [(frame, i, detector) for i, frame in enumerate(frames)]
    results = list(executor.map(process_frame_wrapper, args))
```

### 2. **Batch Processing with GPU**

```python
# Process multiple images at once
import glob
from integrated_detector import IntegratedDetector

detector = IntegratedDetector(device='cuda')

image_paths = glob.glob("images/*.jpg")
batch_size = 4

for i in range(0, len(image_paths), batch_size):
    batch = image_paths[i:i+batch_size]
    
    for path in batch:
        frame = cv2.imread(path)
        results = detector.process_frame(frame)
        # Process results...
```

### 3. **Reduce Resolution**

```python
def resize_for_processing(frame, max_width=1280):
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    return frame

# Use it
frame = cv2.imread("large_image.jpg")
frame = resize_for_processing(frame, max_width=1280)
results = detector.process_frame(frame)
```

### 4. **Selective Component Execution**

```python
class FastDetector(IntegratedDetector):
    """Faster version - skip expensive operations"""
    
    def process_frame(self, frame, frame_id=0, 
                     detect_pose=False,
                     detect_scene=False):
        """
        Custom processing with optional components
        """
        # Person detection (required)
        people_bboxes = self.person_detector.detect_people(frame)
        
        detections = []
        for idx, person_bbox in enumerate(people_bboxes):
            person_data = {
                "id": idx + 1,
                "bbox_person": list(person_bbox),
            }
            
            # Face + attributes (fast)
            face_bbox = self.detect_face_in_person(frame, person_bbox)
            if face_bbox:
                person_data["bbox_face"] = face_bbox
                fx, fy, fw, fh = face_bbox
                face_crop = frame[fy:fy+fh, fx:fx+fw]
                person_data["attributes"] = self.attribute_extractor.analyze(face_crop)
            
            # Clothing (medium speed)
            x, y, w, h = person_bbox
            clothing = analyze_clothing(frame, [x, y, x+w, y+h])
            person_data["cloth"] = clothing["clothing_type"]
            person_data["dress_color"] = clothing["color"]
            
            # Pose (optional - slow)
            if detect_pose:
                kps = self.pose_estimator.estimate(frame, person_bbox)
                person_data["pose"] = self.classify_pose(kps, person_bbox)
            
            detections.append(person_data)
        
        result = {
            "frame_id": frame_id,
            "detections": detections
        }
        
        # Scene (optional - slow)
        if detect_scene:
            result["scene"] = self.scene_classifier.predict_scene(frame)
        
        return result

# Usage
fast_detector = FastDetector()
results = fast_detector.process_frame(
    frame, 
    detect_pose=False,  # Skip pose for speed
    detect_scene=False  # Skip scene for speed
)
```

### 5. **Caching for Video**

```python
class CachedDetector(IntegratedDetector):
    """Cache scene classification for video"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scene_cache = None
        self.scene_cache_frames = 30  # Re-classify every 30 frames
    
    def process_frame(self, frame, frame_id=0):
        # Classify scene only every N frames
        if self.scene_cache is None or frame_id % self.scene_cache_frames == 0:
            self.scene_cache = self.scene_classifier.predict_scene(frame)
        
        # Rest of processing...
        people_bboxes = self.person_detector.detect_people(frame)
        detections = []
        
        for idx, person_bbox in enumerate(people_bboxes):
            # ... process each person ...
            pass
        
        return {
            "frame_id": frame_id,
            "scene": self.scene_cache,
            "detections": detections
        }
```

## 📊 Performance Comparison

| Configuration | Speed (fps) | Accuracy | Use Case |
|--------------|-------------|----------|----------|
| **Full Pipeline (GPU)** | 3-5 | Highest | Production, offline analysis |
| **Full Pipeline (CPU)** | 0.5-1 | Highest | Offline, batch processing |
| **Without Pose (GPU)** | 5-8 | High | Real-time with good GPU |
| **Without Scene (GPU)** | 4-6 | High | Person-focused analysis |
| **Minimal (GPU)** | 10-15 | Medium | Real-time, surveillance |
| **With Frame Skip x3** | 10-15 | Medium | Video analysis |
| **Low Res 720p (GPU)** | 8-12 | Medium-High | Fast processing |

**Minimal = Person + Face + Attributes only**

## 🎨 Custom Visualization

```python
def custom_visualize(frame, results):
    """Custom visualization with different style"""
    vis = frame.copy()
    
    for det in results['detections']:
        x, y, w, h = det['bbox_person']
        
        # Draw semi-transparent rectangle
        overlay = vis.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
        
        # Draw border
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # Info box with background
        if det['attributes']:
            info = f"{det['attributes']['gender']}, {det['attributes']['age']}"
            (text_w, text_h), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(vis, (x, y-text_h-10), (x+text_w+10, y), (0, 255, 0), -1)
            cv2.putText(vis, info, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 0, 0), 2)
    
    return vis
```

## 🔍 Debugging Tips

### Enable Detailed Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# In your code
logger.debug(f"Processing frame {frame_id}")
logger.info(f"Detected {len(people)} persons")
```

### Profile Performance

```python
import time

class ProfiledDetector(IntegratedDetector):
    def process_frame(self, frame, frame_id=0):
        timings = {}
        
        t0 = time.time()
        scene = self.scene_classifier.predict_scene(frame)
        timings['scene'] = time.time() - t0
        
        t0 = time.time()
        people = self.person_detector.detect_people(frame)
        timings['person_detection'] = time.time() - t0
        
        # ... rest of processing with timing ...
        
        print(f"Timings: {timings}")
        return results

# Use it to identify bottlenecks
```

## 🛠️ Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce resolution, use CPU, or enable frame skip |
| Slow processing | Enable GPU, reduce resolution, skip components |
| Missing detections | Lower confidence thresholds |
| False positives | Increase confidence thresholds |
| Poor clothing detection | Ensure good lighting, clear view of torso |
| Pose errors | Ensure full body visible, good lighting |

## 📦 Deployment Recommendations

### Docker Container

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "integrated_detector.py"]
```

### API Service (Flask)

```python
from flask import Flask, request, jsonify
from integrated_detector import IntegratedDetector
import cv2
import numpy as np

app = Flask(__name__)
detector = IntegratedDetector()

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    results = detector.process_frame(frame)
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

This completes the comprehensive detection system! 🎉