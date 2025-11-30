# AnonVision: Real-Time Context-Aware Video Anonymization System

Complete system for intelligent, selective video anonymization with 15+ techniques, natural language queries powered by Gemini AI, and real-time processing capabilities.

---

## 🌟 Key Features

### Core Capabilities
- ✅ **Real-time video processing** with WebSocket streaming
- ✅ **Multiple input sources**: Webcam, video files, and RTSP streams
- ✅ **15+ anonymization techniques** (blur, pixelate, artistic effects)
- ✅ **AI-powered detection** (faces, bodies, attributes)
- ✅ **Natural language queries** powered by Google Gemini ("blur all children", "anonymize people in red")
- ✅ **Batch image processing** with ZIP export
- ✅ **Optimized performance** with GPU acceleration and frame skipping
- ✅ **RESTful API** for seamless integration

### Anonymization Techniques
1. **Gaussian Blur** - Classic smooth blur effect
2. **Pixelation** - Retro pixel mosaic
3. **Mosaic** - Combined pixelate + blur
4. **Black Box** - Complete blackout
5. **Median Blur** - Edge-preserving blur
6. **Bilateral Filter** - Smart edge-aware blur
7. **Mask Overlay** - Semi-transparent mask
8. **Edge Preserve Blur** - Advanced smoothing
9. **Oil Painting** - Artistic oil paint effect
10. **Cartoon** - Comic-style rendering
11. **Negative** - Color inversion
12. **Grayscale** - Black and white conversion
13. **Sepia** - Vintage tone
14. **Brightness** - Darkening effect
15. **Contrast** - Contrast reduction

### Processing Modes
- **Face Only** - Fast face detection and anonymization
- **Body Only** - Full body detection and anonymization
- **Face & Body** - Combined face and body processing
- **Query Based** - AI-powered natural language filtering with Google Gemini

---

## 📋 Table of Contents

1. [Installation](#-installation)
2. [Quick Start](#-quick-start)
3. [API Usage](#-api-usage)
4. [Real-Time Streaming](#-real-time-streaming)
5. [Natural Language Queries](#-natural-language-queries-with-gemini-ai)
6. [Performance Optimization](#-performance-optimization)
7. [Project Structure](#-project-structure)
8. [Configuration](#-configuration)
9. [Deployment](#-deployment)
10. [Troubleshooting](#-troubleshooting)

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for real-time processing)
- Webcam (for live streaming demos)
- Google Gemini API Key (for query-based processing)

### Step 1: Clone Repository
```bash
git clone https://github.com/shubham-dube/anonvision.git
cd anonvision
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies (Only Required to do this and run)
```bash
pip install -r requirements.txt
```

### Step 4: Setup Gemini API Key (Not required for ZIP FIle)

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## ⚡ Quick Start

### 1. Start the API Server

```bash
python api_server.py

# if only checking real time video anonimization (run after running server, you can change technique also)
python stream_client.py --webcam 0 --mode face_only --technique gaussian_blur
```

Server will start at: `http://localhost:8000`

Website: `http://localhost:8000`

Interactive API Docs: `http://localhost:8000/docs`

### 2. Test with Single Image

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/api/process/image" \
  -F "file=@test_image.jpg" \
  -F "mode=face_only" \
  -F "technique=gaussian_blur" \
  -F "intensity=medium" \
  -o output.jpg
```

**Using Python:**
```python
import requests

with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/process/image',
        files={'file': f},
        data={
            'mode': 'face_only',
            'technique': 'pixelate',
            'intensity': 'high'
        }
    )

with open('output.jpg', 'wb') as f:
    f.write(response.content)
```

### 3. Process Video File

```bash
curl -X POST "http://localhost:8000/api/process/video" \
  -F "file=@input_video.mp4" \
  -F "mode=face_only" \
  -F "technique=mosaic" \
  -F "intensity=medium" \
  -F "frame_skip=1"
```

The response includes a download link to the processed video.

### 4. Real-Time Webcam Streaming

```bash
# Start server first
python api_server.py

# In another terminal, start streaming client
python stream_client.py --webcam 0 --mode face_only --technique gaussian_blur
```

---

## 🔌 API Usage

### Image Processing

**Endpoint:** `POST /api/process/image`

**Parameters:**
- `file` (required): Image file
- `mode` (optional): `face_only`, `body_only`, `face_and_body`, `query_based`
- `technique` (optional): Any technique from the list above
- `intensity` (optional): `low`, `medium`, `high`
- `query` (optional): Natural language query (for query_based mode)

**Example:**
```python
import requests

files = {'file': open('image.jpg', 'rb')}
data = {
    'mode': 'query_based',
    'technique': 'gaussian_blur',
    'intensity': 'medium',
    'query': 'blur all children'
}

response = requests.post(
    'http://localhost:8000/api/process/image',
    files=files,
    data=data
)

with open('output.jpg', 'wb') as f:
    f.write(response.content)
```

### Batch Image Processing

**Endpoint:** `POST /api/process/images`

**Parameters:**
- `files` (required): Multiple image files
- `return_zip` (optional): Set to `true` to get ZIP file, `false` for JSON links
- Other parameters same as single image

**Example - Get ZIP file:**
```python
files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb')),
    ('files', open('image3.jpg', 'rb'))
]

data = {
    'mode': 'face_only',
    'technique': 'pixelate',
    'return_zip': 'true'
}

response = requests.post(
    'http://localhost:8000/api/process/images',
    files=files,
    data=data
)

with open('anonymized_images.zip', 'wb') as f:
    f.write(response.content)
```

**Example - Get JSON links:**
```python
data['return_zip'] = 'false'

response = requests.post(
    'http://localhost:8000/api/process/images',
    files=files,
    data=data
)

results = response.json()
for result in results['results']:
    print(f"{result['filename']}: {result['download_url']}")
```

### Video Processing

**Endpoint:** `POST /api/process/video`

**Parameters:**
- `file` (required): Video file
- `mode`, `technique`, `intensity`: Same as image
- `frame_skip` (optional): Process every Nth frame (default: 1 for best quality)
- `query` (optional): Natural language query

**Example:**
```python
with open('video.mp4', 'rb') as f:
    files = {'file': f}
    data = {
        'mode': 'face_only',
        'technique': 'mosaic',
        'intensity': 'high',
        'frame_skip': 1
    }
    
    response = requests.post(
        'http://localhost:8000/api/process/video',
        files=files,
        data=data
    )
    
    result = response.json()
    print(f"Download: {result['download_url']}")
    print(f"Stats: {result['metadata']}")
```

### Get Available Techniques

**Endpoint:** `GET /api/techniques`

```python
response = requests.get('http://localhost:8000/api/techniques')
techniques = response.json()

for tech in techniques['techniques']:
    print(f"{tech['name']} ({tech['category']})")
```

---

## 🎥 Real-Time Streaming

### WebSocket Protocol

Connect to: `ws://localhost:8000/api/stream/websocket`

**Message Flow:**

1. **Send Configuration:**
```json
{
  "type": "config",
  "config": {
    "mode": "face_only",
    "technique": "gaussian_blur",
    "intensity": "medium"
  }
}
```

2. **Send Frame:**
```json
{
  "type": "frame",
  "frame": "base64_encoded_jpeg_image"
}
```

3. **Receive Processed Frame:**
```json
{
  "type": "frame",
  "frame": "base64_encoded_jpeg_image",
  "metadata": {
    "detections": 3,
    "anonymized": 3,
    "processing_time_ms": 45.2,
    "fps": 22.1
  }
}
```

## 🧠 Natural Language Queries with Gemini AI

AnonVision uses Google Gemini AI to understand natural language queries and intelligently select people for anonymization based on visual characteristics.

### How It Works

1. **Person Detection**: YOLOv8 detects all people in the frame
2. **Crop Extraction**: Each detected person is cropped from the image
3. **Gemini Analysis**: Cropped images are sent to Gemini with your query
4. **Smart Filtering**: Gemini determines which people match your criteria
5. **Selective Anonymization**: Only matching people are anonymized

### Supported Query Types

#### Age-Based Queries (depends on Gemini)
```
"blur all children"              # Ages 0-12
"anonymize teenagers"            # Ages 13-19
"blur all adults"                # Ages 20-64
"hide elderly people"            # Ages 65+
"blur everyone except adults"    # Invert: blur only non-adults
```

#### Gender-Based Queries
```
"blur all males"
"anonymize women"
"hide all men"
"blur everyone except females"
```

#### Emotion-Based Queries
```
"blur happy people"
"anonymize sad faces"
"hide angry individuals"
"blur people who look surprised"
```

#### Clothing-Based Queries
```
"blur people wearing red"
"anonymize anyone in blue"
"hide people in black clothes"
"blur everyone in dark clothing"
```

#### Combined Queries
```
"blur all children wearing red"
"anonymize sad teenagers"
"hide elderly men in blue"
"blur happy people wearing bright colors"
```

### API Usage with Queries

```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/process/image',
        files={'file': f},
        data={
            'mode': 'query_based',
            'technique': 'gaussian_blur',
            'intensity': 'high',
            'query': 'blur all children wearing red shirts'
        }
    )

with open('output.jpg', 'wb') as f:
    f.write(response.content)
```

### Query Processing Performance

Query-based processing requires AI analysis, which is slower than standard detection:

- **Face Only**: ~30-50ms per frame
- **Body Only**: ~40-60ms per frame  
- **Query Based**: ~1500-5000 per frame (depends on number of people)

**Optimization Tips:**
- Use specific queries to reduce false positives
- Increase `frame_skip` for videos (e.g., process every 2-3 frames)
- Use standard modes when queries aren't needed
- Consider batch processing for large videos

---

## ⚡ Performance Optimization

### Frame Skipping

Process every Nth frame to speed up video processing:

```python
# Fast: Process every 3rd frame
data = {'frame_skip': 3, 'mode': 'face_only'}

# Balanced: Every 2nd frame
data = {'frame_skip': 2, 'mode': 'face_only'}

# High quality: Every frame (default)
data = {'frame_skip': 1, 'mode': 'face_only'}
```

### Selective Processing Mode

Choose the minimal mode for your needs:

```python
# Fastest: Face detection only (~30ms/frame)
'mode': 'face_only'

# Medium: Body detection only (~80ms/frame)
'mode': 'body_only'

# Slower: Both detections (~80ms/frame)
'mode': 'face_and_body'

# Slowest: AI-powered queries (~1500-5000ms/frame)
'mode': 'query_based'
```

### GPU Acceleration

The system automatically detects and uses CUDA:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

To force CPU mode:
```python
from processor import ProcessingConfig
config = ProcessingConfig(use_gpu=False)
```

### Detection Confidence

Reduce false positives and improve speed:

```python
config = ProcessingConfig(
    confidence_threshold=0.7,  # Higher = fewer detections
    min_face_size=50           # Skip tiny faces
)
```

### Benchmark Results

Tested on NVIDIA RTX 3060, 720 video:

| Mode | Technique | Frame Skip | FPS |
|------|-----------|------------|-----|
| face_only | gaussian_blur | 1 | 28 |
| face_only | gaussian_blur | 2 | 45 |
| face_only | pixelate | 2 | 47 |
| face_and_body | mosaic | 2 | 25 |
| query_based | gaussian_blur | 3 | 12 |
| query_based | pixelate | 3 | 15 |

---

## 📁 Project Structure

```
anonvision/
├── detection/                      # Detection modules
│   ├── __init__.py
│   ├── person_detector.py         # YOLOv8 person detection
│   ├── face_detection.py          # OpenCV DNN face detection
│   ├── attribute_extractor.py     # DeepFace attributes (fallback)
│   ├── gemini_analyzer.py         # Google Gemini AI integration
│   ├── clothing_analyzer.py       # Color extraction (fallback)
│   ├── scene_classifier.py        # Places365 scene context
│   └── models/                    # Model files (auto-download)
│
├── processor.py                   # Core processing engine
├── api_server.py                 # FastAPI server
├── stream_client.py              # Real-time streaming client
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (create this)
├── README.md                     # This file
│
├── static/                       # Web interface
│   └── index.html
│
├── uploads/                      # Temporary upload storage
├── outputs/                      # Processed outputs
└── tests/                        # Test files
    ├── test_image.jpg
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required: Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here
```

### Advanced Configuration

Edit `processor.py` for fine-tuning:

```python
from processor import ProcessingConfig, ProcessingMode, AnonymizationTechnique

config = ProcessingConfig(
    mode=ProcessingMode.FACE_ONLY,
    technique=AnonymizationTechnique.GAUSSIAN_BLUR,
    intensity='medium',
    frame_skip=1,
    face_padding=0.15,           # 15% padding around faces
    body_padding=0.05,            # 5% padding around bodies
    require_context=False,        # Enable scene classification
    require_attributes=False,     # Enable attribute extraction
    min_face_size=30,             # Minimum face size in pixels
    confidence_threshold=0.5,     # Detection confidence
    use_gpu=True,                 # Use GPU if available
    query=None                    # Natural language query
)
```

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
Solution: Reduce batch size or use CPU mode
config = ProcessingConfig(use_gpu=False)
```

**2. Gemini API Key Error**
```
Error: "API key not found"
Solution: Create .env file with GEMINI_API_KEY=your_key_here
```

**3. Webcam Not Found**
```
Solution: Check device ID
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

**4. Slow Processing**
```
Solutions:
- Increase frame_skip for videos
- Use simpler techniques (gaussian_blur, pixelate)
- Use face_only mode instead of query_based
- Enable GPU acceleration
- Reduce video resolution
```

**5. WebSocket Connection Failed**
```
Solutions:
- Verify server is running: python api_server.py
- Check firewall settings
- Use correct URL: ws://localhost:8000/api/stream/websocket
```

**6. Query Not Working**
```
Solutions:
- Verify GEMINI_API_KEY is set in .env
- Check your Gemini API quota
- Use simpler, clearer queries
- Try the fallback rule-based system by removing API key
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📞 Support

- **GitHub Issues**: [Create Issue](https://github.com/shubham-dube/anonvision/issues)
- **Documentation**: [Wiki](https://github.com/shubham-dube/anonvision)
- **Email**: itshubhamofficial@gmail.com

---

## 🎯 Roadmap

- [x] Real-time video processing
- [x] Natural language queries with Gemini AI
- [x] 15+ anonymization techniques
- [x] Batch processing with ZIP export
- [ ] GPU batch processing optimization
- [ ] Cloud storage integration (S3, GCS)
- [ ] Multi-language support
- [ ] Mobile app (React Native)
- [ ] Advanced pose-based filtering
- [ ] Custom model training interface
- [ ] Video analytics dashboard
- [ ] RTMP output streaming
- [ ] Face recognition whitelist/blacklist

---

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics for person detection
- **OpenCV** for face detection and image processing
- **Google Gemini AI** for intelligent query understanding
- **DeepFace** for attribute extraction (fallback)
- **FastAPI** for the REST API framework
- **PyTorch** for GPU acceleration

---

## ⭐ Star History

If you find this project useful, please consider giving it a star!

---

**Built with ❤️ for Privacy and Innovation**

Made possible by cutting-edge AI and computer vision technologies.