# AnonVision Quick Start Guide

Get started with AnonVision in 5 minutes!

## 🚀 Fast Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/anonvision.git
cd anonvision

# 2. Run automated setup
chmod +x setup.sh
./setup.sh

# 3. Activate environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

## ⚡ Quick Test

### Option 1: CLI Tool (Easiest)

```bash
# Install CLI dependencies
pip install rich click

# Run interactive demo
python anonvision_cli.py demo

# Process an image
python anonvision_cli.py process-image input.jpg output.jpg

# Start webcam stream
python anonvision_cli.py stream --webcam 0

# Get help
python anonvision_cli.py --help
```

### Option 2: API Server

**Terminal 1 - Start Server:**
```bash
python api_server.py
```

**Terminal 2 - Test API:**
```bash
# Process image
curl -X POST http://localhost:8000/api/process/image \
  -F "file=@test.jpg" \
  -F "mode=face_only" \
  -o output.jpg

# Get available techniques
curl http://localhost:8000/api/techniques
```

### Option 3: Real-Time Streaming

**Terminal 1 - Start Server:**
```bash
python api_server.py
```

**Terminal 2 - Start Streaming:**
```bash
# Webcam
python stream_client.py --webcam 0 --mode face_only

# Video file
python stream_client.py --video input.mp4 --save output.mp4

# RTSP stream
python stream_client.py --rtsp rtsp://camera.local/stream
```

## 🎯 Common Use Cases

### 1. Blur All Faces in Image
```bash
python anonvision_cli.py process-image \
  input.jpg output.jpg \
  --mode face_only \
  --technique gaussian_blur \
  --intensity medium
```

### 2. Pixelate Full Bodies in Video
```bash
python anonvision_cli.py process-video \
  input.mp4 output.mp4 \
  --mode body_only \
  --technique pixelate \
  --intensity high \
  --frame-skip 2
```

### 3. Smart Query: "Blur All Children"
```bash
python anonvision_cli.py process-image \
  input.jpg output.jpg \
  --mode query_based \
  --query "blur all children"
```

### 4. Real-Time Webcam with Mosaic Effect
```bash
python stream_client.py \
  --webcam 0 \
  --technique mosaic \
  --intensity medium \
  --save recording.mp4
```

### 5. Batch Process Directory
```bash
python anonvision_cli.py process-batch \
  ./input_images/ \
  ./output_images/ \
  --technique gaussian_blur
```

## 📝 Available Techniques

| Technique | Description | Speed | Quality |
|-----------|-------------|-------|---------|
| `gaussian_blur` | Classic smooth blur | Fast | Good |
| `pixelate` | Retro pixel effect | Fast | Good |
| `mosaic` | Pixelate + blur | Fast | Better |
| `black_box` | Complete blackout | Fastest | N/A |
| `median_blur` | Edge-preserving | Medium | Good |
| `bilateral_filter` | Smart blur | Slow | Best |
| `cartoon` | Comic style | Slow | Unique |
| `oil_painting` | Artistic | Slow | Artistic |

See all 15 techniques:
```bash
python anonvision_cli.py techniques
```

## 🎮 Processing Modes

### `face_only` - Fastest
- Only detects and anonymizes faces
- Best for real-time performance
- Use when: You only need face privacy

### `body_only` - Fast
- Anonymizes full body regions
- Good for silhouette privacy
- Use when: Facial features aren't the concern

### `face_and_body` - Balanced
- Processes both faces and bodies
- Moderate performance
- Use when: Maximum privacy needed

### `query_based` - Flexible
- Uses natural language queries
- Slowest (requires attribute extraction)
- Use when: Selective filtering needed

## 💡 Tips for Best Performance

### Real-Time Video (30+ FPS)
```bash
# Use fastest mode and technique
python stream_client.py \
  --webcam 0 \
  --mode face_only \
  --technique pixelate \
  --intensity low
```

### High Quality (10-15 FPS)
```bash
# Use better techniques with frame skip
python stream_client.py \
  --video input.mp4 \
  --mode face_and_body \
  --technique bilateral_filter \
  --intensity high
```

### Batch Processing
```bash
# Process multiple images efficiently
python anonvision_cli.py process-batch \
  ./photos/ ./anonymized/ \
  --mode face_only \
  --technique gaussian_blur
```

## 🔍 Troubleshooting

### Server Won't Start
```bash
# Check if port is in use
lsof -i :8000

# Use different port
python api_server.py --port 8080
```

### Webcam Not Found
```bash
# List available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# Use specific camera
python stream_client.py --webcam 1
```

### Slow Processing
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Increase frame skip
--frame-skip 3

# Use faster technique
--technique pixelate
```

### Out of Memory
```bash
# Reduce resolution before processing
# Or increase frame skip
--frame-skip 5
```

## 📚 Next Steps

1. **Read Full Documentation**: See `README.md`
2. **Try Examples**: Run `python examples.py`
3. **API Integration**: Check `http://localhost:8000/docs`
4. **Custom Techniques**: Edit `processor.py`
5. **Deploy**: See deployment section in README

## 🆘 Getting Help

- **Documentation**: Full README.md
- **API Docs**: http://localhost:8000/docs
- **Examples**: examples.py
- **Issues**: GitHub Issues
- **CLI Help**: `python anonvision_cli.py --help`

---

**Ready to start? Run the interactive demo:**
```bash
python anonvision_cli.py demo
```

🎉 **You're all set! Have fun with AnonVision!**