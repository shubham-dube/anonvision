#!/bin/bash

# AnonVision Setup Script
# Automated installation and configuration

set -e

echo "======================================"
echo "  AnonVision Setup Script"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "⚠️  Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Create directories
echo "Creating project directories..."
mkdir -p detection/models
mkdir -p uploads
mkdir -p outputs
mkdir -p tests
echo "✅ Directories created"
echo ""

# Download models
echo "Downloading detection models..."
cd detection/models

# Face detection model
if [ ! -f "res10_300x300_ssd_iter_140000.caffemodel" ]; then
    echo "Downloading face detection model..."
    wget -q --show-progress https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
    echo "✅ Face model downloaded"
fi

if [ ! -f "deploy.prototxt" ]; then
    echo "Downloading model configuration..."
    wget -q --show-progress https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
    echo "✅ Prototxt downloaded"
fi

# Places365 model (large file - optional for now)
# if [ ! -f "resnet50_places365.pth.tar" ]; then
#     echo "Downloading Places365 model (this may take a while)..."
#     wget -q --show-progress http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar
#     echo "✅ Places365 model downloaded"
# fi

cd ../..
echo ""

# Download categories file
if [ ! -f "detection/categories_places365.txt" ]; then
    echo "Downloading Places365 categories..."
    wget -q --show-progress -O detection/categories_places365.txt \
        https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt
    echo "✅ Categories downloaded"
fi
echo ""

# Create .env file
if [ ! -f ".env" ]; then
    echo "Creating .env configuration file..."
    cat > .env << EOF
# AnonVision Configuration

# Server Settings
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Processing Defaults
DEFAULT_MODE=face_only
DEFAULT_TECHNIQUE=gaussian_blur
DEFAULT_INTENSITY=medium
DEFAULT_FRAME_SKIP=2

# Storage
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
MAX_UPLOAD_SIZE=100

# Performance
USE_GPU=True
CONFIDENCE_THRESHOLD=0.5
MIN_FACE_SIZE=30
EOF
    echo "✅ .env file created"
else
    echo "⚠️  .env file already exists"
fi
echo ""

# Test installation
echo "Testing installation..."
python3 << EOF
try:
    import cv2
    import torch
    import numpy as np
    from ultralytics import YOLO
    print("✅ OpenCV version:", cv2.__version__)
    print("✅ PyTorch version:", torch.__version__)
    print("✅ CUDA available:", torch.cuda.is_available())
    print("✅ All imports successful!")
except Exception as e:
    print("❌ Import error:", str(e))
    exit(1)
EOF
echo ""

# Create test image if not exists
if [ ! -f "tests/test_image.jpg" ]; then
    echo "⚠️  No test image found. Place a test image at: tests/test_image.jpg"
fi
echo ""

echo "======================================"
echo "  Setup Complete! ✅"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Start the API server:"
echo "   python api_server.py"
echo ""
echo "3. Test with webcam:"
echo "   python stream_client.py --webcam 0 --mode face_only"
echo ""
echo "4. View API docs:"
echo "   http://localhost:8000/docs"
echo ""
echo "For more information, see README.md"
echo "======================================"