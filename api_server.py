"""
AnonVision FastAPI Server
Handles image/video upload, real-time streaming, and processing requests
"""

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import cv2
import numpy as np
import asyncio
import uuid
import os
import json
import shutil
from pathlib import Path
import base64
from datetime import datetime

from processor import (
    AnonVisionProcessor, ProcessingConfig, ProcessingMode,
    AnonymizationTechnique
)

# ===== Configuration =====
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ===== FastAPI App =====
app = FastAPI(
    title="AnonVision API",
    description="Real-time context-aware video anonymization system",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve output files
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


# ===== Request Models =====
class ImageProcessRequest(BaseModel):
    mode: str = "face_only"  # face_only, body_only, face_and_body, query_based
    technique: str = "gaussian_blur"
    intensity: str = "medium"  # low, medium, high
    query: Optional[str] = None


class VideoProcessRequest(BaseModel):
    mode: str = "face_only"
    technique: str = "gaussian_blur"
    intensity: str = "medium"
    query: Optional[str] = None
    frame_skip: int = 2  # Process every Nth frame


class StreamConfig(BaseModel):
    mode: str = "face_only"
    technique: str = "gaussian_blur"
    intensity: str = "medium"
    query: Optional[str] = None
    stream_url: Optional[str] = None  # RTSP/HTTP stream URL
    use_webcam: bool = False
    webcam_id: int = 0


# ===== Helper Functions =====
def parse_config(request_data: dict) -> ProcessingConfig:
    """Convert request data to ProcessingConfig"""
    mode_map = {
        'face_only': ProcessingMode.FACE_ONLY,
        'body_only': ProcessingMode.BODY_ONLY,
        'face_and_body': ProcessingMode.FACE_AND_BODY,
        'query_based': ProcessingMode.QUERY_BASED
    }
    
    technique_map = {
        'gaussian_blur': AnonymizationTechnique.GAUSSIAN_BLUR,
        'pixelate': AnonymizationTechnique.PIXELATE,
        'mosaic': AnonymizationTechnique.MOSAIC,
        'black_box': AnonymizationTechnique.BLACK_BOX,
        'median_blur': AnonymizationTechnique.MEDIAN_BLUR,
        'bilateral_filter': AnonymizationTechnique.BILATERAL_FILTER,
        'mask_overlay': AnonymizationTechnique.MASK_OVERLAY,
        'edge_preserve_blur': AnonymizationTechnique.EDGE_PRESERVE_BLUR,
        'oil_painting': AnonymizationTechnique.OIL_PAINTING,
        'cartoon': AnonymizationTechnique.CARTOON,
        'negative': AnonymizationTechnique.NEGATIVE,
        'grayscale': AnonymizationTechnique.GRAYSCALE,
        'sepia': AnonymizationTechnique.SEPIA,
        'brightness': AnonymizationTechnique.BRIGHTNESS,
        'contrast': AnonymizationTechnique.CONTRAST,
    }
    
    return ProcessingConfig(
        mode=mode_map.get(request_data.get('mode', 'face_only'), ProcessingMode.FACE_ONLY),
        technique=technique_map.get(request_data.get('technique', 'gaussian_blur'), 
                                   AnonymizationTechnique.GAUSSIAN_BLUR),
        intensity=request_data.get('intensity', 'medium'),
        frame_skip=request_data.get('frame_skip', 2),
        query=request_data.get('query'),
        require_attributes=request_data.get('mode') == 'query_based'
    )


def save_upload_file(upload_file: UploadFile) -> Path:
    """Save uploaded file to disk"""
    file_id = str(uuid.uuid4())
    extension = Path(upload_file.filename).suffix
    file_path = UPLOAD_DIR / f"{file_id}{extension}"
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    
    return file_path


# ===== API Endpoints =====

@app.get("/")
async def root():
    return {
        "service": "AnonVision API",
        "version": "2.0.0",
        "endpoints": {
            "image": "/api/process/image",
            "images_batch": "/api/process/images",
            "video": "/api/process/video",
            "stream": "/api/stream/websocket",
            "techniques": "/api/techniques",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/techniques")
async def get_techniques():
    """List all available anonymization techniques"""
    return {
        "techniques": [
            {"id": "gaussian_blur", "name": "Gaussian Blur", "category": "blur"},
            {"id": "pixelate", "name": "Pixelation", "category": "blur"},
            {"id": "mosaic", "name": "Mosaic", "category": "blur"},
            {"id": "black_box", "name": "Black Box", "category": "block"},
            {"id": "median_blur", "name": "Median Blur", "category": "blur"},
            {"id": "bilateral_filter", "name": "Bilateral Filter", "category": "blur"},
            {"id": "mask_overlay", "name": "Mask Overlay", "category": "overlay"},
            {"id": "edge_preserve_blur", "name": "Edge Preserve Blur", "category": "blur"},
            {"id": "oil_painting", "name": "Oil Painting", "category": "artistic"},
            {"id": "cartoon", "name": "Cartoon", "category": "artistic"},
            {"id": "negative", "name": "Negative", "category": "effect"},
            {"id": "grayscale", "name": "Grayscale", "category": "effect"},
            {"id": "sepia", "name": "Sepia", "category": "effect"},
            {"id": "brightness", "name": "Brightness Reduction", "category": "effect"},
            {"id": "contrast", "name": "Contrast Reduction", "category": "effect"},
        ],
        "modes": [
            {"id": "face_only", "name": "Face Only", "description": "Anonymize faces only"},
            {"id": "body_only", "name": "Body Only", "description": "Anonymize full bodies"},
            {"id": "face_and_body", "name": "Face & Body", "description": "Anonymize both"},
            {"id": "query_based", "name": "Query Based", "description": "Use natural language query"}
        ],
        "intensities": ["low", "medium", "high"]
    }


@app.post("/api/process/image")
async def process_image(
    file: UploadFile = File(...),
    mode: str = Form("face_only"),
    technique: str = Form("gaussian_blur"),
    intensity: str = Form("medium"),
    query: Optional[str] = Form(None)
):
    """
    Process a single image
    
    Returns: Direct image file download
    """
    try:
        # Save uploaded file
        input_path = save_upload_file(file)
        
        # Read image
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Create config
        config = parse_config({
            'mode': mode,
            'technique': technique,
            'intensity': intensity,
            'query': query
        })
        
        # Process
        processor = AnonVisionProcessor(config)
        processed_frame, metadata = processor.process_frame(frame, force_process=True)
        
        # Save output
        output_id = str(uuid.uuid4())
        output_path = OUTPUT_DIR / f"{output_id}.jpg"
        cv2.imwrite(str(output_path), processed_frame)
        
        # Cleanup input
        input_path.unlink()
        
        # Return file
        return FileResponse(
            str(output_path),
            media_type="image/jpeg",
            filename=f"anonymized_{file.filename}",
            headers={
                "X-Processing-Time": str(metadata.get('processing_time_ms', 0)),
                "X-Detections": str(metadata.get('detections', 0)),
                "X-Anonymized": str(metadata.get('anonymized', 0))
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process/images")
async def process_images(
    files: List[UploadFile] = File(...),
    mode: str = Form("face_only"),
    technique: str = Form("gaussian_blur"),
    intensity: str = Form("medium"),
    query: Optional[str] = Form(None)
):
    """
    Process multiple images in batch
    
    Returns: JSON with links to processed images
    """
    try:
        results = []
        
        config = parse_config({
            'mode': mode,
            'technique': technique,
            'intensity': intensity,
            'query': query
        })
        
        processor = AnonVisionProcessor(config)
        
        for file in files:
            # Save and read
            input_path = save_upload_file(file)
            frame = cv2.imread(str(input_path))
            
            if frame is None:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": "Invalid image file"
                })
                input_path.unlink()
                continue
            
            # Process
            processed_frame, metadata = processor.process_frame(frame, force_process=True)
            
            # Save output
            output_id = str(uuid.uuid4())
            output_path = OUTPUT_DIR / f"{output_id}.jpg"
            cv2.imwrite(str(output_path), processed_frame)
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "output_url": f"/outputs/{output_id}.jpg",
                "metadata": metadata
            })
            
            # Cleanup
            input_path.unlink()
        
        return JSONResponse(content={"results": results})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process/video")
async def process_video(
    file: UploadFile = File(...),
    mode: str = Form("face_only"),
    technique: str = Form("gaussian_blur"),
    intensity: str = Form("medium"),
    frame_skip: int = Form(2),
    query: Optional[str] = Form(None)
):
    """
    Process a video file
    
    Returns: JSON with link to processed video
    """
    try:
        # Save uploaded video
        input_path = save_upload_file(file)
        
        # Create config
        config = parse_config({
            'mode': mode,
            'technique': technique,
            'intensity': intensity,
            'query': query,
            'frame_skip': frame_skip
        })
        
        # Output path
        output_id = str(uuid.uuid4())
        output_path = OUTPUT_DIR / f"{output_id}.mp4"
        
        # Process video
        processor = AnonVisionProcessor(config)
        
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Invalid video file")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, _ = processor.process_frame(frame)
            out.write(processed_frame)
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        # Cleanup input
        input_path.unlink()
        
        # Get stats
        stats = processor.get_stats()
        
        return JSONResponse(content={
            "status": "success",
            "output_url": f"/outputs/{output_id}.mp4",
            "metadata": {
                "total_frames": frame_count,
                "fps": fps,
                "resolution": f"{width}x{height}",
                **stats
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== WebSocket for Real-Time Streaming =====

active_connections: List[WebSocket] = []


@app.websocket("/api/stream/websocket")
async def websocket_stream(websocket: WebSocket):
    """
    Real-time video streaming with anonymization
    
    Protocol:
    1. Client connects
    2. Client sends config: {"mode": "face_only", "technique": "blur", ...}
    3. Client sends frames as base64 encoded JPEG
    4. Server responds with processed frame as base64 encoded JPEG
    """
    await websocket.accept()
    active_connections.append(websocket)
    
    processor = None
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle config message
            if message.get('type') == 'config':
                config = parse_config(message.get('config', {}))
                processor = AnonVisionProcessor(config)
                await websocket.send_json({
                    'type': 'config_ack',
                    'status': 'ready'
                })
                continue
            
            # Handle frame message
            if message.get('type') == 'frame':
                if processor is None:
                    await websocket.send_json({
                        'type': 'error',
                        'message': 'Send config first'
                    })
                    continue
                
                # Decode frame
                frame_b64 = message.get('frame', '')
                frame_bytes = base64.b64decode(frame_b64)
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({
                        'type': 'error',
                        'message': 'Invalid frame'
                    })
                    continue
                
                # Process frame
                processed_frame, metadata = processor.process_frame(frame, force_process=True)
                
                # Encode response
                _, buffer = cv2.imencode('.jpg', processed_frame, 
                                        [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send response
                await websocket.send_json({
                    'type': 'frame',
                    'frame': frame_b64,
                    'metadata': metadata
                })
    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print("Client disconnected")
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({
            'type': 'error',
            'message': str(e)
        })


@app.post("/api/stream/start")
async def start_stream(config: StreamConfig):
    """
    Start processing an external stream (RTSP/HTTP) or webcam
    
    Returns: Stream ID for status checking
    """
    # This would require a background task manager
    # For now, return a simple response
    return JSONResponse(content={
        "status": "not_implemented",
        "message": "Use WebSocket endpoint for real-time streaming"
    })


# ===== Run Server =====
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("  AnonVision API Server")
    print("=" * 60)
    print(f"  Upload directory: {UPLOAD_DIR.absolute()}")
    print(f"  Output directory: {OUTPUT_DIR.absolute()}")
    print("=" * 60)
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )