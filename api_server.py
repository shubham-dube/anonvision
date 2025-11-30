"""
AnonVision FastAPI Server
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
import zipfile
import io
from dotenv import load_dotenv
load_dotenv()


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

# Serve static files
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/static", StaticFiles(directory="static"), name="static")


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
        frame_skip=request_data.get('frame_skip', 1),  # Changed to 1 for better video quality
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
    """Serve the web interface"""
    return FileResponse("static/index.html")


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
    """Process a single image - Returns downloadable file"""
    try:
        input_path = save_upload_file(file)
        frame = cv2.imread(str(input_path))
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        config = parse_config({
            'mode': mode,
            'technique': technique,
            'intensity': intensity,
            'query': query
        })
        
        processor = AnonVisionProcessor(config)
        processed_frame, metadata = processor.process_frame(frame, force_process=True)
        
        output_id = str(uuid.uuid4())
        output_path = OUTPUT_DIR / f"{output_id}.jpg"
        cv2.imwrite(str(output_path), processed_frame)
        
        input_path.unlink()
        
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
    query: Optional[str] = Form(None),
    return_zip: bool = Form(False)
):
    """
    Process multiple images in batch
    
    If return_zip=True: Returns a ZIP file with all processed images
    If return_zip=False: Returns JSON with individual download links
    """
    try:
        results = []
        output_files = []
        
        config = parse_config({
            'mode': mode,
            'technique': technique,
            'intensity': intensity,
            'query': query
        })
        
        processor = AnonVisionProcessor(config)
        
        for file in files:
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
            
            processed_frame, metadata = processor.process_frame(frame, force_process=True)
            
            output_id = str(uuid.uuid4())
            output_filename = f"{output_id}_{file.filename}"
            output_path = OUTPUT_DIR / output_filename
            cv2.imwrite(str(output_path), processed_frame)
            
            output_files.append((output_path, output_filename))
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "download_url": f"/outputs/{output_filename}",
                "metadata": metadata
            })
            
            input_path.unlink()
        
        # Return ZIP file if requested
        if return_zip:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for output_path, output_filename in output_files:
                    zip_file.write(output_path, arcname=f"anonymized_{output_filename}")
            
            zip_buffer.seek(0)
            return StreamingResponse(
                zip_buffer,
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename=anonymized_images.zip"}
            )
        
        # Return JSON with individual links
        return JSONResponse(content={
            "status": "success",
            "total": len(files),
            "processed": len([r for r in results if r['status'] == 'success']),
            "failed": len([r for r in results if r['status'] == 'error']),
            "results": results
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process/video")
async def process_video(
    file: UploadFile = File(...),
    mode: str = Form("face_only"),
    technique: str = Form("gaussian_blur"),
    intensity: str = Form("medium"),
    frame_skip: int = Form(1),
    query: Optional[str] = Form(None)
):
    """
    Process a video file - FIXED VERSION
    Now properly processes and saves anonymized video
    """
    try:
        input_path = save_upload_file(file)
        
        config = parse_config({
            'mode': mode,
            'technique': technique,
            'intensity': intensity,
            'query': query,
            'frame_skip': frame_skip
        })
        
        output_id = str(uuid.uuid4())
        output_path = OUTPUT_DIR / f"{output_id}.mp4"
        
        processor = AnonVisionProcessor(config)
        
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Invalid video file")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use H264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Changed from mp4v to avc1
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            # Fallback to mp4v if avc1 fails
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        
        print(f"Processing video: {total_frames} frames @ {fps}fps")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # CRITICAL FIX: Always process frame with force_process=True
            processed_frame, metadata = processor.process_frame(frame, force_process=True)
            
            # Verify frame was actually processed
            if metadata.get('anonymized', 0) > 0:
                processed_count += 1
            
            out.write(processed_frame)
            frame_count += 1
            
            # Progress logging
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Anonymized: {processed_count} regions")
        
        cap.release()
        out.release()
        
        input_path.unlink()
        
        stats = processor.get_stats()
        
        return JSONResponse(content={
            "status": "success",
            "download_url": f"/outputs/{output_id}.mp4",
            "filename": f"anonymized_{file.filename}",
            "metadata": {
                "total_frames": frame_count,
                "processed_frames": processed_count,
                "fps": fps,
                "resolution": f"{width}x{height}",
                **stats
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== WebSocket for Real-Time Streaming =====

active_connections: dict = {}


@app.websocket("/api/stream/websocket")
async def websocket_stream(websocket: WebSocket):
    """
    Real-time video streaming with anonymization
    """
    await websocket.accept()
    client_id = str(uuid.uuid4())
    active_connections[client_id] = websocket
    
    processor = None
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('type') == 'config':
                # CRITICAL FIX: Always create new processor instance
                config = parse_config(message.get('config', {}))
                processor = AnonVisionProcessor(config)
                processor.reset_stats()  # Reset stats for new session
                
                await websocket.send_json({
                    'type': 'config_ack',
                    'status': 'ready',
                    'client_id': client_id
                })
                continue
            
            if message.get('type') == 'frame':
                if processor is None:
                    await websocket.send_json({
                        'type': 'error',
                        'message': 'Send config first'
                    })
                    continue
                
                try:
                    # Decode frame
                    frame_b64 = message.get('frame', '')
                    if not frame_b64:
                        continue
                    
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
                    
                    # Encode response with lower quality for speed
                    _, buffer = cv2.imencode('.jpg', processed_frame, 
                                            [cv2.IMWRITE_JPEG_QUALITY, 75])
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    await websocket.send_json({
                        'type': 'frame',
                        'frame': frame_b64,
                        'metadata': metadata
                    })
                    
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    await websocket.send_json({
                        'type': 'error',
                        'message': f'Frame processing failed: {str(e)}'
                    })
    
    except WebSocketDisconnect:
        if client_id in active_connections:
            del active_connections[client_id]
        print(f"Client {client_id} disconnected")
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                'type': 'error',
                'message': str(e)
            })
        except:
            pass
    
    finally:
        # Cleanup
        if client_id in active_connections:
            del active_connections[client_id]


# ===== Video Feed Simulator =====

@app.get("/api/simulate/start")
async def start_video_simulation():
    """Start a simulated video feed for testing"""
    return JSONResponse(content={
        "status": "use_websocket",
        "message": "Connect to WebSocket endpoint at /api/stream/websocket",
        "instructions": "Use the web interface at / for easy testing"
    })


if __name__ == "__main__":
    import uvicorn
    
    # Create static directory if not exists
    Path("static").mkdir(exist_ok=True)
    
    print("=" * 60)
    print("  AnonVision API Server")
    print("=" * 60)
    print(f"  Upload directory: {UPLOAD_DIR.absolute()}")
    print(f"  Output directory: {OUTPUT_DIR.absolute()}")
    print("  Web Interface: http://localhost:8000")
    print("=" * 60)
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )