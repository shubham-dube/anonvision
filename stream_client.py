"""
AnonVision Streaming Client
Real-time video processing with live output streaming
"""

import cv2
import asyncio
import websockets
import json
import base64
import numpy as np
from typing import Optional
import argparse
from threading import Thread
import time


class StreamingClient:
    """Client for real-time video anonymization"""
    
    def __init__(self, server_url: str = "ws://localhost:8000/api/stream/websocket"):
        self.server_url = server_url
        self.websocket = None
        self.is_running = False
        self.fps_counter = FPSCounter()
        
    async def connect(self):
        """Connect to WebSocket server"""
        self.websocket = await websockets.connect(self.server_url)
        print(f"✅ Connected to {self.server_url}")
    
    async def send_config(self, config: dict):
        """Send processing configuration"""
        message = {
            'type': 'config',
            'config': config
        }
        await self.websocket.send(json.dumps(message))
        
        # Wait for acknowledgment
        response = await self.websocket.recv()
        ack = json.loads(response)
        
        if ack.get('type') == 'config_ack':
            print("✅ Configuration accepted")
        else:
            print(f"❌ Config error: {ack}")
    
    async def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Send frame for processing and receive result
        
        Returns: (processed_frame, metadata)
        """
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send frame
        message = {
            'type': 'frame',
            'frame': frame_b64
        }
        await self.websocket.send(json.dumps(message))
        
        # Receive processed frame
        response = await self.websocket.recv()
        result = json.loads(response)
        
        if result.get('type') == 'error':
            print(f"❌ Processing error: {result.get('message')}")
            return frame, {}
        
        # Decode processed frame
        frame_b64 = result.get('frame', '')
        frame_bytes = base64.b64decode(frame_b64)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        metadata = result.get('metadata', {})
        
        return processed_frame, metadata
    
    async def stream_webcam(self, config: dict, webcam_id: int = 0, 
                           show_output: bool = True, save_output: Optional[str] = None):
        """
        Stream from webcam with real-time processing
        
        Args:
            config: Processing configuration
            webcam_id: Webcam device ID
            show_output: Display output window
            save_output: Optional path to save output video
        """
        await self.connect()
        await self.send_config(config)
        
        # Open webcam
        cap = cv2.VideoCapture(webcam_id)
        if not cap.isOpened():
            print(f"❌ Cannot open webcam {webcam_id}")
            return
        
        # Get webcam properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"📹 Webcam: {width}x{height} @ {fps}fps")
        
        # Setup video writer if saving
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
            print(f"💾 Saving output to: {save_output}")
        
        self.is_running = True
        self.fps_counter.start()
        
        print("\n🎥 Streaming started! Press 'q' to quit, 's' to save frame\n")
        
        try:
            while self.is_running:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Process frame
                processed_frame, metadata = await self.process_frame(frame)
                
                # Update FPS
                current_fps = self.fps_counter.update()
                
                # Add overlay info
                if show_output:
                    self._draw_overlay(processed_frame, metadata, current_fps)
                
                # Save frame
                if writer:
                    writer.write(processed_frame)
                
                # Display
                if show_output:
                    cv2.imshow('AnonVision - Live Stream', processed_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        filename = f"snapshot_{int(time.time())}.jpg"
                        cv2.imwrite(filename, processed_frame)
                        print(f"📸 Saved snapshot: {filename}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            await self.websocket.close()
            print("\n✅ Stream ended")
    
    async def stream_video_file(self, config: dict, video_path: str,
                               show_output: bool = True, save_output: Optional[str] = None):
        """
        Stream from video file with real-time processing
        """
        await self.connect()
        await self.send_config(config)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎬 Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup writer
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
            print(f"💾 Saving output to: {save_output}")
        
        self.is_running = True
        self.fps_counter.start()
        frame_count = 0
        
        print("\n🎥 Processing video... Press 'q' to quit\n")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, metadata = await self.process_frame(frame)
                
                # Update counters
                frame_count += 1
                current_fps = self.fps_counter.update()
                progress = (frame_count / total_frames) * 100
                
                # Add overlay
                if show_output:
                    self._draw_overlay(processed_frame, metadata, current_fps, progress)
                
                # Save
                if writer:
                    writer.write(processed_frame)
                
                # Display
                if show_output:
                    cv2.imshow('AnonVision - Video Processing', processed_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Print progress
                if frame_count % 30 == 0:
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) @ {current_fps:.1f} FPS")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            await self.websocket.close()
            print(f"\n✅ Processed {frame_count} frames")
    
    async def stream_rtsp(self, config: dict, rtsp_url: str,
                         show_output: bool = True, save_output: Optional[str] = None):
        """
        Stream from RTSP/RTMP source
        """
        await self.connect()
        await self.send_config(config)
        
        # Open RTSP stream
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"❌ Cannot connect to stream: {rtsp_url}")
            return
        
        print(f"📡 Connected to RTSP stream: {rtsp_url}")
        
        # Get properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup writer
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output, fourcc, 25, (width, height))
        
        self.is_running = True
        self.fps_counter.start()
        
        print("\n🎥 Streaming... Press 'q' to quit\n")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Stream disconnected")
                    break
                
                # Process
                processed_frame, metadata = await self.process_frame(frame)
                
                # Update FPS
                current_fps = self.fps_counter.update()
                
                # Overlay
                if show_output:
                    self._draw_overlay(processed_frame, metadata, current_fps)
                
                # Save
                if writer:
                    writer.write(processed_frame)
                
                # Display
                if show_output:
                    cv2.imshow('AnonVision - RTSP Stream', processed_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            await self.websocket.close()
            print("\n✅ Stream ended")
    
    def _draw_overlay(self, frame: np.ndarray, metadata: dict, fps: float, 
                     progress: Optional[float] = None):
        """Draw information overlay on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Text info
        texts = [
            f"FPS: {fps:.1f}",
            f"Detections: {metadata.get('detections', 0)}",
            f"Anonymized: {metadata.get('anonymized', 0)}",
            f"Processing: {metadata.get('processing_time_ms', 0):.1f}ms"
        ]
        
        if progress is not None:
            texts.append(f"Progress: {progress:.1f}%")
        
        y_offset = 35
        for text in texts:
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25


class FPSCounter:
    """Simple FPS counter"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps = []
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
        self.timestamps = []
    
    def update(self) -> float:
        """Update counter and return current FPS"""
        now = time.time()
        self.timestamps.append(now)
        
        # Keep only recent timestamps
        if len(self.timestamps) > self.window_size:
            self.timestamps = self.timestamps[-self.window_size:]
        
        # Calculate FPS
        if len(self.timestamps) < 2:
            return 0.0
        
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return len(self.timestamps) / elapsed if elapsed > 0 else 0.0


# ===== CLI Interface =====

async def main():
    parser = argparse.ArgumentParser(description="AnonVision Real-Time Streaming Client")
    
    parser.add_argument('--server', type=str, default='ws://localhost:8000/api/stream/websocket',
                       help='WebSocket server URL')
    parser.add_argument('--mode', type=str, default='face_only',
                       choices=['face_only', 'body_only', 'face_and_body', 'query_based'],
                       help='Processing mode')
    parser.add_argument('--technique', type=str, default='gaussian_blur',
                       help='Anonymization technique')
    parser.add_argument('--intensity', type=str, default='medium',
                       choices=['low', 'medium', 'high'],
                       help='Processing intensity')
    parser.add_argument('--query', type=str, default=None,
                       help='Natural language query (for query_based mode)')
    
    # Input source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--webcam', type=int, nargs='?', const=0,
                             help='Use webcam (default: 0)')
    source_group.add_argument('--video', type=str,
                             help='Path to video file')
    source_group.add_argument('--rtsp', type=str,
                             help='RTSP/RTMP stream URL')
    
    # Output options
    parser.add_argument('--save', type=str, default=None,
                       help='Save output video to file')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display window')
    
    args = parser.parse_args()
    
    # Build config
    config = {
        'mode': args.mode,
        'technique': args.technique,
        'intensity': args.intensity,
        'query': args.query
    }
    
    # Create client
    client = StreamingClient(server_url=args.server)
    
    # Start streaming based on source
    try:
        if args.webcam is not None:
            await client.stream_webcam(
                config, 
                webcam_id=args.webcam,
                show_output=not args.no_display,
                save_output=args.save
            )
        elif args.video:
            await client.stream_video_file(
                config,
                video_path=args.video,
                show_output=not args.no_display,
                save_output=args.save
            )
        elif args.rtsp:
            await client.stream_rtsp(
                config,
                rtsp_url=args.rtsp,
                show_output=not args.no_display,
                save_output=args.save
            )
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())