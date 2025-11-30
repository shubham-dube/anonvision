"""
AnonVision API Usage Examples
Comprehensive examples for all API endpoints
"""

import requests
import json
import base64
import asyncio
import websockets
import cv2
import numpy as np
from pathlib import Path


# ===== Configuration =====
API_BASE = "http://localhost:8000"
WS_URL = "ws://localhost:8000/api/stream/websocket"


# ===== Example 1: Process Single Image =====
def example_single_image():
    """Process a single image with face anonymization"""
    
    print("Example 1: Processing single image...")
    
    image_path = "test_image.jpg"
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {
            'mode': 'face_only',
            'technique': 'gaussian_blur',
            'intensity': 'medium'
        }
        
        response = requests.post(f"{API_BASE}/api/process/image", 
                               files=files, data=data)
    
    if response.status_code == 200:
        # Save output
        with open('output_example1.jpg', 'wb') as f:
            f.write(response.content)
        
        print("✅ Image processed successfully!")
        print(f"   Processing time: {response.headers.get('X-Processing-Time')}ms")
        print(f"   Detections: {response.headers.get('X-Detections')}")
        print(f"   Saved to: output_example1.jpg")
    else:
        print(f"❌ Error: {response.text}")


# ===== Example 2: Batch Image Processing =====
def example_batch_images():
    """Process multiple images at once"""
    
    print("\nExample 2: Batch image processing...")
    
    image_files = [
        'image1.jpg',
        'image2.jpg',
        'image3.jpg'
    ]
    
    files = [('files', open(img, 'rb')) for img in image_files if Path(img).exists()]
    
    if not files:
        print("⚠️  No image files found")
        return
    
    data = {
        'mode': 'face_only',
        'technique': 'pixelate',
        'intensity': 'high'
    }
    
    response = requests.post(f"{API_BASE}/api/process/images",
                           files=files, data=data)
    
    # Close file handles
    for _, f in files:
        f.close()
    
    if response.status_code == 200:
        results = response.json()['results']
        
        print(f"✅ Processed {len(results)} images")
        for result in results:
            print(f"   {result['filename']}: {result['status']}")
            if result['status'] == 'success':
                print(f"      Output: {API_BASE}{result['output_url']}")
    else:
        print(f"❌ Error: {response.text}")


# ===== Example 3: Video Processing =====
def example_video_processing():
    """Process a video file with anonymization"""
    
    print("\nExample 3: Video processing...")
    
    video_path = "test_video.mp4"
    
    if not Path(video_path).exists():
        print(f"⚠️  Video file not found: {video_path}")
        return
    
    with open(video_path, 'rb') as f:
        files = {'file': f}
        data = {
            'mode': 'face_and_body',
            'technique': 'mosaic',
            'intensity': 'medium',
            'frame_skip': 2
        }
        
        print("Uploading and processing (this may take a while)...")
        response = requests.post(f"{API_BASE}/api/process/video",
                               files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        
        print("✅ Video processed successfully!")
        print(f"   Output URL: {API_BASE}{result['output_url']}")
        print(f"   Metadata:")
        for key, value in result['metadata'].items():
            print(f"      {key}: {value}")
        
        # Download processed video
        output_url = result['output_url']
        video_response = requests.get(f"{API_BASE}{output_url}")
        
        with open('output_example3.mp4', 'wb') as f:
            f.write(video_response.content)
        
        print("   Downloaded to: output_example3.mp4")
    else:
        print(f"❌ Error: {response.text}")


# ===== Example 4: Query-Based Processing =====
def example_query_based():
    """Use natural language query for selective anonymization"""
    
    print("\nExample 4: Query-based processing...")
    
    image_path = "test_image.jpg"
    
    queries = [
        "blur all children",
        "anonymize people wearing red",
        "blur all except adults"
    ]
    
    for i, query in enumerate(queries):
        print(f"\n   Query {i+1}: '{query}'")
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'mode': 'query_based',
                'technique': 'gaussian_blur',
                'intensity': 'medium',
                'query': query
            }
            
            response = requests.post(f"{API_BASE}/api/process/image",
                                   files=files, data=data)
        
        if response.status_code == 200:
            output_file = f'output_example4_{i+1}.jpg'
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            print(f"   ✅ Saved to: {output_file}")
        else:
            print(f"   ❌ Error: {response.text}")


# ===== Example 5: Different Techniques =====
def example_all_techniques():
    """Test all available anonymization techniques"""
    
    print("\nExample 5: Testing all techniques...")
    
    image_path = "test_image.jpg"
    
    if not Path(image_path).exists():
        print(f"⚠️  Image file not found: {image_path}")
        return
    
    # Get available techniques
    response = requests.get(f"{API_BASE}/api/techniques")
    techniques = response.json()['techniques']
    
    print(f"Found {len(techniques)} techniques\n")
    
    for tech in techniques[:5]:  # Test first 5 for demo
        print(f"   Testing: {tech['name']}...")
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'mode': 'face_only',
                'technique': tech['id'],
                'intensity': 'medium'
            }
            
            response = requests.post(f"{API_BASE}/api/process/image",
                                   files=files, data=data)
        
        if response.status_code == 200:
            output_file = f"output_{tech['id']}.jpg"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"      ✅ Saved to: {output_file}")
        else:
            print(f"      ❌ Error")


# ===== Example 6: WebSocket Real-Time Streaming =====
async def example_websocket_stream():
    """Real-time frame processing via WebSocket"""
    
    print("\nExample 6: WebSocket real-time streaming...")
    
    async with websockets.connect(WS_URL) as websocket:
        # Send configuration
        config = {
            'type': 'config',
            'config': {
                'mode': 'face_only',
                'technique': 'gaussian_blur',
                'intensity': 'medium'
            }
        }
        
        await websocket.send(json.dumps(config))
        
        # Wait for acknowledgment
        response = await websocket.recv()
        ack = json.loads(response)
        
        if ack.get('type') == 'config_ack':
            print("✅ Configuration accepted")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        print("📹 Streaming... (Press Ctrl+C to stop)")
        
        frame_count = 0
        max_frames = 30  # Process 30 frames for demo
        
        try:
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Encode frame
                _, buffer = cv2.imencode('.jpg', frame, 
                                        [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send frame
                message = {
                    'type': 'frame',
                    'frame': frame_b64
                }
                await websocket.send(json.dumps(message))
                
                # Receive processed frame
                response = await websocket.recv()
                result = json.loads(response)
                
                if result.get('type') == 'frame':
                    # Decode processed frame
                    frame_b64 = result.get('frame', '')
                    frame_bytes = base64.b64decode(frame_b64)
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Display
                    cv2.imshow('WebSocket Stream', processed_frame)
                    cv2.waitKey(1)
                    
                    frame_count += 1
                    
                    if frame_count % 10 == 0:
                        metadata = result.get('metadata', {})
                        print(f"   Frame {frame_count}: "
                              f"{metadata.get('detections', 0)} detections, "
                              f"{metadata.get('processing_time_ms', 0):.1f}ms")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n✅ Processed {frame_count} frames")


# ===== Example 7: Custom Processing Config =====
def example_custom_config():
    """Use custom processing configuration"""
    
    print("\nExample 7: Custom configuration...")
    
    from processor import (
        AnonVisionProcessor, ProcessingConfig, 
        ProcessingMode, AnonymizationTechnique
    )
    
    # Create custom config
    config = ProcessingConfig(
        mode=ProcessingMode.FACE_ONLY,
        technique=AnonymizationTechnique.MOSAIC,
        intensity='high',
        frame_skip=1,
        face_padding=0.2,  # More padding
        min_face_size=40,  # Larger minimum face size
        confidence_threshold=0.7  # Higher confidence
    )
    
    # Create processor
    processor = AnonVisionProcessor(config)
    
    # Process image
    image = cv2.imread('test_image.jpg')
    
    if image is not None:
        processed, metadata = processor.process_frame(image, force_process=True)
        
        cv2.imwrite('output_example7.jpg', processed)
        
        print("✅ Custom config processing complete!")
        print(f"   Metadata: {metadata}")
        print(f"   Stats: {processor.get_stats()}")
    else:
        print("⚠️  Image not found")


# ===== Example 8: Health Check & System Info =====
def example_system_info():
    """Get system information and health status"""
    
    print("\nExample 8: System information...")
    
    # Health check
    response = requests.get(f"{API_BASE}/api/health")
    
    if response.status_code == 200:
        health = response.json()
        print(f"✅ Server status: {health['status']}")
        print(f"   Timestamp: {health['timestamp']}")
    
    # Get techniques
    response = requests.get(f"{API_BASE}/api/techniques")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n📊 System capabilities:")
        print(f"   Techniques: {len(data['techniques'])}")
        print(f"   Modes: {len(data['modes'])}")
        print(f"   Intensities: {len(data['intensities'])}")


# ===== Main =====
def main():
    """Run all examples"""
    
    print("="*60)
    print("  AnonVision API Examples")
    print("="*60)
    print("\nMake sure the API server is running:")
    print("  python api_server.py\n")
    
    try:
        # Check if server is running
        response = requests.get(f"{API_BASE}/api/health", timeout=2)
        if response.status_code != 200:
            print("❌ Server not responding. Start the server first.")
            return
    except:
        print("❌ Cannot connect to server. Start the server first:")
        print("   python api_server.py")
        return
    
    # Run examples
    print("\nRunning examples...\n")
    
    # Basic examples
    example_single_image()
    example_system_info()
    
    # Uncomment to run more examples:
    # example_batch_images()
    # example_video_processing()
    # example_query_based()
    # example_all_techniques()
    # example_custom_config()
    
    # WebSocket example (requires asyncio)
    # asyncio.run(example_websocket_stream())
    
    print("\n" + "="*60)
    print("  Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()