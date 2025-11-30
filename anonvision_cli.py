#!/usr/bin/env python3
"""
AnonVision CLI Tool
Easy command-line interface for all operations
"""

import click
import cv2
import requests
from pathlib import Path
import json
from rich.console import Console
from rich.table import Table
from rich.progress import track
import time

console = Console()

DEFAULT_SERVER = "http://localhost:8000"


@click.group()
@click.version_option(version="2.0.0")
def cli():
    """AnonVision - Context-Aware Video Anonymization"""
    pass


@cli.command()
@click.option('--host', default='0.0.0.0', help='Server host')
@click.option('--port', default=8000, help='Server port')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host, port, reload):
    """Start the API server"""
    import uvicorn
    
    console.print(f"[green]🚀 Starting AnonVision API server...[/green]")
    console.print(f"[blue]   URL: http://{host}:{port}[/blue]")
    console.print(f"[blue]   Docs: http://{host}:{port}/docs[/blue]")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--mode', default='face_only', 
              type=click.Choice(['face_only', 'body_only', 'face_and_body', 'query_based']))
@click.option('--technique', default='gaussian_blur', help='Anonymization technique')
@click.option('--intensity', default='medium', type=click.Choice(['low', 'medium', 'high']))
@click.option('--query', default=None, help='Natural language query')
@click.option('--server', default=DEFAULT_SERVER, help='API server URL')
def process_image(input_path, output_path, mode, technique, intensity, query, server):
    """Process a single image"""
    
    console.print(f"[cyan]📷 Processing image: {input_path}[/cyan]")
    
    with console.status("[bold green]Processing..."):
        with open(input_path, 'rb') as f:
            files = {'file': f}
            data = {
                'mode': mode,
                'technique': technique,
                'intensity': intensity,
                'query': query
            }
            
            response = requests.post(f"{server}/api/process/image", 
                                    files=files, data=data)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            # Get metadata from headers
            proc_time = response.headers.get('X-Processing-Time', 'N/A')
            detections = response.headers.get('X-Detections', 'N/A')
            anonymized = response.headers.get('X-Anonymized', 'N/A')
            
            console.print(f"[green]✅ Success! Saved to: {output_path}[/green]")
            console.print(f"[blue]   Processing time: {proc_time}ms[/blue]")
            console.print(f"[blue]   Detections: {detections} | Anonymized: {anonymized}[/blue]")
        else:
            console.print(f"[red]❌ Error: {response.text}[/red]")


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--mode', default='face_only')
@click.option('--technique', default='gaussian_blur')
@click.option('--intensity', default='medium', type=click.Choice(['low', 'medium', 'high']))
@click.option('--query', default=None)
@click.option('--server', default=DEFAULT_SERVER)
def process_batch(input_dir, output_dir, mode, technique, intensity, query, server):
    """Process multiple images in a directory"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all images
    image_files = list(input_path.glob('*.jpg')) + \
                  list(input_path.glob('*.jpeg')) + \
                  list(input_path.glob('*.png'))
    
    if not image_files:
        console.print("[yellow]⚠️  No images found in directory[/yellow]")
        return
    
    console.print(f"[cyan]📁 Processing {len(image_files)} images...[/cyan]")
    
    results = []
    
    for img_file in track(image_files, description="Processing..."):
        try:
            with open(img_file, 'rb') as f:
                files = {'file': f}
                data = {
                    'mode': mode,
                    'technique': technique,
                    'intensity': intensity,
                    'query': query
                }
                
                response = requests.post(f"{server}/api/process/image",
                                       files=files, data=data)
            
            if response.status_code == 200:
                output_file = output_path / img_file.name
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                
                results.append({'file': img_file.name, 'status': 'success'})
            else:
                results.append({'file': img_file.name, 'status': 'failed'})
        
        except Exception as e:
            results.append({'file': img_file.name, 'status': f'error: {str(e)}'})
    
    # Print summary
    success = sum(1 for r in results if r['status'] == 'success')
    console.print(f"\n[green]✅ Processed: {success}/{len(image_files)}[/green]")
    console.print(f"[blue]Output directory: {output_path}[/blue]")


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--mode', default='face_only')
@click.option('--technique', default='gaussian_blur')
@click.option('--intensity', default='medium', type=click.Choice(['low', 'medium', 'high']))
@click.option('--frame-skip', default=2, help='Process every Nth frame')
@click.option('--query', default=None)
@click.option('--server', default=DEFAULT_SERVER)
def process_video(input_path, output_path, mode, technique, intensity, frame_skip, query, server):
    """Process a video file"""
    
    console.print(f"[cyan]🎬 Processing video: {input_path}[/cyan]")
    
    with console.status("[bold green]Uploading and processing..."):
        with open(input_path, 'rb') as f:
            files = {'file': f}
            data = {
                'mode': mode,
                'technique': technique,
                'intensity': intensity,
                'frame_skip': frame_skip,
                'query': query
            }
            
            response = requests.post(f"{server}/api/process/video",
                                   files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        output_url = result.get('output_url')
        metadata = result.get('metadata', {})
        
        # Download processed video
        console.print("[cyan]📥 Downloading processed video...[/cyan]")
        video_response = requests.get(f"{server}{output_url}")
        
        with open(output_path, 'wb') as f:
            f.write(video_response.content)
        
        console.print(f"[green]✅ Success! Saved to: {output_path}[/green]")
        console.print(f"[blue]   Total frames: {metadata.get('total_frames', 'N/A')}[/blue]")
        console.print(f"[blue]   FPS: {metadata.get('fps', 'N/A')}[/blue]")
        console.print(f"[blue]   Avg processing time: {metadata.get('avg_processing_time_ms', 'N/A')}ms[/blue]")
    else:
        console.print(f"[red]❌ Error: {response.text}[/red]")


@cli.command()
@click.option('--webcam', default=0, help='Webcam ID')
@click.option('--video', default=None, help='Video file path')
@click.option('--rtsp', default=None, help='RTSP stream URL')
@click.option('--mode', default='face_only')
@click.option('--technique', default='gaussian_blur')
@click.option('--intensity', default='medium', type=click.Choice(['low', 'medium', 'high']))
@click.option('--query', default=None)
@click.option('--save', default=None, help='Save output to file')
@click.option('--server', default='ws://localhost:8000/api/stream/websocket')
def stream(webcam, video, rtsp, mode, technique, intensity, query, save, server):
    """Start real-time streaming"""
    
    console.print("[cyan]🎥 Starting real-time stream...[/cyan]")
    console.print(f"[blue]   Mode: {mode}[/blue]")
    console.print(f"[blue]   Technique: {technique}[/blue]")
    console.print(f"[blue]   Intensity: {intensity}[/blue]")
    
    # Build command
    cmd_parts = ['python', 'stream_client.py']
    
    if video:
        cmd_parts.extend(['--video', video])
    elif rtsp:
        cmd_parts.extend(['--rtsp', rtsp])
    else:
        cmd_parts.extend(['--webcam', str(webcam)])
    
    cmd_parts.extend([
        '--mode', mode,
        '--technique', technique,
        '--intensity', intensity,
        '--server', server
    ])
    
    if query:
        cmd_parts.extend(['--query', query])
    
    if save:
        cmd_parts.extend(['--save', save])
    
    # Execute
    import subprocess
    subprocess.run(cmd_parts)


@cli.command()
@click.option('--server', default=DEFAULT_SERVER)
def techniques(server):
    """List all available techniques"""
    
    response = requests.get(f"{server}/api/techniques")
    
    if response.status_code == 200:
        data = response.json()
        
        # Techniques table
        table = Table(title="Anonymization Techniques")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Category", style="yellow")
        
        for tech in data['techniques']:
            table.add_row(tech['id'], tech['name'], tech['category'])
        
        console.print(table)
        
        # Modes
        console.print("\n[bold cyan]Processing Modes:[/bold cyan]")
        for mode in data['modes']:
            console.print(f"  [green]{mode['id']}[/green]: {mode['description']}")
        
        # Intensities
        console.print("\n[bold cyan]Intensities:[/bold cyan]")
        console.print(f"  {', '.join(data['intensities'])}")
    else:
        console.print(f"[red]❌ Error: {response.text}[/red]")


@cli.command()
@click.option('--server', default=DEFAULT_SERVER)
def health(server):
    """Check server health"""
    
    try:
        response = requests.get(f"{server}/api/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            console.print("[green]✅ Server is healthy[/green]")
            console.print(f"[blue]   Status: {data.get('status')}[/blue]")
            console.print(f"[blue]   Timestamp: {data.get('timestamp')}[/blue]")
        else:
            console.print(f"[yellow]⚠️  Server responded with: {response.status_code}[/yellow]")
    
    except requests.ConnectionError:
        console.print(f"[red]❌ Cannot connect to server: {server}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")


@cli.command()
def demo():
    """Run interactive demo"""
    
    console.print("[bold cyan]AnonVision Interactive Demo[/bold cyan]\n")
    
    # Check server
    console.print("[cyan]1. Checking server...[/cyan]")
    try:
        response = requests.get(f"{DEFAULT_SERVER}/api/health", timeout=2)
        if response.status_code == 200:
            console.print("[green]   ✅ Server is running[/green]\n")
        else:
            console.print("[yellow]   ⚠️  Server not responding[/yellow]")
            console.print("[yellow]   Start server: python api_server.py[/yellow]\n")
            return
    except:
        console.print("[red]   ❌ Server not running[/red]")
        console.print("[yellow]   Start server: python api_server.py[/yellow]\n")
        return
    
    # Demo options
    console.print("[cyan]2. Select demo type:[/cyan]")
    console.print("   [1] Process test image")
    console.print("   [2] Start webcam stream")
    console.print("   [3] Show available techniques")
    
    choice = click.prompt("\n   Enter choice", type=int, default=1)
    
    if choice == 1:
        console.print("\n[cyan]Processing test image...[/cyan]")
        if Path("tests/test_image.jpg").exists():
            process_image.callback(
                "tests/test_image.jpg",
                "output_demo.jpg",
                "face_only",
                "gaussian_blur",
                "medium",
                None,
                DEFAULT_SERVER
            )
        else:
            console.print("[red]❌ Test image not found: tests/test_image.jpg[/red]")
    
    elif choice == 2:
        console.print("\n[cyan]Starting webcam stream...[/cyan]")
        stream.callback(0, None, None, "face_only", "gaussian_blur", 
                       "medium", None, None, "ws://localhost:8000/api/stream/websocket")
    
    elif choice == 3:
        techniques.callback(DEFAULT_SERVER)


if __name__ == "__main__":
    cli()