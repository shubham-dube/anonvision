# ============================================================
#   ANONVISION – Combined CLI + FASTAPI Server (Full System)
# ============================================================

import cv2
import argparse
import numpy as np
import threading
import traceback
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pyngrok import ngrok
import uvicorn

# ---- Your existing modules ----
from detector import IntegratedDetector
from decision.decision_module import DecisionModule
from blurring.blurring_module import FaceBlurrer, SelectiveBlurPipeline


# ============================================================
#      FASTAPI SERVER INITIALIZATION
# ============================================================

app = FastAPI(
    title="AnonVision Full API",
    description="Detection + Decision + Blurring via API",
    version="2.0.0"
)

detector = IntegratedDetector()
decision = DecisionModule(mode="all")
blurrer = FaceBlurrer(blur_type="gaussian", blur_intensity="medium")
pipeline = SelectiveBlurPipeline(detector, decision, blurrer)


def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ============================================================
#   API ENDPOINT 1 → DETECTION-ONLY (from new code)
# ============================================================

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        npimg = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        results = detector.process_frame(frame)
        return JSONResponse(content=convert_numpy(results))

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})


# ============================================================
#   API ENDPOINT 2 → FULL PROCESSING (old pipeline)
# ============================================================

@app.post("/process")
async def process(
    image: UploadFile = File(...),
    blur_type: str = "gaussian",
    mode: str = "all",
    text: str = None
):
    """
    Run FULL AnonVision pipeline (Detect + Decide + Blur)
    """
    try:
        # Decode image
        contents = await image.read()
        frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        # Update settings
        decision.mode = mode
        blurrer.blur_type = blur_type

        # Process frame through FULL pipeline
        blurred_frame, results = pipeline.process_frame(frame, user_text=text)

        # Encode output image
        _, buffer = cv2.imencode(".jpg", blurred_frame)

        return {
            "processed_image": buffer.tobytes().hex(),  # hex string
            "results": convert_numpy(results)
        }

    except Exception:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": "Processing failed"})


# ============================================================
#              NGROK TUNNEL STARTER
# ============================================================

def start_ngrok():
    public_url = ngrok.connect(5000)
    print(f"\n🚀 Public API running at: {public_url}\n")


# ============================================================
#        ORIGINAL CLI DEMOS (image/video/webcam)
# ============================================================

def demo_image(image_path, output_path, blur_type="gaussian", user_text=None, mode="all"):
    print("\n===== RUNNING IMAGE DEMO =====")
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Could not load image.")
        return

    decision.mode = mode
    blurrer.blur_type = blur_type

    blurred_frame, results = pipeline.process_frame(frame, user_text=user_text)
    cv2.imwrite(output_path, blurred_frame)
    print(f"Saved output → {output_path}")


def demo_video(video_path, output_path, blur_type="gaussian", user_text=None, mode="all", frame_skip=1):
    pipeline.process_video(video_path, output_path, user_text=user_text,
                           show_debug=True, frame_skip=frame_skip)


def demo_webcam(blur_type="gaussian", user_text=None, mode="all"):
    pipeline.process_webcam(user_text=user_text, show_debug=True)


# ============================================================
#                     MAIN EXECUTION
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AnonVision Combined CLI + API")

    parser.add_argument("--mode", required=True,
        choices=["image", "video", "webcam", "api"],
        help="Run CLI mode or FastAPI server")

    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--blur", default="gaussian",
                        choices=["gaussian", "pixelate", "black_box", "mosaic"])
    parser.add_argument("--text", type=str)
    parser.add_argument("--decision", default="all")
    parser.add_argument("--skip", type=int, default=1)

    args = parser.parse_args()

    if args.mode == "api":
        threading.Thread(target=start_ngrok, daemon=True).start()
        uvicorn.run(app, host="0.0.0.0", port=5000)
        return

    if args.mode == "image":
        demo_image(args.input, args.output, args.blur, args.text, args.decision)

    elif args.mode == "video":
        demo_video(args.input, args.output, args.blur, args.text, args.decision, args.skip)

    elif args.mode == "webcam":
        demo_webcam(args.blur, args.text, args.decision)


if __name__ == "__main__":
    main()
