from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from detector import IntegratedDetector
from pyngrok import ngrok
import cv2
import numpy as np
import uvicorn
import traceback
import threading

# Initialize FastAPI app
app = FastAPI(
    title="AnonVision Detection API",
    description="Context-Aware Detection API for faces, clothes, and poses.",
    version="1.0.0"
)

# Initialize detector once
detector = IntegratedDetector()

def convert_numpy(obj):
    """Recursively convert NumPy types to native Python types."""
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
    else:
        return obj


@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        npimg = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        results = detector.process_frame(frame)
        results = convert_numpy(results)

        return JSONResponse(content=results)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})


# ==== Ngrok + Uvicorn setup ====
def start_ngrok():
    """Open an ngrok tunnel and print the public URL."""
    public_url = ngrok.connect(5000)
    print(f"\n🚀 Public FastAPI URL: {public_url}\n")


if __name__ == "__main__":
    # Start ngrok tunnel in a background thread (so it doesn’t block uvicorn)
    threading.Thread(target=start_ngrok, daemon=True).start()

    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=5000)
