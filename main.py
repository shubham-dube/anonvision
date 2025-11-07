from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from detector import IntegratedDetector
import cv2
import numpy as np
import uvicorn
import traceback

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
        # Read image bytes
        contents = await image.read()
        npimg = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Run integrated detection
        results = detector.process_frame(frame)

        results = convert_numpy(results)

        return JSONResponse(content=results)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Run the API
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
