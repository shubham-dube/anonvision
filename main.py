from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from detector import IntegratedDetector
import cv2
import numpy as np
import uvicorn

app = FastAPI(
    title="AnonVision Detection API",
    description="Context-Aware Detection API for faces, clothes, and poses.",
    version="1.0.0"
)

# Initialize detector once
detector = IntegratedDetector()

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    try:
        # Read image bytes
        contents = await image.read()
        npimg = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Run integrated detection
        results = detector.process_frame(frame)

        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Run the API
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
