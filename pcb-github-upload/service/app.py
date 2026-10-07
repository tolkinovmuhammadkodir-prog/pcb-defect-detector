"""
service/app.py — FastAPI inference service for the PCB defect detector.

Run locally:   MODEL_PATH=models/best.onnx uvicorn service.app:app --port 8000
Run in Docker: see Dockerfile / README.

Endpoints:
  GET  /health    -> {"status": "ok", "classes": [...]}
  POST /predict   -> multipart upload field "file" (jpg/png), optional ?conf=0.25
"""
import os
import sys
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from infer_onnx import OnnxDetector  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.onnx")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, not per request.
    state["detector"] = OnnxDetector(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="PCB Defect Detector", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "classes": state["detector"].class_names}


@app.post("/predict")
def predict(file: UploadFile = File(...),
            conf: float = Query(0.25, ge=0.01, le=0.99)):
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image larger than 10 MB")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    start = time.perf_counter()
    detections = state["detector"].detect(image, conf=conf)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "image": {"width": image.shape[1], "height": image.shape[0]},
        "conf_threshold": conf,
        "count": len(detections),
        "detections": detections,
        "inference_ms": round(elapsed_ms, 1),
    }
