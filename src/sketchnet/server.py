"""Loopback API: browser vectors and training share one rasterizer."""
import base64
import io
import json
import logging
import random
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, field_validator

from .config import ARTIFACTS_DIR, CHECKPOINT_PATH, CLASSES, INPUT_SIZE, ROOT
from .model import SketchCNN
from .preprocessing import PREPROCESSING_VERSION, preprocess_strokes
from .data import load_dataset

logger = logging.getLogger(__name__)
lock = threading.Lock()
model = None
model_error = "Model has not been loaded."


def load_model():
    global model, model_error
    model = None
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        if tuple(checkpoint["classes"]) != tuple(CLASSES) or checkpoint["input_size"] != INPUT_SIZE:
            raise ValueError("Checkpoint class mapping or input size is incompatible")
        if checkpoint.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError("Checkpoint preprocessing version is incompatible")
        candidate = SketchCNN(num_classes=len(CLASSES))
        candidate.load_state_dict(checkpoint["model_state"])
        candidate.eval()
        model = candidate
        model_error = None
    except FileNotFoundError:
        model_error = "No trained checkpoint. Run the download and training commands in README."
    except Exception as exc:
        logger.exception("Checkpoint loading failed")
        model_error = f"Checkpoint could not be loaded: {type(exc).__name__}. See server log."


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(4)
    load_model()
    yield


app = FastAPI(title="SketchNet Lab", lifespan=lifespan)


@app.middleware("http")
async def limit_body(request: Request, call_next):
    if request.method == "POST":
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 1_000_000:
                return JSONResponse({"detail": "Drawing exceeds 1 MB limit"}, status_code=413)
        request._body = bytes(body)
    return await call_next(request)


class Drawing(BaseModel):
    strokes: list[list[list[float]]]

    @field_validator("strokes")
    @classmethod
    def validate_strokes(cls, strokes):
        if len(strokes) > 256:
            raise ValueError("Maximum 256 strokes")
        points = 0
        for stroke in strokes:
            if len(stroke) != 2 or len(stroke[0]) != len(stroke[1]) or not stroke[0]:
                raise ValueError("Each stroke needs equal nonempty x/y arrays")
            points += len(stroke[0])
            if not all(np.isfinite(v) and abs(v) <= 10000 for axis in stroke for v in axis):
                raise ValueError("Coordinates must be finite and within ±10000")
        if points > 20000:
            raise ValueError("Maximum 20000 points")
        return strokes


def png_url(array):
    array = np.asarray(array, dtype=np.float32)
    image = Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def read_artifact(name, default=None):
    path = ARTIFACTS_DIR / name
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        logger.exception("Unable to read %s", path)
        return default


@app.get("/api/status")
def status():
    return {
        "ready": model is not None, "error": model_error, "classes": CLASSES,
        "architecture": "SketchCNN · three convolution blocks · CPU",
        "parameters": sum(p.numel() for p in model.parameters()) if model else None,
        "checkpoint_bytes": CHECKPOINT_PATH.stat().st_size if CHECKPOINT_PATH.exists() else None,
        "input_size": INPUT_SIZE, "history": read_artifact("history.json", []),
        "validation": read_artifact("metrics_validation.json"),
        "test": read_artifact("metrics_test.json"), "latency": read_artifact("latency.json"),
    }


@app.post("/api/reload")
def reload_checkpoint():
    """Reload a newly trained local checkpoint without restarting the service."""
    load_model()
    return {"ready": model is not None, "error": model_error}


@app.get("/api/gallery")
def gallery():
    """Return a bounded, labelled sample gallery from the local training set."""
    try:
        images, labels, ids = load_dataset()
        if any(str(item).startswith("synthetic-") for item in ids[: min(20, len(ids))]):
            raise ValueError("Current fixture is synthetic; download Quick Draw before showing gallery")
        rng = random.Random(42)
        indices = list(range(len(labels)))
        rng.shuffle(indices)
        selected = indices[:30]
        return {"items": [{"label": CLASSES[int(labels[i])], "id": str(ids[i]),
                           "image": png_url(images[i, 0])} for i in selected]}
    except Exception as exc:
        logger.exception("Gallery unavailable")
        raise HTTPException(503, f"Gallery unavailable: {type(exc).__name__}") from None


@app.post("/api/predict")
def predict(drawing: Drawing):
    start = time.perf_counter()
    x = torch.as_tensor(preprocess_strokes(drawing.strokes), dtype=torch.float32)
    preprocessed = time.perf_counter()
    preview = png_url(x.squeeze().numpy())
    if not bool(x.any()):
        return {"empty": True, "top": [], "probabilities": [], "preview": preview,
                "activations": [], "latency_ms": {"preprocessing": 0, "inference": 0, "total": 0}}
    if model is None:
        raise HTTPException(503, model_error)
    try:
        with lock, torch.inference_mode():
            infer_start = time.perf_counter()
            features = model.features(x.unsqueeze(0))
            logits = model.classifier(features.flatten(1))
            probabilities = logits.softmax(1)[0].numpy()
            infer_end = time.perf_counter()
            maps = features[0, :6].numpy()
            activations = [png_url(m / max(float(m.max()), 1e-8)) for m in maps]
        order = np.argsort(-probabilities)
        return {"empty": False, "preview": preview, "activations": activations,
                "probabilities": probabilities.tolist(),
                "top": [{"label": CLASSES[i], "probability": float(probabilities[i])}
                        for i in order[:5]],
                "latency_ms": {"preprocessing": (preprocessed-start)*1000,
                               "inference": (infer_end-infer_start)*1000,
                               "total": (time.perf_counter()-start)*1000}}
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(500, "Prediction failed. Check the server log and retry.") from None


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "index.html")


ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_DIR), name="artifacts")
app.mount("/static", StaticFiles(directory=ROOT / "app", check_dir=False), name="static")
