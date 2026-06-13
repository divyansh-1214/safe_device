"""
api.py — Module 7: FastAPI REST Backend
========================================
Wraps the full face verification pipeline into a REST API.

Run with:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Interactive docs available at:
    http://localhost:8000/docs

Endpoints:
    POST   /register          — register a new face (name + image file)
    POST   /verify            — verify a face (image file), returns name + score + liveness
    DELETE /delete/{name}     — remove a person from the database
    GET    /list              — list all registered names
    GET    /health            — health check
"""

import io
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from detectors.face_detector import OpenCVFaceDetector
from utlity.matcher import FaceMatcher
from utlity.liveness import LivenessChecker

# ── App & shared resources ────────────────────────────────────────────────────

app = FastAPI(
    title="Safe Device — Face Verification API",
    description="Face registration and real-time verification using ArcFace embeddings.",
    version="1.0.0",
)

# Shared instances (loaded once at startup)
_detector: OpenCVFaceDetector | None = None
_matcher:  FaceMatcher         | None = None


@app.on_event("startup")
async def startup():
    global _detector, _matcher
    _detector = OpenCVFaceDetector()
    _matcher  = FaceMatcher()
    print(f"[API] Detection backend: {_detector.backend}")
    print(f"[API] Registered faces : {_matcher.list_names()}")


@app.on_event("shutdown")
async def shutdown():
    if _detector:
        _detector.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes → BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Send a valid JPEG/PNG.")
    return img


def _detect_one_face(frame: np.ndarray) -> dict:
    """Run Module 1 and return the highest-confidence face detection."""
    faces = _detector.detect(frame)
    if not faces:
        raise HTTPException(status_code=422, detail="No face detected in the provided image.")
    # Return face with highest confidence
    return max(faces, key=lambda f: f.get("confidence") or 0.0)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Health check — confirms the API and models are loaded."""
    return {
        "status": "ok",
        "backend": _detector.backend if _detector else "not loaded",
        "registered_faces": _matcher.list_names() if _matcher else [],
    }


@app.get("/list", tags=["Database"])
async def list_faces():
    """Return the names of all registered people."""
    return {"names": _matcher.list_names()}


@app.post("/register", tags=["Database"])
async def register(
    name:  str        = Form(..., description="Person's name to register"),
    image: UploadFile = File(..., description="Face image (JPEG or PNG)"),
):
    """
    Register a new face.

    Runs the full detection → embedding pipeline on the uploaded image,
    then saves the L2-normalized 512-d embedding to the database.

    Returns the registered name and the number of faces now in the DB.
    """
    raw = await image.read()
    frame = _decode_image(raw)
    face  = _detect_one_face(frame)

    emb = face.get("embedding")
    if emb is None:
        raise HTTPException(
            status_code=500,
            detail="Embedding extraction failed. Ensure InsightFace is installed and the model is downloaded.",
        )

    _matcher.add(name.strip(), emb)

    return {
        "status": "registered",
        "name":   name.strip(),
        "total_registered": len(_matcher.list_names()),
    }


@app.post("/verify", tags=["Verification"])
async def verify(
    image: UploadFile = File(..., description="Face image to verify (JPEG or PNG)"),
):
    """
    Verify a face against the registered database.

    Returns:
        - `name`     — matched person's name ("Unknown" if no match)
        - `score`    — cosine similarity score [0.0, 1.0]
        - `verified` — True if score ≥ threshold (0.4)
        - `live`     — liveness result (single-frame proxy; for full liveness use the webcam loop)
        - `confidence` — face detection confidence
    """
    raw   = await image.read()
    frame = _decode_image(raw)
    face  = _detect_one_face(frame)

    emb = face.get("embedding")
    if emb is None:
        raise HTTPException(
            status_code=500,
            detail="Embedding extraction failed.",
        )

    name, score = _matcher.match(emb)

    # Single-frame liveness check (proxy — blink-based liveness requires video)
    liveness_checker = LivenessChecker()
    liveness_result  = liveness_checker.check(face)

    return {
        "name":       name,
        "score":      round(float(score), 4),
        "verified":   name != "Unknown",
        "live":       liveness_result["live"],
        "ear":        round(liveness_result["ear"], 4),
        "confidence": round(float(face.get("confidence") or 0.0), 4),
    }


@app.delete("/delete/{name}", tags=["Database"])
async def delete(name: str):
    """
    Remove a registered person from the database.

    Returns 404 if the name is not found.
    """
    found = _matcher.remove(name)
    if not found:
        raise HTTPException(status_code=404, detail=f"'{name}' not found in database.")
    return {
        "status":   "deleted",
        "name":     name,
        "remaining": _matcher.list_names(),
    }
