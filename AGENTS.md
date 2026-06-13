# AI Agent Instructions for safe_device

## Project Overview
**safe_device** is a complete face verification system built on computer vision and deep learning. It provides real-time face detection, embedding, matching, liveness detection, and a REST API — all GPU-accelerated via CUDA/PyTorch.

**Core Tech Stack:**
- **Deep Learning**: PyTorch + TensorFlow + Keras
- **Computer Vision**: OpenCV, MediaPipe, MTCNN
- **Face Detection / Embedding**: InsightFace (`buffalo_l`) — primary; MTCNN — fallback
- **Face Matching**: Cosine similarity on L2-normalized 512-d ArcFace embeddings
- **REST API**: FastAPI + Uvicorn (Module 7)
- **GPU Support**: CUDA 128 via PyTorch / ONNX Runtime (InsightFace)
- **Development**: Jupyter notebooks, Python 3.x

---

## Module Architecture

| # | Module | File(s) | Status |
|---|--------|---------|--------|
| M1 | Face Detection | `detectors/face_detector.py` | ✅ Done |
| M2 | Preprocessing | `utlity/preprocessing.py` | ✅ Done |
| M3 | Face Embedding | `utlity/embedding.py` | ✅ Done (norm fixed) |
| M4 | Face Database | `register.py` + `face_db/faces.pkl` | ✅ Done |
| M5 | Face Matching | `utlity/matcher.py` | ✅ Done |
| M6 | Liveness Detection | `utlity/liveness.py` | ✅ Done |
| M7 | FastAPI Backend | `api.py` | ✅ Done |

---

## Environment Setup

### Virtual Environment
```powershell
& d:\work\intership\safe_device\venv\Scripts\Activate.ps1
```

### GPU/CUDA Requirements
- **Critical**: PyTorch with CUDA support (`torch==2.11.0+cu128`)
- InsightFace uses ONNX Runtime with `CUDAExecutionProvider` (falls back to CPU gracefully)

### Installing Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements_api.txt   # for FastAPI (Module 7)
```

---

## Project Structure
```
safe_device/
├── main.py                   # Real-time demo — integrates all 6 modules
├── register.py               # CLI to register a face into the database (M4)
├── api.py                    # FastAPI REST backend (M7)
├── requirements.txt          # Core dependencies
├── requirements_api.txt      # Additional API dependencies (FastAPI, uvicorn)
├── detectors/
│   └── face_detector.py      # M1 — InsightFace + MTCNN fallback
├── utlity/
│   ├── preprocessing.py      # M2 — crop, align, resize, normalize → tensor
│   ├── embedding.py          # M3 — ArcFace embedding (L2-normalized, norm=1.0)
│   ├── matcher.py            # M5 — cosine similarity matching against face_db
│   └── liveness.py           # M6 — blink detection via EAR proxy
├── face_db/
│   └── faces.pkl             # Pickle DB: {"Name": np.ndarray(512,)} — created at runtime
├── model/
│   └── blaze_face_short_range.tflite
├── venv/                     # Virtual environment
└── .git/                     # Version control
```

---

## Development Workflow

### Register a New Face (M4)
```bash
python register.py --name "Divyansh"
# Optional args:
#   --frames 15       (default: 10 frames averaged)
#   --threshold 0.7   (minimum detection confidence)
```
Controls: **SPACE** = capture frame | **Q** = abort

### Run Real-time Verification Demo
```bash
python main.py
```
Controls: **Q** = quit | **R** = reload database from disk

### Run the FastAPI Backend (M7)
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
# Interactive docs: http://localhost:8000/docs
```

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/register` | Upload image + name → saves embedding |
| `POST` | `/verify` | Upload image → returns `{name, score, verified, live}` |
| `DELETE` | `/delete/{name}` | Remove person from database |
| `GET` | `/list` | List all registered names |
| `GET` | `/health` | Health check |

---

## Key Design Decisions

### Embedding Norm Fix (M3)
InsightFace's raw `face.embedding` has a Euclidean norm of ~23.
**All embeddings are L2-normalized to norm=1.0** before storage or comparison.
This means cosine similarity reduces to a simple dot product: `np.dot(a, b)`.
The normalization happens in two places:
- `detectors/face_detector.py` (inline, when cached into the detection dict)
- `utlity/embedding.py` (in `get_embedding` and `get_embedding_direct`)

### No Double Inference
`OpenCVFaceDetector.detect()` returns `embedding` directly from InsightFace's single pass.
`main.py` and `api.py` use `face["embedding"]` directly — ArcFaceEmbedder is only called as a fallback if InsightFace is unavailable.

### Face Database (M4/M5)
- Format: Python `dict` saved as `face_db/faces.pkl`
- Key: person name (str), Value: unit-norm (512,) float32 array
- `register.py` averages N frames to produce a stable embedding before saving
- `FaceMatcher.reload()` enables hot-reloading without restarting `main.py` (press **R**)

### Liveness Detection (M6)
- Uses a proxy EAR (Eye Aspect Ratio) from the 5-point InsightFace landmark model
- A proper 6-point EAR (dlib/MediaPipe) is available via `LivenessChecker.ear_from_6pts()`
- At least 1 confirmed blink is required before `match()` runs in the live demo
- The API `/verify` does a single-frame liveness check (video-based blink not possible via REST)

### Matching Threshold
- Default cosine similarity threshold: **0.4** (configurable in `FaceMatcher.THRESHOLD`)
- With L2-normalized ArcFace embeddings this is conservative — increase to 0.5 if you see false positives

---

## Key Modules & Frameworks

| Framework | Purpose | Notes |
|-----------|---------|-------|
| **InsightFace** | Detection + ArcFace embedding | Single pass returns bbox + kps + embedding |
| **MTCNN** | Face detection fallback | Used if InsightFace unavailable |
| **PyTorch** | Tensor ops + GPU support | CUDA acceleration |
| **OpenCV** | Image processing | Frame reading, drawing |
| **FastAPI** | REST API | Async, auto-docs at /docs |
| **NumPy/SciPy** | Embedding math | Cosine similarity, normalization |

---

## Common Development Tasks

### Adding a New Person
```bash
python register.py --name "Alice"
# Then in main.py, press R to reload the DB without restarting
```

### Removing a Person
```python
from utlity.matcher import FaceMatcher
m = FaceMatcher()
m.remove("Alice")
```

### Adjusting Liveness Sensitivity
```python
# In utlity/liveness.py — class-level constants:
LivenessChecker.EAR_THRESHOLD  = 0.22   # lower = harder to trigger
LivenessChecker.CONSEC_FRAMES  = 2      # frames eye must stay closed
LivenessChecker.REQUIRED_BLINKS = 1    # blinks before "live"
```

### GPU Memory Management
- Use `torch.cuda.empty_cache()` after batch operations
- InsightFace runs ONNX Runtime sessions — memory managed automatically
- Profile with `torch.cuda.memory_allocated()`

---

## Known Pitfalls
- ⚠️ **CUDA Mismatch**: Ensure system CUDA matches PyTorch version (cu128 expected)
- ⚠️ **Norm ~23**: Raw InsightFace embeddings are NOT unit-norm — always normalize before cosine similarity
- ⚠️ **5-point EAR**: The proxy EAR from InsightFace's 5-point model is less accurate than 6-point dlib. For robust liveness, upgrade to MediaPipe Face Mesh
- ⚠️ **Single-frame API liveness**: `/verify` endpoint cannot detect blinks in one image — it returns a static EAR proxy only. Real liveness requires video frames
- ⚠️ **register.py requires webcam**: For API-only deployments, add a `/register` endpoint that accepts multiple frames instead

---
*Last updated: 2026-06-14*
