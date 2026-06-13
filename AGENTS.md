# AI Agent Instructions for safe_device

## Project Overview
**safe_device** is a complete face verification system with desktop access control, built on computer vision and deep learning. It provides real-time face detection, embedding, matching, liveness detection, a REST API, and a background access-control agent that monitors open windows and enforces per-user permission rules — all GPU-accelerated via CUDA/PyTorch.

**Core Tech Stack:**
- **Deep Learning**: PyTorch + TensorFlow + Keras
- **Computer Vision**: OpenCV, MediaPipe, MTCNN
- **Face Detection / Embedding**: InsightFace (`buffalo_l`) — primary; MTCNN — fallback
- **Face Matching**: Cosine similarity on L2-normalized 512-d ArcFace embeddings
- **REST API**: FastAPI + Uvicorn (Module 7)
- **Desktop Access Control**: FaceGuard agent — window monitoring + rule enforcement (Module 8)
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
| M8 | Desktop Access Control | `faceguard/main.py` + `faceguard/*.py` + `profiles.json` | ✅ Done |

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
├── profiles.json             # M8 — per-user permission rules
├── requirements.txt          # Core dependencies
├── requirements_api.txt      # Additional API dependencies (FastAPI, uvicorn)
├── detectors/
│   └── face_detector.py      # M1 — InsightFace + MTCNN fallback
├── utlity/
│   ├── preprocessing.py      # M2 — crop, align, resize, normalize → tensor
│   ├── embedding.py          # M3 — ArcFace embedding (L2-normalized, norm=1.0)
│   ├── matcher.py            # M5 — cosine similarity matching against face_db
│   └── liveness.py           # M6 — blink detection via EAR proxy
├── faceguard/                # M8 — Desktop Access Control (FaceGuard)
│   ├── main.py               # Entry point — face scan → monitor loop
│   ├── face_verify.py        # Calls FastAPI /verify to identify user
│   ├── window_monitor.py     # Reads active window title/process (cross-platform)
│   ├── access_rules.py       # Checks window against allowed/blocked lists
│   └── notifier.py           # Desktop warning popup (via plyer)
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

### Run FaceGuard Desktop Access Control (M8)
```bash
# Terminal 1 — start FastAPI backend first
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — start the desktop monitor
python -m faceguard.main
```

**Runtime flow:**
1. Camera opens → captures a frame → sends to FastAPI `/verify`
2. If verified → loads matching profile from `profiles.json`
3. Background loop polls the active window every 1.5 seconds
4. If the window title matches a blocked keyword → window is minimized + desktop notification shown
5. Session expires after `session_minutes` → requires re-verification

**profiles.json format:**
```json
{
    "Divyansh": {
        "role": "admin",
        "allowed_urls": ["*"],
        "blocked_urls": [],
        "allowed_hours": { "start": 0, "end": 23 },
        "session_minutes": 60
    },
    "Guest": {
        "role": "guest",
        "allowed_urls": ["google", "stackoverflow", "github"],
        "blocked_urls": ["youtube", "instagram", "facebook", "netflix"],
        "allowed_hours": { "start": 9, "end": 18 },
        "session_minutes": 30
    }
}
```
- `role: "admin"` → bypasses all URL and time checks
- `blocked_urls` takes priority over `allowed_urls`
- `allowed_urls: ["*"]` → wildcard, everything allowed
- `allowed_hours` → access denied outside this window
- `session_minutes` → auto-expire after this duration

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

### Desktop Access Control (M8 — FaceGuard)
- Runs as a background script, completely independent from `main.py`
- Requires FastAPI (`api.py`) to be running — communicates via HTTP `POST /verify`
- `face_verify.py` opens webcam once, captures a single frame, sends to API
- `window_monitor.py` uses platform-native APIs: `ctypes` (Windows), `xdotool` (Linux), `AppKit` (macOS)
- `access_rules.py` matches active window title/process against the user's profile from `profiles.json`
- `notifier.py` uses `plyer` for cross-platform desktop notifications with a 5-second cooldown
- Blocked windows are immediately minimized via `ShowWindow(hwnd, SW_MINIMIZE)` on Windows
- Internal windows (Task Manager, Terminal, Python) are auto-skipped to avoid locking the user out
- Session expiry enforced — user must re-verify face after `session_minutes` elapse

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
| **psutil** | Process name from PID | Used by `window_monitor.py` (M8) |
| **plyer** | Cross-platform notifications | Desktop popups in `notifier.py` (M8) |
| **requests** | HTTP client | `face_verify.py` calls FastAPI (M8) |

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
- ⚠️ **FaceGuard requires FastAPI running**: Module 8 calls `http://localhost:8000/verify` — start `api.py` before launching FaceGuard
- ⚠️ **Window monitor is Windows-primary**: Linux requires `xdotool`, macOS requires `AppKit` / PyObjC. Test on target OS
- ⚠️ **psutil permission**: On some systems `psutil.Process(pid).name()` may require elevated privileges
- ⚠️ **Notification spam**: `notifier.py` has a 5-second cooldown, but rapid window switching can still feel noisy — increase `COOLDOWN_SECONDS` if needed

---
*Last updated: 2026-06-14*
