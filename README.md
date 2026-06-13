<p align="center">
  <h1 align="center">🛡️ Safe Device</h1>
  <p align="center">
    <strong>Face Verification System with Desktop Access Control</strong>
  </p>
  <p align="center">
    Real-time face detection · ArcFace embeddings · Liveness detection · REST API · Desktop enforcement
  </p>
</p>

---

## Overview

**Safe Device** is a complete, GPU-accelerated face verification system built with Python. It detects faces via webcam, generates unique 512-dimensional embeddings using ArcFace, matches them against a stored database, performs liveness checks (anti-spoofing), and exposes everything through a FastAPI REST API.

On top of the core verification pipeline, **FaceGuard** — a background desktop agent — watches every window on your PC and blocks or allows it based on per-user permission profiles. If a guest opens YouTube, the window gets minimized and a desktop notification warns them.

### Key Features

- **Real-time face detection** — InsightFace (`buffalo_l`) with MTCNN fallback
- **512-d ArcFace embeddings** — L2-normalized for fast cosine similarity matching
- **Face database** — Register faces via webcam (averaged over N frames for stability)
- **Liveness detection** — Blink-based anti-spoofing using Eye Aspect Ratio (EAR)
- **REST API** — FastAPI with register, verify, delete, list, and health endpoints
- **Desktop access control** — FaceGuard monitors active windows and enforces rules
- **GPU-accelerated** — CUDA 128 via PyTorch + ONNX Runtime
- **Cross-platform** — Windows (primary), Linux, macOS support

---

## Architecture

The system is organized into 8 modules that build on each other:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Safe Device Architecture                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Webcam Frame                                                      │
│       │                                                             │
│       ▼                                                             │
│   ┌──────────────────┐                                              │
│   │  M1: Detection   │  InsightFace buffalo_l / MTCNN fallback      │
│   │  face_detector.py│  → bbox, keypoints, confidence, embedding    │
│   └────────┬─────────┘                                              │
│            │                                                        │
│       ┌────┴────┐                                                   │
│       ▼         ▼                                                   │
│   ┌────────┐ ┌──────────┐                                           │
│   │M2:Prep │ │M3:Embed  │  ArcFace 512-d vector (norm=1.0)          │
│   │process │ │embedding │  L2-normalized from raw norm ~23           │
│   └────────┘ └────┬─────┘                                           │
│                   │                                                  │
│              ┌────┴────┐                                             │
│              ▼         ▼                                             │
│          ┌────────┐ ┌────────┐                                       │
│          │M4: DB  │ │M6:Live │  Blink detection via EAR proxy        │
│          │faces.pk│ │liveness│  Requires 1 blink before verify       │
│          └───┬────┘ └────────┘                                       │
│              ▼                                                       │
│          ┌────────┐                                                  │
│          │M5:Match│  Cosine similarity (dot product, both unit-norm) │
│          │matcher │  Threshold: 0.4                                  │
│          └───┬────┘                                                  │
│              ▼                                                       │
│   ┌──────────────────┐     ┌──────────────────────┐                  │
│   │ M7: FastAPI      │────▶│ M8: FaceGuard        │                  │
│   │ /register /verify│     │ Window monitor +     │                  │
│   │ /delete /list    │     │ access enforcement   │                  │
│   └──────────────────┘     └──────────────────────┘                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
safe_device/
├── main.py                   # Real-time demo — integrates Modules 1–6
├── register.py               # CLI to register a face (Module 4)
├── api.py                    # FastAPI REST backend (Module 7)
├── profiles.json             # Per-user permission rules (Module 8)
├── requirements.txt          # Core dependencies
├── requirements_api.txt      # FastAPI dependencies
│
├── detectors/
│   └── face_detector.py      # M1 — Face detection (InsightFace + MTCNN)
│
├── utlity/
│   ├── preprocessing.py      # M2 — Crop, align, resize, normalize
│   ├── embedding.py          # M3 — ArcFace embedding extraction
│   ├── matcher.py            # M5 — Cosine similarity matching
│   └── liveness.py           # M6 — Blink-based liveness detection
│
├── faceguard/                # M8 — Desktop Access Control
│   ├── main.py               # Entry point — face scan → monitor loop
│   ├── face_verify.py        # Webcam capture → FastAPI /verify
│   ├── window_monitor.py     # Active window reader (cross-platform)
│   ├── access_rules.py       # Allowed/blocked list logic
│   └── notifier.py           # Desktop notifications (via plyer)
│
├── face_db/
│   └── faces.pkl             # Face database (created at runtime)
│
└── model/
    └── blaze_face_short_range.tflite
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) with CUDA 12.8
- Webcam

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/safe_device.git
cd safe_device

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Install FastAPI dependencies (for Module 7 & 8)
pip install -r requirements_api.txt

# Install FaceGuard dependencies (for Module 8)
pip install psutil plyer requests
```

> **Note:** InsightFace will download the `buffalo_l` model (~300 MB) on the first run.

---

## Quick Start

### 1. Register Your Face

```bash
python register.py --name "Divyansh"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--name` | required | Name to store in the database |
| `--frames` | `10` | Number of frames to average |
| `--threshold` | `0.5` | Minimum detection confidence |

**Controls:** `SPACE` = capture frame · `Q` = abort

The script captures N frames, averages the embeddings, L2-normalizes the result, and saves it to `face_db/faces.pkl`.

### 2. Run the Real-Time Demo

```bash
python main.py
```

**Controls:** `Q` = quit · `R` = reload database from disk

The demo window shows:
- 🟢 **Green box** — verified (name + score displayed)
- 🟠 **Orange box** — waiting for blink (liveness gate)
- 🔴 **Red box** — unknown face

### 3. Start the FastAPI Backend

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Start FaceGuard (Desktop Access Control)

```bash
# Terminal 1 — API must be running
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — start the monitor
python -m faceguard.main
```

---

## Module Reference

### Module 1 — Face Detection (`detectors/face_detector.py`)

**Class: `OpenCVFaceDetector`**

Detects faces in a BGR frame and returns structured detection dictionaries.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `()` | Initializes InsightFace (CUDA → CPU fallback) or MTCNN |
| `detect` | `(frame_bgr) → list[dict]` | Returns list of detection dicts for every face found |
| `close` | `()` | Releases model resources |

**Detection dict returned by `detect()`:**
```python
{
    "bbox":             (x, y, w, h),          # tight bounding box
    "expanded_bbox":    (x, y, w, h),          # 1.35× expanded box for cropping
    "confidence":       float,                  # detection score [0.0, 1.0]
    "keypoints":        [{"name": str, "point": (x, y)}, ...],   # 5 landmarks
    "keypoints_by_name": {"left_eye": (x,y), "right_eye": (x,y), ...},
    "face_center":      (x, y),
    "face_area":        int,
    "face_aspect_ratio": float,
    "face_crop":        np.ndarray,            # BGR crop of the face
    "embedding":        np.ndarray or None,    # L2-normalized 512-d (InsightFace only)
}
```

**Backend priority:** `insightface-cuda` → `insightface-cpu` → `mtcnn`

---

### Module 2 — Preprocessing (`utlity/preprocessing.py`)

**Class: `preprocessing`**

Transforms a raw face crop into a model-ready tensor.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(TARGET_SIZE=(160,160))` | Set target resolution |
| `preprocess_face` | `(frame, detection) → Tensor` | Full pipeline: crop → blur → align → resize → normalize → tensor |
| `preprocess_batch` | `(frame, detections) → Tensor` | Batch version — returns `(N, 3, 160, 160)` |

**Pipeline steps:**
1. **Crop** using `expanded_bbox` from Module 1
2. **Noise reduction** — Gaussian blur (3×3, σ=0.5)
3. **Alignment** — rotate to make eyes horizontal using keypoints
4. **Resize** to 160×160
5. **Normalize** — BGR→RGB, pixels from `[0, 255]` to `[-1.0, 1.0]`
6. **Tensor** — HWC→CHW, add batch dim → `(1, 3, 160, 160)` float32

---

### Module 3 — Face Embedding (`utlity/embedding.py`)

**Class: `ArcFaceEmbedder`**

Fallback embedding extractor (used only when InsightFace is unavailable in Module 1).

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `()` | Loads InsightFace `buffalo_l` model |
| `get_embedding` | `(frame, detection) → np.ndarray` | Crops face → runs ArcFace → returns L2-normalized (512,) |
| `get_embedding_direct` | `(face_crop) → np.ndarray` | Same but accepts a pre-cropped BGR image |

**Critical design note:** InsightFace's raw embeddings have a norm of ~23. All embeddings are L2-normalized to `norm=1.0` before storage or comparison:
```python
raw = face.embedding.astype(np.float32)
embedding = raw / np.linalg.norm(raw)
```

---

### Module 4 — Face Database (`register.py`)

CLI script to register a person's face into the persistent database.

**How it works:**
1. Opens webcam with live preview
2. User presses `SPACE` to capture frames (N = 10 by default)
3. Extracts 512-d embedding from each frame
4. Averages all N embeddings → L2-normalizes
5. Saves to `face_db/faces.pkl` via `FaceMatcher.add()`

**Database format (`face_db/faces.pkl`):**
```python
{
    "Divyansh": np.ndarray(shape=(512,), dtype=float32),  # unit-norm
    "Alice":    np.ndarray(shape=(512,), dtype=float32),
}
```

---

### Module 5 — Face Matching (`utlity/matcher.py`)

**Class: `FaceMatcher`**

Loads the face database and matches live embeddings using cosine similarity.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(db_path=DB_PATH)` | Loads database from disk |
| `match` | `(embedding) → (name, score)` | Compares against all stored faces |
| `add` | `(name, embedding)` | Stores a new face (overwrites if exists) |
| `remove` | `(name) → bool` | Deletes a face from the database |
| `list_names` | `() → list[str]` | Returns all registered names |
| `reload` | `()` | Hot-reloads the database from disk |

**Matching logic:**
- Both vectors are unit-norm → cosine similarity = `np.dot(a, b)`
- Default threshold: **0.4** (configurable via `FaceMatcher.THRESHOLD`)
- Returns `("Unknown", score)` if best score < threshold

---

### Module 6 — Liveness Detection (`utlity/liveness.py`)

**Class: `LivenessChecker`**

Stateful blink-based anti-spoofing. Prevents photo/video replay attacks.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `()` | Initializes blink counter and EAR history |
| `check` | `(detection) → dict` | Processes one frame, returns liveness status |
| `reset` | `()` | Resets blink counter (call when face disappears) |
| `ear_from_6pts` | `(eye_pts) → float` | Standard 6-point EAR (for MediaPipe/dlib) |

**`check()` return value:**
```python
{
    "live":        bool,   # True after required blinks detected
    "blink_count": int,    # total blinks seen
    "ear":         float,  # current Eye Aspect Ratio
    "ear_low":     bool,   # True when EAR < threshold this frame
}
```

**Configurable constants:**
| Constant | Default | Description |
|----------|---------|-------------|
| `EAR_THRESHOLD` | `0.22` | Below this = eye closed |
| `CONSEC_FRAMES` | `2` | Consecutive low-EAR frames needed for blink |
| `REQUIRED_BLINKS` | `1` | Blinks required before "live" is declared |

---

### Module 7 — FastAPI Backend (`api.py`)

REST API wrapping the entire verification pipeline.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/register` | Upload image + name → saves embedding to DB |
| `POST` | `/verify` | Upload image → returns name, score, verified, liveness |
| `DELETE` | `/delete/{name}` | Remove a person from the database |
| `GET` | `/list` | List all registered names |
| `GET` | `/health` | Health check (backend status, registered faces) |

**Example `/verify` response:**
```json
{
    "name": "Divyansh",
    "score": 0.8724,
    "verified": true,
    "live": false,
    "ear": 0.0312,
    "confidence": 0.9987
}
```

**Run:**
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

---

### Module 8 — FaceGuard Desktop Access Control (`faceguard/`)

Background agent that enforces per-user window access rules after face verification.

#### How It Works

```
Face verified via FastAPI → Profile loaded from profiles.json
                                    ↓
              Monitor loop (polls active window every 1.5s)
                                    ↓
              Check window title against allowed/blocked lists
                                    ↓
                    ┌───────────┬───────────────┐
                    ↓           ↓               ↓
                 Allowed     Blocked         Session Expired
                (do nothing) (minimize +     (re-verify
                              notify)        required)
```

#### Files

| File | Purpose |
|------|---------|
| `faceguard/main.py` | Entry point — runs face scan, then starts monitor loop |
| `faceguard/face_verify.py` | Opens webcam → captures frame → sends to FastAPI `/verify` |
| `faceguard/window_monitor.py` | Reads active window title/process name (cross-platform) |
| `faceguard/access_rules.py` | Evaluates window against allowed/blocked/time rules |
| `faceguard/notifier.py` | Shows desktop notification via `plyer` (5-second cooldown) |

#### Profile Configuration (`profiles.json`)

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

| Field | Description |
|-------|-------------|
| `role` | `"admin"` bypasses all checks; `"guest"` enforces all rules |
| `allowed_urls` | Keywords to match in window title. `["*"]` = allow everything |
| `blocked_urls` | Keywords to block (**takes priority** over allowed) |
| `allowed_hours` | Hour range (24h format). Access denied outside this window |
| `session_minutes` | Session duration before re-verification is required |

#### Platform Support

| Platform | Window Detection | Minimize |
|----------|-----------------|----------|
| **Windows** | `ctypes` + `user32.dll` + `psutil` | `ShowWindow(hwnd, SW_MINIMIZE)` |
| **Linux** | `xdotool` + `psutil` | `xdotool windowminimize` |
| **macOS** | `AppKit` (`NSWorkspace`) | `osascript` keystroke |

---

## Tech Stack

| Framework | Version | Purpose |
|-----------|---------|---------|
| [InsightFace](https://github.com/deepinsight/insightface) | latest | Face detection + ArcFace embedding (single pass) |
| [MTCNN](https://github.com/ipazc/mtcnn) | latest | Fallback face detector |
| [PyTorch](https://pytorch.org/) | 2.11.0+cu128 | Tensor operations, GPU acceleration |
| [OpenCV](https://opencv.org/) | latest | Image processing, webcam I/O |
| [FastAPI](https://fastapi.tiangolo.com/) | ≥0.111 | Async REST API with auto-docs |
| [NumPy](https://numpy.org/) | latest | Embedding math, L2 normalization |
| [psutil](https://github.com/giampaolo/psutil) | latest | Process name lookup (FaceGuard) |
| [plyer](https://github.com/kivy/plyer) | latest | Cross-platform desktop notifications |
| [ONNX Runtime](https://onnxruntime.ai/) | latest | InsightFace GPU inference via CUDAExecutionProvider |

---

## Configuration

### Matching Threshold

```python
# In utlity/matcher.py
FaceMatcher.THRESHOLD = 0.4   # increase to 0.5 to reduce false positives
```

### Liveness Sensitivity

```python
# In utlity/liveness.py
LivenessChecker.EAR_THRESHOLD   = 0.22  # lower = harder to trigger blink
LivenessChecker.CONSEC_FRAMES   = 2     # frames eye must stay closed
LivenessChecker.REQUIRED_BLINKS = 1     # blinks needed before "live"
```

### GPU Memory

```python
import torch
torch.cuda.empty_cache()              # free unused GPU memory
torch.cuda.memory_allocated()          # check current usage
```

---

## Common Tasks

### Add a New Person
```bash
python register.py --name "Alice" --frames 15
# Then press R in main.py to reload DB without restart
```

### Remove a Person
```python
from utlity.matcher import FaceMatcher
m = FaceMatcher()
m.remove("Alice")
```

### Add a FaceGuard Profile
Edit `profiles.json` and add a new entry:
```json
"Alice": {
    "role": "guest",
    "allowed_urls": ["google", "github"],
    "blocked_urls": ["youtube", "tiktok"],
    "allowed_hours": { "start": 9, "end": 17 },
    "session_minutes": 45
}
```

---

## Known Limitations

| Issue | Details |
|-------|---------|
| **CUDA version** | System CUDA must match PyTorch build (`cu128` expected) |
| **Raw embedding norm** | InsightFace outputs norm ~23, not 1.0 — always L2-normalize |
| **5-point EAR** | Proxy EAR from InsightFace's 5-point model is less accurate than 6-point dlib. Upgrade to MediaPipe Face Mesh for robust liveness |
| **Single-frame liveness** | `/verify` endpoint cannot detect blinks — it returns a static EAR proxy. Real liveness needs video frames |
| **FaceGuard needs API** | Module 8 calls `http://localhost:8000/verify` — start `api.py` first |
| **Windows-primary** | `window_monitor.py` uses `ctypes`. Linux needs `xdotool`, macOS needs `AppKit` |
| **Notification spam** | `notifier.py` has a 5s cooldown. Increase `COOLDOWN_SECONDS` if needed |

---

## License

This project is for educational and research purposes.

---

<p align="center">
  Built with ❤️ using InsightFace, PyTorch, FastAPI, and OpenCV
</p>
