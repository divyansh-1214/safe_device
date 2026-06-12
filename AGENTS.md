# AI Agent Instructions for safe_device

## Project Overview
**safe_device** is a computer vision and deep learning project focused on face recognition, detection, and person tracking. The project uses GPU acceleration (CUDA/PyTorch) for efficient inference and is built for an internship program.

**Core Tech Stack:**
- **Deep Learning**: PyTorch + TensorFlow + Keras
- **Computer Vision**: OpenCV, MediaPipe, MTCNN
- **Face Recognition**: Keras-FaceNet
- **Data Processing**: NumPy, Pandas, SciPy, scikit-learn
- **GPU Support**: CUDA 128 via PyTorch/TensorFlow
- **Development**: Jupyter notebooks, Python 3.x

## Environment Setup

### Virtual Environment
The project uses a Python virtual environment (`venv/`) for dependency isolation.

**Activation (already done in terminals):**
```powershell
& d:\work\intership\safe_device\venv\Scripts\Activate.ps1
```

### GPU/CUDA Requirements
- **Critical**: The project depends on PyTorch with CUDA support (`torch==2.11.0+cu128`)
- Main script (`main.py`) validates GPU availability at startup:
  - `torch.cuda.is_available()` - checks if CUDA is accessible
  - `torch.cuda.get_device_name(0)` - retrieves GPU device name
- If CUDA is unavailable, PyTorch will fall back to CPU (slower for inference)

### Installing Dependencies
```bash
pip install -r requirements.txt
```

## Development Workflow

### Before Running Code
1. **Activate virtual environment** (required)
2. **Verify GPU availability** by running `main.py` or checking `torch.cuda.is_available()`
3. **Check CUDA availability** - if GPU not detected, ensure:
   - NVIDIA drivers are installed
   - CUDA Toolkit is compatible with system
   - PyTorch was installed with the correct CUDA version

### Running the Project
```bash
python main.py
```

## Key Modules & Frameworks

| Framework | Purpose | Notes |
|-----------|---------|-------|
| **PyTorch** | Primary DL framework | GPU acceleration via CUDA |
| **TensorFlow/Keras** | Alternative DL framework | Face recognition models |
| **OpenCV** | Image processing | Frame reading, preprocessing |
| **MediaPipe** | Pose/face detection | Ready-to-use detection pipelines |
| **MTCNN** | Face detection | Robust multi-scale face detection |
| **Keras-FaceNet** | Face embedding | Extract face embeddings for recognition |
| **Jupyter** | Interactive development | For experimentation & visualization |

## Project Structure
```
safe_device/
├── main.py              # Entry point - GPU/CUDA verification
├── requirements.txt     # Python dependencies (CPU & GPU versions)
├── README.md           # (To be filled)
├── venv/               # Virtual environment
└── .git/              # Version control
```

## Common Development Tasks

### Adding New Features
1. Add imports at the top of new modules
2. Ensure CUDA compatibility for any GPU-intensive operations
3. Test with `torch.cuda.is_available()` before GPU operations
4. Use try-except for graceful CPU fallback if GPU unavailable

### Working with Models
- Face detection: Use MTCNN or MediaPipe
- Face recognition: Use Keras-FaceNet for embeddings + scikit-learn for matching
- General inference: PyTorch or TensorFlow as needed

### GPU Memory Management
- Be mindful of batch sizes - large batches on GPU can cause OOM errors
- Use `torch.cuda.empty_cache()` to clear GPU memory if needed
- Profile memory usage with `torch.cuda.memory_allocated()`

## Conventions & Notes

**Naming**: Use snake_case for variables/functions, PascalCase for classes
**GPU-First Design**: Assume GPU will be available; handle CPU fallback gracefully
**Logging**: Use print() for now; consider structured logging (logging module) for production
**Testing**: Run `main.py` first to verify GPU setup before running other scripts

## Known Pitfalls
- ⚠️ **CUDA Mismatch**: Ensure system CUDA version matches PyTorch version (cu128 expected)
- ⚠️ **GPU Memory**: Monitor GPU memory usage - face recognition models can be memory-intensive
- ⚠️ **Driver Issues**: Outdated NVIDIA drivers prevent CUDA detection
- ⚠️ **Dependencies**: Some dependencies (OpenCV, MediaPipe) have CPU/GPU variants - requirements.txt specifies GPU versions

## Next Steps for AI Agents
- When implementing face detection features, reference MTCNN or MediaPipe modules
- For face recognition, use Keras-FaceNet embeddings + nearest neighbor search
- Always validate GPU availability before GPU-intensive operations
- Add unit tests for CPU-GPU compatibility
- Document any new model architectures or custom preprocessing steps

---
*Last updated: 2026-06-12*
