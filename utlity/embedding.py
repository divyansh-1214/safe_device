# embedding.py

import numpy as np
import cv2

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover - handled at runtime
    FaceAnalysis = None

class ArcFaceEmbedder:
    def __init__(self):
        if FaceAnalysis is None:
            raise ImportError("insightface is not installed")

        # Initialize — downloads model weights on first run (~300MB)
        try:
            self.app = FaceAnalysis(
                name='buffalo_l',         # best accuracy model
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception:
            self.app = FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=-1, det_size=(640, 640))

        print("[ArcFace] Model loaded successfully")

    def get_embedding(self, frame: np.ndarray, detection: dict) -> np.ndarray:
        """
        Input:  frame      — raw OpenCV BGR frame (from Module 1)
                detection  — one detection dict (from Module 1)
        Output: embedding  — (512,) float32, L2-normalized
        """
        # Crop face using expanded_bbox from Module 1
        x, y, w, h = detection['expanded_bbox']

        # Clamp to frame boundaries
        x, y  = max(0, x), max(0, y)
        w     = min(w, frame.shape[1] - x)
        h     = min(h, frame.shape[0] - y)

        face_crop = frame[y:y+h, x:x+w]

        if face_crop.size == 0:
            return None

        # ArcFace expects RGB, OpenCV gives BGR
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Run ArcFace — returns list of detected faces with embeddings
        faces = self.app.get(face_rgb)

        if len(faces) == 0:
            print("[ArcFace] Warning: no face found in crop")
            return None

        # Take the highest confidence face
        face = max(faces, key=lambda f: f.det_score)

        # L2-normalize so every embedding sits on the unit sphere.
        # InsightFace's raw output has norm ~23; matching requires norm == 1.0.
        raw = face.embedding.astype(np.float32)   # (512,)
        embedding = raw / np.linalg.norm(raw)

        return embedding

    def get_embedding_direct(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Simpler version — pass the cropped face directly if you already have it.
        Input:  face_crop  — BGR numpy array, any size
        Output: embedding  — (512,) float32
        """
        if face_crop.size == 0:
            return None

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        faces = self.app.get(face_rgb)

        if not faces:
            return None

        raw = max(faces, key=lambda f: f.det_score).embedding.astype(np.float32)
        return raw / np.linalg.norm(raw)