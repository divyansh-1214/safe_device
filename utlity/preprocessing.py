import cv2
from detectors.face_detector import OpenCVFaceDetector
import numpy as np
import torch
from PIL import Image

# preprocessing.py


 

class preprocessing:
    __doc__ = """
    This module contains the `preprocess_face` function, which takes a raw OpenCV frame"""

    def __init__(self,TARGET_SIZE = (160, 160)):
        self.TARGET_SIZE = TARGET_SIZE

    def preprocess_face( self,frame: np.ndarray, detection: dict) -> torch.Tensor:
        """
        Takes a raw OpenCV frame + one detection dict from Module 1.
        Returns a normalized (1, 3, 160, 160) torch.Tensor ready for FaceNet.
        """

        # ── 1. CROP ──────────────────────────────────────────────────
        x, y, w, h = detection['expanded_bbox']
        # Clamp to frame boundaries (safety check)
        x, y = max(0, x), max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        face_crop = frame[y:y+h, x:x+w].copy()

        # ── 2. NOISE REDUCTION (before resize — more effective) ───────
        face_crop = cv2.GaussianBlur(face_crop, (3, 3), sigmaX=0.5)

        # ── 3. ALIGNMENT ──────────────────────────────────────────────
        kps = detection['keypoints_by_name']
        left_eye  = np.array(kps['left_eye'])
        right_eye = np.array(kps['right_eye'])

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Rotate around the face crop center
        crop_h, crop_w = face_crop.shape[:2]
        center = (crop_w // 2, crop_h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        aligned = cv2.warpAffine(
            face_crop, M, (crop_w, crop_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101   # fills edges cleanly
        )

        # ── 4. RESIZE to 160×160 ──────────────────────────────────────
        resized = cv2.resize(aligned, self.TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

        # ── 5. NORMALIZE pixels to [-1.0, 1.0] ───────────────────────
        # Convert BGR (OpenCV) → RGB first
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) / 127.5) - 1.0

        # ── 6. HWC → CHW → add batch dim → Tensor ────────────────────
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1))  # (3, 160, 160)
        tensor = tensor.unsqueeze(0)                               # (1, 3, 160, 160)

        return tensor
    
    def preprocess_batch(self,frame: np.ndarray, detections: list) -> torch.Tensor:
        """Process multiple detections from one frame into a batched tensor."""
        tensors = [self.preprocess_face(frame, d) for d in detections]
        return torch.cat(tensors, dim=0)   # (N, 3, 160, 160)
