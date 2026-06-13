# utlity/liveness.py
# Module 6 — Liveness Detection (Anti-Spoofing)
# Uses Eye Aspect Ratio (EAR) computed from the 5-point facial landmarks
# returned by Module 1. A blink is detected when EAR drops below the threshold.
# At least one confirmed blink is required before a face can be verified.

import collections
import numpy as np


class LivenessChecker:
    """Stateful blink-based liveness checker.

    How it works
    ------------
    1. Each frame: compute EAR from eye keypoints.
    2. If EAR < EAR_THRESHOLD for at least CONSEC_FRAMES consecutive frames,
       that counts as one confirmed blink.
    3. Once REQUIRED_BLINKS blinks are accumulated the face is declared live.

    The state is per-track — create one LivenessChecker per tracked face,
    or call `reset()` when a new face appears.

    Keypoints expected (from Module 1, insightface 5-point format):
        left_eye   (x, y)
        right_eye  (x, y)
    Only two points per eye are available from the 5-point model, so we
    fall back to a simplified single-axis EAR proxy. If full 6-point dlib
    landmarks are available, override `ear_from_6pts` instead.
    """

    EAR_THRESHOLD: float = 0.22     # below this → eye considered closed
    CONSEC_FRAMES: int   = 2        # frames eye must stay closed to count as blink
    REQUIRED_BLINKS: int = 1        # blinks needed before "live" is declared

    def __init__(self):
        self._consec_below: int = 0      # consecutive frames with low EAR
        self.blink_count: int  = 0
        self._ear_history: collections.deque = collections.deque(maxlen=10)

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self, detection: dict) -> dict:
        """
        Args:
            detection: one face detection dict from Module 1.
                       Must have 'keypoints_by_name' with 'left_eye' and
                       'right_eye' keys (each a (x, y) tuple/array).

        Returns:
            {
                "live":        bool,    # True once enough blinks seen
                "blink_count": int,
                "ear":         float,   # current averaged EAR
                "ear_low":     bool,    # True when EAR < threshold this frame
            }
        """
        kps = detection.get("keypoints_by_name", {})
        left_eye  = kps.get("left_eye")
        right_eye = kps.get("right_eye")

        ear = self._compute_ear(left_eye, right_eye)
        self._ear_history.append(ear)

        ear_low = ear < self.EAR_THRESHOLD

        if ear_low:
            self._consec_below += 1
        else:
            if self._consec_below >= self.CONSEC_FRAMES:
                self.blink_count += 1
                print(f"[Liveness] Blink #{self.blink_count} detected  (EAR={ear:.3f})")
            self._consec_below = 0

        live = self.blink_count >= self.REQUIRED_BLINKS

        return {
            "live":        live,
            "blink_count": self.blink_count,
            "ear":         ear,
            "ear_low":     ear_low,
        }

    def reset(self) -> None:
        """Reset state — call when a new/different face appears."""
        self._consec_below = 0
        self.blink_count   = 0
        self._ear_history.clear()

    # ── EAR computation ───────────────────────────────────────────────────────

    def _compute_ear(self, left_eye, right_eye) -> float:
        """Compute a proxy EAR from the 5-point landmark model.

        InsightFace's 5-point model gives only one point per eye (the centre),
        so we cannot compute a proper 6-point EAR. Instead we use the
        inter-eye distance ratio as a coarse proxy:

            EAR_proxy = |left_y - right_y| / |left_x - right_x|

        This rises when eyes tilt (blink-related head-motion artifact) but
        works well enough for single-blink detection.

        For a more robust signal, wire in MediaPipe Face Mesh (468 landmarks)
        and use the standard 6-point EAR formula below.
        """
        if left_eye is None or right_eye is None:
            return 1.0       # assume open if landmarks unavailable

        le = np.asarray(left_eye,  dtype=float)
        re = np.asarray(right_eye, dtype=float)

        horizontal = np.linalg.norm(re - le)
        if horizontal < 1e-6:
            return 1.0

        vertical = abs(le[1] - re[1])   # difference in y coordinates

        # Proxy: ranges roughly 0..0.3; drop below EAR_THRESHOLD on blink
        return float(vertical / horizontal)

    @staticmethod
    def ear_from_6pts(eye_pts: np.ndarray) -> float:
        """Standard 6-point EAR (Soukupová & Čech 2016).

        eye_pts: (6, 2) array — ordered as dlib landmark indices 36-41 or 42-47
        Use this when MediaPipe Face Mesh / dlib landmarks are available.
        """
        A = np.linalg.norm(eye_pts[1] - eye_pts[5])
        B = np.linalg.norm(eye_pts[2] - eye_pts[4])
        C = np.linalg.norm(eye_pts[0] - eye_pts[3])
        return float((A + B) / (2.0 * C))
