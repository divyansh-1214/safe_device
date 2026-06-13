"""
main.py — Real-time Face Verification Demo
==========================================
Integrates all modules:
    M1 — Face Detection  (detectors/face_detector.py)
    M2 — Preprocessing   (utlity/preprocessing.py)
    M3 — Embedding       (utlity/embedding.py)       [via M1 cache]
    M4 — Face Database   (face_db/faces.pkl)          [via FaceMatcher]
    M5 — Face Matching   (utlity/matcher.py)
    M6 — Liveness        (utlity/liveness.py)

Controls:
    Q  — quit
    R  — reload face database from disk (useful after running register.py)
"""

import cv2
import numpy as np

from detectors.face_detector import OpenCVFaceDetector
from utlity.preprocessing import preprocessing
from utlity.matcher import FaceMatcher
from utlity.liveness import LivenessChecker


def draw_hud(frame, face, label, score, liveness):
    """Draw bounding box, keypoints, label, and liveness HUD on frame."""
    x, y, w, h = face["expanded_bbox"]

    # Color encodes liveness + match state
    if not liveness["live"]:
        box_color = (0, 165, 255)       # orange — waiting for blink
    elif label == "Unknown":
        box_color = (0, 0, 255)         # red — unrecognized
    else:
        box_color = (0, 220, 50)        # green — verified

    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

    # Keypoints
    for kp in face["keypoints"]:
        px, py = kp["point"]
        cv2.circle(frame, (px, py), 3, (0, 100, 255), -1)

    # Identity label
    id_text = f"{label}  {score:.2f}" if label != "Unknown" else "Unknown"
    cv2.putText(frame, id_text, (x, y - 12),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, box_color, 1, cv2.LINE_AA)

    # Liveness badge
    live_text  = f"LIVE  blinks:{liveness['blink_count']}" if liveness["live"] else f"BLINK REQUIRED  blinks:{liveness['blink_count']}"
    live_color = (0, 220, 50) if liveness["live"] else (0, 165, 255)
    cv2.putText(frame, live_text, (x, y + h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, live_color, 1, cv2.LINE_AA)

    # EAR debug
    cv2.putText(frame, f"EAR:{liveness['ear']:.3f}", (x, y + h + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)


def main():
    # ── Init ──────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("--(!) Error opening camera")
        raise SystemExit(1)

    detector     = OpenCVFaceDetector()
    preprocessor = preprocessing()
    matcher      = FaceMatcher()
    liveness     = LivenessChecker()   # single checker (single-face scenario)

    print(f"Backend      : {detector.backend}")
    print(f"Known faces  : {matcher.list_names()}")
    print("Controls: Q = quit | R = reload DB\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = detector.detect(frame)

            if not faces:
                liveness.reset()   # reset when face disappears
            else:
                # Use the highest-confidence / first face for demo
                face = faces[0]

                # ── Embedding (cached from M1; no double inference) ───────────
                embedding = face.get("embedding")

                # ── Liveness (M6) ─────────────────────────────────────────────
                live_result = liveness.check(face)

                # ── Matching (M5) ─────────────────────────────────────────────
                if embedding is not None and live_result["live"]:
                    name, score = matcher.match(embedding)
                elif embedding is not None:
                    name, score = "⟳ Blink to verify", 0.0
                else:
                    name, score = "No embedding", 0.0

                # ── HUD ───────────────────────────────────────────────────────
                draw_hud(frame, face, name, score, live_result)

                # Face crop preview
                crop = face.get("face_crop")
                if crop is not None and crop.size > 0:
                    cv2.imshow("Face Crop", crop)

            # ── Global HUD ────────────────────────────────────────────────────
            cv2.putText(frame, f"DB: {matcher.list_names() or ['empty']}",
                        (8, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
            cv2.imshow("Safe Device — Face Verification", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                matcher.reload()
                print(f"[main] DB reloaded — {matcher.list_names()}")

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
