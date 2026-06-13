"""
register.py — Module 4: Face Database Registration
===================================================
Run this script ONCE per person to capture their face embedding and save it to disk.

Usage:
    python register.py --name "Divyansh"
    python register.py --name "Divyansh" --frames 15 --threshold 0.7

What it does:
    1. Opens the webcam.
    2. Detects faces in real-time (Module 1).
    3. Waits for you to press SPACE to capture a frame.
    4. Collects `--frames` good embeddings (skipping low-confidence detections).
    5. Averages them → L2-normalizes → saves to face_db/faces.pkl.

Controls during capture:
    SPACE  — capture current frame
    q      — quit without saving
"""

import argparse
import os
import sys

import cv2
import numpy as np

from detectors.face_detector import OpenCVFaceDetector
from utlity.matcher import FaceMatcher

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Register a face into the database")
parser.add_argument("--name",      required=True,      help="Person's name (used as the database key)")
parser.add_argument("--frames",    type=int, default=10, help="Number of embeddings to average (default: 10)")
parser.add_argument("--threshold", type=float, default=0.5, help="Min detection confidence to accept frame (default: 0.5)")
args = parser.parse_args()

TARGET_FRAMES = args.frames
CONF_THRESHOLD = args.threshold
NAME = args.name.strip()

if not NAME:
    print("[ERROR] --name cannot be empty")
    sys.exit(1)

# ── Init ─────────────────────────────────────────────────────────────────────

print(f"\n=== Face Registration: '{NAME}' ===")
print(f"  Collecting {TARGET_FRAMES} frames.  Press SPACE to capture, Q to quit.\n")

detector = OpenCVFaceDetector()
matcher  = FaceMatcher()

print(f"Detection backend: {detector.backend}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam")
    sys.exit(1)

embeddings_collected: list[np.ndarray] = []

# ── Capture loop ─────────────────────────────────────────────────────────────

try:
    while len(embeddings_collected) < TARGET_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)
        display = frame.copy()

        status_color = (0, 200, 255)   # orange — no face
        status_text  = "No face detected"

        if faces:
            face = faces[0]   # use largest / first face
            x, y, w, h = face["expanded_bbox"]
            conf = face.get("confidence", 0.0) or 0.0

            if conf >= CONF_THRESHOLD or detector.backend == "mtcnn":
                status_text  = f"Face OK (conf={conf:.2f}) — SPACE to capture [{len(embeddings_collected)}/{TARGET_FRAMES}]"
                status_color = (0, 255, 0)   # green
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                status_text  = f"Low confidence ({conf:.2f}) — move closer"
                status_color = (0, 165, 255)
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 165, 255), 2)

        cv2.putText(display, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.putText(display, f"Registering: {NAME}", (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow("Register Face — SPACE to capture, Q to quit", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\n[Aborted] No changes saved.")
            sys.exit(0)

        elif key == ord(" "):   # SPACE — capture frame
            if not faces:
                print("  [Skip] No face in frame")
                continue

            face = faces[0]
            emb  = face.get("embedding")   # already L2-normalized by Module 1/3

            if emb is None:
                print("  [Skip] Embedding unavailable")
                continue

            conf = face.get("confidence", 0.0) or 0.0
            if conf < CONF_THRESHOLD and detector.backend != "mtcnn":
                print(f"  [Skip] Confidence too low ({conf:.2f})")
                continue

            embeddings_collected.append(emb)
            n = len(embeddings_collected)
            print(f"  Captured frame {n}/{TARGET_FRAMES}  (norm={np.linalg.norm(emb):.4f})")

            # Brief flash to confirm capture
            flash = display.copy()
            cv2.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]), (255, 255, 255), 6)
            cv2.imshow("Register Face — SPACE to capture, Q to quit", flash)
            cv2.waitKey(120)

finally:
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

# ── Average + normalize + save ────────────────────────────────────────────────

if len(embeddings_collected) < TARGET_FRAMES:
    print(f"\n[WARNING] Only {len(embeddings_collected)} frames collected (wanted {TARGET_FRAMES})")
    if len(embeddings_collected) == 0:
        print("[ERROR] No embeddings captured. Exiting without saving.")
        sys.exit(1)

stack   = np.stack(embeddings_collected, axis=0)   # (N, 512)
mean_emb = stack.mean(axis=0)                      # (512,)  — average
mean_emb = mean_emb / np.linalg.norm(mean_emb)    # L2-normalize

matcher.add(NAME, mean_emb)

print(f"\n✓ Registered '{NAME}' using {len(embeddings_collected)} frames.")
print(f"  Embedding norm : {np.linalg.norm(mean_emb):.6f}  (should be 1.0)")
print(f"  Database now contains: {matcher.list_names()}")
