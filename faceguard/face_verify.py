# faceguard/face_verify.py
# One-shot face verification via the FastAPI /verify endpoint.
# Opens webcam, captures a single frame (after warm-up), sends to API.

import cv2
import time
import requests

FASTAPI_URL = "http://localhost:8000"


def capture_and_verify() -> dict | None:
    """Open webcam, capture one frame, send to FastAPI /verify.

    Returns:
        API result dict if verified ({"name", "score", "verified", ...}),
        or None on failure / unrecognised face.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[FaceVerify] Cannot open camera")
        return None

    print("[FaceVerify] Look at the camera...")

    # Warm up camera — first few frames are often dark / white-balanced
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[FaceVerify] Failed to capture frame")
        return None

    # Brief preview so user sees what was captured
    cv2.imshow("Scanning face...", frame)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # Encode as JPEG and send to API
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    try:
        response = requests.post(
            f"{FASTAPI_URL}/verify",
            files={"image": ("face.jpg", img_bytes, "image/jpeg")},
            timeout=10,
        )
        result = response.json()

        if result.get("verified"):
            print(f"[FaceVerify] Verified as: {result['name']} "
                  f"(score: {result['score']:.3f})")
            return result
        else:
            print("[FaceVerify] Face not recognized")
            return None

    except requests.exceptions.ConnectionError:
        print("[FaceVerify] FastAPI not running — start it first:\n"
              "  uvicorn api:app --reload --host 0.0.0.0 --port 8000")
        return None
    except Exception as e:
        print(f"[FaceVerify] Error: {e}")
        return None
