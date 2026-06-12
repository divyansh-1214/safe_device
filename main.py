import cv2
from detectors.face_detector import OpenCVFaceDetector


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("--(!)Error opening camera")
        raise SystemExit(1)

    detector = OpenCVFaceDetector()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = detector.detect(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Video Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
