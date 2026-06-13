import cv2
import numpy as np
from detectors.face_detector import OpenCVFaceDetector
from utlity.preprocessing import  preprocessing
from utlity.embedding import ArcFaceEmbedder 


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("--(!)Error opening camera")
        raise SystemExit(1)

    detector = OpenCVFaceDetector()
    preprocessor = preprocessing()
    try:
        embedder = ArcFaceEmbedder()
    except Exception as exc:
        embedder = None
        print(f"[ArcFace] Embedding disabled: {exc}")
    
    print("Backend:", detector.backend)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = detector.detect(frame)

            for face in faces:
                x, y, w, h = face["expanded_bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                for keypoint in face["keypoints"]:
                    px, py = keypoint["point"]
                    cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)
                if face["face_crop"].size > 0:
                    cv2.imshow("Face Crop", face["face_crop"])

                face_tensor = preprocessor.preprocess_face(frame, face)
                print("Face tensor shape:", face_tensor.shape)

                if embedder is not None:
                    embedding = embedder.get_embedding(frame, face)
                    if embedding is not None:
                        print(f"Shape : {embedding.shape}")          # (512,)
                        print(f"Norm  : {np.linalg.norm(embedding):.6f}")  # ~1.0
                        print(f"Range : [{embedding.min():.4f}, {embedding.max():.4f}]")
                        print(f"First 5: {embedding[:5].round(4)}")

            cv2.imshow("Video Stream", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
