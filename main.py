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
    
    known_embeddings = []
    
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
                
                # Use pre-calculated embedding if available (avoids double inference)
                if face.get("embedding") is not None:
                    embedding = face["embedding"]
                elif embedder is not None:
                    embedding = embedder.get_embedding(frame, face)
                else:
                    embedding = None

                face["current_embedding"] = embedding

                if embedding is not None:
                    # Identification
                    best_match_id = -1
                    best_similarity = -1
                    for i, known_emb in enumerate(known_embeddings):
                        # Cosine similarity
                        similarity = np.dot(embedding, known_emb) / (np.linalg.norm(embedding) * np.linalg.norm(known_emb))
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_id = i
                    
                    # Threshold for recognition (0.5 is a common starting point for Cosine Similarity with ArcFace)
                    if best_similarity > 0.5:
                        label = f"ID: {best_match_id} ({best_similarity:.2f})"
                        color = (0, 255, 0)
                    else:
                        label = "Unknown (Press 's' to save)"
                        color = (0, 0, 255)
                        
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imshow("Video Stream", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                # Store unknown faces
                for face in faces:
                    if face.get("current_embedding") is not None:
                        known_embeddings.append(face["current_embedding"])
                        print(f"Stored face with ID: {len(known_embeddings) - 1}")
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
