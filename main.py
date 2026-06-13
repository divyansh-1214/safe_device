import cv2
from detectors.face_detector import OpenCVFaceDetector
import numpy as np
import torch
from PIL import Image

from utlity.preprocessing import  preprocessing


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("--(!)Error opening camera")
        raise SystemExit(1)

    detector = OpenCVFaceDetector()
    preprocessor = preprocessing()
    print("Backend:", detector.backend)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = detector.detect(frame)
            
            for det in faces:
                face_tensor = preprocessor.preprocess_face(frame, det)
                print("Face tensor shape:", face_tensor.shape)
                # print(face_tensor)
            # print(
            #     [
            #         {
            #             "bbox": face["bbox"],
            #             "expanded_bbox": face["expanded_bbox"],
            #             "confidence": face["confidence"],
            #             "keypoints": face["keypoints"],
            #             "keypoints_by_name": face["keypoints_by_name"],
            #             "face_center": face["face_center"],
            #             "face_area": face["face_area"],
            #             "face_aspect_ratio": face["face_aspect_ratio"],
            #             "face_crop_shape": face["face_crop"].shape,
            #         }
            #         for face in faces
            #     ]
            # )
            for face in faces:
                x, y, w, h = face["expanded_bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                for keypoint in face["keypoints"]:
                    px, py = keypoint["point"]
                    cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)
                if face["face_crop"].size > 0:
                    cv2.imshow("Face Crop", face["face_crop"])

            cv2.imshow("Video Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
