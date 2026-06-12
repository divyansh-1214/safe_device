import os

import cv2
import mediapipe as mp


class OpenCVFaceDetector:
    def __init__(self, model_path: str | None = None, min_detection_confidence: float = 0.5):
        self.detector = None
        self.backend = "opencv"

        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "model",
                "blaze_face_short_range.tflite",
            )

        try:
            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = mp.tasks.vision.FaceDetectorOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                min_detection_confidence=min_detection_confidence,
            )
            self.detector = mp.tasks.vision.FaceDetector.create_from_options(options)
            self.backend = "mediapipe"
        except Exception:
            # Fallback for environments where MediaPipe Tasks runtime is unavailable.
            self.cascade = cv2.CascadeClassifier()
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt.xml")
            if not self.cascade.load(cascade_path):
                raise RuntimeError("Failed to initialize both MediaPipe and OpenCV face detectors")

    def detect(self, frame_bgr):
        if self.backend == "opencv":
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        print(mp_image)
        result = self.detector.detect(mp_image)

        boxes = []
        for detection in result.detections:
            bbox = detection.bounding_box
            
            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            w = int(bbox.width)
            h = int(bbox.height)
            boxes.append((x, y, w, h))
        return boxes

    def close(self):
        if self.detector is not None:
            self.detector.close()
