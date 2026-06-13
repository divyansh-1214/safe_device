import cv2
from mtcnn import MTCNN


class OpenCVFaceDetector:
    def __init__(self):
        self.detector = MTCNN()
        self.backend = "mtcnn"

    def _expand_bbox(self, frame_bgr, bbox, scale=1.35):
        frame_height, frame_width = frame_bgr.shape[:2]
        x, y, w, h = bbox

        center_x = x + w / 2.0
        center_y = y + h / 2.0
        half_width = (w * scale) / 2.0
        half_height = (h * scale) / 2.0

        left = max(0, int(center_x - half_width))
        top = max(0, int(center_y - half_height))
        right = min(frame_width, int(center_x + half_width))
        bottom = min(frame_height, int(center_y + half_height))

        return (left, top, right - left, bottom - top)

    def _clamp_bbox(self, frame_bgr, bbox):
        frame_height, frame_width = frame_bgr.shape[:2]
        x, y, w, h = bbox

        left = max(0, int(x))
        top = max(0, int(y))
        right = min(frame_width, int(x + w))
        bottom = min(frame_height, int(y + h))

        return (left, top, max(0, right - left), max(0, bottom - top))

    def _normalize_keypoints(self, keypoints):
        return [
            {
                "name": name,
                "point": (int(point[0]), int(point[1])),
            }
            for name, point in keypoints.items()
        ]

    def _build_detection(self, frame_bgr, bbox, confidence=None, keypoints=None):
        x, y, w, h = bbox
        clipped_bbox = self._clamp_bbox(frame_bgr, bbox)
        expanded_bbox = self._expand_bbox(frame_bgr, clipped_bbox)
        ex, ey, ew, eh = expanded_bbox
        face_crop = frame_bgr[ey:ey + eh, ex:ex + ew].copy()

        return {
            "bbox": clipped_bbox,
            "expanded_bbox": expanded_bbox,
            "confidence": confidence,
            "keypoints": keypoints or [],
            "keypoints_by_name": {
                keypoint["name"]: keypoint["point"] for keypoint in (keypoints or [])
            },
            "face_center": (int(x + w / 2.0), int(y + h / 2.0)),
            "face_area": max(0, int(w)) * max(0, int(h)),
            "face_aspect_ratio": (float(w) / float(h)) if h else None,
            "face_crop": face_crop,
        }

    def detect(self, frame_bgr):
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(rgb_frame)

        detections_data = []

        if not results:
            return detections_data

        for face in results:
            x, y, w, h = face["box"]
            bbox = (x, y, w, h)
            keypoints = self._normalize_keypoints(face.get("keypoints", {}))

            detections_data.append(
                self._build_detection(
                    frame_bgr,
                    bbox,
                    confidence=float(face.get("confidence", 0.0)),
                    keypoints=keypoints,
                )
            )

        return detections_data

    def close(self):
        self.detector = None