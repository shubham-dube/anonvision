# detection/pose_estimator.py
# Robust pose estimator: YOLOv8-pose (ultralytics) primary, MediaPipe fallback.
# Install: pip install ultralytics opencv-python-headless mediapipe numpy

import cv2
import numpy as np
import traceback

# try ultralytics YOLOv8 pose
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False

# MediaPipe fallback
try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except Exception:
    _HAS_MEDIAPIPE = False

class PoseEstimator:
    """
    PoseEstimator returns a list of detections:
    [
      {
        "person_box": (x1, y1, x2, y2),   # pixel coords in original frame
        "keypoints": {idx: (x_px, y_px, conf), ...}  # pixel coords + confidence
      }, ...
    ]
    Methods:
      - estimate(frame, person_bbox=None)
      - draw(frame, detections)  # draws keypoints and boxes on a copy and returns it
    """

    def __init__(self, model_path="yolov8n-pose.pt", device=None, conf=0.25):
        self.device = device or ("cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
        self.conf = conf

        self.use_yolo = False
        self.yolo = None
        if _HAS_YOLO:
            try:
                self.yolo = YOLO(model_path)  # will auto-download if missing
                # set default conf threshold
                self.yolo.overrides = getattr(self.yolo, "overrides", {})
                self.yolo.overrides["conf"] = conf
                self.use_yolo = True
            except Exception:
                self.yolo = None
                self.use_yolo = False

        # media pipe setup
        if _HAS_MEDIAPIPE:
            self.mp_pose = mp.solutions.pose
            self.mp_pose_instance = self.mp_pose.Pose(
                static_image_mode=False, model_complexity=1,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
        else:
            self.mp_pose = None
            self.mp_pose_instance = None

    # ---------- helpers to extract arrays safely ----------
    @staticmethod
    def _kp_obj_to_numpy(kp_obj):
        """
        Convert ultralytics Keypoints-like obj to numpy array shape (n_persons, n_kp, 3)
        """
        try:
            # new API: Keypoints has .data (torch tensor)
            if hasattr(kp_obj, "data"):
                arr = kp_obj.data.cpu().numpy()
                return arr
            # older API possibilities
            if hasattr(kp_obj, "xy"):
                # xy might be (n, k, 3) tensor
                arr = kp_obj.xy.cpu().numpy()
                return arr
            # fallback: maybe it's already array-like
            arr = np.array(kp_obj)
            return arr
        except Exception:
            # final fallback: try to coerce to numpy
            try:
                return np.asarray(kp_obj)
            except Exception:
                return None

    @staticmethod
    def _boxes_obj_to_numpy(boxes_obj):
        """
        Convert ultralytics Boxes-like obj to numpy array (n_persons, 4) in xyxy pixel coords
        """
        try:
            if boxes_obj is None:
                return None
            if hasattr(boxes_obj, "xyxy"):
                a = boxes_obj.xyxy
                try:
                    return a.cpu().numpy()
                except Exception:
                    return np.array(a)
            if hasattr(boxes_obj, "xywh"):
                a = boxes_obj.xywh
                try:
                    xywh = a.cpu().numpy()
                except Exception:
                    xywh = np.array(a)
                # convert to xyxy
                xyxy = np.zeros((xywh.shape[0], 4))
                xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
                xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
                xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
                xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
                return xyxy
            # fallback
            return np.array(boxes_obj)
        except Exception:
            return None

    @staticmethod
    def _map_kps_to_frame(kps, crop_box, img_shape):
        """
        kps: (num_kp, 3) array where x,y either pixels or normalized [0,1]
        crop_box: (x1, y1, x2, y2) coordinates of the crop in original frame
        img_shape: (h, w) shape of the image that kps are relative to (the cropped image)
        Returns dict idx -> (x_px, y_px, conf)
        """
        x1, y1, x2, y2 = crop_box
        crop_h, crop_w = img_shape
        mapped = {}
        for idx, (x, y, c) in enumerate(kps):
            # detect whether x,y are pixels (>1.5) or normalized (<=1.0)
            if x > 1.5 or y > 1.5:
                # pixel coords relative to cropped image
                x_px = float(x1 + float(x))
                y_px = float(y1 + float(y))
            else:
                # normalized coords
                x_px = float(x1 + float(x) * max(1, crop_w))
                y_px = float(y1 + float(y) * max(1, crop_h))
            mapped[idx] = (x_px, y_px, float(c))
        return mapped

    # ---------- main API ----------
    def estimate(self, frame, person_bbox=None):
        """
        frame: BGR image (numpy)
        person_bbox: optional (x, y, w, h) or (x1,y1,x2,y2) — if provided, run pose only on that crop.
        returns list of detection dicts (see class docstring)
        """
        try:
            h_frame, w_frame = frame.shape[:2]

            if person_bbox is not None:
                # allow (x,y,w,h) or (x1,y1,x2,y2)
                if len(person_bbox) == 4:
                    x, y, a, b = person_bbox
                    # detect whether it's xywh (w small relative) or xyxy
                    if a <= w_frame and b <= h_frame and (x + a <= w_frame and y + b <= h_frame):
                        # treat as xywh
                        x1, y1, x2, y2 = int(x), int(y), int(x + a), int(y + b)
                    else:
                        # assume xyxy
                        x1, y1, x2, y2 = int(x), int(y), int(a), int(b)
                else:
                    # fallback: full frame
                    x1, y1, x2, y2 = 0, 0, w_frame, h_frame
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    crop = frame.copy()
                    crop_box = (0, 0, w_frame, h_frame)
                else:
                    crop_box = (x1, y1, x2, y2)
                return self._infer_on_crop(crop, crop_box)
            else:
                # infer on whole frame
                return self._infer_on_crop(frame, (0, 0, w_frame, h_frame))
        except Exception as e:
            print("[PoseEstimator] estimate error:", e)
            print(traceback.format_exc())
            return []

    def _infer_on_crop(self, crop_img, crop_box):
        """
        Perform inference on the image crop (BGR), return standardized detections list.
        """
        # ensure crop is BGR numpy
        h, w = crop_img.shape[:2]

        # first try YOLOv8 pose
        if self.use_yolo and self.yolo is not None:
            try:
                img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                # ultralytics accept either path or numpy array
                results = self.yolo.predict(img_rgb, device=self.device, imgsz=640, conf=self.conf, verbose=False)

                detections = []
                for r in results:  # r is a Results object
                    boxes_np = self._boxes_obj_to_numpy(getattr(r, "boxes", None))
                    kps_np = self._kp_obj_to_numpy(getattr(r, "keypoints", None))

                    if kps_np is None:
                        continue

                    # kps_np shape could be (n_persons, n_kp, 3) or (n_kp, 3)
                    if kps_np.ndim == 3:
                        n_persons = kps_np.shape[0]
                        for i in range(n_persons):
                            person_kps = kps_np[i]  # (num_kp, 3)
                            # get bbox for this person if available
                            if boxes_np is not None and len(boxes_np) > i:
                                x1, y1, x2, y2 = [float(v) for v in boxes_np[i]]
                            else:
                                xs = person_kps[:, 0]
                                ys = person_kps[:, 1]
                                # if x,y are normalized, compute min/max in normalized then scale
                                if xs.max() <= 1.01 and ys.max() <= 1.01:
                                    x1 = float(xs.min() * w)
                                    y1 = float(ys.min() * h)
                                    x2 = float(xs.max() * w)
                                    y2 = float(ys.max() * h)
                                else:
                                    x1 = float(xs.min())
                                    y1 = float(ys.min())
                                    x2 = float(xs.max())
                                    y2 = float(ys.max())

                            mapped = self._map_kps_to_frame(person_kps, crop_box, (h, w))
                            detections.append({
                                "person_box": (float(crop_box[0] + x1), float(crop_box[1] + y1),
                                               float(crop_box[0] + x2), float(crop_box[1] + y2)),
                                "keypoints": mapped
                            })
                    elif kps_np.ndim == 2:
                        person_kps = kps_np
                        xs = person_kps[:, 0]
                        ys = person_kps[:, 1]
                        if xs.max() <= 1.01 and ys.max() <= 1.01:
                            x1 = float(xs.min() * w)
                            y1 = float(ys.min() * h)
                            x2 = float(xs.max() * w)
                            y2 = float(ys.max() * h)
                        else:
                            x1 = float(xs.min()); y1 = float(ys.min()); x2 = float(xs.max()); y2 = float(ys.max())
                        mapped = self._map_kps_to_frame(person_kps, crop_box, (h, w))
                        detections.append({
                            "person_box": (float(crop_box[0] + x1), float(crop_box[1] + y1),
                                           float(crop_box[0] + x2), float(crop_box[1] + y2)),
                            "keypoints": mapped
                        })
                    else:
                        # unknown format, skip
                        continue

                if detections:
                    return detections
            except Exception as e:
                print("[PoseEstimator] YOLO inference error:", e)
                print(traceback.format_exc())
                # continue to fallback

        # Fallback to MediaPipe if YOLO unavailable or failed
        if self.mp_pose_instance is not None:
            try:
                img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                res = self.mp_pose_instance.process(img_rgb)
                if not res.pose_landmarks:
                    return []
                lm = res.pose_landmarks.landmark
                mp_kps = []
                for l in lm:
                    mp_kps.append((l.x, l.y, l.visibility))
                person_kps = np.array(mp_kps)  # normalized coords
                mapped = self._map_kps_to_frame(person_kps, crop_box, (h, w))
                xs = np.array([p[0] for p in mp_kps]) * w
                ys = np.array([p[1] for p in mp_kps]) * h
                x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
                return [{
                    "person_box": (float(crop_box[0] + x1), float(crop_box[1] + y1),
                                   float(crop_box[0] + x2), float(crop_box[1] + y2)),
                    "keypoints": mapped
                }]
            except Exception as e:
                print("[PoseEstimator] MediaPipe error:", e)
                print(traceback.format_exc())

        # nothing detected or all failed
        return []

    # ---------- utility: draw results ----------
    @staticmethod
    def draw(frame, detections, kp_color=(0,255,0), box_color=(0,128,255), kp_radius=3, box_thickness=2):
        """
        Draw boxes and keypoints on a copy of the frame and return it.
        detections: list returned by estimate()
        """
        out = frame.copy()
        for det in detections:
            x1,y1,x2,y2 = map(int, det.get("person_box", (0,0,0,0)))
            cv2.rectangle(out, (x1,y1), (x2,y2), box_color, thickness=box_thickness)
            kps = det.get("keypoints", {})
            for idx, (x_px, y_px, conf) in kps.items():
                if conf < 0.05:
                    continue
                cv2.circle(out, (int(x_px), int(y_px)), kp_radius, kp_color, -1)
        return out
