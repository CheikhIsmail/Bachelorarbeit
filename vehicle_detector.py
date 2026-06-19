from typing import Dict, List
import numpy as np
from yolo_utils import preprocess, scale_boxes_to_original, nms

VEHICLE_CLASSES = ["BUS", "CARS", "MINIBUS", "SUV", "TRUCK", "VAN"]


class VehicleDetector:
    def __init__(self, session, imgsz: int = 640, conf: float = 0.55, iou: float = 0.45):
        self.session = session
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    def detect(self, img_bgr) -> List[Dict]:
        x, scale, pad_x, pad_y = preprocess(img_bgr, self.imgsz)
        out = self.session.run([self.output_name], {self.input_name: x})[0]

        if out.ndim != 3 or out.shape[0] != 1:
            raise ValueError(f"Unexpected vehicle model output shape: {out.shape}")

        pred = out[0].transpose(1, 0)
        boxes_xywh = pred[:, :4]
        class_logits = pred[:, 4:]
        num_classes = class_logits.shape[1]
        names = VEHICLE_CLASSES if len(VEHICLE_CLASSES) == num_classes else [f"class{i}" for i in range(num_classes)]

        # Same scoring logic as the original vehicle script.
        cls_scores = 1.0 / (1.0 + np.exp(-class_logits))
        boxes = scale_boxes_to_original(boxes_xywh, scale, pad_x, pad_y, img_bgr.shape)

        detections = []
        for c in range(num_classes):
            scores = cls_scores[:, c]
            mask = scores >= self.conf
            if not np.any(mask):
                continue
            b = boxes[mask]
            s = scores[mask]
            keep = nms(b, s, self.iou)
            for idx in keep:
                detections.append({
                    "box": b[idx].astype(float),
                    "score": float(s[idx]),
                    "class_id": int(c),
                    "class_name": names[c],
                    "plates": []
                })

        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections
