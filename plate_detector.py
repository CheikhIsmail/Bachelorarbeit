from typing import Dict, List
import numpy as np
from yolo_utils import preprocess, scale_boxes_to_original, nms


class PlateDetector:
    def __init__(self, session, imgsz: int = 640, conf: float = 0.40, iou: float = 0.45):
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
            raise ValueError(f"Unexpected plate model output shape: {out.shape}")

        pred = out[0].transpose(1, 0)
        if pred.shape[1] < 5:
            raise ValueError(f"Plate model output has too few channels: {pred.shape}")

        boxes_xywh = pred[:, :4]
        scores = pred[:, 4]

        # If scores are logits, convert them. If already probabilities, keep them.
        if scores.min() < 0 or scores.max() > 1:
            scores = 1.0 / (1.0 + np.exp(-scores))

        boxes = scale_boxes_to_original(boxes_xywh, scale, pad_x, pad_y, img_bgr.shape)
        mask = scores >= self.conf
        if not np.any(mask):
            return []

        b = boxes[mask]
        s = scores[mask]
        keep = nms(b, s, self.iou)

        detections = []
        for idx in keep:
            detections.append({
                "box": b[idx].astype(float),
                "score": float(s[idx]),
                "text": "",
                "vehicle": None
            })

        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections
