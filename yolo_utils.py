from typing import List, Tuple
import cv2
import numpy as np
import onnxruntime as ort


def create_session(model_path: str) -> ort.InferenceSession:
    """Create an ONNX Runtime session. Uses CUDA if available, otherwise CPU."""
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    providers = [p for p in preferred if p in available]

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=options, providers=providers)


def letterbox(img: np.ndarray, new_size: int = 640, color: Tuple[int, int, int] = (114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_w = new_size - nw
    pad_h = new_size - nh
    left = pad_w // 2
    top = pad_h // 2
    right = pad_w - left
    bottom = pad_h - top

    boxed = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return boxed, scale, left, top


def preprocess(img_bgr: np.ndarray, imgsz: int):
    img_lb, scale, pad_x, pad_y = letterbox(img_bgr, imgsz)
    x = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]
    return x, scale, pad_x, pad_y


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_boxes_to_original(boxes_xywh: np.ndarray, scale: float, pad_x: int, pad_y: int, img_shape):
    boxes = xywh2xyxy(boxes_xywh)
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= scale

    h, w = img_shape[:2]
    boxes[:, 0] = boxes[:, 0].clip(0, w - 1)
    boxes[:, 2] = boxes[:, 2].clip(0, w - 1)
    boxes[:, 1] = boxes[:, 1].clip(0, h - 1)
    boxes[:, 3] = boxes[:, 3].clip(0, h - 1)
    return boxes


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        denom = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / denom
        order = order[1:][iou <= iou_thres]

    return keep
