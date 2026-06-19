from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from ocr_utils import recognize_plate


def safe_crop_xyxy(img: np.ndarray, box, pad: int = 8):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None
    return img[y1:y2, x1:x2].copy()


def box_center(box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def contains_point(box, point) -> bool:
    x1, y1, x2, y2 = box
    px, py = point
    return x1 <= px <= x2 and y1 <= py <= y2


def assign_plates_to_vehicles(vehicles: List[Dict], plates: List[Dict]) -> None:
    for vehicle in vehicles:
        vehicle["plates"] = []

    for plate in plates:
        center = box_center(plate["box"])
        candidates = [v for v in vehicles if contains_point(v["box"], center)]
        if candidates:
            best_vehicle = min(candidates, key=lambda v: box_area(v["box"]))
            best_vehicle["plates"].append(plate)
            plate["vehicle"] = best_vehicle
        else:
            plate["vehicle"] = None


def draw_label(img: np.ndarray, text: str, x: int, y: int, color, scale: float = 0.75):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x0 = max(2, x)
    y0 = y - th - baseline - 8
    if y0 < 2:
        y0 = y + 2

    x1 = min(img.shape[1] - 2, x0 + tw + 10)
    y1 = min(img.shape[0] - 2, y0 + th + baseline + 8)

    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.putText(img, text, (x0 + 5, y1 - baseline - 4), font, scale, color, thickness)


def draw_results(img_bgr: np.ndarray, vehicles: List[Dict], plates: List[Dict]) -> np.ndarray:
    vis = img_bgr.copy()

    for vehicle in vehicles:
        x1, y1, x2, y2 = map(int, vehicle["box"])
        assigned_plates = vehicle.get("plates", [])
        if assigned_plates:
            best_plate = max(assigned_plates, key=lambda p: p["score"])
            plate_text = best_plate.get("text", "") or "NO_OCR"
        else:
            plate_text = "NO_PLATE"

        label = f"{vehicle['class_name']} | {plate_text}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        draw_label(vis, label, x1, y1, (0, 255, 255))

    for plate in plates:
        x1, y1, x2, y2 = map(int, plate["box"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return vis


class ALPRVehiclePipeline:
    def __init__(self, vehicle_detector, plate_detector, plate_crop_pad: int = 8):
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.plate_crop_pad = plate_crop_pad

    def process_image(self, image_path: Path, out_dir: Path) -> List[Dict]:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[warn] Cannot read image: {image_path}")
            return []

        vehicles = self.vehicle_detector.detect(img)
        plates = self.plate_detector.detect(img)

        for plate in plates:
            crop = safe_crop_xyxy(img, plate["box"], pad=self.plate_crop_pad)
            plate["text"] = recognize_plate(crop, out_dir) if crop is not None else "NO_CROP"

        assign_plates_to_vehicles(vehicles, plates)
        vis = draw_results(img, vehicles, plates)

        out_path = out_dir / f"{image_path.stem}_combined.jpg"
        cv2.imwrite(str(out_path), vis)

        rows = []
        for vehicle in vehicles:
            assigned = vehicle.get("plates", [])
            plate_text = "NO_PLATE"
            plate_conf = None
            if assigned:
                best_plate = max(assigned, key=lambda p: p["score"])
                plate_text = best_plate.get("text", "NO_OCR")
                plate_conf = best_plate.get("score")

            rows.append({
                "image": image_path.name,
                "vehicle_type": vehicle["class_name"],
                "vehicle_conf": round(vehicle["score"], 4),
                "plate_text": plate_text,
                "plate_conf": round(plate_conf, 4) if plate_conf is not None else None,
                "output": out_path.name,
            })

        print(f"[ok] {image_path.name} -> {out_path.name} | vehicles={len(vehicles)} plates={len(plates)}")
        for row in rows:
            print(f"     {row['vehicle_type']} | {row['plate_text']}  vehicle_conf={row['vehicle_conf']}")

        return rows
