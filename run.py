import argparse
import csv
import time
from pathlib import Path
import os

import onnxruntime as ort

from yolo_utils import create_session
from vehicle_detector import VehicleDetector
from plate_detector import PlateDetector
from pipeline import ALPRVehiclePipeline
from ocr_utils import OCR_AVAILABLE

DEFAULT_VEHICLE_MODEL = "models/vehicules.onnx"
DEFAULT_PLATE_MODEL = "models/ALPR.onnx"
DEFAULT_SOURCE = r"C:\Users\ismai\OneDrive\Desktop\thesis_inference_pipeline\test\images"
DEFAULT_OUT_DIR = "results"


def collect_images(source: str):
    path = Path(source)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if path.is_file() and path.suffix.lower() in exts:
        return [path]
    if path.is_dir():
        return sorted([p for p in path.glob("*") if p.suffix.lower() in exts])

    raise FileNotFoundError(f"Image source not found: {source}")


def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle type + ALPR inference pipeline")
    parser.add_argument("--vehicle-model", default=DEFAULT_VEHICLE_MODEL)
    parser.add_argument("--plate-model", default=DEFAULT_PLATE_MODEL)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Image file or folder")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--vehicle-conf", type=float, default=0.55)
    parser.add_argument("--plate-conf", type=float, default=0.40)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--plate-crop-pad", type=int, default=8)
    parser.add_argument("--save-csv", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[info] Loading ONNX models...")
    vehicle_session = create_session(args.vehicle_model)
    plate_session = create_session(args.plate_model)
    print(f"[info] ONNX available providers: {ort.get_available_providers()}")

    if not OCR_AVAILABLE:
        print("[warn] fast_plate_ocr is not installed. Plate text will be NO_OCR_LIB.")

    vehicle_detector = VehicleDetector(
        vehicle_session,
        imgsz=args.imgsz,
        conf=args.vehicle_conf,
        iou=args.iou,
    )
    plate_detector = PlateDetector(
        plate_session,
        imgsz=args.imgsz,
        conf=args.plate_conf,
        iou=args.iou,
    )
    pipeline = ALPRVehiclePipeline(
        vehicle_detector,
        plate_detector,
        plate_crop_pad=args.plate_crop_pad,
    )

    image_paths = collect_images(args.source)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.source}")

    print(f"[info] Processing {len(image_paths)} image(s)")
    start = time.time()
    all_rows = []

    for image_path in image_paths:
        all_rows.extend(pipeline.process_image(image_path, out_dir))

    if args.save_csv and all_rows:
        csv_path = out_dir / "combined_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[ok] CSV saved: {csv_path}")

    print(f"[done] Finished in {time.time() - start:.2f}s")
    print(f"[done] Results saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
