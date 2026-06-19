# Thesis Inference Pipeline

Combined inference pipeline for:

1. vehicle type detection: `BUS`, `CARS`, `MINIBUS`, `SUV`, `TRUCK`, `VAN`
2. license plate detection
3. OCR recognition
4. plate-to-vehicle association
5. final visualization using this format:

```text
TRUCK | 1234AA06
```

## Structure

```text
thesis_inference_pipeline/
├── models/
│   ├── vehicules.onnx
│   └── ALPR.onnx
├── data.yaml
├── yolo_utils.py
├── vehicle_detector.py
├── plate_detector.py
├── ocr_utils.py
├── pipeline.py
├── run.py
├── requirements.txt
└── README.md
```

## Run

Put test images in a folder, for example:

```text
test_images/
├── image1.jpg
└── image2.jpg
```

Then run:

```bash
python run.py --source test_images --out-dir results --save-csv
```

Or run on a single image:

```bash
python run.py --source path/to/image.jpg --out-dir results --save-csv
```

## Output

Annotated images are saved in:

```text
results/
```

If `--save-csv` is used, the script also saves:

```text
results/combined_results.csv
```
