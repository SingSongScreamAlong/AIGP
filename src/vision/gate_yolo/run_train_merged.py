"""Train on the full merged 2000-image dataset."""
from ultralytics import YOLO
from pathlib import Path

data_path = Path("datasets/gates_merged/data.yaml")
if not data_path.exists():
    print("Merged dataset not found! Run merge_datasets.py first.")
    exit(1)

print("Loading YOLOv8n pretrained model...")
model = YOLO("yolov8n.pt")

print(f"Starting training on MERGED dataset (2000 images, 50 epochs)...")
results = model.train(
    data=str(data_path),
    epochs=50,
    imgsz=640,
    batch=8,
    project="runs/detect",
    name="gate_detector_v2",
    exist_ok=True,
    patience=15,
    device="cpu",
    workers=0,
    verbose=True,
)

best = Path("runs/detect/gate_detector_v2/weights/best.pt")
print(f"\nDone! Best model: {best}")

if best.exists():
    val_model = YOLO(str(best))
    metrics = val_model.val(data=str(data_path))
    print(f"\nmAP50:     {metrics.box.map50:.3f}")
    print(f"mAP50-95:  {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")
