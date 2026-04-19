"""Quick training launcher — reduced epochs for CPU."""
from ultralytics import YOLO
from pathlib import Path

data_path = Path("datasets/gates/data.yaml")
if not data_path.exists():
    print("Dataset not found! Run generate_data.py first.")
    exit(1)

print("Loading YOLOv8n pretrained model...")
model = YOLO("yolov8n.pt")

print("Starting training (30 epochs on CPU — this will take ~15-20 min)...")
results = model.train(
    data=str(data_path),
    epochs=30,
    imgsz=640,
    batch=8,
    project="runs/detect",
    name="gate_detector",
    exist_ok=True,
    patience=10,
    device="cpu",
    workers=0,
    verbose=True,
)

best = Path("runs/detect/gate_detector/weights/best.pt")
print(f"\nDone! Best model: {best}")
print(f"Exists: {best.exists()}")

# Quick validation
if best.exists():
    val_model = YOLO(str(best))
    metrics = val_model.val(data=str(data_path))
    print(f"\nmAP50:    {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")
