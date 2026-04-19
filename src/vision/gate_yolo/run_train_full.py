"""Training on full merged dataset (1600 train, 400 val)."""
from ultralytics import YOLO
from pathlib import Path
import shutil

data_path = Path("datasets/gates_merged/data.yaml")
if not data_path.exists():
    print("Merged dataset not found!")
    exit(1)

print("Loading YOLOv8n pretrained model...")
model = YOLO("yolov8n.pt")

print("Starting training: 1600 images, 50 epochs, merged dataset")
print("This will take ~45 min on CPU...")
print()

results = model.train(
    data=str(data_path),
    epochs=50,
    imgsz=640,
    batch=8,
    project="runs/detect",
    name="gate_detector_full",
    exist_ok=True,
    patience=15,
    device="cpu",
    workers=0,
    verbose=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15.0,
    translate=0.1,
    scale=0.5,
    flipud=0.1,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    lr0=0.01,
    lrf=0.01,
    warmup_epochs=3,
)

best = Path("runs/detect/gate_detector_full/weights/best.pt")
print(f"\nTraining complete! Best model: {best}")

# Copy best model to models/
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
if best.exists():
    shutil.copy2(best, models_dir / "gate_detector_v2.pt")
    print(f"Copied to models/gate_detector_v2.pt")

    # Validate
    val_model = YOLO(str(best))
    metrics = val_model.val(data=str(data_path))
    print(f"\n=== FINAL VALIDATION ===")
    print(f"mAP50:     {metrics.box.map50:.3f}")
    print(f"mAP50-95:  {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")
