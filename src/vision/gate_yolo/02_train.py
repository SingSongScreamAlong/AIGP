"""
Step 2: Train YOLOv8 to detect racing gates.
==============================================
This fine-tunes a pretrained YOLOv8 model on the gate dataset.
YOLOv8n (nano) is the smallest/fastest — good for drones where
inference speed matters. You can try larger models (s, m, l, x)
if you have more GPU power.

Prerequisites:
    pip install ultralytics
    Run 01_setup_dataset.py first to get the dataset.

Usage:
    python 02_train.py

On a decent GPU (RTX 3060+), training takes ~15-30 minutes.
On CPU it'll be much slower but still works for learning.

The trained model goes to: runs/detect/gate_detector/weights/best.pt
"""

from pathlib import Path
from ultralytics import YOLO


def train(
    data_yaml: str = "datasets/gates/data.yaml",
    model_size: str = "n",      # n=nano, s=small, m=medium, l=large, x=xlarge
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 16,       # Reduce to 8 or 4 if you run out of GPU memory
    device: str = "",           # "" = auto-detect, "0" = GPU 0, "cpu" = CPU
):
    """
    Train YOLOv8 on the gate detection dataset.

    Args:
        data_yaml: Path to the dataset YAML file.
        model_size: YOLOv8 variant. 'n' is fastest, 'x' is most accurate.
        epochs: Number of training epochs. 100 is a good starting point.
        img_size: Input image size. 640 is the default. Larger = more detail but slower.
        batch_size: Batch size. Reduce if GPU memory is limited.
        device: Training device. Empty string auto-detects GPU.
    """
    # Load a pretrained YOLOv8 model
    model_name = f"yolov8{model_size}.pt"
    print(f"Loading pretrained model: {model_name}")
    model = YOLO(model_name)

    # Train on our gate dataset
    print(f"\nStarting training:")
    print(f"  Dataset:    {data_yaml}")
    print(f"  Model:      YOLOv8{model_size}")
    print(f"  Epochs:     {epochs}")
    print(f"  Image size: {img_size}")
    print(f"  Batch size: {batch_size}")
    print()

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device or None,
        project="runs/detect",
        name="gate_detector",
        exist_ok=True,

        # Data augmentation — helps the model generalize.
        hsv_h=0.015,       # Hue variation
        hsv_s=0.7,         # Saturation variation
        hsv_v=0.4,         # Brightness variation
        degrees=15.0,      # Random rotation
        translate=0.1,     # Random shift
        scale=0.5,         # Random zoom
        flipud=0.1,        # Vertical flip (occasional)
        fliplr=0.5,        # Horizontal flip
        mosaic=1.0,        # Mosaic augmentation (combines 4 images)
        mixup=0.1,         # Mix two images together

        # Training hyperparameters
        lr0=0.01,          # Initial learning rate
        lrf=0.01,          # Final learning rate (fraction of lr0)
        warmup_epochs=3,   # Learning rate warmup
        patience=20,       # Early stopping patience (stop if no improvement)
        save_period=10,    # Save checkpoint every N epochs

        # Logging
        verbose=True,
    )

    # Print results
    best_weights = Path("runs/detect/gate_detector/weights/best.pt")
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best model saved to: {best_weights}")
    print(f"{'='*50}")

    # Validate the best model
    print("\nRunning validation on the best model...")
    best_model = YOLO(str(best_weights))
    metrics = best_model.val(data=data_yaml)
    print(f"\nValidation results:")
    print(f"  mAP50:    {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall:    {metrics.box.mr:.3f}")

    return results


if __name__ == "__main__":
    # Check if dataset exists
    data_path = Path("datasets/gates/data.yaml")
    if not data_path.exists():
        print("Dataset not found! Run 01_setup_dataset.py first.")
        print("Or set the correct path to your data.yaml below.")
        exit(1)

    train(data_yaml=str(data_path))
