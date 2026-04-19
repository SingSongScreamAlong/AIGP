"""
Phase 2: Train YOLOv8-pose for gate corner keypoint detection.
==============================================================
Trains a YOLOv8-pose model to detect gates AND predict 4 corner
keypoints per gate in a single forward pass.

This is the bridge between Phase 1 (detection) and Phase 3 (PnP pose).
One model gives us both the bounding box and the exact corner locations
we need for 3D pose estimation via solvePnP.

Prerequisites:
    pip install ultralytics
    Run generate_data_keypoints.py first to create the dataset.

Usage:
    # Train on keypoint dataset
    python train_keypoints.py

    # Train with custom settings
    python train_keypoints.py --epochs 100 --batch 16 --device 0

    # Resume training
    python train_keypoints.py --resume runs/pose/gate_corners/weights/last.pt

Author: Conrad Weeden
Date: April 2026
"""

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO


def train_keypoints(
    data_yaml: str = "datasets/gates_keypoints/data.yaml",
    model_size: str = "n",          # n=nano, s=small
    epochs: int = 80,
    img_size: int = 640,
    batch_size: int = 8,            # Conservative for CPU, bump to 16-32 on GPU
    device: str = "",               # "" = auto, "0" = GPU 0, "cpu" = CPU
    resume: str = None,
):
    if resume:
        print(f'Resuming training from: {resume}')
        model = YOLO(resume)
    else:
        model_name = f'yolov8{model_size}-pose.pt'
        print(f'Loading pretrained pose model: {model_name}')
        model = YOLO(model_name)

    print('\\n' + '='*60)
    print('  AI Grand Prix - Phase 2: Gate Corner Keypoint Training')
    print('='*60)
    print(f'  Dataset:     {data_yaml}')
    print(f'  Model:       YOLOv8{model_size}-pose')
    print(f'  Epochs:      {epochs}')
    print(f'  Image size:  {img_size}')
    print(f'  Batch size:  {batch_size}')
    print(f'  Device:      {device or "auto"}')
    print('='*60 + '\\n')

    results = model.train(
        data=data_yaml, epochs=epochs, imgsz=img_size, batch=batch_size,
        device=device or None, project='runs/pose', name='gate_corners', exist_ok=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=15.0, translate=0.1, scale=0.5,
        flipud=0.0, fliplr=0.0, mosaic=1.0, mixup=0.1,
        lr0=0.01, lrf=0.01, warmup_epochs=3, patience=25, save_period=10,
        pose=12.0, verbose=True,
    )

    best_weights = Path('runs/pose/gate_corners/weights/best.pt')
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    out_model = model_dir / 'gate_corners_v1.pt'
    shutil.copy2(best_weights, out_model)
    print(f'\\nBest model copied to: {out_model}')

    print('\\nRunning validation on best model...')
    best_model = YOLO(str(best_weights))
    metrics = best_model.val(data=data_yaml)

    print('\\n' + '='*60)
    print('  Training Complete')
    print('='*60)
    print(f'  Box mAP50:      {metrics.box.map50:.3f}')
    print(f'  Box mAP50-95:   {metrics.box.map:.3f}')
    print(f'  Box Precision:  {metrics.box.mp:.3f}')
    print(f'  Box Recall:     {metrics.box.mr:.3f}')
    if hasattr(metrics, 'pose'):
        print(f'  Pose mAP50:     {metrics.pose.map50:.3f}')
    print(f'\\n  Model saved:    {out_model}')
    print('='*60)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv8-pose for gate corners')
    parser.add_argument('--data', type=str, default='datasets/gates_keypoints/data.yaml')
    parser.add_argument('--model_size', type=str, default='n', choices=['n', 's', 'm'])
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_image', type=str, default=None)
    parser.add_argument('--test_only', action='store_true')
    args = parser.parse_args()

    if args.test_only:
        model_path = args.resume or 'models/gate_corners_v1.pt'
        from ultralytics import YOLO
        m = YOLO(model_path)
        print(f'Loaded: {model_path}')
    else:
        if not Path(args.data).exists():
            print(f'Dataset not found: {args.data}')
            print('Run generate_data_keypoints.py first!')
            exit(1)
        train_keypoints(data_yaml=args.data, model_size=args.model_size,
            epochs=args.epochs, img_size=args.imgsz, batch_size=args.batch,
            device=args.device, resume=args.resume)
