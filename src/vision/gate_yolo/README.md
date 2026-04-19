# Gate Detection with YOLOv8

Train a neural network to spot racing gates from camera frames.
This is the foundation of the entire autonomous racing pipeline —
the drone needs to see gates before it can fly through them.

## Quick Start

### 1. Get the Dataset

Sign up at [roboflow.com](https://roboflow.com) (free), then grab your API key
from [app.roboflow.com/settings/api](https://app.roboflow.com/settings/api).

```bash
cd ~/ai-grand-prix/src/vision/gate_yolo
ROBOFLOW_API_KEY=your_key_here python 01_setup_dataset.py
```

This downloads ~190 annotated images of racing gates in YOLOv8 format.

### 2. Train the Model

**On your Mac (CPU, slower but works for testing):**
```bash
cd ~/ai-grand-prix/src/vision/gate_yolo
python 02_train.py
```
This will be slow (~hours) but lets you verify everything works.

**On your PC with GPU (recommended for real training):**

1. Install Python 3.11+ from [python.org](https://python.org)
2. Install PyTorch with CUDA:
   ```bash
   # Check your CUDA version: nvidia-smi
   # Then install matching PyTorch from https://pytorch.org/get-started
   # Example for CUDA 12.4:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
3. Install ultralytics:
   ```bash
   pip install ultralytics roboflow opencv-python
   ```
4. Copy the gate_yolo folder to your PC (USB, git, Google Drive, whatever)
5. Download the dataset on PC:
   ```bash
   ROBOFLOW_API_KEY=your_key python 01_setup_dataset.py
   ```
6. Train:
   ```bash
   python 02_train.py
   ```
   With an RTX 3060+, this takes ~15-30 minutes instead of hours.

### 3. Run Detection

After training, test on an image:
```bash
python 03_detect.py --source ~/ai-grand-prix/gate_frame.png
```

Or on a video:
```bash
python 03_detect.py --source path/to/video.mp4
```

## What Each Script Does

| Script | Purpose |
|--------|---------|
| `01_setup_dataset.py` | Downloads gate images + labels from Roboflow |
| `02_train.py` | Fine-tunes YOLOv8-nano on gate dataset |
| `03_detect.py` | Runs trained model on images/video/webcam |

## Training Tips

- **Start with YOLOv8n** (nano). It is the fastest and good enough to learn with.
- **If GPU runs out of memory**, reduce batch_size in 02_train.py (try 8 or 4).
- **Early stopping** is on by default (patience=20). Training stops if no improvement.
- **TensorBoard** logs are in `runs/detect/gate_detector/`. View with:
  ```bash
  tensorboard --logdir runs/detect/gate_detector
  ```

## Next Steps After This Works

1. **More data**: Capture frames from the MuJoCo sim and label them
2. **Domain randomization**: Train on varied lighting/textures for sim-to-real transfer
3. **Corner detection**: Upgrade from bounding box to gate corner keypoints (like MonoRace)
4. **Pose estimation**: Use PnP to estimate 3D gate position from 2D corners
5. **Integration**: Feed detections into the racing controller
