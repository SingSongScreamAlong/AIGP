# DCL Capture → YOLO Retrain Pipeline

Capture FPV footage from **DCL The Game**, extract frames, label gates,
and retrain the YOLOv8 gate detector on real-looking drone racing imagery.

## Why

The current YOLO model is trained on synthetic MuJoCo renders. DCL The
Game provides photorealistic FPV drone footage with gates that look much
closer to what the DCL AI Race League simulator will ship. Mixing real
DCL frames into training data bridges the sim-to-real gap.

## Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  1. Capture   │───▶│  2. Extract  │───▶│  3. Label    │───▶│  4. Train    │
│  gameplay.py  │    │  frames.py   │    │  assist.py   │    │  02_train.py │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
   Record screen       Pull frames        Pre-label with     Retrain YOLO
   while flying        @ 2 fps,           existing model,    on DCL + synth
   in DCL              dedup menus        human review       merged data
```

## Step-by-Step

### 1. Record gameplay

Launch DCL The Game and fly some races/free flight.  In another terminal:

```bash
cd ~/ai-grand-prix

# Record 2 minutes of gameplay at 15 fps
python src/vision/dcl_capture/capture_gameplay.py --duration 120 --fps 15

# Or just list capture devices first
python src/vision/dcl_capture/capture_gameplay.py --list-devices
```

Output: `data/dcl_raw/dcl_<timestamp>.mp4`

> **Tip:** On macOS, the first run will prompt for screen recording
> permission. Grant it to Terminal / your IDE.

### 2. Extract frames

```bash
# Extract at 2 fps (one frame every 0.5s), skip near-duplicate frames
python src/vision/dcl_capture/extract_frames.py data/dcl_raw/dcl_*.mp4

# Or process all recordings at once
python src/vision/dcl_capture/extract_frames.py --all
```

Output: `data/dcl_frames/<video_name>/dcl_00000.jpg ...`

### 3. Label gates

**Option A — Model-assist (recommended):**

Transfer frames to your GPU machine, then:

```bash
# Pre-label using existing YOLO model
python src/vision/dcl_capture/label_assist.py data/dcl_frames/dcl_20260428/ \
    --model models/gate_detector_latest.pt --conf 0.25

# Review pre-labels interactively (OpenCV window)
python src/vision/dcl_capture/label_assist.py data/dcl_frames/dcl_20260428/ \
    --review
```

**Option B — External labeling tool:**

Upload `data/dcl_frames/<session>/` to [Label Studio](https://labelstud.io)
or [CVAT](https://www.cvat.ai/) and label manually. Export in YOLO format
and drop the labels into the `labels/` subdirectory.

### 4. Build dataset

```bash
# Package accepted labels into YOLO dataset format
python src/vision/dcl_capture/label_assist.py data/dcl_frames/dcl_20260428/ \
    --build-dataset
```

Output: `datasets/dcl_real/` with `data.yaml`, `train/`, `valid/`.

### 5. Merge and retrain

```bash
# Merge DCL real data with synthetic data
python src/vision/gate_yolo/merge_datasets.py \
    --output datasets/combined \
    --dirs datasets/gates_keypoints datasets/dcl_real

# Retrain
python src/vision/gate_yolo/02_train.py \
    --data datasets/combined/data.yaml
```

## Directory Layout

```
ai-grand-prix/
├── data/
│   ├── dcl_raw/              # Raw MP4 recordings
│   │   └── dcl_20260428.mp4
│   └── dcl_frames/           # Extracted frames per session
│       └── dcl_20260428/
│           ├── dcl_00000.jpg
│           ├── labels/        # YOLO-format pre-labels
│           ├── viz/           # Label overlays for QA
│           └── accepted/      # Human-reviewed frames
├── datasets/
│   ├── dcl_real/             # Built YOLO dataset
│   │   ├── data.yaml
│   │   ├── train/images/  train/labels/
│   │   └── valid/images/  valid/labels/
│   └── combined/             # Merged synth + real
└── src/vision/dcl_capture/
    ├── capture_gameplay.py
    ├── extract_frames.py
    └── label_assist.py
```

## Tips

- **Fly through gates slowly at first** — gives more frames per gate
  from different angles, better training data.
- **Vary your flight path** — approach gates from different angles,
  distances, and lighting conditions.
- **Include failure cases** — partial gate views, far-away gates, gates
  at screen edges are all valuable for training robustness.
- **2 fps extraction is usually enough** — DCL runs at 60+ fps, so
  at 2 fps you get ~120 unique frames per minute of flying.
- **Quality over quantity** — 200 well-labeled DCL frames mixed with
  2000 synthetic frames is better than 2000 poorly-labeled ones.
