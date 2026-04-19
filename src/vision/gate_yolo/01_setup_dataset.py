"""
Step 1: Download and prepare the gate detection dataset.
=========================================================
We use the Roboflow API to download the AirSim Drone Racing Lab
Gates dataset in YOLOv8 format. This gives us annotated images of
racing gates — exactly what the drone camera will see.

Prerequisites:
    pip install roboflow ultralytics

Usage:
    1. Go to https://universe.roboflow.com/drone-racing/airsim-drone-racing-lab-gates
    2. Create a free Roboflow account
    3. Get your API key from https://app.roboflow.com/settings/api
    4. Set it below or as an environment variable

    python 01_setup_dataset.py
"""

import os
from pathlib import Path


def download_roboflow_dataset(api_key: str, output_dir: str = "datasets/gates"):
    """Download the AirSim gate dataset from Roboflow in YOLOv8 format."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Install roboflow: pip install roboflow")
        return None

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("drone-racing").project("airsim-drone-racing-lab-gates")
    version = project.version(1)

    # Download in YOLOv8 format — ready to train
    dataset = version.download("yolov8", location=output_dir)
    print(f"\nDataset downloaded to: {output_dir}")
    print(f"  Train images: {output_dir}/train/images/")
    print(f"  Val images:   {output_dir}/valid/images/")
    print(f"  data.yaml:    {output_dir}/data.yaml")
    return dataset


def create_sample_data_yaml(output_dir: str = "datasets/gates"):
    """
    If you don't want to use Roboflow, create a data.yaml manually
    and organize your images + labels in the YOLO format.

    Directory structure:
        datasets/gates/
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   │   ├── img001.jpg
        │   │   └── ...
        │   └── labels/
        │       ├── img001.txt
        │       └── ...
        └── valid/
            ├── images/
            └── labels/

    Label format (each .txt file, one line per gate):
        <class_id> <x_center> <y_center> <width> <height>
        All values normalized 0-1 relative to image dimensions.
        Example: 0 0.5 0.5 0.3 0.4
    """
    os.makedirs(output_dir, exist_ok=True)

    data_yaml = f"""# Gate Detection Dataset
# YOLOv8 format

train: {output_dir}/train/images
val: {output_dir}/valid/images

nc: 1  # number of classes
names: ['gate']  # class names
"""

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(data_yaml)

    print(f"Sample data.yaml created at: {yaml_path}")
    print("\nTo use your own images:")
    print("  1. Put training images in datasets/gates/train/images/")
    print("  2. Put corresponding YOLO label .txt files in datasets/gates/train/labels/")
    print("  3. Same for validation in datasets/gates/valid/")
    print("  4. Each label line: <class_id> <x_center> <y_center> <width> <height>")
    return yaml_path


if __name__ == "__main__":
    # --- OPTION A: Download from Roboflow (recommended) ---
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")

    if api_key:
        print("Downloading dataset from Roboflow...")
        download_roboflow_dataset(api_key)
    else:
        print("No ROBOFLOW_API_KEY set.")
        print()
        print("To download the dataset:")
        print("  1. Sign up at https://roboflow.com (free)")
        print("  2. Get your API key from https://app.roboflow.com/settings/api")
        print("  3. Run: ROBOFLOW_API_KEY=your_key python 01_setup_dataset.py")
        print()
        print("--- OR ---")
        print()
        print("Creating a sample data.yaml for manual dataset setup...")
        create_sample_data_yaml()
