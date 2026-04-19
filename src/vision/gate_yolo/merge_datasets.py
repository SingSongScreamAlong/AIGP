"""Merge multiple dataset batches into one unified YOLOv8 dataset."""
import os
import shutil
from pathlib import Path


def merge(output_dir="datasets/gates_merged", batch_dirs=None):
    if batch_dirs is None:
        # Auto-discover all gates* directories
        base = Path("datasets")
        batch_dirs = sorted([str(d) for d in base.iterdir() if d.is_dir() and d.name.startswith("gates")])

    print(f"Merging {len(batch_dirs)} batches into {output_dir}")
    for d in batch_dirs:
        print(f"  - {d}")

    for split in ["train", "valid"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    img_count = 0
    label_count = 0

    for batch_idx, batch_dir in enumerate(batch_dirs):
        for split in ["train", "valid"]:
            img_dir = os.path.join(batch_dir, split, "images")
            lbl_dir = os.path.join(batch_dir, split, "labels")

            if not os.path.exists(img_dir):
                continue

            for fname in sorted(os.listdir(img_dir)):
                if not fname.endswith((".jpg", ".png")):
                    continue

                stem = Path(fname).stem
                suffix = Path(fname).suffix

                # Rename to avoid collisions: batch0_frame_00001.jpg
                new_name = f"b{batch_idx}_{stem}"
                new_img = new_name + suffix
                new_lbl = new_name + ".txt"

                shutil.copy2(
                    os.path.join(img_dir, fname),
                    os.path.join(output_dir, split, "images", new_img),
                )
                img_count += 1

                lbl_file = os.path.join(lbl_dir, stem + ".txt")
                if os.path.exists(lbl_file):
                    shutil.copy2(
                        lbl_file,
                        os.path.join(output_dir, split, "labels", new_lbl),
                    )
                    label_count += 1

    # Write data.yaml
    abs_out = os.path.abspath(output_dir)
    yaml_content = f"""# Merged Gate Detection Dataset
# {len(batch_dirs)} batches combined

train: {abs_out}/train/images
val: {abs_out}/valid/images

nc: 1
names: ['gate']
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    train_count = len(os.listdir(os.path.join(output_dir, "train", "images")))
    val_count = len(os.listdir(os.path.join(output_dir, "valid", "images")))

    print(f"\nMerge complete!")
    print(f"  Total images: {img_count} ({train_count} train, {val_count} val)")
    print(f"  Total labels: {label_count}")
    print(f"  data.yaml:    {yaml_path}")
    print(f"\nTo retrain: python 02_train.py  (update data_yaml path to datasets/gates_merged/data.yaml)")


if __name__ == "__main__":
    merge()
