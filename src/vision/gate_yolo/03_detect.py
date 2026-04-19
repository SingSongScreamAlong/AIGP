"""
Step 3: Run gate detection on images or video.
================================================
Use your trained model to detect gates in new images.
This is what will eventually run onboard the drone,
taking camera frames and finding where the gates are.

Prerequisites:
    pip install ultralytics opencv-python
    Run 02_train.py first to get a trained model.

Usage:
    # Detect gates in an image
    python 03_detect.py --source path/to/image.jpg

    # Detect in a video
    python 03_detect.py --source path/to/video.mp4

    # Detect using webcam (for fun)
    python 03_detect.py --source 0

    # Use with the MuJoCo sim frames
    python 03_detect.py --source ~/ai-grand-prix/gate_frame.png
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class GateDetectorYOLO:
    """
    Neural network gate detector using YOLOv8.

    This is the upgrade from the simple color-based detector.
    Works in any lighting, any angle, any environment — because
    it learned what gates look like from hundreds of examples.
    """

    def __init__(self, model_path: str = "runs/detect/gate_detector/weights/best.pt"):
        """Load the trained YOLO model."""
        self.model = YOLO(model_path)
        print(f"Loaded model: {model_path}")

    def detect(self, frame, conf_threshold: float = 0.5):
        """
        Detect gates in a camera frame.

        Args:
            frame: BGR image (numpy array)
            conf_threshold: Minimum confidence to report a detection

        Returns:
            list of dicts with:
                - 'bbox': (x1, y1, x2, y2) bounding box
                - 'center': (cx, cy) center point
                - 'confidence': detection confidence 0-1
                - 'area': bounding box area in pixels
        """
        results = self.model(frame, conf=conf_threshold, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)

                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "center": (int(cx), int(cy)),
                    "confidence": conf,
                    "area": float(area),
                })

        # Sort by area (biggest/closest first)
        detections.sort(key=lambda d: d["area"], reverse=True)
        return detections

    def annotate(self, frame, detections):
        """Draw detections on the frame."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cx, cy = det["center"]

            # Bounding box
            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Center point
            cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)

            # Label
            label = f"Gate {conf:.0%}"
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return annotated

    def detect_and_annotate(self, frame, conf_threshold: float = 0.5):
        """Detect + draw in one call. Returns (detections, annotated_frame)."""
        detections = self.detect(frame, conf_threshold)
        annotated = self.annotate(frame, detections)
        return detections, annotated


def run_on_source(source, model_path: str = "runs/detect/gate_detector/weights/best.pt"):
    """Run gate detection on an image, video, or webcam."""
    detector = GateDetectorYOLO(model_path)

    # Check if source is an image
    source_path = Path(str(source))
    if source_path.exists() and source_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        print(f"\nDetecting gates in: {source}")
        frame = cv2.imread(str(source))
        if frame is None:
            print(f"Could not read image: {source}")
            return

        detections, annotated = detector.detect_and_annotate(frame)
        print(f"Found {len(detections)} gate(s)")
        for i, det in enumerate(detections):
            print(f"  Gate {i+1}: center={det['center']}, "
                  f"conf={det['confidence']:.1%}, area={det['area']:.0f}px")

        # Save annotated image
        out_path = source_path.stem + "_detected" + source_path.suffix
        cv2.imwrite(out_path, annotated)
        print(f"\nSaved annotated image: {out_path}")
        return

    # Video or webcam
    source_val = int(source) if str(source).isdigit() else str(source)
    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"Could not open: {source}")
        return

    print(f"\nRunning detection on: {source}")
    print("Press 'q' to quit\n")

    frame_count = 0
    total_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        detections, annotated = detector.detect_and_annotate(frame)
        dt = time.time() - t0

        frame_count += 1
        total_time += dt
        fps = frame_count / total_time

        # Add FPS counter
        cv2.putText(annotated, f"FPS: {fps:.1f} | Gates: {len(detections)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Gate Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames at {fps:.1f} FPS average")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect racing gates in images/video")
    parser.add_argument("--source", default="gate_frame.png",
                        help="Image path, video path, or camera index (0)")
    parser.add_argument("--model", default="runs/detect/gate_detector/weights/best.pt",
                        help="Path to trained YOLO model")
    args = parser.parse_args()

    run_on_source(args.source, args.model)
