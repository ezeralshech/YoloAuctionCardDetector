#!/usr/bin/env python3
r"""
train.py — train a YOLO-nano card detector on CPU.

First time only, install Ultralytics into your venv:
    pip install ultralytics

Smoke test first (3 epochs, confirms the whole pipeline runs end to end):
    python train.py --epochs 3 --name smoke

Real run (defaults are tuned for a CPU laptop on ~400 training frames):
    python train.py

The model is saved to:
    runs/detect/<name>/weights/best.pt      <- THIS is "the engine"
    runs/detect/<name>/weights/last.pt
Metrics/plots: runs/detect/<name>/results.png, confusion_matrix.png, *_curve.png

Notes for CPU:
  - This WILL take a while (often a few minutes per epoch). The smoke test tells
    you the per-epoch time; multiply to estimate the real run before committing.
  - To go faster at some accuracy cost: --imgsz 416  (or 512), fewer --epochs.
  - workers=0 and the __main__ guard below are required for stability on Windows.
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="pretrained weights to start from (auto-downloads). "
                         "Use yolov8n.pt if yolo11n.pt is unavailable.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=25,
                    help="early-stop if val mAP doesn't improve for N epochs")
    ap.add_argument("--name", default="card_v1",
                    help="run name -> runs/detect/<name>/ (bump it each iteration)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"{args.data} not found — run prepare_dataset.py first.")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        name=args.name,
        workers=0,          # Windows/CPU stability
        seed=0,             # reproducible splits/augmentation
        plots=True,
    )
    print("\nDone. Best weights: runs/detect/%s/weights/best.pt" % args.name)
    print("Next: python eval.py --weights runs/detect/%s/weights/best.pt" % args.name)


if __name__ == "__main__":
    main()
