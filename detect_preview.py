#!/usr/bin/env python3
r"""
detect_preview.py — run the trained model on frames and save annotated images so
you can EYEBALL where it's right and wrong. Numbers don't tell you *how* it fails;
this does.

    # preview the held-out test auction
    python detect_preview.py --source dataset/images/test

    # preview a brand-new, never-labeled clip's frames
    python detect_preview.py --source frames_new

    # a single image
    python detect_preview.py --source frames/action05__002160.jpg

Annotated images are written to runs/detect/<name>/. Open that folder and look:
  - boxes on the right featured card = good
  - boxes on binder/background/overlay = false positives (need more negatives)
  - no box where a card is clearly featured = misses (need more of that format)

--conf controls the confidence threshold (default 0.25). Raise it to see only
confident boxes; lower it to see everything the model is tempted by.

This is also the seed of the bootstrap pre-labeling step: once this model is
decent, we extend it to emit Label Studio predictions so you CORRECT boxes on
new clips instead of drawing them from scratch.
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/card_v1/weights/best.pt")
    ap.add_argument("--source", required=True,
                    help="image file OR a folder of frames")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default="preview")
    ap.add_argument("--max-det", type=int, default=1,
                    help="max boxes per frame; this detector expects 1 featured card")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"{args.weights} not found — train first, or pass --weights.")
    if not Path(args.source).exists():
        raise SystemExit(f"source not found: {args.source}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        save=True,
        name=args.name,
    )
    print(f"\nAnnotated images saved under runs/detect/{args.name}/ — open and inspect.")


if __name__ == "__main__":
    main()
