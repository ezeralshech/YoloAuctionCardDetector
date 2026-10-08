#!/usr/bin/env python3
r"""
eval.py — score a trained model on the HELD-OUT test auctions (the honest check).

    python eval.py
    python eval.py --weights runs/detect/card_v1/weights/best.pt --split test

What the numbers mean for this project:
  - mAP50          : main number to watch. Overlap-based; 1 box per frame, so this
                     tracks "did it put a tight box on the right card".
  - precision      : of the boxes it drew, how many were correct (false-positive rate).
                     LOW precision = firing on binder/background/overlay junk.
  - recall         : of the cards that were there, how many it found (misses).
                     LOW recall = missing the featured card.

Because test = auctions never seen in training, these scores are the real-world
estimate. A big gap between val and test mAP means it's overfitting to sellers.
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/card_v1/weights/best.pt")
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"{args.weights} not found — train first, or pass --weights.")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz,
                  device=args.device, workers=0, plots=True)

    b = m.box
    print("\n" + "=" * 48)
    print(f"RESULTS on '{args.split}' split")
    print("=" * 48)
    print(f"  mAP50     : {b.map50:.3f}")
    print(f"  mAP50-95  : {b.map:.3f}")
    print(f"  precision : {b.mp:.3f}")
    print(f"  recall    : {b.mr:.3f}")


if __name__ == "__main__":
    main()
