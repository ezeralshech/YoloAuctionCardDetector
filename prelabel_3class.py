#!/usr/bin/env python3
r"""
prelabel_3class.py — pre-label auction frames with 3 classes (card / label / slab)
for a 3-class auction detector, using BOTH models at their strengths:

  auction best.pt  -> finds the featured item in the full scene (it's good at this)
  scan   best.pt   -> decomposes the CROP into card(art)/label/slab (it's good at this
                      on close-ups, but weak on full frames — so we only run it on the crop)

For each frame: auction locates the featured item -> crop (padded) -> scan decomposes ->
the scan boxes are mapped back to full-frame coordinates and written as predictions.
You then correct them in a 3-class Label Studio project.

    python prelabel_3class.py --auction-weights best.pt ^
        --scan-weights "..\YoloCardDetector\best.pt" --frames frames_new

IMPORTANT correction discipline (so the model learns the right thing):
  - slab/label ONLY on a real graded case (hard shell + PSA/BGS/CGC grading banner).
  - a penny-sleeve or toploader is NOT a slab — on those the scan model tends to
    over-call 'slab'; delete that box, keep only 'card'.
  - 'card' = the card art (for a slab, the art window; for a raw card, the whole card).
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auction-weights", default="best.pt")
    ap.add_argument("--scan-weights", required=True,
                    help="path to the scan detector best.pt (3-class)")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="predictions_3class.json")
    ap.add_argument("--conf-auction", type=float, default=0.20)
    ap.add_argument("--conf-scan", type=float, default=0.30)
    ap.add_argument("--pad", type=float, default=0.08, help="crop padding fraction")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--doc-root", default=".")
    ap.add_argument("--from-name", default="label")
    ap.add_argument("--to-name", default="image")
    ap.add_argument("--model-version", default="hybrid-3class")
    args = ap.parse_args()

    frames_dir = Path(args.frames)
    if not frames_dir.is_dir():
        raise SystemExit(f"frames folder not found: {frames_dir}")
    for w in (args.auction_weights, args.scan_weights):
        if not Path(w).exists():
            raise SystemExit(f"weights not found: {w}")

    doc_root = Path(args.doc_root).resolve()
    imgs = sorted(frames_dir.glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"no .jpg frames in {frames_dir}")

    from ultralytics import YOLO
    import cv2
    from collections import Counter

    auction = YOLO(args.auction_weights)
    scan = YOLO(args.scan_weights)
    snames = scan.names
    print(f"scan classes: {snames}")

    tasks = []
    n_box = n_empty = 0
    class_counts = Counter()

    for ip in imgs:
        img = cv2.imread(str(ip))
        if img is None:
            print("skip unreadable:", ip.name)
            continue
        H, W = img.shape[:2]
        try:
            rel = ip.resolve().relative_to(doc_root).as_posix()
        except ValueError:
            raise SystemExit(f"{ip} is not under --doc-root {doc_root}.")
        data_image = f"/data/local-files/?d={rel}"

        results = []
        # 1) locate featured item in the full frame
        ar = auction.predict(str(ip), conf=args.conf_auction, imgsz=args.imgsz,
                             device=args.device, max_det=1, verbose=False)[0]
        if len(ar.boxes):
            x1, y1, x2, y2 = ar.boxes[0].xyxy[0].tolist()
            bw, bh = x2 - x1, y2 - y1
            cx1 = max(0, int(x1 - args.pad * bw)); cy1 = max(0, int(y1 - args.pad * bh))
            cx2 = min(W, int(x2 + args.pad * bw)); cy2 = min(H, int(y2 + args.pad * bh))
            crop = img[cy1:cy2, cx1:cx2]
            # 2) decompose the crop with the scan model
            sr = scan.predict(crop, conf=args.conf_scan, imgsz=args.imgsz,
                             device=args.device, max_det=10, verbose=False)[0]
            for b in sr.boxes:
                sx1, sy1, sx2, sy2 = b.xyxy[0].tolist()
                # map crop coords -> full-frame coords
                fx1, fy1 = cx1 + sx1, cy1 + sy1
                fx2, fy2 = cx1 + sx2, cy1 + sy2
                cls_name = snames[int(b.cls)]
                class_counts[cls_name] += 1
                results.append({
                    "from_name": args.from_name, "to_name": args.to_name,
                    "type": "rectanglelabels",
                    "original_width": W, "original_height": H, "image_rotation": 0,
                    "score": round(float(b.conf[0]), 4),
                    "value": {
                        "x": max(0.0, fx1 / W * 100), "y": max(0.0, fy1 / H * 100),
                        "width": min(100.0, (fx2 - fx1) / W * 100),
                        "height": min(100.0, (fy2 - fy1) / H * 100),
                        "rotation": 0, "rectanglelabels": [cls_name],
                    },
                })
        if results:
            n_box += 1
        else:
            n_empty += 1
        tasks.append({"data": {"image": data_image},
                      "predictions": [{"model_version": args.model_version,
                                       "score": max((x["score"] for x in results), default=0.0),
                                       "result": results}]})

    Path(args.out).write_text(json.dumps(tasks, indent=1))
    print(f"\nWrote {args.out}: {len(tasks)} tasks ({n_box} with >=1 box, {n_empty} empty)")
    print(f"predicted boxes by class: {dict(class_counts)}")
    print("\nLabel Studio: start with local-files serving on, add Local Storage for the")
    print(f"frames folder ({frames_dir.resolve()}), then import {args.out} into a 3-CLASS project.")


if __name__ == "__main__":
    main()
