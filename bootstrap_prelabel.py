#!/usr/bin/env python3
r"""
bootstrap_prelabel.py — use a trained best.pt to PRE-LABEL new frames, so in
Label Studio you just correct boxes instead of drawing them from scratch.

It runs the model over a folder of frames and writes a Label Studio import file
(predictions.json). Every frame becomes a task; frames where the model found a
card carry a pre-drawn box you can nudge or delete, frames where it found nothing
come in empty (draw one, or submit empty if it's a true negative).

    pip install ultralytics            # once, if not already installed
    python bootstrap_prelabel.py --weights best.pt --frames frames_new

Key options:
    --conf 0.15     lower than usual on purpose — recall is the weak spot on new
                    sellers, and a slightly loose box is faster to fix than a
                    missing one is to draw. Lower = more boxes to correct.
    --doc-root      the Label Studio LOCAL_FILES_DOCUMENT_ROOT (default: the
                    project folder, i.e. the parent you run this from).

The frames are referenced via Label Studio's local-files URL, so you must start
Label Studio with local file serving on and add a Local Storage pointing at the
frames folder — see the printed instructions at the end.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--frames", required=True, help="folder of frames to pre-label")
    ap.add_argument("--out", default="predictions.json")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-det", type=int, default=10,
                    help="max boxes per frame (a slab frame can have card+label+slab)")
    ap.add_argument("--doc-root", default=".",
                    help="Label Studio LOCAL_FILES_DOCUMENT_ROOT")
    ap.add_argument("--from-name", default="label",
                    help="must match the RectangleLabels name in your LS config")
    ap.add_argument("--to-name", default="image")
    ap.add_argument("--model-version", default="bootstrap")
    args = ap.parse_args()

    frames_dir = Path(args.frames)
    if not frames_dir.is_dir():
        raise SystemExit(f"frames folder not found: {frames_dir}")
    if not Path(args.weights).exists():
        raise SystemExit(f"weights not found: {args.weights}")

    doc_root = Path(args.doc_root).resolve()
    imgs = sorted(p for p in frames_dir.glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"no .jpg frames in {frames_dir}")

    from ultralytics import YOLO
    import cv2
    from collections import Counter

    model = YOLO(args.weights)
    names = model.names   # {0: 'card', 1: 'label', 2: 'slab'} for the scan model
    print(f"model classes: {names}")

    tasks = []
    n_box = n_empty = 0
    class_counts = Counter()
    for ip in imgs:
        img = cv2.imread(str(ip))
        if img is None:
            print("skip unreadable:", ip.name)
            continue
        h, w = img.shape[:2]

        # local-files URL, relative to the document root, forward slashes
        try:
            rel = ip.resolve().relative_to(doc_root).as_posix()
        except ValueError:
            raise SystemExit(
                f"{ip} is not under --doc-root {doc_root}. Set --doc-root to a "
                f"parent folder of your frames (usually the project folder).")
        data_image = f"/data/local-files/?d={rel}"

        r = model.predict(str(ip), conf=args.conf, imgsz=args.imgsz,
                          device=args.device, max_det=args.max_det, verbose=False)[0]

        results = []
        if len(r.boxes):
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                score = float(b.conf[0])
                cls_name = names[int(b.cls)]      # card / label / slab
                class_counts[cls_name] += 1
                results.append({
                    "from_name": args.from_name,
                    "to_name": args.to_name,
                    "type": "rectanglelabels",
                    "original_width": w,
                    "original_height": h,
                    "image_rotation": 0,
                    "score": round(score, 4),
                    "value": {
                        "x": max(0.0, x1 / w * 100),
                        "y": max(0.0, y1 / h * 100),
                        "width": min(100.0, (x2 - x1) / w * 100),
                        "height": min(100.0, (y2 - y1) / h * 100),
                        "rotation": 0,
                        "rectanglelabels": [cls_name],
                    },
                })
            n_box += 1
        else:
            n_empty += 1

        tasks.append({
            "data": {"image": data_image},
            "predictions": [{
                "model_version": args.model_version,
                "score": max((x["score"] for x in results), default=0.0),
                "result": results,
            }],
        })

    Path(args.out).write_text(json.dumps(tasks, indent=1))

    print(f"\nWrote {args.out}: {len(tasks)} tasks "
          f"({n_box} with >=1 box, {n_empty} empty)")
    print(f"predicted boxes by class: {dict(class_counts)}")
    print("\n--- To use in Label Studio ---")
    print("1. Stop Label Studio, then restart it with local file serving on:")
    print('   $env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"')
    print(f'   $env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="{doc_root}"')
    print('   $env:DATA_UPLOAD_MAX_NUMBER_FILES="10000"')
    print("   label-studio start")
    print("2. New project (or your existing one) -> Settings -> Cloud Storage ->")
    print("   Add Source Storage -> Local files -> Absolute local path =")
    print(f"       {frames_dir.resolve()}")
    print("   Save it (you do NOT need to click Sync).")
    print(f"3. Project -> Import -> upload {args.out}.")
    print("   Tasks appear with the model's boxes already drawn — correct and submit.")


if __name__ == "__main__":
    main()
