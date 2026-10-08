#!/usr/bin/env python3
r"""
sample_frames.py — turn a folder of short auction clips into a set of frames ready to label
in Label Studio.

Key design points for a good detector dataset:
  * Samples ~1 frame/second (configurable) — enough variety without labeling 60 near-identical
    frames.
  * Skips near-duplicate consecutive frames (perceptual-hash diff) so you don't label the same
    static shot 5 times.
  * Names every frame by its SOURCE VIDEO, e.g.  auc03__000123.jpg  — this is essential: it lets
    us later split train/val/test BY AUCTION (whole auctions never cross splits), which is the
    only honest way to prove the detector generalizes.

SETUP:
  python -m venv venv ; .\venv\Scripts\Activate.ps1
  pip install opencv-python pillow imagehash numpy

RUN:
  put your clips in .\clips\   (auc01.mp4, auc02.mp4, ...)
  python sample_frames.py --in clips --out frames --fps 1

OUTPUT:
  frames\<video>__<frame6>.jpg     one image per kept frame, prefixed by source video
  frames\_manifest.csv             video, frame index, timestamp, path (for bookkeeping)
"""
import argparse, csv, os, sys, glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="clips", help="folder of .mp4/.mov clips")
    ap.add_argument("--out", default="frames")
    ap.add_argument("--fps", type=float, default=1.0, help="frames to sample per second")
    ap.add_argument("--dup-dist", type=int, default=4,
                    help="skip a frame if its phash is within this Hamming distance of the last kept one")
    ap.add_argument("--max-per-video", type=int, default=0, help="0 = no cap")
    args = ap.parse_args()

    try:
        import cv2
        from PIL import Image
        import imagehash
    except Exception as e:
        print("Missing dependency:", e)
        print("pip install opencv-python pillow imagehash numpy")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    # set() dedups: on Windows the filesystem is case-insensitive, so *.mp4 and
    # *.MP4 both match the same file — without this each clip is processed twice.
    clips = sorted(set(glob.glob(os.path.join(args.inp, "*.mp4")) +
                       glob.glob(os.path.join(args.inp, "*.mov")) +
                       glob.glob(os.path.join(args.inp, "*.MP4")) +
                       glob.glob(os.path.join(args.inp, "*.MOV"))))
    if not clips:
        print(f"No clips found in {args.inp}\\  (put auc01.mp4, auc02.mp4, ... there)")
        sys.exit(1)

    man = open(os.path.join(args.out, "_manifest.csv"), "w", newline="", encoding="utf-8")
    mw = csv.writer(man); mw.writerow(["video", "frame", "t_sec", "path"])
    total_kept = 0

    for clip in clips:
        name = os.path.splitext(os.path.basename(clip))[0]
        name = "".join(c if c.isalnum() else "_" for c in name)      # safe prefix
        cap = cv2.VideoCapture(clip)
        if not cap.isOpened():
            print("  ! could not open", clip); continue
        vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(vfps / max(0.1, args.fps))))
        last_hash, kept, idx = None, 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                h = imagehash.phash(pil)
                if last_hash is None or (h - last_hash) > args.dup_dist:
                    fn = f"{name}__{idx:06d}.jpg"
                    cv2.imwrite(os.path.join(args.out, fn), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    mw.writerow([name, idx, round(idx / vfps, 2), fn])
                    last_hash = h; kept += 1; total_kept += 1
                    if args.max_per_video and kept >= args.max_per_video:
                        break
            idx += 1
        cap.release()
        print(f"  {name}: kept {kept} frames")
    man.close()
    print(f"\nDone. {total_kept} frames in {args.out}\\  ->  import this folder into Label Studio.")


if __name__ == "__main__":
    main()
