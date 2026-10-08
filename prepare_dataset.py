#!/usr/bin/env python3
r"""
prepare_dataset.py — turn your Label Studio YOLO export + frames into a YOLO
training dataset, split BY AUCTION (never mixing one auction across splits).

Why by-auction: the whole point of this project is to know whether the detector
generalizes to auctions it has NEVER seen. If frames from the same auction land
in both train and test, the test score is a lie (the model has effectively seen
that seller/lighting/format). So every split gets WHOLE auctions, never a mix.

What it does:
  1. Reads every label .txt in export/labels/ (strips the Label Studio hash
     prefix like "004045ac-" to recover the real frame name).
  2. Pairs each label with its image in frames/ (the export omits the images;
     they live in your frames/ folder).
  3. Assigns each auction to train / val / test per the config below.
  4. Writes dataset/images/{train,val,test} + dataset/labels/{train,val,test}
     and a data.yaml. Empty-label frames (your negatives) are carried through
     as empty .txt files so YOLO learns them as background.
  5. Prints a summary and refuses to run if any image is missing or if an
     auction would appear in more than one split.

Run it from the project folder:
    python prepare_dataset.py
Override the split without editing the file:
    python prepare_dataset.py --test action05 --val action06
Rebuild after adding/relabeling (safe — it wipes dataset/ and rebuilds):
    python prepare_dataset.py
"""
import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ----------------------------------------------------------------------------
# CONFIG — edit these, or override on the command line.
# ----------------------------------------------------------------------------
FRAMES_DIR = Path("frames")            # where sample_frames.py wrote the .jpg frames
EXPORT_DIR = Path("export")            # the unzipped YOLO export (labels found recursively)
OUT_DIR = Path("dataset")              # output dataset root (wiped & rebuilt each run)
FRAME_EXT = ".jpg"

# Which WHOLE auctions are held out. Everything not listed here goes to train.
# Pick a TEST auction that is a DIFFERENT SELLER from every training auction —
# that is the honest generalization check. Same for VAL (used for early stopping).
TEST_AUCTIONS = ["action05"]
VAL_AUCTIONS = ["action06"]

# Recovers the auction id and frame name from any filename, with or without the
# Label Studio hash prefix. Handles both "action_01" and "action05" styles.
AUCTION_RE = re.compile(r"(action_?\d+)__\d+")
FRAMENAME_RE = re.compile(r"(action_?\d+__\d+)")
# ----------------------------------------------------------------------------


def canonical_frame_name(label_stem: str) -> str | None:
    m = FRAMENAME_RE.search(label_stem)
    return m.group(1) if m else None


def auction_of(name: str) -> str | None:
    m = AUCTION_RE.search(name)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(FRAMES_DIR))
    ap.add_argument("--export", default=str(EXPORT_DIR),
                    help="unzipped export root; label .txt files are found recursively")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--test", nargs="*", default=TEST_AUCTIONS,
                    help="auction ids for the test split")
    ap.add_argument("--val", nargs="*", default=VAL_AUCTIONS,
                    help="auction ids for the val split")
    args = ap.parse_args()

    frames_dir = Path(args.frames)
    export_dir = Path(args.export)
    out_dir = Path(args.out)
    test_aucs = set(args.test)
    val_aucs = set(args.val)

    overlap = test_aucs & val_aucs
    if overlap:
        sys.exit(f"ERROR: auction(s) {sorted(overlap)} listed in BOTH test and val.")

    if not export_dir.is_dir():
        sys.exit(f"ERROR: export dir not found: {export_dir}")
    if not frames_dir.is_dir():
        sys.exit(f"ERROR: frames dir not found: {frames_dir}")

    # Find label files anywhere under the export (handles export/labels/,
    # export/<nested>/labels/, or a flat export/), keeping only real frame
    # labels — classes.txt and notes.json are skipped by the name pattern.
    label_files = sorted(p for p in export_dir.rglob("*.txt")
                         if FRAMENAME_RE.search(p.stem))
    if not label_files:
        sys.exit(f"ERROR: no frame label .txt files found under {export_dir}. "
                 f"Make sure the YOLO export is unzipped there (it should contain "
                 f"a labels/ folder with files like <hash>-action_01__001680.txt).")
    print(f"Found {len(label_files)} label files under {export_dir}")

    # Build the full table: auction -> list of (label_path, image_path, is_boxed)
    records = []
    missing_images = []
    bad_names = []
    for lf in label_files:
        frame = canonical_frame_name(lf.stem)
        if not frame:
            bad_names.append(lf.name)
            continue
        auc = auction_of(frame)
        img = frames_dir / f"{frame}{FRAME_EXT}"
        if not img.exists():
            missing_images.append(f"{lf.name}  ->  {img}")
            continue
        is_boxed = lf.stat().st_size > 0
        records.append((auc, lf, img, is_boxed))

    if bad_names:
        print("WARNING: could not parse auction/frame from these labels:")
        for n in bad_names:
            print("  ", n)

    if missing_images:
        print(f"ERROR: {len(missing_images)} label(s) have no matching image in "
              f"{frames_dir}. First few:")
        for m in missing_images[:10]:
            print("  ", m)
        print("Fix: make sure frames/ holds the same frames you labeled "
              "(re-run sample_frames.py on all clips), then try again.")
        sys.exit(1)

    all_aucs = sorted({r[0] for r in records})
    unknown = (test_aucs | val_aucs) - set(all_aucs)
    if unknown:
        sys.exit(f"ERROR: test/val auction(s) {sorted(unknown)} not in the data. "
                 f"Available: {all_aucs}")

    def split_of(auc):
        if auc in test_aucs:
            return "test"
        if auc in val_aucs:
            return "val"
        return "train"

    # Wipe & recreate output tree (clean rebuild every run).
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            (out_dir / sub / split).mkdir(parents=True, exist_ok=True)

    # Copy files into their split, using the CANONICAL frame name (hash dropped),
    # so image and label basenames match exactly as YOLO requires.
    stats = defaultdict(lambda: {"img": 0, "boxed": 0, "empty": 0,
                                 "aucs": set(), "boxes": 0})
    for auc, lf, img, is_boxed in records:
        split = split_of(auc)
        frame = canonical_frame_name(lf.stem)
        shutil.copy2(img, out_dir / "images" / split / f"{frame}{FRAME_EXT}")
        shutil.copy2(lf, out_dir / "labels" / split / f"{frame}.txt")
        s = stats[split]
        s["img"] += 1
        s["aucs"].add(auc)
        if is_boxed:
            s["boxed"] += 1
            s["boxes"] += sum(1 for ln in lf.read_text().splitlines() if ln.strip())
        else:
            s["empty"] += 1

    # data.yaml — absolute path, forward slashes so it's valid YAML on Windows.
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: card\n"
    )

    # Summary
    print("\n" + "=" * 64)
    print("DATASET BUILT  ->", out_dir.resolve())
    print("=" * 64)
    for split in ("train", "val", "test"):
        s = stats[split]
        aucs = ", ".join(sorted(s["aucs"])) or "(none)"
        print(f"{split:5}  images={s['img']:4}  boxed={s['boxed']:4}  "
              f"empty={s['empty']:4}  boxes={s['boxes']:4}  | auctions: {aucs}")
    total = sum(stats[s]["img"] for s in ("train", "val", "test"))
    print("-" * 64)
    print(f"total images: {total}")
    print(f"data.yaml:    {data_yaml.resolve()}")

    # Leakage guard: no auction in more than one split.
    placement = defaultdict(set)
    for auc, lf, img, is_boxed in records:
        placement[auc].add(split_of(auc))
    leaked = {a: sorted(sp) for a, sp in placement.items() if len(sp) > 1}
    if leaked:
        print("\nERROR: auction(s) span multiple splits (leakage!):", leaked)
        sys.exit(1)
    print("\nOK: every auction is confined to a single split (no leakage).")
    if stats["val"]["img"] == 0 or stats["test"]["img"] == 0:
        print("WARNING: val or test split is empty — check your --val / --test ids.")


if __name__ == "__main__":
    main()
