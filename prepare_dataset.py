#!/usr/bin/env python3
r"""
prepare_dataset.py (3-class) — merge your 1-class and 3-class Label Studio exports
into one YOLO dataset (card / label / slab), split BY AUCTION.

Merge logic:
  - OLD export (1-class): every box is class 0 = card. For RAW cards that's correct
    under the 3-class scheme (card = the card). Slab-heavy auctions must be EXCLUDED
    (their boxes are whole-slab-as-card, which is wrong now) — see EXCLUDE_AUCTIONS.
  - NEW export (3-class): card/label/slab, used as-is (class ids already line up).
  - Images are found in any of FRAME_DIRS by the canonical frame name.

Why by-auction: test = whole unseen auctions, so the score reflects generalization.
NOTE this round: all slabs are in one auction (action07), so it must go in TRAIN —
slab/label has no held-out test yet. Card is tested honestly on a held-out raw seller.

    python prepare_dataset.py
    python prepare_dataset.py --test action05 --val action09
"""
import argparse, re, shutil, sys
from collections import defaultdict
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
EXPORT_DIRS = ["Export", "export_3class"]   # all your export folders (labels read recursively)
FRAME_DIRS  = ["frames", "frames_new"]       # where the .jpg frames live
OUT_DIR     = "dataset"
FRAME_EXT   = ".jpg"
EXCLUDE_AUCTIONS = ["action06"]              # old slabs mislabeled as card — drop this round
TEST_AUCTIONS = ["action05"]
VAL_AUCTIONS  = ["action09"]
CLASS_NAMES = {0: "card", 1: "label", 2: "slab"}

FRAMENAME_RE = re.compile(r"(action_?\d+__\d+)")
AUCTION_RE   = re.compile(r"(action_?\d+)__\d+")
# ────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", nargs="*", default=EXPORT_DIRS)
    ap.add_argument("--frames", nargs="*", default=FRAME_DIRS)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--exclude", nargs="*", default=EXCLUDE_AUCTIONS)
    ap.add_argument("--test", nargs="*", default=TEST_AUCTIONS)
    ap.add_argument("--val", nargs="*", default=VAL_AUCTIONS)
    args = ap.parse_args()

    out_dir = Path(args.out)
    test_aucs, val_aucs, excl = set(args.test), set(args.val), set(args.exclude)
    if test_aucs & val_aucs:
        sys.exit(f"ERROR: {sorted(test_aucs & val_aucs)} in both test and val.")

    frame_dirs = [Path(d) for d in args.frames if Path(d).is_dir()]
    if not frame_dirs:
        sys.exit(f"ERROR: none of these frame dirs exist: {args.frames}")

    # index every image by canonical name across all frame dirs
    img_index = {}
    for fd in frame_dirs:
        for p in fd.glob(f"*{FRAME_EXT}"):
            m = FRAMENAME_RE.search(p.stem)
            if m:
                img_index.setdefault(m.group(1), p)

    # collect labels from all export dirs, keyed by canonical name (dedup)
    records = {}          # canonical -> (label_path, image_path, auction)
    dup, missing_img, excluded = [], [], 0
    for ed in args.exports:
        edp = Path(ed)
        if not edp.is_dir():
            print(f"(skip missing export dir: {ed})"); continue
        for lf in edp.rglob("*.txt"):
            m = FRAMENAME_RE.search(lf.stem)
            if not m:
                continue                      # classes.txt, notes, etc.
            canon = m.group(1)
            auc = AUCTION_RE.search(canon).group(1)
            if auc in excl:
                excluded += 1; continue
            if canon in records:
                dup.append(canon); continue
            img = img_index.get(canon)
            if not img:
                missing_img.append(canon); continue
            records[canon] = (lf, img, auc)

    if missing_img:
        print(f"ERROR: {len(missing_img)} labels have no image in {args.frames}. First few:")
        for c in missing_img[:10]:
            print("  ", c)
        sys.exit(1)
    if dup:
        print(f"(note: {len(dup)} duplicate frame names across exports — kept first)")
    print(f"excluded {excluded} labels from {sorted(excl)}")

    def split_of(auc):
        return "test" if auc in test_aucs else "val" if auc in val_aucs else "train"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for sub in ("images", "labels"):
        for sp in ("train", "val", "test"):
            (out_dir / sub / sp).mkdir(parents=True, exist_ok=True)

    stats = defaultdict(lambda: {"img": 0, "empty": 0,
                                 "card": 0, "label": 0, "slab": 0, "aucs": set()})
    for canon, (lf, img, auc) in records.items():
        sp = split_of(auc)
        shutil.copy2(img, out_dir / "images" / sp / f"{canon}{FRAME_EXT}")
        shutil.copy2(lf, out_dir / "labels" / sp / f"{canon}.txt")
        s = stats[sp]; s["img"] += 1; s["aucs"].add(auc)
        lines = [l for l in lf.read_text().splitlines() if l.strip()]
        if not lines:
            s["empty"] += 1
        for l in lines:
            s[CLASS_NAMES[int(l.split()[0])]] += 1

    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: 3\nnames:\n  0: card\n  1: label\n  2: slab\n")

    print("\n" + "=" * 72)
    print("DATASET BUILT ->", out_dir.resolve())
    print("=" * 72)
    hdr = f"{'split':5} {'imgs':>5} {'empty':>6} {'card':>5} {'label':>6} {'slab':>5}  auctions"
    print(hdr)
    for sp in ("train", "val", "test"):
        s = stats[sp]
        print(f"{sp:5} {s['img']:5} {s['empty']:6} {s['card']:5} {s['label']:6} "
              f"{s['slab']:5}  {', '.join(sorted(s['aucs'])) or '(none)'}")
    # leakage guard
    placement = defaultdict(set)
    for _, (_, _, auc) in records.items():
        placement[auc].add(split_of(auc))
    leaked = {a: sorted(v) for a, v in placement.items() if len(v) > 1}
    if leaked:
        sys.exit(f"\nERROR: auctions span splits (leakage): {leaked}")
    print("\nOK: no auction spans multiple splits.")
    if stats["train"]["slab"] == 0:
        print("WARNING: no slabs in TRAIN — the model can't learn slab/label.")
    if stats["test"]["slab"] == 0:
        print("NOTE: no slabs in TEST — slab/label is train-only this round (expected).")


if __name__ == "__main__":
    main()
