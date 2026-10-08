# Auction Card Detector

A standalone, local YOLO detector that finds the **featured card** in live-auction
video frames and (now) classifies **card / label / slab**. It produces the clean
crop + grade context that feeds the Cards of Iris recognition/pricing pipeline.

Separate from the **Scan Detector** (`../YoloCardDetector`), which is the on-device
iOS/Android single-card/binder model. Same 3-class scheme; different job and dataset.

## Classes
- `card` — the card art (for a slab, the art window; for a raw card, the whole card)
- `label` — the grading banner (PSA/BGS/CGC) — present ⇒ it's a slab
- `slab` — the whole graded case

## Pipeline (offline → training)
1. `sample_frames.py` — clips → deduped frames (named by auction, for by-auction split)
2. `bootstrap_prelabel.py` / `prelabel_3class.py` — pre-label with the current model(s)
   so you correct boxes in Label Studio instead of drawing them
3. correct in Label Studio (config: `configs/labeling_config_3class.xml`)
4. `prepare_dataset.py` — export → train/val/test split **by whole auction** + `data.yaml`
5. `train.py` (or `notebooks/train_colab.ipynb` on a GPU) → `best.pt`
6. `eval.py` — score on **held-out** auctions (the honest check)
7. `detect_preview.py` — eyeball where it fires

## Runtime (production, in the app)
ONE 3-class model, one pass per sampled frame → crop/deskew art → embed → match → price.
Expensive embed+price runs **once per card lock**, not per frame. Target < 2–3 s/card.

## Folder guide
Tracked: scripts, configs, notebooks, docs, this README.
Not tracked (see `.gitignore`): `clips/ frames/ dataset/ export/ runs/` (big/regenerable)
and `*.pt` weights — **weights live in Google Drive** (`card_detector/output_*`).
See `models/README.md` for which weight is which.

## Honest-split rule
Train/val/test are split by WHOLE auction — never mix one auction across splits, so the
test score reflects generalization to unseen sellers.
