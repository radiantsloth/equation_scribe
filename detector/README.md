# Detector: Equation Detection Baseline (YOLOv8)

This `detector/` directory contains an initial scaffold to train and run an equation detector (two classes: `display` and `inline`) over PDF page images. This is a YOLOv8-based starter pipeline and is intended to be extended.

**Key points**
- Local-only data: this scaffold assumes you store images and annotations locally under `detector/data/`.
- Do **not** check large datasets or model weights into git. See `.gitignore`.
- The scripts help convert your existing `equations.jsonl` profiles into COCO annotations and train a YOLOv8 baseline. For scanned PDFs, focus on preprocessing (denoise, deskew, binarize, super-resolution) and tiling.

## Files
- `requirements.txt` — lighter python deps
- `detector.yaml` — dataset YAML for ultralytics
- `.gitignore` — detector ignores
- `render_latex.py` — quick synthetic render utilities
- `data_prep.py` — convert `equations.jsonl` → COCO annotations (uses your repo's pdf helpers if available)
- `train_detector.sh` — example YOLOv8 training command
- `inference.py` — a simple inference helper and PDF coordinate conversion snippet

## Quickstart

1. Create and activate environment:
```bash
conda activate eqscribe
pip install -r detector/requirements.txt

2. Prepare images and COCO annotations under:
detector/data/images/train
detector/data/images/val
detector/data/annotations/instances_train.json
detector/data/annotations/instances_val.json

3. Train a quick baseline:
bash detector/train_detector.sh

4. Run inference on a page image:
python detector/inference.py --image detector/data/images/val/page_001.png
