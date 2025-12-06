
```markdown
# Equation Scribe

Equation Scribe is a small toolchain to detect equations in technical papers (PDFs / scanned pages), extract and render them, and convert equation images into LaTeX. The project is split into a backend detector and utilities for synthetic data generation and training.

**This repo contains:**
- `equation_scribe/detector/` — detector utilities: synthetic data generator, preprocessing, tiling,
  COCO conversion and quick training/inference scripts.
- `tools/` — helper scripts (system checks, small utilities).
- `docs/` — docs for deployment and other notes.
- `tools/run_demo.ps1` — quick end-to-end demo helper (Windows Powershell script).

---

## Goals & Spiral Plan

- **Spiral 1**: basic detector and pipeline, synthetic training data, quick YOLOv8 training to detect equations.
- **Spiral 2**: robust equation detection across scanned PDFs, rotation-aware tight bounding boxes, conversion of equation images to LaTeX (im2latex), GUI improvements for human-in-the-loop editing.

See `docs/roadmap.md` (or your saved roadmap) for the full Spiral 2 plan.

---

## Quickstart (developer)

1. Create environment:

```bash
conda create -n eqscribe python=3.11 -y
conda activate eqscribe
pip install -r requirements.txt

2. Run system check:

python tools/check_prereqs.py


Generate quick synthetic data:

python -m equation_scribe.detector.synthetic_coco \
  --out-images detector/data/images/synth \
  --out-anns detector/data/annotations/instances_all.json \
  --n-pages 5 --eqs-per-page 6 --dpi 150


Split COCO annotations into tiles and train:

python -m equation_scribe.detector.split_coco_by_paper --in detector/data/annotations/instances_all.json --out detector/data/annotations --val-frac 0.2
# quick YOLO training via ultralytics
yolo detect train data=equation_scribe/detector/detector.yaml model=yolov8s.pt epochs=5 imgsz=1024


(If yolo CLI is not on PATH, run python -m ultralytics.)

Run inference on a single image:

python equation_scribe/detector/inference.py --model runs/detect/eq_detector_quick/weights/best.pt --image detector/data/images/synth_pre/page_0000.png --conf 0.25

Helpful scripts

tools/check_prereqs.py — local environment checks (binaries and Python packages).

tools/check_files_exist.py — verify COCO image paths exist (useful for debugging dataset generation).

tools/run_demo.ps1 — windows demo that runs through synthetic generation, train and inference steps.

Synthetic dataset generation

The synthetic_coco.py generator:

Renders LaTeX math using pdflatex where possible, else falls back to matplotlib’s mathtext.

Pastes multiple equation images per page with a non-overlap constraint and produces COCO-style annotations.

Supports rotation augmentation (--rotate-aug, --rotate-max) and deskewing options.

If you see rotated equations where the bounding box is too small, enable the tight-rotated-bbox code path (now included in the generator) that computes the correct rotated bounding box. See the generator docstring for flags.

Notes about training & metrics

We use YOLOv8 (Ultralytics) for quick experiments. The detector.yaml dataset file should point to the tiled images and the instances_tiles_train.json and instances_tiles_val.json COCO files.

Good indicators of training quality:

mAP@0.50 and mAP@[0.50:0.95] over the validation set.

Precision / recall and PR curves printed by Ultralytics.

For synthetic-only training, expect the detector to generalize poorly to real scanned papers. To improve:

Add real annotated papers (arXiv / IEEE style) to training/validation.

Increase synthetic data variety and augmentations (noise, blur, deskew, lighting).

Fine-tune detection to prefer tight bounding boxes (rotate-aware).

Where to look for outputs

Synthetic images: detector/data/images/synth (and synth_pre if deskew/rotation options used).

COCO annotations: detector/data/annotations/*

YOLO training runs: runs/detect/<run_name>

Trained weights: runs/detect/<run_name>/weights/

Contributing and cleanup

Please avoid committing large training data into the repo. Keep data in detector/data/ locally.

When regenerating synthetic data, clear detector/data/images/synth and detector/data/annotations/* before running the generator.

Run pytest -q to run the unit tests in detector/tests/.

License & Contacts

(Add license here)

Maintainer: Rick Spangler (spanglermobile@gmail.com)


---

## Additional notes & suggestions

- **Docstrings**: you mentioned docstrings should be updated. The obvious places to update are:
  - `equation_scribe/detector/synthetic_coco.py` — update the module docstring to show the new CLI flags (`--rotate-aug`, `--rotate-max`, `--deskew`, `--clear-data`) and describe the placement and rotation algorithm and the `place_and_annotate_on_page` function contract.
  - `equation_scribe/detector/tiling.py` and `split_coco_by_paper.py` — ensure docstrings describe expected input COCO structure and how outputs are named (`*_tiles_train.json`, etc).
  - `tools/run_demo.ps1` doc at the top describing what each step does and the required preconditions (e.g., pdflatex on PATH, ultralytics installed).

- **Additions to `run_demo.ps1`**:
  - Add an early step to call `python tools/check_prereqs.py` and fail early with a readable message if a critical binary is missing.
  - Add optional flags to `run_demo.ps1` to control cleanup (clear `detector/data/images/*`, `detector/data/annotations/*`, and `runs/detect/*`) prior to running steps.
  - Keep the `--deskew` and `--rotate-aug` flags in the synthetic generator documented