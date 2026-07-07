
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
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt
```
2. Run system check:

python tools/check_prereqs.py


Generate quick synthetic data:

python -m equation_scribe.detector.synthetic_coco \
  --out-images detector/data/images/synth \
  --out-anns detector/data/annotations/instances_all.json \
  --n-pages 5 --eqs-per-page 6 --dpi 150


3. Split COCO annotations into tiles and train:
```bash
python -m equation_scribe.detector.split_coco_by_paper --in detector/data/annotations/instances_all.json --out detector/data/annotations --val-frac 0.2
# quick YOLO training via ultralytics
yolo detect train data=equation_scribe/detector/detector.yaml model=yolov8s.pt epochs=5 imgsz=1024
```

(If yolo CLI is not on PATH, run python -m ultralytics.)

Run inference on a single image:
```bash
python equation_scribe/detector/inference.py --model runs/detect/eq_detector_quick/weights/best.pt --image detector/data/images/synth_pre/page_0000.png --conf 0.25
```

4. Helpful scripts

tools/check_prereqs.py — local environment checks (binaries and Python packages).

tools/check_files_exist.py — verify COCO image paths exist (useful for debugging dataset generation).

tools/run_demo.ps1 — windows demo that runs through synthetic generation, train and inference steps.

5. Synthetic dataset generation

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

6. Where to look for outputs

Synthetic images: detector/data/images/synth (and synth_pre if deskew/rotation options used).

COCO annotations: detector/data/annotations/*

YOLO training runs: runs/detect/<run_name>

Trained weights: runs/detect/<run_name>/weights/

7. Contributing and cleanup

Please avoid committing large training data into the repo. Keep data in detector/data/ locally.

When regenerating synthetic data, clear detector/data/images/synth and detector/data/annotations/* before running the generator.

Run pytest -q to run the unit tests in detector/tests/.

# Equation Scribe Web (React + PDF.js + Konva + FastAPI)

A PDF-based equation annotation tool (frontend + backend) that provides:
- interactive PDF viewing and zoom
- drawing / moving / resizing equation bounding boxes
- LaTeX editing and KaTeX preview
- LaTeX validation (SymPy/ANTLR)
- saving per-paper structured JSONL「equations.jsonl」profiles
- an index that maps PDFs → `paper_id` profiles for consistent loading

---

## Repo layout

equation_scribe/apps/web/
├── backend/ # FastAPI backend
├── frontend/ # React + Vite + Konva frontend
├── docs/ # Spiral roadmaps & documentation
└── paper_profiles/ # saved JSONL profiles (local example)


---

## Quickstart (development)

These examples assume Windows PowerShell and a Conda environment named `eqscribe`. Adjust paths and shell commands for Linux/macOS.

### Prereqs
- Python 3.10+ (3.11 tested)
- Conda (recommended)
- Node.js 18+ and npm 9+
- Optional: Tesseract OCR installed & on PATH (for heuristic OCR in autodetect)

### Environment
Create/activate the conda environment (if you have `environment.yml`):

```powershell
conda env create -f environment.yml -n eqscribe
conda activate eqscribe

or manually

conda activate eqscribe
pip install -r requirements.txt

# temporary for current session
$env:PAPERS_ROOT = "C:\[BASEDIR]\papers"
$env:PROFILES_ROOT = "C:\[BASEDIR]\paper_profiles"

To make permanent on Windows

setx PAPERS_ROOT "C:\[BASEDIR]\papers"
setx PROFILES_ROOT "C:\[BASEDIR]\paper_profiles"

Backend (FastAPI)

From repo root (equation_scribe_web), ensure dependencies installed:

conda activate eqscribe
pip install -r requirements.txt

2) Start the backend:

cd [EQSCRIBE_ROOT]\equation_scribe
uvicorn apps.web.backend.main:app --reload --reload-dir apps/web/backend

Backend endpoints of interest:

GET /papers/index — profiles index JSON.

GET /papers/find_by_pdf?basename=<name> — find a profile by PDF basename.

GET /papers/{paper_id}/equations — list equations for a paper.

POST /papers/{paper_id}/equations — append a new equation.

PUT /papers/{paper_id}/equations/{eq_uid} — update an equation record.

DELETE /papers/{paper_id}/equations/{eq_uid} — delete an equation.

GET /papers/{paper_id}/page/{idx}/image and /meta — page image and metadata.

POST /validate — validate LaTeX with SymPy.

CORS: The backend allows the default dev origin (http://127.0.0.1:5173). If your frontend runs elsewhere, update CORS settings in backend/main.py.

Notes:

If you see ModuleNotFoundError: No module named 'backend', run uvicorn from the repo root (as shown).

For LaTeX parsing, SymPy requires antlr4-python3-runtime==4.11 (install if you see ANTLR errors).

If you already have the wrong `fitz` package installed, uninstall it before installing dependencies:

```powershell
python -m pip uninstall -y fitz
python -m pip install pymupdf
```

Frontend (React + Vite + Konva)

Install and run:

cd frontend
npm install
npm run dev
Vite will show a local URL (commonly http://127.0.0.1:5173). Open in the browser.

If you see Cannot find @vitejs/plugin-react, run:

powershell
Copy code
cd frontend
npm install @vitejs/plugin-react --save-dev
Windows note: fsevents warnings are normal and can be ignored.

Autodetector CLI (equation_scribe repo)
The heuristic autodetector and profile registration are in the equation_scribe project (separate repo). Example usage:

powershell
Copy code
# in equation_scribe repo
conda activate eqscribe
python -m equation_scribe.autodetect_equations `
  --pdf "C:\[BASEDIR]\papers\MyPaper.pdf" `
  --paper-id "MyPaper" `
  --data-root "C:\[BASEDIR]\paper_profiles" `
  --min-score 0.6 --force
This writes PROFILES_ROOT/MyPaper/equations.jsonl and updates PROFILES_ROOT/index.json.


---

# 📅 Roadmap (Next Spirals)

* Spiral 2:
  Reloading saved boxes + editing existing datasets
* Spiral 3:
  Auto-detect candidate equations (ML + heuristics)
* Spiral 4:
  Symbol extraction + glossary building
* Spiral 5:
  Full RAG pipeline: equation search + explanation + consistency checking

---



## License & Contacts

(Add license here)

Maintainer: Rick Spangler (spanglermobile@gmail.com)

