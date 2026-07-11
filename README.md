# Equation Scribe

Equation Scribe is a PDF equation annotation and training toolkit. The purpose is to vet papers for use in a RAG. It has two main functions:

- Detector and recognition utilities for generating data, preparing datasets,
  and training models for recognizing equations in PDFs
- A web app for uploading PDFs, reviewing detected equations, editing boxes,
  validating LaTeX, and saving paper profiles

Shared storage, models, index management, runtime settings, and path utilities
now live in `packages/core` under the `equation_scribe_core` package.

## Repo Layout

- `apps/web/backend/` FastAPI backend for the annotation app
- `apps/web/frontend/` React + Vite frontend for the annotation app
- `equation_scribe/` detector, recognition, PDF, and CLI utilities
- `packages/core/` shared core package introduced by the migration
- `tools/` helper scripts for prerequisite checks and dataset conversion
- `docs/` architecture and planning notes

## Environment Setup

### Python

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

`pip install -e .` is recommended so `equation_scribe_core` is importable
without relying on ad hoc `PYTHONPATH` setup.

If you prefer Conda, the repo also includes `environment.yml`:

```conda
conda env create -f environment.yml -n eqscribe
conda activate eqscribe
python -m pip install -e .
```

### Optional Training Dependencies

Some training and rendering paths depend on packages that are not installed by
`requirements-dev.txt` alone.

- `ultralytics` for YOLO training and detector inference
- `torch` for YOLO and recognition inference/training
- `matplotlib` for the synthetic generator fallback renderer
- `pdf2image` for PDF-to-image conversion helpers
- `pytesseract` and the Tesseract binary for OCR-assisted flows

Example:

```conda
python -m pip install ultralytics matplotlib pdf2image pytesseract
```

Install PyTorch using the command recommended for your platform from the
official PyTorch installer.

### External Tools

Depending on which workflows you use, you may also need:

- `pdflatex` for higher-fidelity LaTeX rendering in synthetic data generation
- `pdftoppm` from Poppler for `pdf2image`
- `tesseract` for OCR fallback
- `node` and `npm` for the web frontend

You can check the local environment with:

```conda
python tools/check_prereqs.py
```

## Running The Web GUI

The active GUI is the web app under `apps/web`. The old
`equation_scribe/ui_gradio.py` module is deprecated and should not be treated
as the primary UI.

### Backend

The backend defaults to:

- `data/pdfs` for uploaded PDFs
- `data/profiles` for saved equation profiles and `index.json`

You can override those locations with environment variables:

```powershell
$env:PAPERS_ROOT = "C:\path\to\pdfs"
$env:PROFILES_ROOT = "C:\path\to\profiles"
```

Start the backend from the repo root:

```Anaconda Shell
cd [equation_scribe root]
conda activate eqscribe
uvicorn apps.web.backend.main:app --reload
```

The backend listens on `http://127.0.0.1:8000`.

### Frontend

Start the frontend from `apps/web/frontend`:

```powershell
cd apps/web/frontend
npm install
npm run dev
```

Vite runs on `http://127.0.0.1:5173`, which matches the hardcoded API target in
`apps/web/frontend/src/api/client.ts`.

### Typical GUI Flow

1. Start the backend.
2. Start the frontend.
3. Open `http://127.0.0.1:5173`.
4. Upload a PDF through the UI.
5. Review pages, draw or edit boxes, validate LaTeX, and save equation records.
6. Optionally use the "Scan Entire Paper" action to run backend autodetection
   across the uploaded PDF.

### Backend Endpoints In Current Use

- `POST /upload`
  Uploads a PDF into `PAPERS_ROOT`, derives a `paper_id` from the filename, and
  returns that `paper_id` so the frontend can load the paper.
- `GET /papers/index`
  Returns the shared `index.json` contents as JSON so clients can inspect the
  known paper/profile registry.
- `GET /papers/{paper_id}/pages`
  Returns the page count for one uploaded PDF.
- `GET /papers/{paper_id}/page/{idx}/image`
  Renders one PDF page as a PNG for display in the frontend canvas.
- `GET /papers/{paper_id}/page/{idx}/meta`
  Returns page metadata used by the frontend to align PDF-space boxes with the
  rendered page image.
- `GET /papers/{paper_id}/equations`
  Loads the saved `equations.jsonl` profile for a paper and returns its records.
- `POST /papers/{paper_id}/equations`
  Appends a new equation record to the paper profile. The record must include at
  least one box.
- `PUT /papers/{paper_id}/equations/{eq_uid}`
  Rewrites an existing equation record identified by `eq_uid`.
- `DELETE /papers/{paper_id}/equations/{eq_uid}`
  Removes an existing equation record. The backend returns `404` if the record
  does not exist.
- `POST /validate`
  Validates a LaTeX string and returns parser/validation results for the editor
  workflow.
- `POST /papers/{paper_id}/rescan_box`
  Crops a user-selected PDF region and runs the recognition model to propose a
  LaTeX string for that one box.
- `POST /papers/{paper_id}/autodetect_all`
  Runs detector inference across the full uploaded paper, deduplicates
  overlapping results, and appends the detected equations into the saved paper
  profile.

## Detector Training Workflow

The training path in this repo is:

1. generate synthetic pages and COCO annotations
2. split the COCO file into train and validation sets by paper
3. optionally preprocess page images
4. tile images into detector-sized crops
5. convert tiled COCO annotations to YOLO label files
6. train YOLO with a dataset yaml that points at the tiled images

### 1. Generate Synthetic Data

```conda
python -m equation_scribe.detector.synthetic_coco `
  --out-images detector/data/images/synth `
  --out-anns detector/data/annotations/instances_all.json `
  --n-pages 50 `
  --n-papers 200 `
  --eqs-per-page 4 `
  --dpi 150
```

Notes:

- The generator can use `pdflatex` when available and otherwise falls back to
  `matplotlib`.
- Output image names are grouped by synthetic paper id, which is what the split
  script expects.

### 2. Split Train / Validation By Paper

```conda
python -m equation_scribe.detector.split_coco_by_paper ^
  --coco detector/data/annotations/instances_all.json ^
  --out-dir detector/data/annotations ^
  --val-frac 0.2 ^
  --seed 0
```

This produces:

- `detector/data/annotations/instances_train.json`
- `detector/data/annotations/instances_val.json`

### 3. Optional Preprocessing

If you want a more scan-like image set before tiling:

```conda
python -m equation_scribe.detector.preprocess ^
  --input detector/data/images/synth ^
  --output detector/data/images/synth_pre ^
  --denoise --deskew --clahe --binarize 
```

If you skip this step, use `detector/data/images/synth` as the image root in
the tiling step below.

### 4. Tile The Train And Validation Sets

```conda
python -m equation_scribe.detector.tiling ^
  --coco detector/data/annotations/instances_train.json ^
  --images-root detector/data/images/synth_pre ^
  --out-images detector/data/images/tiles_train ^
  --out-annotations detector/data/annotations/instances_tiles_train.json ^
  --tile-size 1024 ^
  --stride 512

python -m equation_scribe.detector.tiling ^
  --coco detector/data/annotations/instances_val.json ^
  --images-root detector/data/images/synth_pre ^
  --out-images detector/data/images/tiles_val ^
  --out-annotations detector/data/annotations/instances_tiles_val.json ^
  --tile-size 1024 ^
  --stride 512
```

### 5. Convert Tiled COCO To YOLO Labels

```conda
python tools/convert_coco_to_yolo.py ^
  --coco detector/data/annotations/instances_tiles_train.json ^
  --dataset-root detector/data ^
  --out-labels detector/data/labels

python tools/convert_coco_to_yolo.py ^
  --coco detector/data/annotations/instances_tiles_val.json ^
  --dataset-root detector/data ^
  --out-labels detector/data/labels
```

### 6. Point YOLO At The Tiled Dataset

Before training, verify the dataset yaml. The checked-in
`equation_scribe/detector/detector.yaml` currently contains a machine-specific
absolute path and should be reviewed locally.

For the tiled workflow above, the important values should look like:

```yaml
path: C:/path/to/your/repo/equation_scribe/detector/data
train: images/tiles_train
val: images/tiles_val
nc: 1
names:
  0: equation
```

### 7. Train YOLO

```conda
yolo task=detect mode=train ^
  model=yolov8s.pt ^
  data=equation_scribe/detector/detector.yaml ^
  epochs=5 ^
  imgsz=1024 ^
  batch=4 ^
  name=eq_detector_quick
```

If the `yolo` CLI is not on `PATH`, use the Python entrypoint provided by your
Ultralytics install instead.

### 8. Run Detector Inference

```conda
python -m equation_scribe.detector.inference ^
  --model runs/detect/eq_detector_quick/weights/best.pt ^
  --image detector/data/images/synth_pre/paper000_page_0000.png ^
  --conf 0.25
```

## Heuristic Autodetect CLI

The heuristic autodetector remains available as a repo-local CLI:

```conda
python -m equation_scribe.autodetect_equations ^
  --pdf "C:\path\to\papers\MyPaper.pdf" ^
  --paper-id "MyPaper" ^
  --data-root "C:\path\to\profiles" ^
  --min-score 0.6 ^
  --force
```

This writes:

- `<data-root>/<paper-id>/equations.jsonl`
- `<data-root>/index.json`

This CLI now uses the shared core persistence and index layer introduced during
the migration.

## Useful Scripts

- `python tools/check_prereqs.py` verify local binaries and Python packages
- `conda -ExecutionPolicy Bypass -File .\tools\run_demo.ps1` smoke test
  the synthetic-data and detector pipeline

## Testing

Run the full test suite from the repo root:

```conda
python -m pytest -q
```

## Notes

- Avoid committing generated training data and model outputs.
- The web app is the supported GUI. Gradio is legacy-only.
- If imports in VS Code fail for `equation_scribe_core`, the workspace already
  includes `.vscode/settings.json` with the extra analysis path for
  `packages/core/src`.

## License

Add license information here.
