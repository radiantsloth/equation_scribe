<#
.SYNOPSIS
    Run end-to-end demo for Equation Scribe detector + synthetic data + quick training.

.DESCRIPTION
    This demo orchestrates:
      1. Synthetic dataset generation (render LaTeX -> PNG, compose pages, COCO annotations)
      2. Split into train/val, produce tiles, and create YOLO dataset
      3. Quick YOLOv8 training (small number of epochs)
      4. Inference on a sample page (detect equation boxes)
      5. Optional downstream recognition steps (pair creation)

    Purpose: quick smoke test of the detector pipeline. The script is NOT for full
    production training — adjust parameters (epochs, batch size, dataset size) as needed.

.PREREQUISITES
    On Windows:
      - Anaconda/Miniconda (recommended). Create the `eqscribe` environment and install
        Python dependencies (see README).
      - MiKTeX or TeX Live (pdflatex) for LaTeX rendering of complex math.
      - poppler (pdftoppm) for pdf -> png conversion when using pdflatex route.
      - Tesseract OCR for OCR fallback when processing scanned PDFs.
      - Node.js & npm (for the frontend; optional for run_demo).
      - Ultralytics (yolo) package for YOLOv8 training (CPU/GPU).
      - CUDA + compatible PyTorch for GPU training (optional but faster).

    Important PATHs:
      - Ensure pdflatex, pdftoppm (poppler), and tesseract are in PATH.
      - If not in PATH, set the appropriate variables or add them in your system environment.

.USAGE
    Open a PowerShell console with ExecutionPolicy set to allow script execution:
      powershell -ExecutionPolicy Bypass

    Activate your environment (example):
      conda activate eqscribe

    Run the demo (quick run):
      powershell -ExecutionPolicy Bypass -File ".\tools\run_demo.ps1"

    If you want a reproducible synthetic run:
      powershell -ExecutionPolicy Bypass -File ".\tools\run_demo.ps1" -nPages 5 -eqsPerPage 6 -rotateAug $true -rotateMax 12 -seed 123

NOTES AND TROUBLESHOOTING
    - If pdflatex fails: check `pdflatex --version`. MiKTeX on Windows sometimes needs packages installed on demand; run MiKTeX Console and make sure shell calls to pdflatex succeed.
    - If pdf2image fails: ensure poppler's `pdftoppm` is installed and on PATH.
    - If tesseract fails: check `tesseract --version` and ensure its path is set.
    - If YOLO/Ultralytics fails or complains about dataset paths, verify the detector YAML `detector/detector.yaml` paths and that `detector/data/images/...` and `detector/data/annotations` exist and are non-empty.
    - To reset state: the script can be re-run after cleaning output directories. See the CLEANUP section below.
    - For Windows fsevents / node errors: these are Mac-only packages; ignore on Windows. If `npm install` throws fsevents 404, remove optional dependencies or run `npm install --no-optional`.

CLEANUP (manual commands)
    # Remove synthetic images and annotations; useful before re-running the demo:
    Remove-Item -Recurse -Force detector\data\images\synth* -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force detector\data\annotations\* -ErrorAction SilentlyContinue
    # Remove YOLO run directory:
    Remove-Item -Recurse -Force runs\detect\eq_detector_quick* -ErrorAction SilentlyContinue

#>


$ErrorActionPreference = 'Stop'

Write-Host "=== Equation Scribe detector/recognizer demo ===" -ForegroundColor Cyan
Write-Host "Make sure your 'eqscribe' conda env is active and 'yolo' (ultralytics) is installed."

# Helper: run python and abort on failure
function Run-Python([string]$args) {
    Write-Host "python $args" -ForegroundColor Yellow
    & python $args
    if ($LASTEXITCODE -ne 0) { throw "Command failed: python $args" }
}

# --------------- LaTeX rendering step ---------------
# This step uses pdflatex -> pdf2image. If pdflatex fails, check:
#   pdflatex --version
# and ensure poppler's pdftoppm is installed (pdf2image depends on it).
# On Windows, MiKTeX may need packages installed via MiKTeX Console.

# 1) Generate synthetic pages (small dataset)
Write-Host "`n1) Generating synthetic data..." -ForegroundColor Green
python equation_scribe\detector\synthetic_coco.py --out-images detector/data/images/synth --out-anns detector/data/annotations/instances_all.json --n-pages 50 --eqs-per-page 4 --dpi 150 --n-papers 200


# 2) Split COCO by paper -> instances_train.json / instances_val.json
Write-Host "`n2) Splitting COCO by paper..." -ForegroundColor Green
python equation_scribe\detector\split_coco_by_paper.py --coco detector/data/annotations/instances_all.json --out-dir detector/data/annotations --val-frac 0.2 --seed 0

# 3) Preprocess (optional; good for scan-like images)
Write-Host "`n3) Preprocessing pages (denoise, deskew, CLAHE, binarize)..." -ForegroundColor Green
python equation_scribe\detector\preprocess.py --input detector/data/images/synth --output detector/data/images/synth_pre --denoise --deskew --clahe --binarize

# --------------- YOLO quick training ---------------
# Uses Ultralytics YOLOv8. For GPU training ensure torch + CUDA are installed.
# If "images not found" or dataset errors occur, check the generated COCO files:
#   detector/data/annotations/instances_tiles_*.json
# and the image folders under detector/data/images/tiles_{train,val}.

# 4) Tile the train set (creates tiles and tile-level COCO)
Write-Host "`n4) Tiling (train set)..." -ForegroundColor Green
python equation_scribe\detector\tiling.py --coco detector/data/annotations/instances_val.json --images-root detector/data/images/synth_pre --out-images detector/data/images/tiles_val --out-annotations detector/data/annotations/instances_tiles_val.json --tile-size 1024 --stride 512 --min-area-frac 0.25 --keep-empty-prob 0.05

python equation_scribe/detector/tiling.py --coco detector/data/annotations/instances_train.json --images-root detector/data/images/synth_pre --out-images detector/data/images/tiles_train --out-annotations detector/data/annotations/instances_tiles_train.json --tile-size 1024 --stride 512 --min-area-frac 0.25 --keep-empty-prob 0.05

# Create labels for train
python tools/convert_coco_to_yolo.py --coco detector/data/annotations/instances_tiles_train.json --dataset-root detector/data --out-labels detector/data/labels

# Create labels for val
python tools/convert_coco_to_yolo.py --coco detector/data/annotations/instances_tiles_val.json --dataset-root detector/data --out-labels detector/data/labels


# 5) Ensure detector/detector.yaml points to tiled dataset
Write-Host "`n5) Verify detector/detector.yaml" -ForegroundColor Green
Write-Host "If necessary, edit equation_scribe/detector/detector.yaml to point to the tiled images:" -ForegroundColor Yellow
Write-Host "  path: detector/data" -ForegroundColor Yellow
Write-Host "  train: images/tiles_train" -ForegroundColor Yellow
Write-Host "  val: images/tiles_train  # for demo or set to instances_val.json" -ForegroundColor Yellow

# 6) Quick YOLOv8 training (small smoke run)
Write-Host "`n6) Quick YOLOv8 training (5 epochs). Adjust batch/imgsz/device as needed..." -ForegroundColor Green
try {
    Write-Host "Running YOLOv8 training..." -ForegroundColor Yellow
    & yolo task=detect mode=train model=yolov8s.pt data=equation_scribe\detector\detector.yaml epochs=5 imgsz=1024 batch=4 device=0 name=eq_detector_quick
    if ($LASTEXITCODE -ne 0) { throw "yolo returned non-zero exit code $LASTEXITCODE" }
} catch {
    Write-Warning "Failed to run 'yolo' CLI. If ultralytics is installed but 'yolo' isn't on PATH, try: `python -m ultralytics ...` or install ultralytics and ensure yolo is accessible."
    Write-Warning "Skipping YOLO training step."
}

# 7) Inference on a sample page using trained weights (if produced)
Write-Host "`n7) Inference on a sample page (if training produced weights)..." -ForegroundColor Green
$bestWeights = "runs\detect\eq_detector_quick\weights\best.pt"
if (Test-Path $bestWeights) {    
    python equation_scribe\detector\inference.py --model $bestWeights --image detector/data/images/synth_pre/paper000_page_0000.png --conf 0.25
} else {
    Write-Warning "Best weights not found at $bestWeights -- skipping inference."
}

# 8) Make recognition pairs (crop images + gold LaTeX) using .meta.json produced by synthetic generator
Write-Host "n8) Create recognition pairs (crops -> latex)..." -ForegroundColor Green
python equation_scribe\detector\make_pairs.py --coco detector/data/annotations/instances_all.json --out-images detector/data/recognition/images --out-jsonl detector/data/recognition/pairs.jsonl --page-images-root detector/data/images/synth_pre

Write-Host "nDemo finished." -ForegroundColor Cyan
Write-Host "Inspect:" -ForegroundColor Cyan
Write-Host "  detector/data/images/synth_pre    (preprocessed pages)"
Write-Host "  detector/data/images/tiles_train (tiles)"
Write-Host "  detector/data/annotations/*      (COCO files)"
Write-Host "  runs/detect/eq_detector_quick    (YOLO run logs/weights if training ran)"
Write-Host "  detector/data/recognition/       (recognition crops and pairs.jsonl)" -ForegroundColor Cyan
