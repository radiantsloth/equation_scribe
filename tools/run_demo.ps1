# # tools/run_demo.ps1
# # Usage: from repo root
# #   .\tools\run_demo.ps1

# $ErrorActionPreference = 'Stop'

# # Activate environment first (do it in your shell)
# Write-Host "Ensure eqscribe conda env is active."

# # 1) Generate synthetic pages (small)
# Write-Host "1) Generating synthetic data..."
# python .\equation_scribe\equation_scribe\detector\synthetic_coco.py `
#   --out-images equation_scribe/detector/data/images/synth `
#   --out-anns equation_scribe/detector/data/annotations/instances_all.json `
#   --n-pages 20 --eqs-per-page 4 --dpi 150

# # 2) Split by paper -> train/val
# Write-Host "2) Splitting COCO by paper..."
# python .\equation_scribe\detector\split_coco_by_paper.py `
#   --coco equation_scribe/detector/data/annotations/instances_all.json `
#   --out-dir equation_scribe/detector/data/annotations --val-frac 0.2 --seed 0

# # 3) Preprocess (optional, good for scanned style)
# Write-Host "3) Preprocessing pages..."
# python .\equation_scribe\equation_scribe\detector\preprocess.py `
#   --input equation_scribe/detector/data/images/synth `
#   --output equation_scribe/detector/data/images/synth_pre `
#   --denoise --deskew --clahe --binarize

# # 4) Tile the train set
# Write-Host "4) Tiling (train set) - this can generate many images..."
# python .\equation_scribe\detector\tiling.py `
#   --coco equation_scribe/detector/data/annotations/instances_train.json `
#   --images-root equation_scribe/detector/data/images/synth_pre `
#   --out-images equation_scribe/detector/data/images/tiles_train `
#   --out-annotations equation_scribe/detector/data/annotations/instances_tiles_train.json `
#   --tile-size 1024 --stride 512 --min-area-frac 0.25 --keep-empty-prob 0.05

# # 5) Prepare detector YAML (update if necessary)
# Write-Host "5) Ensure equation_scribe/detector/detector.yaml points to tiled images. Example:"
# Write-Host "   path: equation_scribe/detector/data"
# Write-Host "   train: equation_scribe/images/tiles_train"
# Write-Host "   val: equation_scribe/images/tiles_train   # for demo use train as val"

# # 6) Quick YOLOv8 training (5 epochs) - adjust device if necessary
# Write-Host "6) Quick YOLO train (5 epochs) - check GPU memory and adjust batch size"
# bash -c "yolo task=detect mode=train model=yolov8s.pt data=equation_scribe/detector/detector.yaml epochs=5 imgsz=1024 batch=4 device=0 name=eq_detector_quick" 

# # 7) Inference on a sample page
# Write-Host "7) Inference on a sample page"
# python .\equation_scribe\detector\inference.py `
#   --model runs/detect/eq_detector_quick/weights/best.pt `
#   --image detector/data/images/synth_pre/page_0000.png --conf 0.25

# # 8) Create recognition pairs from full COCO (uses .meta.json)
# Write-Host "8) Make recognition pairs (crops with gold latex)"
# python .\equation_scribe\detector\make_pairs.py `
#   --coco detector/data/annotations/instances_all.json `
#   --out-images detector/data/recognition/images `
#   --out-jsonl detector/data/recognition/pairs.jsonl `
#   --page-images-root detector/data/images/synth_pre

# Write-Host "Demo complete. Inspect detector/data/... and run further training as required."

# tools/run_demo.ps1
# PowerShell demo orchestration for Windows.
# Run from repo root after activating the 'eqscribe' conda env:
#   conda activate eqscribe
#   .\tools\run_demo.ps1

$ErrorActionPreference = 'Stop'

Write-Host "=== Equation Scribe detector/recognizer demo ===" -ForegroundColor Cyan
Write-Host "Make sure your 'eqscribe' conda env is active and 'yolo' (ultralytics) is installed."

# Helper: run python and abort on failure
function Run-Python([string]$args) {
    Write-Host "python $args" -ForegroundColor Yellow
    & python $args
    if ($LASTEXITCODE -ne 0) { throw "Command failed: python $args" }
}

# 1) Generate synthetic pages (small dataset)
Write-Host "`n1) Generating synthetic data..." -ForegroundColor Green
python equation_scribe\detector\synthetic_coco.py --out-images detector/data/images/synth --out-anns detector/data/annotations/instances_all.json --n-pages 20 --eqs-per-page 4 --dpi 150

# 2) Split COCO by paper -> instances_train.json / instances_val.json
Write-Host "`n2) Splitting COCO by paper..." -ForegroundColor Green
python equation_scribe\detector\split_coco_by_paper.py --coco detector/data/annotations/instances_all.json --out-dir detector/data/annotations --val-frac 0.2 --seed 0

# 3) Preprocess (optional; good for scan-like images)
Write-Host "`n3) Preprocessing pages (denoise, deskew, CLAHE, binarize)..." -ForegroundColor Green
python equation_scribe\detector\preprocess.py --input detector/data/images/synth --output detector/data/images/synth_pre --denoise --deskew --clahe --binarize

# 4) Tile the train set (creates tiles and tile-level COCO)
Write-Host "`n4) Tiling (train set)..." -ForegroundColor Green
python equation_scribe\detector\tiling.py --coco detector/data/annotations/instances_val.json --images-root detector/data/images/synth_pre --out-images detector/data/images/tiles_val --out-annotations detector/data/annotations/instances_tiles_val.json --tile-size 1024 --stride 512 --min-area-frac 0.25 --keep-empty-prob 0.05

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

# # 7) Inference on a sample page using trained weights (if produced)
# Write-Host "`n7) Inference on a sample page (if training produced weights)..." -ForegroundColor Green
# $bestWeights = "runs\detect\eq_detector_quick\weights\best.pt"
# if (Test-Path $bestWeights) {
#     python equation_scribe\detector\inference.py --model $bestWeights --image detector/data/images/synth_pre/page_0000.png --conf 0.25
# } else {
#     Write-Warning "Best weights not found at $bestWeights — skipping inference."
# }

# # 8) Make recognition pairs (crop images + gold LaTeX) using .meta.json produced by synthetic generator
# Write-Host "`n8) Create recognition pairs (crops -> latex)..." -ForegroundColor Green
# python equation_scribe\detector\make_pairs.py --coco detector/data/annotations/instances_all.json --out-images detector/data/recognition/images --out-jsonl detector/data/recognition/pairs.jsonl --page-images-root detector/data/images/synth_pre

# Write-Host "`nDemo finished." -ForegroundColor Cyan
# Write-Host "Inspect:" -ForegroundColor Cyan
# Write-Host "  detector/data/images/synth_pre    (preprocessed pages)"
# Write-Host "  detector/data/images/tiles_train (tiles)"
# Write-Host "  detector/data/annotations/*      (COCO files)"
# Write-Host "  runs/detect/eq_detector_quick    (YOLO run logs/weights if training ran)"
# Write-Host "  detector/data/recognition/       (recognition crops and pairs.jsonl)" -ForegroundColor Cyan

