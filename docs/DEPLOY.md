# Deployment & System Setup (Equation Scribe)

This document describes how to prepare a development machine (Windows / Linux)
to run the Equation Scribe detector pipeline: synthetic generation, training,
and inference. The important external tools are **pdflatex** (LaTeX), **poppler**
(pdftoppm), **Tesseract** (OCR), and Node.js (frontend). We also list Python
dependencies and recommended commands.

---

## 1. Create Python environment

We recommend using `conda`:

```bash
conda create -n eqscribe python=3.11 -y
conda activate eqscribe
pip install --upgrade pip
pip install -r requirements.txt

If you don't have requirements.txt, install the main packages:

```bash
pip install pillow numpy matplotlib pdf2image pytesseract ultralytics portalocker sympy pdfplumber PyMuPDF torch torchvision

Note: torch installation depends on your CUDA / CPU. See https://pytorch.org/get-started/locally/  for the correct command.

## 2. System tools
## Windows

### 1. MiKTeX (for pdflatex to render LaTeX formulas):

* Install from https://miktex.org/download

* After install, open MiKTeX Console and ensure packages are allowed to be installed on-the-fly or pre-install common math packages (amsmath, amssymb, bm, etc.)

### 2. Poppler (for pdftoppm):

* Download the pre-built binaries (e.g. poppler-utils) or via Chocolatey: choco install poppler -y

* Add the bin folder to PATH if not done by the installer.

### 3. Tesseract:

* Install from https://github.com/tesseract-ocr/tesseract (Windows installer), or choco install tesseract -y

* Add tesseract to PATH (the installer usually does this).

### 4. Node.js & npm (front-end):

* Install Node from https://nodejs.org/

* Verify with node --version and npm --version.

### 5. Other Notes:

* Ensure the installed executables are visible with where pdflatex, where pdftoppm, where tesseract.

# Linux (Ubuntu / Debian)
```bash
sudo apt update
sudo apt install -y texlive-latex-base texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra poppler-utils tesseract-ocr nodejs npm


Adjust package names depending on distro.

## 3. Configure Python utilities

* pdf2image requires pdftoppm (poppler) on PATH.

* pytesseract requires tesseract on PATH.

* Optionally, install pdfplumber, PyMuPDF (fitz), sympy, and other packages.

## 4. Ultralytics / YOLO

* We use Ultralytics yolo CLI for quick detector training. After installing:

* pip install ultralytics
* run training (example)
yolo detect train data=detector/detector.yaml model=yolov8s.pt epochs=5 imgsz=1024


* If yolo is not on PATH, run python -m ultralytics or python -m ultralytics.yolo.

* The quick demo script tools/run_demo.ps1 expects yolo available; otherwise, update the script to use python -m ultralytics.

## 5. Recommended repo-level setup

* Set up your conda env and install requirements.

* Run the system check:

* python tools/check_prereqs.py


* Run unit tests:

pytest -q


If you plan to render LaTeX formulas with pdflatex, make sure your MiKTeX or TeX Live configuration allows non-interactive runs (the scripts call pdflatex with -interaction=nonstopmode).

## 6. Useful commands

* Clear synthetic data and re-run generator:

* Windows Powershell:

Remove-Item -Recurse -Force detector\data\images\synth
Remove-Item -Recurse -Force detector\data\annotations\instances_all.json


* Linux:

rm -rf detector/data/images/synth
rm -f detector/data/annotations/instances_all.json


* Remove YOLO runs (example):

rm -rf runs/detect/eq_detector_quick*


* Run the demo script:

* Windows:

powershell -ExecutionPolicy Bypass -File .\tools\run_demo.ps1

## 7. Troubleshooting

* pdflatex not found: ensure MiKTeX/TeX Live is installed and pdflatex is on PATH.

* Matplotlib MathText errors: try installing latex via MiKTeX plus amsmath and bm. For some complex LaTeX constructs you must use pdflatex flow (the generator attempts matplotlib mathtext first and falls back to full LaTeX).

* Ultralytics dataset errors: ensure detector/data/annotations/instances_tiles_train.json and instances_tiles_val.json exist and paths in detector/detector.yaml are correct.

* GPU/torch errors: check torch installation and CUDA compatibility.

## 8. Notes for CI / headless servers

* If running on a headless Linux server, ensure pdflatex is installed (TeX Live) and poppler installed for pdftoppm.

* Configure ultralytics runs directory if you want artifact outputs to go somewhere specific; the yolo settings command or the settings JSON in your profile can be used.