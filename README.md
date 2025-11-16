# equation-scribe

Human-in-the-loop technical paper reader: PDF ingest + LaTeX validator.

## Project layout

- `equation_scribe/` – core library:
  - `pdf_ingest.py` – PDF loading, page images, layout extraction.
  - `validate.py` – LaTeX validation and normalization helpers.
- `tests/` – pytest test suite.
- `data/` – sample PDFs used in local testing.
- `notebooks/` – exploratory / spiral notebooks.
- `environment.yml` – optional Conda environment for reproducible setup.

---

## Development setup

### Option A: Using Conda (recommended)

```bash
# From the repo root
conda env create -f environment.yml
conda activate eqscribe

Then install package in editable mode
pip install -e .

### Option B: Using a virtualvenv + pip
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install --upgrade pip
pip install -e .[dev]  # if you later define an extra "dev" in pyproject

or just 

pip install -e .
pip install pytest black jupyter

### Running tests
pytest -q

to run only ingestion tests:
pytest tests/test_pdf_ingest.py -q

### Sample PDF for tests
By default, tests look for a sample under data/, e.g.:

data/Research_on_SAR_Imaging_Simulation_Based_on_Time-Domain_Shooting_and_Bouncing_Ray_Algorithm.pdf

You can override this by setting an environment variable:
export PDF_SAMPLE=/path/to/your/test.pdf
pytest tests/test_pdf_ingest.py -q

on Windows:
$env:PDF_SAMPLE = "C:\path\to\your\test.pdf"
pytest tests/test_pdf_ingest.py -q

