from pathlib import Path
import os
import hashlib
from math import ceil, floor
from typing import Callable, List, Dict, Any, Optional, Tuple
import sys

from fastapi import Body, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from .services.pdf import page_count, render_page_png, page_meta
from .services.validate import validate_latex
from .adjudication import AdjudicationManager

try:
    from equation_scribe_core.config import get_runtime_settings
    from equation_scribe_core.io import append_equation, delete_equation, load_index, read_equations, update_equation
    from equation_scribe_core.models import EquationRecord
except ModuleNotFoundError:
    core_src = Path(__file__).resolve().parents[3] / "packages" / "core" / "src"
    if str(core_src) not in sys.path:
        sys.path.insert(0, str(core_src))
    from equation_scribe_core.config import get_runtime_settings
    from equation_scribe_core.io import append_equation, delete_equation, load_index, read_equations, update_equation
    from equation_scribe_core.models import EquationRecord

from equation_scribe.recognition.inference import image_to_latex
from equation_scribe.pdf_ingest import load_pdf, page_image, page_layout, page_size_points, pdf_to_px_transform
from equation_scribe.detector.inference import detect_image 
from equation_scribe.detect import find_equation_candidates
import uuid 

APP_ROOT = Path(__file__).resolve().parents[1]
YOLO_MODEL_PATH = Path(__file__).parent / "models" / "best.pt" 
RUNTIME_SETTINGS = get_runtime_settings()
PROFILES_ROOT = RUNTIME_SETTINGS.profiles_root
PAPERS_ROOT = RUNTIME_SETTINGS.papers_root

app = FastAPI(title="Equation Scribe API")
adjudicator = AdjudicationManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LatexPayload(BaseModel):
    latex: str

class AutoDetectRequest(BaseModel):
    page_index: int

class UploadResponse(BaseModel):
    paper_id: str

class RescanRequest(BaseModel):
    page_index: int
    bbox: List[float]

def slugify(name: str) -> str:
    stem = Path(name).stem
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return safe or "paper"

def pdf_bbox_to_crop_box(
    pdf2px: Callable[[float, float], Tuple[int, int]],
    bbox_pdf: List[float] | Tuple[float, float, float, float],
    image_size: Tuple[int, int],
    pad: int = 5,
) -> Optional[Tuple[int, int, int, int]]:
    """Convert a PDF-space bbox into a clipped PIL crop box.

    Returns None when the box is fully outside the rendered image or collapses
    to an empty rectangle after clipping.
    """
    try:
        if len(bbox_pdf) != 4:
            return None
        x0, y0, x1, y1 = (float(v) for v in bbox_pdf)
    except (TypeError, ValueError):
        return None

    px0, py0 = pdf2px(x0, y0)
    px1, py1 = pdf2px(x1, y1)

    left = floor(min(px0, px1) - pad)
    top = floor(min(py0, py1) - pad)
    right = ceil(max(px0, px1) + pad)
    bottom = ceil(max(py0, py1) + pad)

    w, h = image_size
    left = max(0, min(w, left))
    top = max(0, min(h, top))
    right = max(0, min(w, right))
    bottom = max(0, min(h, bottom))

    if right <= left or bottom <= top:
        return None

    return (left, top, right, bottom)

def pdf_path_for(paper_id: str) -> Path:
    p = PAPERS_ROOT / f"{paper_id}.pdf"
    if p.exists(): return p
    raise HTTPException(404, f"PDF for paper_id '{paper_id}' not found")

# --- HELPER: IoU for Deduplication ---
def calculate_iou(boxA, boxB):
    # box: [x0, y0, x1, y1]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0: return 0
    return interArea / unionArea

# --- ENDPOINTS ---

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    paper_id = slugify(file.filename)
    dest = PAPERS_ROOT / f"{paper_id}.pdf"
    contents = await file.read()
    dest.write_bytes(contents)
    return UploadResponse(paper_id=paper_id)

@app.get("/papers/{paper_id}/pages")
def get_pages(paper_id: str):
    return {"pages": page_count(pdf_path_for(paper_id))}

@app.get("/papers/index")
def get_profiles_index_endpoint():
    return load_index(PROFILES_ROOT).model_dump(mode="json")

@app.get("/papers/{paper_id}/page/{idx}/image")
def get_page_image_endpoint(paper_id: str, idx: int, zoom: float = 1.5):
    p = pdf_path_for(paper_id)
    data = render_page_png(p, idx, zoom=zoom)
    return Response(content=data, media_type="image/png")

@app.get("/papers/{paper_id}/page/{idx}/meta")
def get_page_meta_endpoint(paper_id: str, idx: int, zoom: float = 1.5):
    return page_meta(pdf_path_for(paper_id), idx, zoom=zoom)

@app.get("/papers/{paper_id}/equations")
def list_equations(paper_id: str) -> Dict[str, Any]:
    items = read_equations(PROFILES_ROOT, paper_id)
    return {"items": items}

@app.post("/validate")
def validate(payload: LatexPayload):
    return validate_latex(payload.latex or "")

@app.post("/papers/{paper_id}/equations")
def save_equation(paper_id: str, rec: EquationRecord):
    if not rec.boxes:
        raise HTTPException(400, "At least one box is required")
    append_equation(PROFILES_ROOT, rec)
    try:
        _adjudicate_record(paper_id, rec)
    except Exception as e:
        print(f"Adjudication error: {e}")
    return {"ok": True}

@app.put("/papers/{paper_id}/equations/{eq_uid}")
def update_equation_endpoint(paper_id: str, eq_uid: str, rec: EquationRecord):
    update_equation(PROFILES_ROOT, paper_id, eq_uid, rec.model_dump())
    try:
        _adjudicate_record(paper_id, rec)
    except Exception as e:
        print(f"Adjudication error: {e}")
    return {"ok": True}

def _adjudicate_record(paper_id: str, rec: EquationRecord):
    if not rec.boxes: return
    pdf_path = pdf_path_for(paper_id)
    doc = load_pdf(pdf_path)
    page_ix = rec.boxes[0].page
    bbox_pdf = rec.boxes[0].bbox_pdf
    
    full_page_img = page_image(doc, page_ix, dpi=150)
    pdf2px, _ = pdf_to_px_transform(doc, page_ix, dpi=150)

    crop_box = pdf_bbox_to_crop_box(pdf2px, bbox_pdf, full_page_img.size)
    if crop_box is None:
        print(f"Skipping adjudication for invalid bbox on {paper_id} page {page_ix}: {bbox_pdf}")
        return

    crop_img = full_page_img.crop(crop_box)
    
    adjudicator.save_correction(
        image=crop_img,
        latex=rec.latex,
        source_file=f"{paper_id}.pdf",
        bbox=bbox_pdf
    )

@app.delete("/papers/{paper_id}/equations/{eq_uid}")
def delete_equation_endpoint(paper_id: str, eq_uid: str):
    ok = delete_equation(PROFILES_ROOT, paper_id, eq_uid)
    if not ok:
        raise HTTPException(404, "Equation not found")
    return {"ok": True}

@app.post("/papers/{paper_id}/rescan_box")
def rescan_box(paper_id: str, payload: RescanRequest):
    try:
        pdf_path = pdf_path_for(paper_id)
    except NameError:
        pdf_path = PAPERS_ROOT / f"{paper_id}.pdf"

    doc = load_pdf(pdf_path)
    full_page_img = page_image(doc, payload.page_index, dpi=150)
    pdf2px, _ = pdf_to_px_transform(doc, payload.page_index, dpi=150)

    crop_box = pdf_bbox_to_crop_box(pdf2px, payload.bbox, full_page_img.size)
    if crop_box is None:
        raise HTTPException(400, "Bounding box is outside the rendered page")

    crop_img = full_page_img.crop(crop_box)
    latex_result = image_to_latex(crop_img)
    return {"latex": latex_result}

@app.post("/papers/{paper_id}/autodetect_all")
def autodetect_all(paper_id: str):
    """
    Run detection on ALL pages.
    Includes DEDUPLICATION to prevent overlapping boxes on the same equation.
    """
    pdf_path = pdf_path_for(paper_id)
    doc = load_pdf(pdf_path)
    detected_count = 0
    
    # 1. Load EXISTING equations to check for duplicates
    existing_items = read_equations(PROFILES_ROOT, paper_id)
    
    # Map: page_index -> list of [x0, y0, x1, y1]
    existing_boxes = {}
    total_equations_before = 0
    
    if existing_items:
        total_equations_before = len(existing_items)
        for eq in existing_items:
            # FIX: Use dict access (.get) instead of attribute access
            boxes = eq.get("boxes", [])
            for b in boxes:
                # FIX: 'b' is a dict
                page = b.get("page")
                bbox = b.get("bbox_pdf")
                if page is not None and bbox:
                    existing_boxes.setdefault(page, []).append(bbox)

    for page_ix in range(doc.num_pages):
        candidates = []
        
        # YOLO Detection
        if YOLO_MODEL_PATH.exists():
            img_path = PAPERS_ROOT / f"temp_{paper_id}_{page_ix}.png"
            page_img = page_image(doc, page_ix, dpi=150) 
            page_img.save(img_path)
            
            try:
                yolo_boxes = detect_image(str(YOLO_MODEL_PATH), str(img_path), conf_thresh=0.25)
                pdf2px, px2pdf = pdf_to_px_transform(doc, page_ix, dpi=150)
                
                for box in yolo_boxes:
                    px_coords = box['xyxy']
                    x0, y0 = px2pdf(px_coords[0], px_coords[1])
                    x1, y1 = px2pdf(px_coords[2], px_coords[3])
                    candidates.append({
                        "bbox_pdf": (min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1)),
                        "score": box['conf']
                    })
            finally:
                if img_path.exists(): img_path.unlink()

        # Heuristic Fallback
        if not candidates:
             from equation_scribe.detect import find_equation_candidates
             spans = page_layout(doc, page_ix)
             width, _ = page_size_points(doc, page_ix)
             candidates = find_equation_candidates(spans, width)

        # Recognition & Save with DEDUPLICATION
        if candidates:
            full_page_img = page_image(doc, page_ix, dpi=150)
            pdf2px, _ = pdf_to_px_transform(doc, page_ix, dpi=150)
            
            page_existing = existing_boxes.get(page_ix, [])

            for cand in candidates:
                cand_box = cand["bbox_pdf"]
                
                # --- CHECK FOR DUPLICATES ---
                is_duplicate = False
                for ex_box in page_existing:
                    if calculate_iou(cand_box, ex_box) > 0.5: # 50% overlap threshold
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue
                # -----------------------------

                crop_box = pdf_bbox_to_crop_box(pdf2px, cand_box, full_page_img.size)
                if crop_box is None:
                    print(f"Skipping invalid candidate bbox on {paper_id} page {page_ix}: {cand_box}")
                    continue

                crop_img = full_page_img.crop(crop_box)
                latex = image_to_latex(crop_img)
                
                rec = EquationRecord(
                    eq_uid=str(uuid.uuid4())[:16],
                    paper_id=paper_id,
                    latex=latex,
                    notes=f"Auto (YOLO {cand['score']:.2f})",
                    boxes=[{"page": page_ix, "bbox_pdf": cand_box}]
                )
                append_equation(PROFILES_ROOT, rec)
                
                # Add to local exclusion list so we don't add overlapping boxes within the same run
                page_existing.append(cand_box)
                detected_count += 1

    total_after = total_equations_before + detected_count
    return {
        "message": f"Scanned {doc.num_pages} pages", 
        "equations_found": detected_count,
        "total_equations": total_after
    }
