# equation_scribe/ui_gradio.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Dict, Any
import sys
import json
import os

import gradio as gr
from PIL import Image, ImageDraw

from .pdf_ingest import (
    load_pdf,
    page_image,
    page_layout,
    pdf_to_px_transform,
    page_size_points,
)
from .detect import find_equation_candidates
from .validate import validate_latex
from .store import save_equation, canonical_hash


# -------------------- drawing & geometry helpers --------------------

def _draw_boxes(
    img: Image.Image,
    boxes_px: List[Tuple[int, int, int, int]],
    color: str = "red",
    width: int = 3,
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for (x0, y0, x1, y1) in boxes_px:
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    return out


def _pdfbbox_to_px(doc, page_index: int, bbox_pdf):
    pdf2px, _ = pdf_to_px_transform(doc, page_index)
    x0p, y0p = pdf2px(bbox_pdf[0], bbox_pdf[1])
    x1p, y1p = pdf2px(bbox_pdf[2], bbox_pdf[3])
    x_left, x_right = sorted((x0p, x1p))
    y_top, y_bottom = sorted((y0p, y1p))
    return x_left, y_top, x_right, y_bottom


def _pxbbox_to_pdf(doc, page_index: int, bbox_px):
    _, px2pdf = pdf_to_px_transform(doc, page_index)
    x0, y0, x1, y1 = bbox_px
    x0p, y0p = px2pdf(x0, y0)
    x1p, y1p = px2pdf(x1, y1)
    x_left, x_right = (x0p, x1p) if x0p <= x1p else (x1p, x0p)
    y_bottom, y_top = (y0p, y1p) if y0p <= y1p else (y1p, y0p)
    return (x_left, y_bottom, x_right, y_top)


def _point_in_px_box(px: Tuple[int, int], box: Tuple[int, int, int, int]) -> bool:
    x, y = px
    x0, y0, x1, y1 = box
    return (x0 <= x <= x1) and (y0 <= y <= y1)


def _clamp_pdf_bbox_to_page(doc, page_index: int, bbox_pdf):
    w_pt, h_pt = page_size_points(doc, page_index)
    x0, y0, x1, y1 = bbox_pdf
    x0 = max(0.0, min(x0, w_pt))
    x1 = max(0.0, min(x1, w_pt))
    y0 = max(0.0, min(y0, h_pt))
    y1 = max(0.0, min(y1, h_pt))
    x_left, x_right = (x0, x1) if x0 <= x1 else (x1, x0)
    y_bottom, y_top = (y0, y1) if y0 <= y1 else (y1, y0)
    return (x_left, y_bottom, x_right, y_top)


# -------------------- load existing saved boxes (read-only overlay) --------------------

def _load_existing_boxes(store_root: Path, paper_id: str) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """
    Return: page_index -> list of bbox_pdf (tuples)
    """
    d = store_root / paper_id
    path = d / "equations.jsonl"
    by_page: Dict[int, List[Tuple[float, float, float, float]]] = {}
    if not path.exists():
        return by_page
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Spiral 1.1 schema: rec["boxes"] is a list of {"page": int, "bbox_pdf": [x0,y0,x1,y1]}
                boxes = rec.get("boxes") or []
                for b in boxes:
                    pg = int(b.get("page", 0))
                    bb = b.get("bbox_pdf")
                    if not bb or len(bb) != 4:
                        continue
                    tup = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
                    by_page.setdefault(pg, []).append(tup)
    except Exception as e:
        print(f"[ui] warning: failed reading existing boxes: {e}", file=sys.stderr)
    return by_page


# -------------------- UI app --------------------

def launch_app(pdf_path: str | Path, store_root: str | Path, paper_id: str):
    """
    Spiral 1.2 UI:
    - Page navigation
    - Candidate detection (assist)
    - Click-to-draw (two clicks) + click-to-select box
    - Resize via "Start Resize" (two clicks) OR numeric/nudge controls
    - Show existing (saved) boxes in gray (read-only)
    - Current equation boxes in red (editable), ≥1 required to save
    """
    pdf_path = Path(pdf_path)
    store_root = Path(store_root)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"[ui] launch_app: pdf={pdf_path}", file=sys.stderr)
    doc = load_pdf(pdf_path, dpi=200)
    print(f"[ui] loaded pdf: pages={doc.num_pages}", file=sys.stderr)

    # Persistent-but-local UI state
    state: Dict[str, Any] = {
        "page": 0,
        "cands_by_page": {},              # page_index -> list of candidates
        "boxes": [],                      # current equation boxes: list of {"page": int, "bbox_pdf": (x0,y0,x1,y1)}
        "selected_idx": None,             # index into state["boxes"] for current focus
        "mode": "idle",                   # "idle" | "draw" | "resize"
        "pending_point_px": None,         # first click (x,y) in pixels for draw/resize
        "existing_by_page": _load_existing_boxes(store_root, paper_id),  # read-only overlays
    }

    def _get_candidates(page_idx: int) -> List[Dict[str, Any]]:
        if page_idx in state["cands_by_page"]:
            return state["cands_by_page"][page_idx]
        w_pt, _ = page_size_points(doc, page_idx)
        spans = page_layout(doc, page_idx)
        cands = find_equation_candidates(spans, page_width=w_pt)
        state["cands_by_page"][page_idx] = cands
        print(f"[ui] page {page_idx}: {len(cands)} candidates", file=sys.stderr)
        return cands

    def _render_page_all(page_idx: int) -> Image.Image:
        """
        Draw:
          - existing saved boxes (gray)
          - current equation boxes (red, thicker)
          - if a box is selected, draw it in red (we draw all red anyway)
        """
        base = page_image(doc, page_idx)
        # draw existing (read-only) in gray
        gray_boxes_px = []
        for bb in state["existing_by_page"].get(page_idx, []):
            gray_boxes_px.append(_pdfbbox_to_px(doc, page_idx, bb))
        if gray_boxes_px:
            base = _draw_boxes(base, gray_boxes_px, color="gray", width=2)
        # draw current equation boxes (editable) in red
        red_boxes_px = []
        for k, b in enumerate(state["boxes"]):
            if b["page"] == page_idx:
                red_boxes_px.append(_pdfbbox_to_px(doc, page_idx, b["bbox_pdf"]))
        if red_boxes_px:
            base = _draw_boxes(base, red_boxes_px, color="red", width=3)
        return base

    # --------------- UI callbacks ---------------

    def on_page_change(page_idx: int):
        page_idx = int(page_idx)
        state["page"] = page_idx
        img = _render_page_all(page_idx)
        cands = _get_candidates(page_idx)
        cand_choices = [f"cand {i+1} (score={c['score']:.2f})" for i, c in enumerate(cands)]
        status = f"Page {page_idx+1}/{doc.num_pages}: {len(cands)} candidate(s). Mode={state['mode']}"
        return img, status, gr.update(choices=cand_choices, value=None)

    def on_pick_candidate(cand_label: str):
        if not cand_label:
            return "", "", "", ""
        cands = _get_candidates(state["page"])
        try:
            idx = int(cand_label.split()[1]) - 1
        except Exception:
            return "", "", "", ""
        if idx < 0 or idx >= len(cands):
            return "", "", "", ""
        x0, y0, x1, y1 = cands[idx]["bbox_pdf"]
        return str(x0), str(y0), str(x1), str(y1)

    def on_add_box_from_fields(x0: float, y0: float, x1: float, y1: float):
        page_idx = state["page"]
        try:
            x0f, y0f, x1f, y1f = float(x0), float(y0), float(x1), float(y1)
        except Exception:
            return "❌ Invalid bbox values.", _render_page_all(page_idx)
        bbox = _clamp_pdf_bbox_to_page(doc, page_idx, (x0f, y0f, x1f, y1f))
        state["boxes"].append({"page": page_idx, "bbox_pdf": bbox})
        state["selected_idx"] = len(state["boxes"]) - 1
        return f"✅ Added box on page {page_idx+1}. Total boxes: {len(state['boxes'])}.", _render_page_all(page_idx)

    def on_remove_selected():
        if state["selected_idx"] is None:
            return "No selected box.", _render_page_all(state["page"])
        idx = state["selected_idx"]
        if 0 <= idx < len(state["boxes"]):
            removed = state["boxes"].pop(idx)
            state["selected_idx"] = None
            msg = f"Removed box from page {removed['page']+1}. Remaining: {len(state['boxes'])}."
        else:
            msg = "No selected box."
        return msg, _render_page_all(state["page"])

    def on_clear_current_eq_boxes():
        state["boxes"].clear()
        state["selected_idx"] = None
        return "Cleared all boxes for current equation.", _render_page_all(state["page"])

    def on_validate_render(latex: str):
        res = validate_latex(latex or "")
        ok = "✅ OK" if res.ok else "❌"
        err = "; ".join(res.errors) if res.errors else ""
        rendered = f"$$ {latex} $$" if (latex or "").strip() else ""
        return f"{ok} {err}", rendered

    def on_save(latex: str, notes: str):
        latex = (latex or "").strip()
        if len(state["boxes"]) == 0:
            return "❌ Please add at least one box before saving.", _render_page_all(state["page"])
        rec = {
            "eq_uid": canonical_hash(latex) if latex else canonical_hash(
                json.dumps(state["boxes"], sort_keys=True)
            ),
            "paper_id": paper_id,
            "latex": latex,
            "notes": notes or "",
            "boxes": state["boxes"][:],
        }
        try:
            save_equation(store_root, paper_id, rec)
            # refresh existing overlays cache
            state["existing_by_page"] = _load_existing_boxes(store_root, paper_id)
        except Exception as e:
            return f"❌ Save failed: {e}", _render_page_all(state["page"])
        state["boxes"].clear()
        state["selected_idx"] = None
        return f"✅ Saved equation with {len(rec['boxes'])} box(es).", _render_page_all(state["page"])

    # ----- Click-to-draw / select / resize -----

    def set_mode_draw():
        state["mode"] = "draw"
        state["pending_point_px"] = None
        return f"Mode set to DRAW. Click two points on the image.", _render_page_all(state["page"])

    def set_mode_resize():
        if state["selected_idx"] is None:
            return "Select a box first (click inside it), then Start Resize.", _render_page_all(state["page"])
        state["mode"] = "resize"
        state["pending_point_px"] = None
        return f"Mode set to RESIZE. Click two points for new corners.", _render_page_all(state["page"])

    def cancel_mode():
        state["mode"] = "idle"
        state["pending_point_px"] = None
        return f"Mode set to IDLE.", _render_page_all(state["page"])

    def on_image_select(evt: gr.SelectData):
        """
        Image click handler.
        - idle: select a box if click falls inside one (focus it)
        - draw:  first click => start; second click => create box
        - resize: first click => start; second click => replace selected box
        """
        page_idx = state["page"]
        x_px = int(evt.index[0]) if isinstance(evt.index, (list, tuple)) else getattr(evt, "x", None)
        y_px = int(evt.index[1]) if isinstance(evt.index, (list, tuple)) else getattr(evt, "y", None)
        # Gradio SelectData can vary; try common forms:
        if x_px is None or y_px is None:
            # Fall back to evt.x/evt.y if available
            try:
                x_px = int(evt.x); y_px = int(evt.y)
            except Exception:
                return _render_page_all(page_idx), f"Click ignored (couldn't parse coordinates). Mode={state['mode']}"

        mode = state["mode"]

        # Always try to select a box if clicked inside (for focus),
        # unless we're in the middle of the 2-click sequence.
        if mode == "idle":
            # Check current equation boxes first
            hit_idx = None
            for i, b in enumerate(state["boxes"]):
                if b["page"] != page_idx: 
                    continue
                box_px = _pdfbbox_to_px(doc, page_idx, b["bbox_pdf"])
                if _point_in_px_box((x_px, y_px), box_px):
                    hit_idx = i
                    break
            state["selected_idx"] = hit_idx
            focus_msg = f"Selected box #{hit_idx}" if hit_idx is not None else "No box selected."
            return _render_page_all(page_idx), f"{focus_msg} Mode=IDLE."

        # DRAW: two clicks make a new box
        if mode == "draw":
            if state["pending_point_px"] is None:
                state["pending_point_px"] = (x_px, y_px)
                return _render_page_all(page_idx), "DRAW: first point set. Click second corner."
            else:
                x0, y0 = state["pending_point_px"]
                x1, y1 = x_px, y_px
                # convert to PDF bbox and clamp
                bbox_pdf = _pxbbox_to_pdf(doc, page_idx, (x0, y0, x1, y1))
                bbox_pdf = _clamp_pdf_bbox_to_page(doc, page_idx, bbox_pdf)
                state["boxes"].append({"page": page_idx, "bbox_pdf": bbox_pdf})
                state["selected_idx"] = len(state["boxes"]) - 1
                state["pending_point_px"] = None
                state["mode"] = "idle"
                return _render_page_all(page_idx), f"✅ Added box. Mode=IDLE. Total boxes: {len(state['boxes'])}."

        # RESIZE: replace selected box with new two-click box
        if mode == "resize":
            if state["selected_idx"] is None:
                return _render_page_all(page_idx), "No box selected. Click inside a box first."
            if state["pending_point_px"] is None:
                state["pending_point_px"] = (x_px, y_px)
                return _render_page_all(page_idx), "RESIZE: first point set. Click second corner."
            else:
                x0, y0 = state["pending_point_px"]
                x1, y1 = x_px, y_px
                bbox_pdf = _pxbbox_to_pdf(doc, page_idx, (x0, y0, x1, y1))
                bbox_pdf = _clamp_pdf_bbox_to_page(doc, page_idx, bbox_pdf)
                idx = state["selected_idx"]
                if 0 <= idx < len(state["boxes"]):
                    state["boxes"][idx] = {"page": page_idx, "bbox_pdf": bbox_pdf}
                    state["pending_point_px"] = None
                    state["mode"] = "idle"
                    return _render_page_all(page_idx), f"✅ Resized box #{idx}. Mode=IDLE."
                else:
                    state["pending_point_px"] = None
                    state["mode"] = "idle"
                    return _render_page_all(page_idx), "Resize canceled (invalid index). Mode=IDLE."

        # Fallback
        return _render_page_all(page_idx), f"Click handled. Mode={state['mode']}."

    # ----- update numeric fields from selection, and apply numeric edits -----

    def refresh_bbox_fields_from_selection():
        idx = state["selected_idx"]
        if idx is None or idx < 0 or idx >= len(state["boxes"]):
            return "", "", "", ""
        b = state["boxes"][idx]
        x0, y0, x1, y1 = b["bbox_pdf"]
        return str(x0), str(y0), str(x1), str(y1)

    def apply_bbox_fields(x0: float, y0: float, x1: float, y1: float):
        idx = state["selected_idx"]
        if idx is None or not (0 <= idx < len(state["boxes"])):
            return "No selected box.", _render_page_all(state["page"])
        page_idx = state["page"]
        try:
            x0f, y0f, x1f, y1f = float(x0), float(y0), float(x1), float(y1)
        except Exception:
            return "❌ Invalid bbox values.", _render_page_all(page_idx)
        bbox = _clamp_pdf_bbox_to_page(doc, page_idx, (x0f, y0f, x1f, y1f))
        state["boxes"][idx]["bbox_pdf"] = bbox
        return f"✅ Updated box #{idx}.", _render_page_all(page_idx)

    def nudge_selected(dx: float = 0, dy: float = 0, grow: float = 0):
        idx = state["selected_idx"]
        if idx is None or not (0 <= idx < len(state["boxes"])):
            return "No selected box.", _render_page_all(state["page"])
        page_idx = state["page"]
        x0, y0, x1, y1 = state["boxes"][idx]["bbox_pdf"]
        # translate
        x0 += dx; x1 += dx
        y0 += dy; y1 += dy
        # grow/shrink (expand equally in all directions)
        x0 -= grow; y0 -= grow
        x1 += grow; y1 += grow
        bbox = _clamp_pdf_bbox_to_page(doc, page_idx, (x0, y0, x1, y1))
        state["boxes"][idx]["bbox_pdf"] = bbox
        return f"Moved/resized box #{idx}.", _render_page_all(page_idx)

    # --------------- Build Gradio UI ---------------

    with gr.Blocks(title="Equation Scribe – Spiral 1.2") as demo:
        with gr.Row():
            with gr.Column(scale=3):
                page_slider = gr.Slider(minimum=0, maximum=max(0, doc.num_pages - 1), step=1,
                                        value=0, label="Page", interactive=True)
                img = gr.Image(label="Page (gray = saved, red = current equation)",
                               interactive=True, type="pil")
            with gr.Column(scale=2):
                status = gr.Textbox(label="Status", interactive=False)
                cand_dd = gr.Dropdown(label="Detected candidates (assist)", choices=[], value=None)
                with gr.Row():
                    b_mode_draw = gr.Button("Start Draw")
                    b_mode_resize = gr.Button("Start Resize")
                    b_mode_cancel = gr.Button("Cancel Mode")
                with gr.Row():
                    x0 = gr.Textbox(label="x0 (PDF pts)")
                    y0 = gr.Textbox(label="y0 (PDF pts)")
                    x1 = gr.Textbox(label="x1 (PDF pts)")
                    y1 = gr.Textbox(label="y1 (PDF pts)")
                with gr.Row():
                    b_add_fields = gr.Button("Add Box (from fields)")
                    b_apply_fields = gr.Button("Apply to Selected")
                    b_remove_sel = gr.Button("Remove Selected")
                    b_clear_current = gr.Button("Clear Current Equation Boxes")

                with gr.Row():
                    b_nudge_left = gr.Button("◀︎")
                    b_nudge_right = gr.Button("▶︎")
                    b_nudge_up = gr.Button("▲")
                    b_nudge_down = gr.Button("▼")
                    b_grow = gr.Button("+")
                    b_shrink = gr.Button("−")

                notes = gr.Textbox(label="Notes / discussion", lines=2)

        with gr.Row():
            latex = gr.Textbox(label="LaTeX (edit before saving)", lines=3)
            rendered = gr.Markdown(label="Rendered")

        with gr.Row():
            b_validate = gr.Button("Check & Render")
            b_save = gr.Button("Approve & Save")

        # Wiring
        page_slider.change(on_page_change, inputs=[page_slider], outputs=[img, status, cand_dd])
        cand_dd.change(on_pick_candidate, inputs=[cand_dd], outputs=[x0, y0, x1, y1])

        b_mode_draw.click(set_mode_draw, inputs=None, outputs=[status, img])
        b_mode_resize.click(set_mode_resize, inputs=None, outputs=[status, img])
        b_mode_cancel.click(cancel_mode, inputs=None, outputs=[status, img])

        # Image click handler
        img.select(on_image_select, outputs=[img, status])

        # Fields / box mgmt
        b_add_fields.click(on_add_box_from_fields, inputs=[x0, y0, x1, y1], outputs=[status, img])
        b_apply_fields.click(apply_bbox_fields, inputs=[x0, y0, x1, y1], outputs=[status, img])
        b_remove_sel.click(on_remove_selected, inputs=None, outputs=[status, img])
        b_clear_current.click(on_clear_current_eq_boxes, inputs=None, outputs=[status, img])

        # Nudge / grow
        b_nudge_left.click(lambda: nudge_selected(dx=-5, dy=0, grow=0), outputs=[status, img])
        b_nudge_right.click(lambda: nudge_selected(dx=+5, dy=0, grow=0), outputs=[status, img])
        b_nudge_up.click(lambda: nudge_selected(dx=0, dy=-5, grow=0), outputs=[status, img])
        b_nudge_down.click(lambda: nudge_selected(dx=0, dy=+5, grow=0), outputs=[status, img])
        b_grow.click(lambda: nudge_selected(dx=0, dy=0, grow=3), outputs=[status, img])
        b_shrink.click(lambda: nudge_selected(dx=0, dy=0, grow=-3), outputs=[status, img])

        b_validate.click(on_validate_render, inputs=[latex], outputs=[status, rendered])
        b_save.click(on_save, inputs=[latex, notes], outputs=[status, img])

        # Initial load
        demo.load(fn=lambda: on_page_change(0), inputs=None, outputs=[img, status, cand_dd])

    return demo


# ---------------- CLI entry point ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Equation Scribe Spiral 1.2 UI")
    parser.add_argument("--pdf", required=True, help="Path to PDF")
    parser.add_argument("--store", default="paper_profiles", help="Directory to store paper profiles")
    parser.add_argument("--paper-id", default=None, help="ID for this paper (default: PDF filename stem)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")), help="Server port")
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"), help="Server host")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    paper_id = args.paper_id or pdf_path.stem

    try:
        app = launch_app(pdf_path, args.store, paper_id)
    except Exception as e:
        print(f"[ui] Fatal: {e}", file=sys.stderr)
        sys.exit(1)

    app.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
        show_error=True,
    )
