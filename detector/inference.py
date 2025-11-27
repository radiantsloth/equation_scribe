# detector/inference.py
import argparse
from ultralytics import YOLO
from pathlib import Path
import json
from PIL import Image

def detect_image(model_path, image_path, conf_thresh=0.25, iou=0.5):
    model = YOLO(model_path)
    results = model.predict(source=str(image_path), conf=conf_thresh, iou=iou, max_det=300)
    r = results[0]
    boxes = []
    if hasattr(r, "boxes"):
        for box in r.boxes:
            xyxy = box.xyxy[0].numpy().tolist()  # [x1,y1,x2,y2]
            conf = float(box.conf[0])
            clsid = int(box.cls[0])
            boxes.append({"xyxy": xyxy, "conf": conf, "cls": clsid})
    return boxes

def px_boxes_to_pdf_coords(pdf_path, page_index, px_boxes):
    """Convert px box coordinates to PDF coordinates using equation_scribe.pdf_ingest functions."""
    try:
        from equation_scribe.pdf_ingest import load_pdf, pdf_to_px_transform
    except Exception as e:
        raise RuntimeError("Could not import equation_scribe.pdf_ingest. Run this inside the repo or adjust PYTHONPATH.") from e

    doc = load_pdf(Path(pdf_path))
    pdf2px, px2pdf = pdf_to_px_transform(doc, page_index)
    converted = []
    for b in px_boxes:
        x1, y1, x2, y2 = b["xyxy"]
        # note: px2pdf expects x,y pairs
        x0_pdf, y0_pdf = px2pdf(x1, y1)
        x1_pdf, y1_pdf = px2pdf(x2, y2)
        converted.append({
            "bbox_pdf": [x0_pdf, y0_pdf, x1_pdf, y1_pdf],
            "conf": b["conf"],
            "cls": b["cls"]
        })
    return converted

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/detect/eq_detector_yolov8s/weights/best.pt")
    parser.add_argument("--image", help="Input page image path", required=True)
    parser.add_argument("--pdf", help="Optional PDF path (for px->pdf conversion)")
    parser.add_argument("--page", type=int, default=0, help="Page index in PDF")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    boxes = detect_image(args.model, args.image, conf_thresh=args.conf)
    if args.pdf:
        converted = px_boxes_to_pdf_coords(args.pdf, args.page, boxes)
        print(json.dumps({"detections": converted}, indent=2))
    else:
        print(json.dumps({"detections": boxes}, indent=2))
