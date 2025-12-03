from PIL import Image, ImageDraw
import json
from pathlib import Path
from ultralytics import YOLO

def draw_boxes(img_path, preds, gt_boxes, out_path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    W,H = img.size
    # preds: list of [x1,y1,x2,y2, conf, cls]
    for x1,y1,x2,y2,conf,clsid in preds:
        draw.rectangle([x1,y1,x2,y2], outline="lime", width=2)
        draw.text((x1, y1-10), f"{conf:.2f}", fill="lime")
    # gt_boxes: list of [x,y,w,h] (COCO) or [x0,y0,x1,y1] adjust accordingly
    for b in gt_boxes:
        if len(b)==4: # assume COCO x,y,w,h
            x0,y0,w,h = b
            draw.rectangle([x0,y0,x0+w,y0+h], outline="red", width=2)
        else:
            x0,y0,x1,y1 = b
            draw.rectangle([x0,y0,x1,y1], outline="red", width=2)
    img.save(out_path)
    print("Wrote", out_path)

# Example usage
model = YOLO("runs/detect/eq_detector_quick/weights/best.pt")
img = "detector/data/images/synth_pre/paper000_page_0000.png"
r = model.predict(source=img, conf=0.2, imgsz=1024)[0]
preds = []
for box in r.boxes:
    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().tolist()
    conf = float(box.conf[0].cpu().item())
    clsid = int(box.cls[0].cpu().item())
    preds.append((x1,y1,x2,y2,conf,clsid))

# Load GT for this image (example: find corresponding entry in your COCO/annotations)
coco = json.load(open("detector/data/annotations/instances_all.json"))
# find image entry by file_name
img_entry = next(im for im in coco["images"] if im["file_name"].endswith("paper000_page_0000.png"))
img_id = img_entry["id"]
gt = [a["bbox"] for a in coco["annotations"] if a["image_id"]==img_id]  # COCO x,y,w,h

draw_boxes(img, preds, gt, "tmp_preds_vs_gt.png")
