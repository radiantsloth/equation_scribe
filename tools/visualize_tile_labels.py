# tools/visualize_tile_labels.py
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import random

DATA_ROOT = Path("detector/data")
IM_DIR = DATA_ROOT / "images" / "tiles_train"
LABEL_ROOT = DATA_ROOT / "labels" / "tiles_train"

def load_label(lbl_path):
    lines = []
    with open(lbl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line=line.strip()
            if not line:
                continue
            parts=line.split()
            cls=int(parts[0])
            x=float(parts[1]); y=float(parts[2]); w=float(parts[3]); h=float(parts[4])
            lines.append((cls,x,y,w,h))
    return lines

# pick random sample images
candidates=list(IM_DIR.glob("*.png"))
random.shuffle(candidates)
sample=candidates[:8]

fig, axs = plt.subplots(2,4, figsize=(16,8))
axs=axs.flatten()
for ax, img_path in zip(axs, sample):
    img=Image.open(img_path).convert("RGB")
    ax.imshow(img)
    ax.set_title(img_path.name)
    lbl_path = LABEL_ROOT / (img_path.stem + ".txt")
    if lbl_path.exists():
        labs = load_label(lbl_path)
        W,H = img.size
        for cls,x,y,w,h in labs:
            # convert normalized center -> pixel box
            x0 = (x - w/2) * W
            y0 = (y - h/2) * H
            x1 = (x + w/2) * W
            y1 = (y + h/2) * H
            rect = plt.Rectangle((x0,y0), x1-x0, y1-y0, edgecolor='red', facecolor='none', lw=2)
            ax.add_patch(rect)
            ax.text(x0, y0-6, str(cls), color='red', fontsize=12, backgroundcolor='white')
    else:
        ax.text(0.5,0.5, "no label", transform=ax.transAxes, ha="center")
    ax.axis("off")
plt.tight_layout()
plt.show()
