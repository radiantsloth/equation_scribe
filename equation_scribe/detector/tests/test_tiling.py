import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
import json
from detector.tiling import generate_tiles_from_coco

def make_sample_image(img_path):
    img = Image.new("RGB", (800, 1200), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([100,200,200,300], outline="black", width=2)
    img.save(img_path)

def make_sample_coco(tmpdir, img_path):
    images = [{"id":1, "file_name": str(img_path), "width":800, "height":1200}]
    annotations = [{"id":1, "image_id":1, "category_id":1, "bbox":[100,200,100,100], "area":10000, "iscrowd":0}]
    coco = {"images": images, "annotations": annotations, "categories":[{"id":1,"name":"display"},{"id":2,"name":"inline"}]}
    ann_path = Path(tmpdir) / "instances.json"
    json.dump(coco, open(ann_path, "w"))
    return ann_path

def test_tiling(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "paperA_page_0001.png"
    make_sample_image(img_path)
    coco = make_sample_coco(tmp_path, img_path)
    out_images = tmp_path / "tiles"
    out_ann = tmp_path / "tiles_instances.json"
    out = generate_tiles_from_coco(coco, img_dir, out_images, out_ann, tile_size=512, stride=256, min_area_frac=0.1)
    assert out.exists()
    data = json.load(open(out))
    assert "images" in data and "annotations" in data
    # ensure at least one tile was produced
    assert len(data["images"]) > 0

