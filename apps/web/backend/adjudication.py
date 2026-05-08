import os
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime
from PIL import Image

logger = logging.getLogger("adjudicator")
logger.setLevel(logging.INFO)

class AdjudicationManager:
    """
    Manages Human-in-the-Loop data collection.
    Saves cropped images and verified LaTeX to a dataset for future training.
    """
    def __init__(self, data_root: str = "../data/adjudicated"):
        # Resolve relative to this file
        self.root = Path(__file__).resolve().parent.parent / "data" / "adjudicated"
        self.images_dir = self.root / "images"
        self.manifest_path = self.root / "dataset.jsonl"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def save_correction(self, image: Image.Image, latex: str, source_file: str, bbox: list = None):
        """Save a verified equation sample."""
        sample_id = uuid.uuid4().hex
        timestamp = datetime.now().isoformat()
        
        # Save Crop
        image_filename = f"{sample_id}.png"
        image_path = self.images_dir / image_filename
        image.save(image_path, format="PNG")
        
        entry = {
            "id": sample_id,
            "timestamp": timestamp,
            "source_file": source_file,
            "latex_gt": latex,
            "image_path": str(image_filename),
            "bbox": bbox,
            "is_figure": False,
            "verified": True
        }

        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
        logger.info(f"Adjudicated: {sample_id}")

    def flag_as_figure(self, image: Image.Image, source_file: str, bbox: list = None):
        """Save a false positive (figure/chart detection)."""
        sample_id = uuid.uuid4().hex
        image_filename = f"FIG_{sample_id}.png"
        image_path = self.images_dir / image_filename
        image.save(image_path, format="PNG")

        entry = {
            "id": sample_id,
            "timestamp": datetime.now().isoformat(),
            "source_file": source_file,
            "latex_gt": None,
            "image_path": str(image_filename),
            "bbox": bbox,
            "is_figure": True,
            "verified": True
        }

        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
        logger.info(f"Flagged Figure: {sample_id}")