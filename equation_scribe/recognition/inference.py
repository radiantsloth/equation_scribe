# equation_scribe/recognition/inference.py
import warnings

# --- Add these lines to suppress the noise ---
# 1. Block the "New version available" spam from albumentations
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

# 2. Block the internal Pydantic serializer warnings from within the model libraries
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")
# ---------------------------------------------
from PIL import Image
import torch
from pix2tex.cli import LatexOCR

_MODEL = None

def get_model():
    """Singleton loader for the model (it's heavy)"""
    global _MODEL
    if _MODEL is None:
        # This automatically downloads weights (~200MB) on first run
        _MODEL = LatexOCR()
    return _MODEL

def image_to_latex(image: Image.Image) -> str:
    """
    Takes a PIL Image (equation crop) and returns LaTeX string.
    """
    model = get_model()
    try:
        # pix2tex expects a PIL image
        result = model(image)
        return result
    except Exception as e:
        print(f"Error in recognition: {e}")
        return ""

if __name__ == "__main__":
    # Quick CLI test
    import sys
    if len(sys.argv) > 1:
        print(image_to_latex(Image.open(sys.argv[1])))