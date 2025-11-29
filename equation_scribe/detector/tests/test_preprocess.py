import numpy as np
from PIL import Image
from pathlib import Path
from equation_scribe.detector.preprocess import preprocess_image
import tempfile

def test_preprocess(tmp_path):
    # make synthetic noisy image
    w,h = 640, 800
    arr = np.zeros((h,w,3), dtype=np.uint8) + 255
    # draw dark rectangle/text-like structure
    import cv2
    cv2.putText(arr, "E = mc^2", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
    noisy = arr.copy()
    # add noise
    noise = (np.random.randn(*arr.shape) * 20).astype(np.int16)
    noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    out = preprocess_image(noisy, denoise=True, deskew=True, clahe=True, binarize=True)
    # result should be 2D array (grayscale)
    assert out.ndim == 2
    assert out.shape[0] == h and out.shape[1] == w
