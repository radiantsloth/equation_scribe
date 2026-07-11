"""Stable shared numeric constants for Equation Scribe components.

These values are durable configuration primitives used by detector and data
generation utilities. They are separate from runtime settings because they do
not depend on environment variables or filesystem state.
"""

DEFAULT_DPI = 150
PAGE_WIDTH_IN = 8.5
PAGE_HEIGHT_IN = 11.0
DEFAULT_PAGE_W_PX = int(PAGE_WIDTH_IN * DEFAULT_DPI)
DEFAULT_PAGE_H_PX = int(PAGE_HEIGHT_IN * DEFAULT_DPI)

ROTATION_AUG_MAX_ANGLE = 15
DESKEW_THRESHOLD_DEG = 0.75

MAX_EQ_WIDTH_FRAC = 0.6
MAX_EQ_HEIGHT_FRAC = 0.25

# If IoU exceeds this value, synthetic placement treats the boxes as overlapping.
NON_OVERLAP_IOU = 0.0

MAX_PLACEMENT_ATTEMPTS = 1000

__all__ = [
    "DEFAULT_DPI",
    "PAGE_WIDTH_IN",
    "PAGE_HEIGHT_IN",
    "DEFAULT_PAGE_W_PX",
    "DEFAULT_PAGE_H_PX",
    "ROTATION_AUG_MAX_ANGLE",
    "DESKEW_THRESHOLD_DEG",
    "MAX_EQ_WIDTH_FRAC",
    "MAX_EQ_HEIGHT_FRAC",
    "NON_OVERLAP_IOU",
    "MAX_PLACEMENT_ATTEMPTS",
]
