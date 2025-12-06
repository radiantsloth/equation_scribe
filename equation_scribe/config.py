# equation_scribe/config.py
"""
Central configuration / constants for equation_scribe projects.
Helps avoid hard-coded literal constants scattered through the code.
"""

DEFAULT_DPI = 150
PAGE_WIDTH_IN = 8.5
PAGE_HEIGHT_IN = 11.0
DEFAULT_PAGE_W_PX = int(PAGE_WIDTH_IN * DEFAULT_DPI)
DEFAULT_PAGE_H_PX = int(PAGE_HEIGHT_IN * DEFAULT_DPI)

# Rotation augmentation: maximum rotation range (± degrees).
ROTATION_AUG_MAX_ANGLE = 15

# If an estimated deskew angle is smaller (in absolute value) than this,
# we consider the page already upright and skip deskewing.
DESKEW_THRESHOLD_DEG = 0.75

# Limits for equation image size relative to page when generating synthetic pages.
MAX_EQ_WIDTH_FRAC = 0.6
MAX_EQ_HEIGHT_FRAC = 0.25

# For placing boxes, maximum allowed IoU when `require_non_overlap=True`.
# If IoU > NON_OVERLAP_IOU, we consider that overlapping and will retry placement.
NON_OVERLAP_IOU = 0.0  # strict; change to 0.05 or 0.1 to allow tiny overlaps

# Number of attempts before giving up placing a single box on a page
MAX_PLACEMENT_ATTEMPTS = 1000
