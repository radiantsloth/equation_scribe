"""Shared domain models for paper profile storage.

These models describe the durable records written to and read from the
paper-profile storage layer. They are intentionally framework-agnostic so both
the web backend and future CLI or detector tooling can share the same schema.
"""

from typing import List, Tuple

from pydantic import BaseModel, Field

# A PDF-space bounding box stored as (x0, y0, x1, y1) in page points.
BBox = Tuple[float, float, float, float]


class Box(BaseModel):
    """Location of one equation box on a specific PDF page.

    A single equation can have one or more boxes. In the current data model,
    each box records the page number and the rectangle in PDF coordinates.
    """

    page: int = Field(..., description="Zero-based page index in the source PDF.")
    bbox_pdf: BBox = Field(
        ...,
        description="Bounding box in PDF points as (x0, y0, x1, y1).",
    )


class EquationRecord(BaseModel):
    """One equation record stored in ``equations.jsonl``.

    This model is the shared durable record for manual and automatic equation
    extraction. Route-specific request and response models should stay outside
    core unless they are reused by multiple application surfaces.
    """

    eq_uid: str = Field(..., description="Stable equation identifier within a paper.")
    paper_id: str = Field(..., description="Paper identifier used for profile storage.")
    latex: str = Field("", description="Current LaTeX text for the equation.")
    notes: str = Field("", description="Human or pipeline notes about the equation.")
    boxes: List[Box] = Field(
        ...,
        description="One or more PDF-space boxes associated with the equation.",
    )
