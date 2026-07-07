from typing import List, Tuple
from pydantic import BaseModel, Field

BBox = Tuple[float, float, float, float]

class Box(BaseModel):
    page: int = Field(..., description="Zero-based page index")
    bbox_pdf: BBox

class EquationRecord(BaseModel):
    eq_uid: str
    paper_id: str
    latex: str = ""
    notes: str = ""
    boxes: List[Box]