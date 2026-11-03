from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class OCRBBox(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    box: List[float]  # [x1, y1, x2, y2]
    text: str
    confidence: float

class DocumentResult(BaseModel):
    engine_name: str
    text: str
    markdown: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None
    elements: List[OCRBBox] = []
