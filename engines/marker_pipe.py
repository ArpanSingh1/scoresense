from marker.convert import convert_single_pdf
from marker.models import load_all_models
from core.base import BaseOCREngine
from core.schema import DocumentResult

class MarkerEngine(BaseOCREngine):
    def __init__(self):
        self.model_lst = load_all_models()

    def process(self, file_path: str, **kwargs) -> DocumentResult:
        full_text, images, out_metadata = convert_single_pdf(file_path, self.model_lst)
        return DocumentResult(
            engine_name="marker",
            text=full_text,
            markdown=full_text,
            structured_data=out_metadata
        )
