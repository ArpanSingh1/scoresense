from docling.document_converter import DocumentConverter
from core.base import BaseOCREngine
from core.schema import DocumentResult

class DoclingEngine(BaseOCREngine):
    def __init__(self):
        self.converter = DocumentConverter()

    def process(self, file_path: str, **kwargs) -> DocumentResult:
        conv_result = self.converter.convert(file_path)
        markdown_text = conv_result.document.export_to_markdown()
        return DocumentResult(
            engine_name="docling",
            text=markdown_text,
            markdown=markdown_text,
            structured_data=conv_result.document.export_to_dict()
        )
