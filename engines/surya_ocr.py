from html.parser import HTMLParser

from PIL import Image
from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor

from core.base import BaseOCREngine
from core.schemas import DocumentResult, OCRBBox

class SuryaOCREngine(BaseOCREngine):
    def __init__(self):
        self.manager = SuryaInferenceManager()
        self.recognizer = RecognitionPredictor(self.manager)

    def process(self, file_path: str, **kwargs) -> DocumentResult:
        image = Image.open(file_path).convert("RGB")
        prediction = self.recognizer([image])[0]

        elements = []
        full_text = []
        for block in prediction.blocks:
            parser = _TextExtractor()
            parser.feed(block.html)
            text = " ".join(" ".join(parser.parts).split())
            if not text or block.skipped or block.error:
                continue
            elements.append(
                OCRBBox(
                    box=block.bbox,
                    text=text,
                    confidence=float(block.confidence or 0.0),
                )
            )
            full_text.append(text)

        return DocumentResult(
            engine_name="surya",
            text="\n".join(full_text),
            markdown="\n\n".join(full_text),
            elements=elements
        )


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        if data.strip():
            self.parts.append(data.strip())
