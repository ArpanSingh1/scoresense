from paddleocr import PaddleOCR
from core.base import BaseOCREngine
from core.schemas import DocumentResult, OCRBBox

class PaddleOCREngine(BaseOCREngine):
    def __init__(self, lang='en'):
        # Clean initialization without deprecated arguments
        # Work around the PaddlePaddle 3.3.x CPU oneDNN/PIR conversion bug.
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, enable_mkldnn=False)

    def process(self, file_path: str, **kwargs) -> DocumentResult:
        elements = []
        full_text = []

        for page in self.ocr.predict(file_path):
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            boxes = page.get("rec_boxes", [])
            if len(boxes) == 0:
                boxes = page.get("rec_polys", [])

            for index, text in enumerate(texts):
                if not text or index >= len(boxes):
                    continue

                box = boxes[index]
                if hasattr(box, "tolist"):
                    box = box.tolist()
                if box and isinstance(box[0], (list, tuple)):
                    flat_box = [float(coord) for point in box for coord in point]
                else:
                    flat_box = [float(coord) for coord in box]

                confidence = float(scores[index]) if index < len(scores) else 0.0
                elements.append(
                    OCRBBox(box=flat_box, text=str(text), confidence=confidence)
                )
                full_text.append(str(text))

        return DocumentResult(
            engine_name="paddle_ocr",
            text="\n".join(full_text),
            markdown="\n".join(full_text),
            elements=elements
        )
