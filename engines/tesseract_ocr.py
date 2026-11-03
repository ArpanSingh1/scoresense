import pytesseract
from PIL import Image
from core.base import BaseOCREngine
from core.schemas import DocumentResult, OCRBBox

class TesseractEngine(BaseOCREngine):
    def process(self, file_path: str, **kwargs) -> DocumentResult:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        elements = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if data['text'][i].strip():
                box = [
                    data['left'][i], data['top'][i],
                    data['left'][i] + data['width'][i],
                    data['top'][i] + data['height'][i]
                ]
                elements.append(OCRBBox(
                    box=[float(b) for b in box],
                    text=data['text'][i],
                    confidence=float(data['conf'][i]) / 100.0
                ))
        return DocumentResult(
            engine_name="tesseract",
            text=text,
            elements=elements
        )
