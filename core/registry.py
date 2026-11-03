from importlib import import_module

class OCRPipelineManager:
    def __init__(self):
        self.engines = {
            "paddle_ocr": ("engines.paddle_ocr", "PaddleOCREngine"),
            "open_ocr": ("engines.open_ocr", "OpenOCREngine"),
            "surya": ("engines.surya_ocr", "SuryaOCREngine"),
            "tesseract": ("engines.tesseract_ocr", "TesseractEngine"),
            "pp_structure": ("engines.pp_structure", "PPStructureEngine"),
            "mineru": ("engines.mineru_pipe", "MinerUEngine"),
            "marker": ("engines.marker_pipe", "MarkerEngine"),
            "docling": ("engines.docling", "DoclingEngine"),
        }

    def run(self, engine_name: str, file_path: str, **kwargs):
        if engine_name not in self.engines:
            raise ValueError(f"Engine '{engine_name}' not found. Choose from: {list(self.engines.keys())}")
        module_name, class_name = self.engines[engine_name]
        engine_class = getattr(import_module(module_name), class_name)
        engine_instance = engine_class()
        return engine_instance.process(file_path, **kwargs)
