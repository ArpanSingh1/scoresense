from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from core.base import BaseOCREngine
from core.schema import DocumentResult

class MinerUEngine(BaseOCREngine):
    def process(self, file_path: str, **kwargs) -> DocumentResult:
        # Read file bytes & initialize MinerU UniPipe
        image_writer, md_writer = DiskReaderWriter("."), DiskReaderWriter(".")
        pipe = UNIPipe(file_path, image_writer, md_writer)
        pipe.pipe_classify()
        pipe.pipe_parse()
        
        markdown_content = pipe.pipe_mk_markdown(".")
        return DocumentResult(
            engine_name="mineru",
            text=markdown_content,
            markdown=markdown_content
        )
