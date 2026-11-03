from abc import ABC, abstractmethod
from core.schemas import DocumentResult

class BaseOCREngine(ABC):
    @abstractmethod
    def process(self, file_path: str, **kwargs) -> DocumentResult:
        """Process an image or PDF file and return standardized results."""
        pass
