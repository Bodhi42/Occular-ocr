"""OCR Skeleton Package"""

__version__ = "0.1.0"

from .registry import Registry
from .detector import CRAFTDetector
from .dbnet_detector import DBNetDetector
from .recognizer import CRNNRecognizer
from .vitstr_recognizer import ViTSTRRecognizer
from .pipeline import OCRPipeline

# Auto-register default implementations
Registry.register_detector("craft", CRAFTDetector)
Registry.register_detector("dbnet", DBNetDetector)
Registry.register_recognizer("crnn", CRNNRecognizer)
Registry.register_recognizer("vitstr", ViTSTRRecognizer)

__all__ = [
    "Registry",
    "CRAFTDetector",
    "DBNetDetector",
    "CRNNRecognizer",
    "ViTSTRRecognizer",
    "OCRPipeline",
]
