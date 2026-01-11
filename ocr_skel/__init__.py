"""
Occular OCR Package

# Для чайников (4 строки):
    from ocr_skel import ocr
    text = ocr("photo.png")
    print(text)

# Для продвинутых:
    from ocr_skel import OCRPipeline
    pipeline = OCRPipeline(onnx=True, gpu=False)
    results = pipeline.process_pdf("doc.pdf", dpi=300, workers=8)

# CLI:
    ocr photo.png
    ocr document.pdf --onnx --workers 4
"""

__version__ = "0.1.0"

from typing import Union, List, Dict, Optional
from pathlib import Path

from .registry import Registry
from .pipeline import OCRPipeline as _OCRPipelineBase, get_optimal_workers


# ============================================================================
# Регистрация компонентов
# ============================================================================

from .dbnet_detector import DBNetDetector
from .recognizer import CRNNRecognizer

Registry.register_detector("dbnet", DBNetDetector)
Registry.register_recognizer("crnn", CRNNRecognizer)

# ONNX версии (если доступны)
try:
    from .dbnet_detector_onnx import DBNetDetectorONNX
    from .recognizer_onnx import CRNNRecognizerONNX
    Registry.register_detector("dbnet-onnx", DBNetDetectorONNX)
    Registry.register_recognizer("crnn-onnx", CRNNRecognizerONNX)
    _onnx_available = True
except ImportError:
    _onnx_available = False


def _ensure_registered():
    """Для совместимости — компоненты уже зарегистрированы при импорте"""
    pass


# ============================================================================
# Простой API для чайников
# ============================================================================

def ocr(file_path: str, *, onnx: bool = True) -> Union[str, List[str]]:
    """
    Распознать текст из изображения или PDF.

    Args:
        file_path: путь к файлу (изображение или PDF)
        onnx: использовать ONNX Runtime (быстрее на AMD/без MKL)

    Returns:
        Для изображения: строка с распознанным текстом
        Для PDF: список строк (по одной на страницу)

    Example:
        >>> from ocr_skel import ocr
        >>> text = ocr("photo.png")
        >>> print(text)
        Привет мир
    """
    _ensure_registered()

    detector = "dbnet-onnx" if onnx else "dbnet"
    recognizer = "crnn-onnx" if onnx else "crnn"

    pipeline = _OCRPipelineBase(
        detector_name=detector,
        recognizer_name=recognizer
    )

    path = Path(file_path)
    if path.suffix.lower() == '.pdf':
        results = pipeline.process_pdf(file_path)
        # Возвращаем текст по страницам
        pages_text = []
        for page in results:
            # Сортируем по Y (сверху вниз)
            sorted_results = sorted(page["results"], key=lambda r: r["quad"][0][1])
            page_text = "\n".join(item["text"] for item in sorted_results)
            pages_text.append(page_text)
        return pages_text
    else:
        results = pipeline.process_image(file_path)
        # Сортируем по Y (сверху вниз)
        sorted_results = sorted(results, key=lambda r: r["quad"][0][1])
        return "\n".join(item["text"] for item in sorted_results)


def ocr_detailed(file_path: str, *, onnx: bool = True) -> Union[List[Dict], List[Dict]]:
    """
    Распознать текст с полной информацией (координаты, confidence).

    Args:
        file_path: путь к файлу
        onnx: использовать ONNX Runtime

    Returns:
        Список словарей {"quad": [...], "text": str, "confidence": float}
        Для PDF: список страниц с результатами

    Example:
        >>> results = ocr_detailed("photo.png")
        >>> for r in results:
        ...     print(f"{r['text']} ({r['confidence']:.2f})")
    """
    _ensure_registered()

    detector = "dbnet-onnx" if onnx else "dbnet"
    recognizer = "crnn-onnx" if onnx else "crnn"

    pipeline = _OCRPipelineBase(
        detector_name=detector,
        recognizer_name=recognizer
    )

    path = Path(file_path)
    if path.suffix.lower() == '.pdf':
        return pipeline.process_pdf(file_path)
    else:
        return pipeline.process_image(file_path)


# ============================================================================
# Продвинутый API
# ============================================================================

class OCRPipeline:
    """
    OCR Pipeline для продвинутых пользователей.

    Args:
        onnx: использовать ONNX Runtime (быстрее на AMD/без MKL)
        gpu: использовать GPU (если доступен)
        detector: явно указать детектор ("dbnet", "dbnet-onnx")
        recognizer: явно указать распознаватель ("crnn", "crnn-onnx")

    Example:
        >>> pipeline = OCRPipeline(onnx=True)
        >>> results = pipeline.process_image("photo.png")
        >>> text = pipeline.get_text("photo.png")
    """

    def __init__(
        self,
        *,
        onnx: bool = True,
        gpu: bool = False,
        detector: Optional[str] = None,
        recognizer: Optional[str] = None
    ):
        _ensure_registered()

        # Определяем компоненты
        if detector is None:
            detector = "dbnet-onnx" if onnx else "dbnet"
        if recognizer is None:
            recognizer = "crnn-onnx" if onnx else "crnn"

        self._pipeline = _OCRPipelineBase(
            detector_name=detector,
            recognizer_name=recognizer,
            detector_kwargs={"gpu": gpu},
            recognizer_kwargs={"gpu": gpu}
        )

    def process_image(self, image_path: str) -> List[Dict]:
        """
        Обработать изображение.

        Returns:
            Список {"quad": [...], "text": str, "confidence": float}
        """
        return self._pipeline.process_image(image_path)

    def process_pdf(
        self,
        pdf_path: str,
        *,
        dpi: int = 200,
        force_ocr: bool = False,
        workers: Optional[int] = None
    ) -> List[Dict]:
        """
        Обработать PDF.

        Args:
            pdf_path: путь к PDF
            dpi: разрешение рендеринга (default: 200)
            force_ocr: принудительно OCR даже для векторных PDF
            workers: количество воркеров (None = auto, max 4)

        Returns:
            Список страниц с результатами
        """
        return self._pipeline.process_pdf(
            pdf_path,
            dpi=dpi,
            force_ocr=force_ocr,
            workers=workers
        )

    def get_text(self, file_path: str, **kwargs) -> Union[str, List[str]]:
        """
        Получить только текст (без координат).

        Returns:
            Для изображения: строка
            Для PDF: список строк по страницам
        """
        path = Path(file_path)
        if path.suffix.lower() == '.pdf':
            results = self.process_pdf(file_path, **kwargs)
            return [
                "\n".join(item["text"] for item in page["results"])
                for page in results
            ]
        else:
            results = self.process_image(file_path)
            return "\n".join(item["text"] for item in results)


# ============================================================================
# Экспорт
# ============================================================================

__all__ = [
    # Простой API
    "ocr",
    "ocr_detailed",
    # Продвинутый API
    "OCRPipeline",
    # Утилиты
    "get_optimal_workers",
    # Низкоуровневый доступ
    "Registry",
]
