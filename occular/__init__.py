"""
Occular OCR Package

# Для чайников (4 строки):
    from occular import ocr
    text = ocr("photo.png")
    print(text)

# Для продвинутых:
    from occular import OCRPipeline
    pipeline = OCRPipeline()
    results = pipeline.process_pdf("doc.pdf", dpi=300, workers=8)

# CLI:
    ocr photo.png
    ocr document.pdf --workers 4
"""

__version__ = "0.3.0"

import os
from typing import Union, List, Dict, Optional
from pathlib import Path

from .registry import Registry
from .pipeline import OCRPipeline as _OCRPipelineBase, get_optimal_workers
from .model_files import model_info, download_reading_order, reading_order_ready, WEIGHTS_DIR
from .settings import Settings


# ============================================================================
# Регистрация компонентов
# ============================================================================

# Библиотека CPU-only ONNX.
from .dbnet_detector_onnx import DBNetDetectorONNX
from .recognizer_onnx import CRNNRecognizerONNX
Registry.register_detector("dbnet-onnx", DBNetDetectorONNX)
Registry.register_recognizer("crnn-onnx", CRNNRecognizerONNX)


def TableRecognizer(*args, **kwargs):
    """Распознавание таблиц (детекция + структура + объединённые ячейки). Ленивая обёртка —
    torch подгружается только при вызове. См. occular.tables.TableRecognizer."""
    from .tables import TableRecognizer as _TR
    return _TR(*args, **kwargs)


def _ensure_registered():
    """Для совместимости — компоненты уже зарегистрированы при импорте"""
    pass


# ============================================================================
# Простой API для чайников
# ============================================================================

def _build_pipeline(deskew, lm, num_threads, gpu, reading_order=False):
    """Собрать конвейер с русским распознавателем (ru/en) по умолчанию."""
    _ensure_registered()
    return _OCRPipelineBase(detector_name="dbnet-onnx", recognizer_name="crnn-onnx",
                            deskew=deskew, reading_order=reading_order, lm=lm,
                            num_threads=num_threads, gpu=gpu)


def ocr(file_path: str, *, deskew: bool = True, lm: bool = True,
        num_threads: Optional[int] = None, gpu: bool = False) -> Union[str, List[str]]:
    """
    Распознать текст из изображения или PDF.

    Args:
        file_path: путь к файлу (изображение или PDF)
        deskew: выпрямлять наклон скана (по умолчанию True)
        lm: beam-CTC + языковая модель для декодирования (по умолчанию True; False = быстрый greedy)
        num_threads: число CPU-ядер (None = min(доступные, 4))
        gpu: исполнять на GPU/CUDA (нужен пакет onnxruntime-gpu; иначе CPU)

    Returns:
        Для изображения: строка с распознанным текстом
        Для PDF: список строк (по одной на страницу)

    Example:
        >>> from occular import ocr
        >>> text = ocr("photo.png")
        >>> print(text)
        Привет мир
    """
    pipeline = _build_pipeline(deskew, lm, num_threads, gpu)

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


def ocr_detailed(file_path: str, *, deskew: bool = True, lm: bool = True,
                 num_threads: Optional[int] = None, gpu: bool = False) -> List[Dict]:
    """
    Распознать текст с полной информацией (координаты, confidence).

    Args:
        file_path: путь к файлу
        deskew: выпрямлять наклон скана (по умолчанию True)
        lm: beam-CTC + языковая модель для декодирования (по умолчанию True; False = быстрый greedy)
        num_threads: число CPU-ядер (None = min(доступные, 4))
        gpu: исполнять на GPU/CUDA (нужен пакет onnxruntime-gpu; иначе CPU)

    Returns:
        Список словарей {"quad": [...], "text": str, "confidence": float}
        Для PDF: список страниц с результатами

    Example:
        >>> results = ocr_detailed("photo.png")
        >>> for r in results:
        ...     print(f"{r['text']} ({r['confidence']:.2f})")
    """
    pipeline = _build_pipeline(deskew, lm, num_threads, gpu)

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

    Настройки — через объект Settings или отдельные kwargs (deskew, reading_order, num_threads).

    Example:
        >>> pipeline = OCRPipeline()
        >>> results = pipeline.process_image("photo.png")
        >>> text = pipeline.get_text("photo.png")
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        deskew: bool = True,
        reading_order: bool = False,
        lm: bool = True,
        num_threads: Optional[int] = None,
        gpu: bool = False,
        detector: Optional[str] = None,
        recognizer: Optional[str] = None
    ):
        _ensure_registered()

        # Единые настройки: можно передать объект Settings, либо отдельные kwargs (совместимость).
        if settings is None:
            settings = Settings(deskew=deskew, reading_order=reading_order, lm=lm,
                                num_threads=num_threads, gpu=gpu,
                                detector=detector, recognizer=recognizer)
        self.settings = settings
        s = settings

        # Порядок чтения (layout) — опционально, ВЫКЛ по умолчанию. Модель качается с HuggingFace.
        self.reading_order = s.reading_order
        if s.reading_order and not reading_order_ready():
            raise RuntimeError(
                "reading_order=True, но модель порядка чтения не скачана.\n"
                "Скачайте её:  from occular import download_reading_order; download_reading_order()\n"
                "Или посмотрите статус:  from occular import model_info; model_info()"
            )

        detector = s.detector or "dbnet-onnx"
        recognizer = s.recognizer or "crnn-onnx"

        self._pipeline = _OCRPipelineBase(
            detector_name=detector,
            recognizer_name=recognizer,
            deskew=s.deskew,
            reading_order=s.reading_order,
            lm=s.lm,
            num_threads=s.num_threads,
            gpu=s.gpu
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
        dpi: int = 300,
        force_ocr: bool = False,
        workers: Optional[int] = None
    ) -> List[Dict]:
        """
        Обработать PDF.

        Args:
            pdf_path: путь к PDF
            dpi: разрешение рендеринга (default: 300)
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
    "Settings",
    # Таблицы (детекция + структура)
    "TableRecognizer",
    # Модели: где лежат и как скачать порядок чтения (для чайников)
    "model_info",
    "download_reading_order",
    "reading_order_ready",
    "WEIGHTS_DIR",
    # Утилиты
    "get_optimal_workers",
    # Низкоуровневый доступ
    "Registry",
]
