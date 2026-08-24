"""Единые настройки Occular-OCR — один объект на всё.

    from occular import Settings, OCRPipeline

    cfg = Settings(num_threads=8, deskew=True, reading_order=False)
    p = OCRPipeline(cfg)

Точечно:
    cfg = Settings()
    cfg.num_threads = 2
    cfg.deskew = False

Посмотреть все параметры и дефолты:
    print(Settings())
"""
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class Settings:
    """Все настройки OCR в одном месте. Меняй что нужно, остальное — разумные дефолты."""

    # --- производительность ---
    num_threads: Optional[int] = None    # число CPU-ядер; None = min(доступные, 4)
    gpu: bool = False                     # исполнять на GPU/CUDA (нужен пакет onnxruntime-gpu; иначе CPU)

    # --- препроцессинг ---
    deskew: bool = True                   # выпрямлять наклон скана перед детекцией

    # --- декодирование ---
    lm: bool = True                       # beam-CTC + языковая модель (−25% WER; качает LM с HF при 1-м запуске)

    # --- постпроцессинг ---
    reading_order: bool = False           # порядок чтения (нужна докачка layout-модели с HF)

    # --- явный выбор моделей (обычно не нужно, авто) ---
    detector: Optional[str] = None
    recognizer: Optional[str] = None

    def resolved_threads(self) -> int:
        """Сколько ядер реально будет использовано."""
        return int(self.num_threads) if self.num_threads else min(os.cpu_count() or 1, 4)

    def __str__(self) -> str:
        return (
            "Настройки Occular-OCR:\n"
            f"  num_threads   = {self.num_threads}  (эффективно {self.resolved_threads()}; None = min(ядра,4))\n"
            f"  gpu           = {self.gpu}  (GPU/CUDA; нужен onnxruntime-gpu, иначе CPU)\n"
            f"  deskew        = {self.deskew}   (препроцессинг: выпрямление наклона)\n"
            f"  lm            = {self.lm}   (beam-CTC + языковая модель; −25% WER, качает LM с HF)\n"
            f"  reading_order = {self.reading_order}  (постпроцессинг: порядок чтения, нужна докачка)\n"
            f"  detector      = {self.detector or 'авто'}\n"
            f"  recognizer    = {self.recognizer or 'авто'}"
        )
