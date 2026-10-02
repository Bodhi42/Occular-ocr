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
from typing import Optional, Union, List
import os


@dataclass
class Settings:
    """Все настройки OCR в одном месте. Меняй что нужно, остальное — разумные дефолты."""

    # --- производительность ---
    num_threads: Optional[int] = None    # число CPU-ядер; None = min(доступные, 4)
    gpu: bool = False                     # исполнять на GPU/CUDA (нужен пакет onnxruntime-gpu; иначе CPU)

    # --- языки ---
    # None = русская модель (ru/en). Список кодов (напр. ['uk'] или ['ru','kk','uk']) или 'auto' =
    # мультиязычный роутер (русская + 12 кир-языков: ba be bg cv kk ky mk mn sr tg tt uk).
    languages: Union[None, str, List[str]] = None

    # --- препроцессинг ---
    orientation: bool = False             # определять поворот страницы (0/90/180/270°) и выпрямлять; ВЫКЛ по умолчанию
    deskew: bool = True                   # выпрямлять наклон скана перед детекцией
    # Нейро-расправление изгиба (UVDoc): None = ВЫКЛ (по умолч). Строка = HF model_id или локальная
    # папка с весами (напр. "Shivin11/occular-uvdoc"), либо готовый UVDocUnwarper. Нужен extra [unwarp].
    unwarp: Union[None, str, object] = None
    unwarp_gate: Union[bool, float] = True   # True=readability-гейт×unwarp_margin; False/None=всегда; число=свой порог; callable(res_orig,res_dew)->bool
    unwarp_margin: float = 1.05           # порог-множитель читаемости для дефолтного гейта (unwarp_gate=True)

    # --- декодирование ---
    lm: bool = True                       # beam-CTC + языковая модель (−25% WER; качает LM с HF при 1-м запуске)

    # --- постпроцессинг ---
    reading_order: bool = False           # порядок чтения (нужна докачка layout-модели с HF)

    # --- выбор моделей ---
    # recognizer: архитектура распознавателя — 'svtr_lcnet' (по умолчанию: лёгкая, ~4x быстрее,
    # качество 99.6% от svtr_t) или 'svtr_t' (крупная, исходная). None = svtr_lcnet.
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
            f"  languages     = {self.languages or 'ru/en (по умолчанию)'}  (None=ru/en; список кодов или 'auto'=мультиязычный)\n"
            f"  orientation   = {self.orientation}  (препроцессинг: поворот страницы 0/90/180/270°)\n"
            f"  deskew        = {self.deskew}   (препроцессинг: выпрямление наклона)\n"
            f"  unwarp        = {self.unwarp if self.unwarp is not None else 'None (выкл)'}  (препроцессинг: UVDoc-расправление изгиба; нужен extra [unwarp])\n"
            f"  lm            = {self.lm}   (beam-CTC + языковая модель; −25% WER, качает LM с HF)\n"
            f"  reading_order = {self.reading_order}  (постпроцессинг: порядок чтения, нужна докачка)\n"
            f"  detector      = {self.detector or 'авто'}\n"
            f"  recognizer    = {self.recognizer or 'svtr_lcnet (по умолчанию)'}"
        )
