"""Единые настройки Occular-OCR — один объект на всё.

    from occular import Settings, OCRPipeline

    cfg = Settings(num_threads=8, deskew=True, tables=True)
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
    num_threads: Optional[int] = None    # потоков на страницу; None = min(доступные, 8). Много страниц — workers (процессы)
    gpu: bool = False                     # исполнять на GPU/CUDA через ONNX Runtime (библиотеки CUDA: occular-ocr[gpu]; иначе CPU)

    # --- языки ---
    # None = русская модель (ru/en). Список кодов (напр. ['uk'] или ['ru','kk','uk']) или 'auto' =
    # мультиязычный роутер (русская + 12 кир-языков: ba be bg cv kk ky mk mn sr tg tt uk).
    languages: Union[None, str, List[str]] = None

    # --- препроцессинг ---
    orientation: bool = False             # определять поворот страницы (0/90/180/270°) и выпрямлять; ВЫКЛ по умолчанию
    deskew: bool = True                   # выпрямлять наклон скана перед детекцией
    # Нейро-расправление изгиба (UVDoc): None = ВЫКЛ (по умолч). Строка = HF model_id или локальная
    # папка с весами (напр. "Shivin11/occular-uvdoc"), либо готовый UVDocUnwarper. Исполняется через ONNX.
    unwarp: Union[None, str, object] = None
    unwarp_gate: Union[bool, float] = True   # True=readability-гейт×unwarp_margin; False/None=всегда; число=свой порог; callable(res_orig,res_dew)->bool
    unwarp_margin: float = 1.05           # порог-множитель читаемости для дефолтного гейта (unwarp_gate=True)

    # --- декодирование ---
    lm: bool = True                       # beam-CTC + языковая модель (−25% WER; качает LM с HF при 1-м запуске)

    # --- постпроцессинг ---
    reading_order: bool = False           # порядок чтения (нужна докачка layout-модели с HF)
    # ЭКСПЕРИМЕНТ: порядок чтения по layout-регионам (D-FINE-S). ВЫКЛ; не вместе с reading_order.
    experimental_layout_order: bool = False
    # чем упорядочивать блоки в этом режиме: 'model' (обученная модель порядка, по умолч.) или 'xycut' (правило)
    layout_order_method: str = "model"

    # --- таблицы ---
    # Находить таблицы и раскладывать строки по ячейкам: строки получают ключи 'table'/'cell', структура —
    # в .tables (изображение) или 'tables' (страница PDF), текст ocr()/get_text — таблицы в Markdown.
    # Модели таблиц качаются с HF при первом использовании.
    tables: bool = False

    # --- выбор моделей ---
    # recognizer: архитектура распознавателя — 'svtr_lcnet' (по умолчанию: лёгкая, ~4x быстрее,
    # качество 99.6% от svtr_t) или 'svtr_t' (крупная, исходная). None = svtr_lcnet.
    detector: Optional[str] = None
    recognizer: Optional[str] = None

    def resolved_threads(self) -> int:
        """Сколько ядер реально будет использовано."""
        if self.num_threads:
            return int(self.num_threads)
        from .parallel import available_cpus
        from .pipeline import DEFAULT_THREADS
        return min(available_cpus(), DEFAULT_THREADS)

    def __str__(self) -> str:
        return (
            "Occular-OCR settings:\n"
            f"  num_threads   = {self.num_threads}  (effective {self.resolved_threads()}; None = min(cores, 8))\n"
            f"  gpu           = {self.gpu}  (GPU/CUDA; CUDA libraries: occular-ocr[gpu], otherwise CPU)\n"
            f"  languages     = {self.languages or 'ru/en (default)'}  (None = ru/en; a list of codes or 'auto' = multilingual)\n"
            f"  orientation   = {self.orientation}  (preprocessing: page rotation 0/90/180/270°)\n"
            f"  deskew        = {self.deskew}   (preprocessing: straighten skew)\n"
            f"  unwarp        = {self.unwarp if self.unwarp is not None else 'None (off)'}  (preprocessing: UVDoc dewarping, ONNX)\n"
            f"  lm            = {self.lm}   (beam-CTC + language model; -25% WER, downloads the LM from HF)\n"
            f"  reading_order = {self.reading_order}  (postprocessing: reading order, extra download)\n"
            f"  experimental_layout_order = {self.experimental_layout_order}  (EXPERIMENTAL: order by layout regions)\n"
            f"  layout_order_method = {self.layout_order_method}  ('model' = trained block order, 'xycut' = rule)\n"
            f"  tables        = {self.tables}  (tables: lines into cells, Markdown tables in text)\n"
            f"  detector      = {self.detector or 'auto'}\n"
            f"  recognizer    = {self.recognizer or 'svtr_lcnet (default)'}"
        )
