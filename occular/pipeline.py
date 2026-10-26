"""OCR Pipeline: detect -> recognize"""

import os
import numpy as np
from typing import List, Dict, Union, Optional
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2

from .registry import Registry


def pymupdf():
    """PyMuPDF под новым именем модуля (с 1.24.3); `import fitz` в свежих версиях печатает
    предупреждение об устаревании — в т.ч. в каждом воркере."""
    try:
        import pymupdf as mod
    except ImportError:
        import fitz as mod
    return mod


DEFAULT_THREADS = 8          # потоков ONNX Runtime на страницу по умолчанию (см. замер в __init__)


def get_optimal_workers(max_workers: int = 4) -> int:
    """
    Определить оптимальное количество воркеров.

    Берёт min(доступные CPU, max_workers).
    По умолчанию max_workers=4, чтобы не перегружать систему.
    """
    try:
        # os.cpu_count() возвращает логические ядра
        available = os.cpu_count() or 1
        return min(available, max_workers)
    except Exception:
        return 1


class PageResult(list):
    """Строки страницы (обычный список словарей) + .tables — таблицы при tables=True.
    Ведёт себя как прежний список: итерация, len, индексы, json.dumps (без .tables)."""

    def __init__(self, lines=(), tables=None):
        super().__init__(lines)
        self.tables = tables if tables is not None else []


class OCRPipeline:
    """Pipeline для OCR: детекция + распознавание"""

    def __init__(self, detector_name: str = None, recognizer_name: str = None,
                 detector_kwargs: dict = None, recognizer_kwargs: dict = None,
                 deskew: bool = True, reading_order: bool = False, lm: bool = True,
                 num_threads: int = None, gpu: bool = False, orientation: bool = False,
                 oriented_lines: bool = None, unwarp=None, unwarp_gate=True,
                 unwarp_margin: float = 1.05, experimental_layout_order: bool = False,
                 layout_order_method: str = "model", tables: bool = False):
        """
        Args:
            detector_name: имя детектора (по умолчанию 'dbnet-onnx')
            recognizer_name: имя распознавателя (по умолчанию 'crnn-onnx')
            detector_kwargs: параметры для детектора
            recognizer_kwargs: параметры для распознавателя
            deskew: геометрическое выпрямление строк — FHT-наклон всей страницы (единицы градусов)
                И per-crop поворот кропов (снятие наклона/±90 + арбитраж 0/180/vertical-stack по
                CTC-уверенности). ВКЛ по умолчанию. Две прежние ручки (deskew + oriented_lines) сведены
                в эту одну. Для повёрнутых на 90/180/270° СТРАНИЦ нужен отдельно orientation.
            orientation: нейро-определение поворота страницы (0/90/180/270°) и выпрямление; ВЫКЛ по умолч.
            oriented_lines: УСТАРЕЛО. None (по умолч) = следовать deskew. Задавай True/False только
                чтобы принудительно оторвать per-crop ориентацию от deskew (обратная совместимость).
            num_threads: потоков ONNX Runtime на страницу (None = min(доступные, 8))
            gpu: исполнять на GPU/CUDA (ONNX Runtime; библиотеки CUDA — pip install "occular-ocr[gpu]"; иначе фолбэк на CPU)
            unwarp: опц. нейросетевое расправление изгиба (UVDoc). None = ВЫКЛ (по умолч).
                Строка = HF model_id/локальная папка с весами (ленивая загрузка), либо готовый
                UVDocUnwarper (ONNX). Тяжёлый препроцесс для фото/смятых страниц.
            unwarp_gate: чем решать, применять ли дьюарп (когда unwarp включён). Варианты:
                • True (по умолч) — readability self-gate с порогом unwarp_margin: берём дьюарп ТОЛЬКО
                  если он повышает читаемость (Σ len·conf строк) минимум в unwarp_margin раз. На нашем
                  домене (мостовые плоские сканы) blanket-дьюарп вредит (+0.006 CER), а этот гейт
                  разворачивает в плюс (−0.0018 CER, на сработавших медиана ≈ −0.04); стоит доп.
                  прогона OCR на кандидате.
                • False/None — blanket: применять unwarp всегда (дешевле на 1 OCR, но вредит плоским).
                • число (напр. 1.10) — тот же readability-гейт, но со своим порогом-margin (перебивает
                  unwarp_margin). Больше порог → реже срабатывает, но чище выигрыш.
                • callable(res_orig, res_dew) -> bool — свой гейт: True = взять дьюарп. res_* это списки
                  результатов OCR ({text, confidence, quad}) для оригинала и дьюарпа.
            experimental_layout_order: ЭКСПЕРИМЕНТАЛЬНЫЙ порядок чтения по layout-регионам (D-FINE-S;
                строки без региона — отдельными объектами). ВЫКЛ по умолч. Не совместим с reading_order=True.
                Строки и их распознавание те же, что в обычном режиме; меняется только порядок, к строкам
                добавляются ключи layout_region / layout_label. Ничего не удаляется и не пропускается.
            layout_order_method: чем упорядочивать блоки в режиме experimental_layout_order:
                'model' (по умолч.) — обученная модель порядка блоков (ONNX, ~5 мс/стр на CPU);
                'xycut' — геометрическое правило XY-cut. Объекты и правило «внутри блока сверху вниз»
                у обоих одинаковые. Модель качается с HuggingFace при первом использовании.
            tables: искать таблицы и раскладывать распознанные строки по их ячейкам. ВЫКЛ по умолч.
                Строкам внутри таблицы добавляются ключи 'table' и 'cell'; структура таблиц — в .tables
                у результата process_image и в ключе 'tables' у страницы PDF. Строку, которую детектор
                склеил через границу ячеек, делим по пробелу у границы (ключ 'split'); текст кусков —
                части исходного текста строки.
            unwarp_margin: порог-множитель для дефолтного readability-гейта (unwarp_gate=True).
                Дьюарп берётся если read(дьюарп) > read(оригинал)·unwarp_margin. По умолч 1.05
                (осторожно: ~3% срабатываний, лучшее соотношение луч/хуж ≈ 3:1).
        """
        # аргументы конструктора как есть — процессы-воркеры (occular.parallel) строят по ним свой конвейер
        self._init_kwargs = dict(
            detector_name=detector_name, recognizer_name=recognizer_name,
            detector_kwargs=dict(detector_kwargs or {}), recognizer_kwargs=dict(recognizer_kwargs or {}),
            deskew=deskew, reading_order=reading_order, lm=lm, num_threads=num_threads, gpu=gpu,
            orientation=orientation, oriented_lines=oriented_lines, unwarp=unwarp, unwarp_gate=unwarp_gate,
            unwarp_margin=unwarp_margin, experimental_layout_order=experimental_layout_order,
            layout_order_method=layout_order_method, tables=tables)
        if reading_order and experimental_layout_order:
            raise ValueError("reading_order=True and experimental_layout_order=True are mutually exclusive: "
                             "choose one reading-order mode.")
        if layout_order_method not in ("model", "xycut"):
            raise ValueError(f"layout_order_method must be 'model' or 'xycut', not {layout_order_method!r}")
        detector_kwargs = detector_kwargs or {}
        recognizer_kwargs = recognizer_kwargs or {}
        self.gpu = bool(gpu)

        # Число потоков на страницу: по умолчанию min(доступные ядра, 8). Замер (EPYC, 64 ядра):
        # 4 потока — 2.35 с/стр, 8 — 1.64, 16 — 1.31, дальше не быстрее, а с 64 медленнее (1.53, 128 — 2.16)
        if num_threads is None:
            from .parallel import available_cpus
            num_threads = min(available_cpus(), DEFAULT_THREADS)
        self.num_threads = max(1, int(num_threads))
        # резолвим дефолтное имя ДО проверки (иначе None не пройдёт "onnx" in ...)
        det_name = detector_name or Registry._default_detector
        rec_name = recognizer_name or Registry._default_recognizer
        if "onnx" in det_name:
            detector_kwargs.setdefault("num_threads", self.num_threads)
            detector_kwargs.setdefault("gpu", self.gpu)
        if "onnx" in rec_name or rec_name == "multilingual":
            recognizer_kwargs.setdefault("num_threads", self.num_threads)
            recognizer_kwargs.setdefault("gpu", self.gpu)
            recognizer_kwargs.setdefault("lm", bool(lm))

        self.detector = Registry.get_detector(detector_name, **detector_kwargs)
        self.recognizer = Registry.get_recognizer(recognizer_name, **recognizer_kwargs)
        self.deskew = deskew   # геом. выпрямление строк: FHT-наклон страницы + per-crop поворот кропов
        # Ориентация страницы (0/90/180/270°) — ВЫКЛ по умолчанию, модель ленивая.
        self.orientation = bool(orientation)
        self._orient = None
        # per-crop ориентация строк (детекция сохраняет угол региона, рекогнайзер-арбитраж снимает
        # наклон/±90 и решает 0/180/vertical-stack) СВЕДЕНА в флаг deskew — отдельной ручки нет.
        # oriented_lines оставлен ТОЛЬКО как устаревший override: None = следовать deskew (норма).
        # Требует detect_regions() у детектора (dbnet-onnx умеет); иначе тихий фолбэк на detect().
        _ol = self.deskew if oriented_lines is None else oriented_lines
        self.oriented_lines = bool(_ol) and hasattr(self.detector, "detect_regions")
        self.reading_order = reading_order   # порядок чтения через layout-модель (по умолчанию ВЫКЛ)
        self._ro = None
        # ЭКСПЕРИМЕНТ: порядок по регионам LayoutDetector + XY-cut. Модель создаётся сразу (не лениво),
        # чтобы параллельные потоки PDF не гонялись за инициализацией.
        self.experimental_layout_order = bool(experimental_layout_order)
        self.layout_order_method = layout_order_method
        self._layout = None
        self._block_order = None
        if self.experimental_layout_order:
            from .layout import LayoutDetector
            self._layout = LayoutDetector(num_threads=self.num_threads, gpu=self.gpu)
            if layout_order_method == "model":
                from .layout_order_model import BlockOrderModel
                self._block_order = BlockOrderModel(num_threads=self.num_threads, gpu=self.gpu)
        # Таблицы: модели создаются сразу (как layout) — параллельные потоки PDF их делят
        self.tables = bool(tables)
        self._tables = None
        if self.tables:
            from .tables import TableRecognizer
            self._tables = TableRecognizer(num_threads=self.num_threads, gpu=self.gpu)
        if reading_order:
            from .reading_order import ReadingOrderModel
            self._ro = ReadingOrderModel(num_threads=self.num_threads, gpu=self.gpu)
        # UVDoc unwarp: None = выкл; строка = ленивый model_id/папка; иначе готовый инстанс.
        self._unwarp_arg = unwarp
        self._unwarp = unwarp if (unwarp is not None and not isinstance(unwarp, str)) else None
        # True=readability×margin, False/None=blanket, число=readability×это, callable=свой гейт
        self.unwarp_gate = unwarp_gate
        self.unwarp_margin = float(unwarp_margin)

    def process_image(self, image_path: str) -> List[Dict]:
        """
        Обработать изображение: детекция + распознавание

        Args:
            image_path: путь к изображению

        Returns:
            Список словарей {"quad": [[x,y], ...], "text": str, "confidence": float}
        """
        # Загрузка изображения
        image = self._load_image(image_path)

        # Порядок эшелонов поворота: (1) FHT-deskew мелкого наклона -> (2) нейро 90-кратная
        # ориентация страницы -> (3) per-crop ориентация строк (внутри _ocr_page, последней).
        # FHT дешёв и правит общий наклон; 90-кратная чинит грубую ориентацию; остаточный
        # наклон после разворота добирает per-crop (снятие наклона кропа) как финальная сетка.
        if self.deskew:
            from .deskew import deskew_image
            image, _ = deskew_image(image)

        if self.orientation:
            if self._orient is None:
                from .orientation import OrientationDetector
                self._orient = OrientationDetector(num_threads=self.num_threads, gpu=self.gpu)
            image, _applied, _conf = self._orient.correct(image)

        # UVDoc unwarp (опц.): расправить изгиб перед OCR. С self-gate — берём оригинал ИЛИ дьюарп
        # по читаемости (не хуже оригинала на плоских). Без гейта — blanket-дьюарп.
        if self._unwarp_arg is not None:
            self._ensure_unwarp()
            # False/None = blanket: применяем дьюарп без гейта (1 OCR)
            if not self.unwarp_gate:
                image = self._unwarp.unwarp(image)
                return self._with_tables(image, self._ocr_page(image))
            # гейт: OCR оригинала vs дьюарпа, решаем через _use_dewarp
            res_orig = self._ocr_page(image)
            try:
                dew = self._unwarp.unwarp(image)
                res_dew = self._ocr_page(dew)
            except Exception:
                return self._with_tables(image, res_orig)
            if self._use_dewarp(res_orig, res_dew):
                return self._with_tables(dew, res_dew)
            return self._with_tables(image, res_orig)

        return self._with_tables(image, self._ocr_page(image))

    def _find_tables(self, image_rgb: np.ndarray, lines: List[Dict], scale: float = 1.0,
                     split: bool = True):
        """Таблицы на той же картинке, по которой распознаны строки (после deskew/orientation/unwarp):
        строки, склеенные детектором через границу ячеек, делятся по пробелу у границы (split=False —
        текстовый слой PDF, там строки не наши), затем раскладываются по ячейкам.
        image_rgb — RGB, модели таблиц ждут BGR. Возвращает (строки, таблицы)."""
        from .tables import attach_lines, split_crossing_lines
        found = self._tables(np.ascontiguousarray(image_rgb[:, :, ::-1]))
        if split and found:
            lines = split_crossing_lines(found, lines, image_rgb, self.recognizer)
        return lines, attach_lines(found, lines, scale=scale)

    def _with_tables(self, image: np.ndarray, results: List[Dict]) -> List[Dict]:
        if not self.tables:
            return results
        lines, tables = self._find_tables(image, results)
        return PageResult(lines, tables=tables)

    def _use_dewarp(self, res_orig: List[Dict], res_dew: List[Dict]) -> bool:
        """Решение гейта: брать ли дьюарп. callable → свой предикат; число → readability×это;
        True → readability×self.unwarp_margin (дефолт 1.05)."""
        gate = self.unwarp_gate
        if callable(gate):
            return bool(gate(res_orig, res_dew))
        margin = self.unwarp_margin if gate is True else float(gate)
        return self._readability(res_dew) > self._readability(res_orig) * margin

    def _ocr_page(self, image: np.ndarray) -> List[Dict]:
        """OCR уже препроцессенной страницы: reading_order по регионам или обычный проход."""
        if self.reading_order and self._ro is not None:
            return self._ocr_by_regions(image)
        if self.experimental_layout_order and self._layout is not None:
            return self._ocr_layout_order(image)
        return self._ocr_whole(image)

    def _ocr_layout_order(self, image: np.ndarray) -> List[Dict]:
        """ЭКСПЕРИМЕНТ: обычный OCR всей страницы (те же строки и текст), затем порядок по layout-регионам
        ТОЙ ЖЕ картинки (после deskew/orientation/unwarp) — координаты строк и регионов совпадают."""
        import logging
        from .layout_order import order_lines
        results = self._ocr_whole(image)
        layout = self._layout.detect(image)
        boxes = []
        for r in results:
            q = np.asarray(r["quad"], dtype=float)
            boxes.append([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()])
        if self._block_order is not None:
            order, line_region, stats = self._block_order.order_lines(boxes, layout.regions,
                                                                      layout.width, layout.height)
        else:
            order, line_region, stats = order_lines(boxes, layout.regions)
        for r, ri in zip(results, line_region):
            r["layout_region"] = int(ri) if ri >= 0 else None
            r["layout_label"] = layout.regions[ri].label if ri >= 0 else None
        logging.getLogger("occular.layout").info(
            "layout_order[%s]: lines=%d regions=%d coverage_80=%.3f unassigned_30=%.3f",
            self.layout_order_method, stats["lines"], stats["regions"], stats["coverage_80"], stats["unassigned_30"])
        return [results[i] for i in order]

    def _ensure_unwarp(self):
        """Ленивая инициализация UVDocUnwarper из model_id-строки при первом использовании."""
        if self._unwarp is None and isinstance(self._unwarp_arg, str):
            from .unwarp_uvdoc import UVDocUnwarper
            self._unwarp = UVDocUnwarper(model_id=self._unwarp_arg, gpu=self.gpu)

    @staticmethod
    def _readability(results: List[Dict]) -> float:
        """Читаемость страницы = Σ len(text)·confidence по строкам. Растёт, когда прочитано
        БОЛЬШЕ текста и УВЕРЕННЕЕ — ровно то, что должен максимизировать выбор оригинал/дьюарп."""
        return float(sum(len(r.get("text", "")) * float(r.get("confidence", 0.0)) for r in results))

    def _ocr_whole(self, image: np.ndarray) -> List[Dict]:
        """Обычный OCR всей страницы: детекция всех строк -> распознавание -> сортировка сверху-вниз."""
        if self.oriented_lines:
            return self._ocr_whole_oriented(image)
        quads = self.detector.detect(image)
        texts_and_confidences = self.recognizer.recognize(image, quads)
        results = [
            {"quad": quad.tolist(), "text": text, "confidence": float(conf)}
            for quad, (text, conf) in zip(quads, texts_and_confidences)
        ]
        results.sort(key=lambda r: r["quad"][0][1])
        return results

    def _ocr_whole_oriented(self, image: np.ndarray) -> List[Dict]:
        """OCR с ориентацией-по-умолчанию: детекция сохраняет угол региона, затем рекогнайзер-арбитраж
        (deskew длинной оси + выбор 0/180/vertical-stack по CTC-уверенности). Ловит наклонённые/±90/
        вертикальные строки, которые осевой crop терял. Цена: арбитраж — по 1-3 прогона рекогнайзера
        на регион (не батч), fast-path для уверенной горизонтали = 1 прогон."""
        from .orientation_resolve import resolve_region
        regions = self.detector.detect_regions(image)
        results = []
        for reg in regions:
            res = resolve_region(image, reg, self.recognizer)
            q = reg.oriented_quad if getattr(reg, "oriented_quad", None) is not None else reg.axis_aligned_quad()
            results.append({
                "quad": np.asarray(q, dtype=float).tolist(),
                "text": res.text,
                "confidence": float(res.confidence),
                "orientation_deg": res.orientation_deg,
                "layout": res.layout_type,
            })
        # сортировка сверху-вниз по верхней точке региона (как в осевом пути)
        results.sort(key=lambda r: min(p[1] for p in r["quad"]))
        return results

    def _ocr_by_regions(self, image: np.ndarray) -> List[Dict]:
        """reading_order (вариант C): детект ВСЕЙ страницы ОДИН раз (полнота), затем строки
        группируются по регионам layout В ПОРЯДКЕ ЧТЕНИЯ. Ничего не теряется, порядок правильный.
        Строки вне всех регионов (номера страниц, поля) — в конец, по Y."""
        # 1) обычный OCR всей страницы — все строки
        quads = self.detector.detect(image)
        texts = self.recognizer.recognize(image, quads)
        lines = [
            {"quad": q.tolist(), "text": t, "confidence": float(c)}
            for q, (t, c) in zip(quads, texts)
        ]
        if not lines:
            return lines
        # 2) регионы в порядке чтения
        regions = self._ro.regions_in_order(image)
        if not regions:
            lines.sort(key=lambda r: r["quad"][0][1])            # layout пусто -> просто по Y
            return lines

        # доля строки, покрытая регионом
        def covered(lb, rb):
            x1 = max(lb[0], rb[0]); y1 = max(lb[1], rb[1])
            x2 = min(lb[2], rb[2]); y2 = min(lb[3], rb[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            return inter / max(1e-6, (lb[2] - lb[0]) * (lb[3] - lb[1]))

        # 3) назначить строку региону (макс. покрытие ≥30%), непривязанные -> индекс len(regions)
        keyed = []
        for ln in lines:
            q = np.asarray(ln["quad"], np.float32)
            lb = (float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max()))
            best, best_ov = len(regions), 0.3
            for i, rb in enumerate(regions):
                ov = covered(lb, rb)
                if ov > best_ov:
                    best_ov, best = ov, i
            keyed.append((best, (lb[1] + lb[3]) / 2, ln))
        # 4) порядок: (индекс региона в порядке чтения, затем Y внутри)
        keyed.sort(key=lambda k: (k[0], k[1]))
        return [k[2] for k in keyed]

    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Загрузить изображение в формате RGB

        Args:
            image_path: путь к изображению

        Returns:
            numpy array (H, W, C) в RGB

        Raises:
            FileNotFoundError: если файла нет
            ValueError: если файл не изображение / повреждён
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {image_path}")
        try:
            with Image.open(image_path) as img:
                return np.array(img.convert('RGB'))
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(
                f"Could not open image '{image_path}': "
                f"the file is damaged or not an image ({e})"
            ) from e

    def process_pdf(self, pdf_path: str, dpi: int = 300, force_ocr: bool = False,
                    workers: Optional[int] = None) -> List[Dict]:
        """
        Обработать PDF: извлечение текста или OCR

        Args:
            pdf_path: путь к PDF файлу
            dpi: разрешение рендеринга для OCR (по умолчанию 300)
            force_ocr: принудительно использовать OCR даже если есть текстовый слой
            workers: сколько страниц распознавать параллельно (отдельными процессами, см. occular.parallel)
                     None = авто: по ядрам и свободной памяти (≈1.1 ГБ на процесс), не больше 16
                     1 = последовательно в этом процессе
                     N = N процессов (на больших машинах — до числа ядер). Процессы нужны, чтобы
                         не упираться в GIL; без `if __name__ == "__main__":` в главном скрипте
                         — потоки (не больше 8)

        Returns:
            Список словарей с результатами по страницам:
            [{"page": 1, "method": "text"|"ocr", "results": [...]}, ...]
        """
        try:
            fitz = pymupdf()
        except ImportError as e:
            # PyMuPDF — обязательная зависимость; если импорт упал, причина в нём самом (битое колесо,
            # нет системной библиотеки) — показываем её, а не совет «установите pymupdf».
            raise ImportError(f"PyMuPDF fails to import ({e}). Reinstall: "
                              f"pip install --force-reinstall \"pymupdf>=1.23.1\"") from e

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        num_pages = len(doc)

        # 1) классификация в главном процессе (дёшево, без рендера): страницы с текстовым слоем читаются
        #    сразу, остальные — в OCR. PyMuPDF не thread-safe при общем Document: OCR-страницы каждый
        #    воркер открывает и рендерит сам, ПО ОДНОЙ (иначе все пиксмапы @300 DPI разом в RAM → OOM).
        all_results = [None] * num_pages
        ocr_pages = []
        for page_num in range(num_pages):
            if not force_ocr:
                text_result = self._extract_text_from_page(doc[page_num], page_num)
                if text_result:
                    all_results[page_num] = self._tables_for_text_page(doc[page_num], text_result, dpi)
                    continue
            ocr_pages.append(page_num)
        doc.close()
        if not ocr_pages:
            return all_results

        from .parallel import (auto_workers, can_use_processes, processes_safe, run_tasks,
                               threads_per_worker, THREAD_WORKERS_MAX)
        num_workers = auto_workers(len(ocr_pages)) if workers is None else max(1, min(workers, len(ocr_pages)))
        pdf_str = str(pdf_path)

        # 2) OCR: один воркер — здесь же; иначе процессы (не упираются в GIL), а если их нельзя
        #    (см. occular.parallel.processes_safe/can_use_processes) — потоки, не больше
        #    THREAD_WORKERS_MAX: дальше потоки под GIL только мешают друг другу
        if num_workers == 1:
            for page_num in ocr_pages:
                all_results[page_num] = self._render_and_ocr_page(pdf_str, page_num, dpi)
        elif can_use_processes(self) and processes_safe():
            tpw = threads_per_worker(num_workers, self.num_threads)
            done = run_tasks(self, [("pdf", pdf_str, p, dpi, True) for p in ocr_pages], num_workers, tpw)
            for page_num, res in zip(ocr_pages, done):
                if isinstance(res, Exception):
                    raise res
                all_results[page_num] = res
        else:
            with ThreadPoolExecutor(max_workers=min(num_workers, THREAD_WORKERS_MAX)) as executor:
                futures = {
                    executor.submit(self._render_and_ocr_page, pdf_str, page_num, dpi): page_num
                    for page_num in ocr_pages
                }
                for future in as_completed(futures):
                    all_results[futures[future]] = future.result()
        return all_results

    def _render_and_ocr_page(self, pdf_path: str, page_num: int, dpi: int) -> Dict:
        """Открыть свой хендл документа, отрендерить ОДНУ страницу и распознать (для параллельного пути).
        Свой fitz.Document на задачу = потокобезопасно; в RAM живёт максимум num_workers пиксмапов."""
        fitz = pymupdf()
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_num]
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            image = image[:, :, :3].copy() if pix.n == 4 else image.copy()
        finally:
            doc.close()
        return self._ocr_image(image, page_num)

    def _extract_text_from_page(self, page, page_num: int) -> Optional[Dict]:
        """Извлечь текст из векторной страницы PDF"""
        text_blocks = page.get_text("dict")["blocks"]
        text_results = []

        for block in text_blocks:
            if block["type"] == 0:  # text block
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")

                    if line_text.strip():
                        bbox = line["bbox"]
                        quad = [
                            [bbox[0], bbox[1]],
                            [bbox[2], bbox[1]],
                            [bbox[2], bbox[3]],
                            [bbox[0], bbox[3]]
                        ]
                        text_results.append({
                            "quad": quad,
                            "text": line_text.strip(),
                            "confidence": 1.0
                        })

        if text_results:
            # Сортируем по Y (сверху вниз)
            text_results.sort(key=lambda r: r["quad"][0][1])
            return {
                "page": page_num + 1,
                "method": "text",
                "results": text_results
            }
        return None

    def _ocr_image(self, image: np.ndarray, page_num: int) -> Dict:
        """OCR для одного изображения страницы (deskew -> reading_order/обычный)."""
        if self.deskew:
            from .deskew import deskew_image
            image, _ = deskew_image(image)
        if self.reading_order and self._ro is not None:
            page_results = self._ocr_by_regions(image)
        elif self.experimental_layout_order and self._layout is not None:
            page_results = self._ocr_layout_order(image)
        else:
            page_results = self._ocr_whole(image)
        out = {
            "page": page_num + 1,
            "method": "ocr",
            "results": page_results
        }
        if self.tables:
            out["results"], out["tables"] = self._find_tables(image, page_results)
        return out

    def _tables_for_text_page(self, page, result: Dict, dpi: int) -> Dict:
        """Текстовый слой PDF: строки в точках PDF, таблицы ищутся на отрендеренной странице и
        переводятся в точки (пиксели / zoom)."""
        if not self.tables:
            return result
        fitz = pymupdf()
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        image = image[:, :, :3].copy() if pix.n == 4 else image.copy()
        _, result["tables"] = self._find_tables(image, result["results"], scale=1.0 / zoom, split=False)
        return result

    def _process_pdf_page(self, doc, page_num: int, dpi: int, force_ocr: bool) -> Dict:
        """Обработать одну страницу PDF (последовательный режим)"""
        page = doc[page_num]

        # Пробуем извлечь текст
        if not force_ocr:
            result = self._extract_text_from_page(page, page_num)
            if result:
                return self._tables_for_text_page(page, result, dpi)

        # OCR
        zoom = dpi / 72
        fitz = pymupdf()
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # np.frombuffer разделяет память с pix.samples; .copy() — чтобы image владел данными,
        # иначе после GC pix буфер может освободиться под живым image.
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        image = image[:, :, :3].copy() if pix.n == 4 else image.copy()

        return self._ocr_image(image, page_num)
