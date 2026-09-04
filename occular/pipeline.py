"""OCR Pipeline: detect -> recognize"""

import os
import numpy as np
from typing import List, Dict, Union, Optional
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2

from .registry import Registry


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


class OCRPipeline:
    """Pipeline для OCR: детекция + распознавание"""

    def __init__(self, detector_name: str = None, recognizer_name: str = None,
                 detector_kwargs: dict = None, recognizer_kwargs: dict = None,
                 deskew: bool = True, reading_order: bool = False, lm: bool = True,
                 num_threads: int = None, gpu: bool = False, orientation: bool = False):
        """
        Args:
            detector_name: имя детектора (по умолчанию 'dbnet-onnx')
            recognizer_name: имя распознавателя (по умолчанию 'crnn-onnx')
            detector_kwargs: параметры для детектора
            recognizer_kwargs: параметры для распознавателя
            orientation: определять поворот страницы (0/90/180/270°) и выпрямлять; ВЫКЛ по умолчанию
            num_threads: число CPU-ядер для инференса (None = min(доступные, 4))
            gpu: исполнять на GPU/CUDA (нужен PyTorch: pip install occular-ocr[gpu]; иначе фолбэк на CPU)
        """
        detector_kwargs = detector_kwargs or {}
        recognizer_kwargs = recognizer_kwargs or {}
        self.gpu = bool(gpu)

        # Число потоков: по умолчанию min(доступные ядра, 4) — чтобы не занять всю машину
        if num_threads is None:
            num_threads = min(os.cpu_count() or 1, 4)
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
        self.deskew = deskew   # выпрямление наклона скана перед детекцией (по умолчанию ВКЛ)
        # Ориентация страницы (0/90/180/270°) — ВЫКЛ по умолчанию, модель ленивая.
        self.orientation = bool(orientation)
        self._orient = None
        self.reading_order = reading_order   # порядок чтения через layout-модель (по умолчанию ВЫКЛ)
        self._ro = None
        if reading_order:
            from .reading_order import ReadingOrderModel
            self._ro = ReadingOrderModel(num_threads=self.num_threads, gpu=self.gpu)

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

        # Препроцессинг: сначала поворот на 90° (если включён), потом мелкий наклон.
        # Порядок важен: deskew правит единицы градусов и на боку работать не должен.
        if self.orientation:
            if self._orient is None:
                from .orientation import OrientationDetector
                self._orient = OrientationDetector(num_threads=self.num_threads)
            image, _applied, _conf = self._orient.correct(image)

        # Препроцессинг: выпрямление наклона (deskew), по умолчанию ВКЛ
        if self.deskew:
            from .deskew import deskew_image
            image, _ = deskew_image(image)

        # reading_order: layout делит страницу на регионы В ПОРЯДКЕ ЧТЕНИЯ (ДО детекции),
        # затем OCR каждого региона по очереди. Иначе — обычный OCR всей страницы, сверху-вниз.
        if self.reading_order and self._ro is not None:
            return self._ocr_by_regions(image)
        return self._ocr_whole(image)

    def _ocr_whole(self, image: np.ndarray) -> List[Dict]:
        """Обычный OCR всей страницы: детекция всех строк -> распознавание -> сортировка сверху-вниз."""
        quads = self.detector.detect(image)
        texts_and_confidences = self.recognizer.recognize(image, quads)
        results = [
            {"quad": quad.tolist(), "text": text, "confidence": float(conf)}
            for quad, (text, conf) in zip(quads, texts_and_confidences)
        ]
        results.sort(key=lambda r: r["quad"][0][1])
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
            raise FileNotFoundError(f"Файл не найден: {image_path}")
        try:
            with Image.open(image_path) as img:
                return np.array(img.convert('RGB'))
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(
                f"Не удалось открыть изображение '{image_path}': "
                f"файл повреждён или это не изображение ({e})"
            ) from e

    def process_pdf(self, pdf_path: str, dpi: int = 300, force_ocr: bool = False,
                    workers: Optional[int] = None) -> List[Dict]:
        """
        Обработать PDF: извлечение текста или OCR

        Args:
            pdf_path: путь к PDF файлу
            dpi: разрешение рендеринга для OCR (по умолчанию 300)
            force_ocr: принудительно использовать OCR даже если есть текстовый слой
            workers: количество воркеров для параллельной обработки
                     None = автоопределение (min(CPU, 4))
                     1 = последовательная обработка
                     N = N воркеров

        Returns:
            Список словарей с результатами по страницам:
            [{"page": 1, "method": "text"|"ocr", "results": [...]}, ...]
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF required for PDF processing. Install: pip install pymupdf")

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        num_pages = len(doc)

        # Определяем количество воркеров
        if workers is None:
            num_workers = get_optimal_workers(max_workers=4)
        else:
            num_workers = max(1, workers)

        # Для 1 страницы или 1 воркера — последовательная обработка
        if num_pages == 1 or num_workers == 1:
            all_results = []
            for page_num in range(num_pages):
                result = self._process_pdf_page(doc, page_num, dpi, force_ocr)
                all_results.append(result)
            doc.close()
            return all_results

        # Параллельная обработка.
        # PyMuPDF не thread-safe при общем Document, поэтому:
        #  1) классифицируем страницы (текст vs OCR) в главном потоке — это дёшево, без рендера;
        #  2) OCR-страницы рендерим ПО ОДНОЙ внутри воркера (каждый открывает свой хендл документа),
        #     а не держим все пиксмапы страниц в RAM разом (иначе OOM на больших сканах @300 DPI).
        all_results = [None] * num_pages
        ocr_pages = []
        for page_num in range(num_pages):
            if not force_ocr:
                text_result = self._extract_text_from_page(doc[page_num], page_num)
                if text_result:
                    all_results[page_num] = text_result
                    continue
            ocr_pages.append(page_num)
        doc.close()

        if ocr_pages:
            pdf_str = str(pdf_path)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(self._render_and_ocr_page, pdf_str, page_num, dpi): page_num
                    for page_num in ocr_pages
                }
                for future in as_completed(futures):
                    page_num = futures[future]
                    all_results[page_num] = future.result()

        return all_results

    def _render_and_ocr_page(self, pdf_path: str, page_num: int, dpi: int) -> Dict:
        """Открыть свой хендл документа, отрендерить ОДНУ страницу и распознать (для параллельного пути).
        Свой fitz.Document на задачу = потокобезопасно; в RAM живёт максимум num_workers пиксмапов."""
        import fitz
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
        else:
            page_results = self._ocr_whole(image)
        return {
            "page": page_num + 1,
            "method": "ocr",
            "results": page_results
        }

    def _process_pdf_page(self, doc, page_num: int, dpi: int, force_ocr: bool) -> Dict:
        """Обработать одну страницу PDF (последовательный режим)"""
        page = doc[page_num]

        # Пробуем извлечь текст
        if not force_ocr:
            result = self._extract_text_from_page(page, page_num)
            if result:
                return result

        # OCR
        zoom = dpi / 72
        import fitz
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # np.frombuffer разделяет память с pix.samples; .copy() — чтобы image владел данными,
        # иначе после GC pix буфер может освободиться под живым image.
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        image = image[:, :, :3].copy() if pix.n == 4 else image.copy()

        return self._ocr_image(image, page_num)
