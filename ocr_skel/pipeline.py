"""OCR Pipeline: detect -> recognize"""

import os
import numpy as np
from typing import List, Dict, Union, Optional
from pathlib import Path
from PIL import Image
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
                 detector_kwargs: dict = None, recognizer_kwargs: dict = None):
        """
        Args:
            detector_name: имя детектора из реестра (по умолчанию 'dbnet')
            recognizer_name: имя распознавателя из реестра (по умолчанию 'crnn')
            detector_kwargs: параметры для детектора
            recognizer_kwargs: параметры для распознавателя
        """
        detector_kwargs = detector_kwargs or {}
        recognizer_kwargs = recognizer_kwargs or {}

        self.detector = Registry.get_detector(detector_name, **detector_kwargs)
        self.recognizer = Registry.get_recognizer(recognizer_name, **recognizer_kwargs)

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

        # Детекция
        quads = self.detector.detect(image)

        # Распознавание
        texts_and_confidences = self.recognizer.recognize(image, quads)

        # Формирование результата
        results = []
        for quad, (text, confidence) in zip(quads, texts_and_confidences):
            results.append({
                "quad": quad.tolist(),  # numpy array -> list для JSON
                "text": text,
                "confidence": float(confidence)
            })

        # Сортируем по Y (сверху вниз)
        results.sort(key=lambda r: r["quad"][0][1])
        return results

    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Загрузить изображение в формате RGB

        Args:
            image_path: путь к изображению

        Returns:
            numpy array (H, W, C) в RGB
        """
        # Используем PIL для загрузки
        img = Image.open(image_path)
        img = img.convert('RGB')
        return np.array(img)

    def process_pdf(self, pdf_path: str, dpi: int = 200, force_ocr: bool = False,
                    workers: Optional[int] = None) -> List[Dict]:
        """
        Обработать PDF: извлечение текста или OCR

        Args:
            pdf_path: путь к PDF файлу
            dpi: разрешение рендеринга для OCR (по умолчанию 200)
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

        # Параллельная обработка
        # Примечание: PyMuPDF не thread-safe для одного документа,
        # поэтому рендерим страницы в главном потоке, а OCR — параллельно
        pages_data = []
        for page_num in range(num_pages):
            page = doc[page_num]

            # Пробуем извлечь текст напрямую
            if not force_ocr:
                text_result = self._extract_text_from_page(page, page_num)
                if text_result:
                    pages_data.append(("text", page_num, text_result))
                    continue

            # Рендерим страницу в изображение (в главном потоке)
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                image = image[:, :, :3].copy()  # copy() для thread-safety
            else:
                image = image.copy()
            pages_data.append(("ocr", page_num, image))

        doc.close()

        # Параллельный OCR
        all_results = [None] * num_pages

        # Сначала добавляем текстовые страницы (уже готовы)
        ocr_tasks = []
        for item in pages_data:
            if item[0] == "text":
                _, page_num, result = item
                all_results[page_num] = result
            else:
                ocr_tasks.append(item)

        # Параллельно обрабатываем OCR страницы
        if ocr_tasks:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(self._ocr_image, image, page_num): page_num
                    for _, page_num, image in ocr_tasks
                }
                for future in as_completed(futures):
                    page_num = futures[future]
                    result = future.result()
                    all_results[page_num] = result

        return all_results

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
        """OCR для одного изображения страницы"""
        quads = self.detector.detect(image)
        texts_and_confidences = self.recognizer.recognize(image, quads)

        page_results = []
        for quad, (text, confidence) in zip(quads, texts_and_confidences):
            page_results.append({
                "quad": quad.tolist(),
                "text": text,
                "confidence": float(confidence)
            })

        # Сортируем по Y (сверху вниз)
        page_results.sort(key=lambda r: r["quad"][0][1])

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

        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            image = image[:, :, :3]

        return self._ocr_image(image, page_num)

    def process(self, file_path: str, dpi: int = 200) -> Union[List[Dict], List[Dict]]:
        """
        Универсальный метод: обработать изображение или PDF

        Args:
            file_path: путь к файлу (изображение или PDF)
            dpi: разрешение для PDF (игнорируется для изображений)

        Returns:
            Для изображения: список результатов OCR
            Для PDF: список страниц с результатами
        """
        path = Path(file_path)
        if path.suffix.lower() == '.pdf':
            return self.process_pdf(file_path, dpi=dpi)
        else:
            return self.process_image(file_path)
