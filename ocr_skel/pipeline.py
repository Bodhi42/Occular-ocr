"""OCR Pipeline: detect -> recognize"""

import numpy as np
from typing import List, Dict
from PIL import Image
import cv2

from .registry import Registry


class OCRPipeline:
    """Pipeline для OCR: детекция + распознавание"""

    def __init__(self, detector_name: str = None, recognizer_name: str = None,
                 detector_kwargs: dict = None, recognizer_kwargs: dict = None):
        """
        Args:
            detector_name: имя детектора из реестра (по умолчанию 'craft')
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
