"""Ориентация страницы: определяет поворот на 0/90/180/270° и выпрямляет скан.

По умолчанию ВЫКЛЮЧЕНА (`Settings(orientation=True)` включает). Нужна там, где сканы
приходят повёрнутыми: телефонные фото документов, пакетная оцифровка, архивы.

Как работает: изображение приводится к квадрату 320 (letterbox, белая заливка),
прогоняется четырьмя поворотами (орбита C4), ответы усредняются со сдвигом. Такая
агрегация — часть модели: она снимает ~40% ошибок одиночного прогона.

Порог уверенности. Ниже `min_confidence` поворот НЕ применяется: замер показал, что
0.8 отделяет верные ответы от ошибочных на документах, а на не-документах (логотипы,
фотографии сцен, коллажи) модель бывает уверенно неправа — там лучше ничего не трогать.
"""
import sys
import threading
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

SIZE = 320                      # вход модели (квадрат после letterbox)
ANGLES = (0, 90, 180, 270)      # класс k -> изображение повёрнуто на ANGLES[k] по часовой
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
MIN_CONFIDENCE = 0.8

_IDX = (np.arange(4).reshape(4, 1) + np.arange(4).reshape(1, 4)) % 4


class OrientationDetector:
    """OriNet (ONNX): класс поворота + уверенность. Модель ~4.8 МБ, ~28 мс на 8 потоках."""

    def __init__(self, num_threads: int = 4, min_confidence: float = MIN_CONFIDENCE):
        import onnxruntime as ort
        from .model_files import ensure_weight

        self.min_confidence = float(min_confidence)
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(num_threads))
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(ensure_weight("orientation_orinet_fp32.onnx"),
                                            sess_options=so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self._lock = threading.Lock()
        print(f"Loaded orientation model (CPU, threads={so.intra_op_num_threads})", file=sys.stderr)

    # --- препроцессинг: letterbox в квадрат, белая заливка, норма ImageNet ---
    @staticmethod
    def _letterbox(image: np.ndarray) -> np.ndarray:
        """Ресайз строго через PIL BILINEAR: контракт модели задан на нём, а у cv2.INTER_LINEAR
        другая реализация — на сверке с эталоном это переворачивало одну страницу из 160."""
        im = Image.fromarray(image)
        w, h = im.size
        if w >= h:
            nw, nh = SIZE, max(1, round(h * SIZE / w))
        else:
            nw, nh = max(1, round(w * SIZE / h)), SIZE
        canvas = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
        canvas.paste(im.resize((nw, nh), Image.BILINEAR), ((SIZE - nw) // 2, (SIZE - nh) // 2))
        return np.asarray(canvas)

    @classmethod
    def _prep(cls, image: np.ndarray) -> np.ndarray:
        x = cls._letterbox(image).astype(np.float32).transpose(2, 0, 1) / 255.0
        return (x - MEAN) / STD

    @staticmethod
    def _rotate(image: np.ndarray, k: int) -> np.ndarray:
        """Поворот на ANGLES[k] по часовой (k-й элемент орбиты)."""
        if k == 0:
            return image
        return cv2.rotate(image, {1: cv2.ROTATE_90_CLOCKWISE,
                                  2: cv2.ROTATE_180,
                                  3: cv2.ROTATE_90_COUNTERCLOCKWISE}[k])

    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        """-> (класс поворота 0..3, уверенность 0..1). Класс k = изображение повёрнуто на ANGLES[k]."""
        batch = np.stack([self._prep(self._rotate(image, k)) for k in range(4)])
        with self._lock:                      # один сеанс ORT на несколько PDF-воркеров
            logits = self.session.run(None, {self.input_name: batch})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        agg = np.take_along_axis(probs, _IDX, axis=1).mean(axis=0)   # сдвиг ответа k-го поворота на -k
        k = int(agg.argmax())
        return k, float(agg[k])

    def correct(self, image: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """Выпрямить страницу. -> (изображение, применённый поворот в градусах, уверенность).

        При уверенности ниже порога изображение возвращается без изменений.
        """
        k, conf = self.predict(image)
        if k == 0 or conf < self.min_confidence:
            return image, 0, conf
        # изображение повёрнуто на ANGLES[k] по часовой -> компенсируем обратным поворотом
        return self._rotate(image, (4 - k) % 4), ANGLES[k], conf
