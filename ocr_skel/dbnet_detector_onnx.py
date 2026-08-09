"""Text detector (ONNX Runtime), ResNet50 backbone.

Вход 'x', RGB + ImageNet-норма, resize длинной стороной до 1280 + белый паддинг до квадрата.
Полные боксы (мало клиппинга) → надёжно на плотном тексте: и русские документы (180+ строк),
и мелкий латинский текст (40+ строк).
"""

import numpy as np
import onnxruntime as ort
import cv2
from typing import List
from pathlib import Path
import pyclipper
from shapely.geometry import Polygon


VAL = 1280                                                  # длинная сторона (как на валидации)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# CANON нашего детектора (canon-metric.md — не менять без записи в вики)
BIN_T = 0.3          # бинаризация prob-карты
SCORE_T = 0.5        # средний prob внутри контура
MIN_SIZE = 3
UNCLIP = 2.0
MAX_CAND = 500


class DBNetDetectorONNX:
    """Text detector using ONNX Runtime (ResNet50 backbone, RGB+ImageNet)."""

    def __init__(self, num_threads: int = 4, gpu: bool = False):
        from .model_files import ensure_weight
        onnx_path = Path(ensure_weight("detector_dbnet_fp32.onnx"))   # локально или с HF

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # FIX v0.2: было intra_op_num_threads=0 (=все ядра) → жрало всю CPU (× воркеры пайплайна).
        sess_options.intra_op_num_threads = max(1, int(num_threads))
        sess_options.inter_op_num_threads = 1
        try:
            cv2.setNumThreads(max(1, int(num_threads)))     # OpenCV тоже по умолч. берёт все ядра
        except Exception:
            pass

        from ._runtime import resolve_providers
        providers = resolve_providers(gpu)     # CPU по умолчанию; CUDA при gpu=True (нужен onnxruntime-gpu)
        self.session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name  # 'x'

        dev = "GPU/CUDA" if providers[0].startswith("CUDA") else "CPU"
        print(f"Loaded DBNet detector ({dev}, threads={sess_options.intra_op_num_threads})")

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect text regions. image — RGB (как отдаёт пайплайн; наш детектор училс на RGB)."""
        orig_h, orig_w = image.shape[:2]
        input_tensor, scale = self._preprocess(image)
        prob_map = self.session.run(None, {self.input_name: input_tensor})[0][0]  # 'prob' → (H, W)
        return self._postprocess(prob_map, scale, orig_w, orig_h)

    def _preprocess(self, image: np.ndarray):
        """Resize длинной стороной до VAL, белый паддинг до квадрата, RGB+ImageNet (как resize_pad)."""
        h, w = image.shape[:2]
        s = VAL / max(h, w)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        resized = cv2.resize(image, (nw, nh))
        canvas = np.full((VAL, VAL, 3), 255, np.uint8)
        canvas[:nh, :nw] = resized
        tensor = ((canvas.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        return tensor, s

    def _postprocess(self, prob_map, scale, orig_w, orig_h):
        """Точный порт canon_decode (постпроц, которым детектор ВАЛИДИРОВАЛСЯ → canon-F1 0.9016):
        бинаризация → отсев area/score → Vatti-unclip СЫРОГО контура → axis-aligned boundingRect.
        """
        binary = (prob_map > BIN_T).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > MAX_CAND:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_CAND]

        quads = []
        for c in contours:
            if cv2.contourArea(c) < MIN_SIZE * MIN_SIZE:
                continue

            mk = np.zeros(prob_map.shape, dtype=np.uint8)
            cv2.drawContours(mk, [c], -1, 1, -1)
            if not mk.any() or float(prob_map[mk > 0].mean()) < SCORE_T:
                continue

            pts = c.reshape(-1, 2)
            if len(pts) < 3:
                continue
            area = cv2.contourArea(c)
            perim = cv2.arcLength(c, True)
            if perim < 1:
                continue

            pco = pyclipper.PyclipperOffset()
            pco.AddPath(pts.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            ex = pco.Execute(area * UNCLIP / perim)
            if not ex:
                continue

            x, y, w, h = cv2.boundingRect(np.array(ex[0]).astype(np.int32))
            # координаты 1280-канваса → оригинал; axis-aligned квадрат (как canon_decode)
            x0, y0 = max(0.0, x / scale), max(0.0, y / scale)
            x1, y1 = min(float(orig_w), (x + w) / scale), min(float(orig_h), (y + h) / scale)
            quads.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32))

        return quads
