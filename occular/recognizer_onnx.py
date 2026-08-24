"""Text recognizer (ONNX Runtime).

Компактный распознаватель строк: вход H48, норма [-1,1], BGR (cv2), выход [B,T,186], CTC blank=0.
Имя класса оставлено для совместимости с реестром.
"""

import threading
import warnings
import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Tuple
from pathlib import Path
from itertools import groupby


INPUT_HEIGHT = 48       # высота входа
MAX_WIDTH = 640         # var-width cap


class CRNNRecognizerONNX:
    """Text recognizer via ONNX Runtime (var-width, blank=0)."""

    def __init__(self, languages: List[str] = None, num_threads: int = 4, gpu: bool = False,
                 lm: bool = True, onnx_file: str = "recognizer_svtr_fp32.onnx",
                 charset_file: str = "recognizer_charset.txt"):
        self.languages = languages or ['ru', 'en']
        self.lm = bool(lm)          # beam-CTC + языковая модель (по умолчанию ВКЛ)
        self._onnx_file = onnx_file  # вариант модели: русская (дефолт) или кириллическая-12
        self._lm_decoder = None
        self._lm_lock = threading.Lock()   # ленивая инициализация LM — потокобезопасно (общий инстанс у PDF-воркеров)
        self.session = None
        self._torch = None          # torch-модель на CUDA, если gpu=True и бэкенд доступен
        self._device = None

        from .model_files import ensure_weight
        charset_path = Path(ensure_weight(charset_file))
        # Charset: индекс 0 = CTC blank; символ для класса k = vocab[k-1] (как rec_data.Charset)
        self.vocab = charset_path.read_text(encoding='utf-8').split('\n')
        try: cv2.setNumThreads(max(1, int(num_threads)))  # OpenCV по умолч. берёт все ядра
        except Exception: pass

        # GPU-путь: нативный torch на CUDA (веса .pth). Надёжнее onnxruntime-gpu.
        if gpu:
            from ._torch_backend import cuda_available, torch_missing_reason, load_recognizer
            if cuda_available():
                self._torch = load_recognizer(len(self.vocab) + 1, "cuda"); self._device = "cuda"
                print(f"Loaded SVTR recognizer (vocab={len(self.vocab)} chars, GPU/CUDA, PyTorch)")
                return
            warnings.warn(f"gpu=True, но GPU-бэкенд недоступен: {torch_missing_reason()}. "
                          f"Откат на CPU (ONNX).")

        # CPU-путь (по умолчанию и как фолбэк): ONNX Runtime, только CPU.
        onnx_path = Path(ensure_weight(self._onnx_file))
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = max(1, int(num_threads))
        sess_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx_path), sess_options=sess_options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name  # 'x'
        print(f"Loaded SVTR recognizer (vocab={len(self.vocab)} chars, CPU, threads={sess_options.intra_op_num_threads})")

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        """batch [B,3,48,W] fp32 -> логиты [B,T,C]. torch/CUDA если включён, иначе ONNX/CPU."""
        if self._torch is not None:
            from ._torch_backend import infer
            return infer(self._torch, batch, self._device).cpu().numpy()
        return self.session.run(None, {self.input_name: batch})[0]

    def _ensure_lm(self):
        """Лениво построить beam+LM-декодер при первом реальном распознавании.
        LM (~268 МБ + униграммы) НЕ грузится, если распознавание так и не понадобилось
        (например, чисто-векторный PDF). Потокобезопасно."""
        if self._lm_decoder is not None:
            return self._lm_decoder
        with self._lm_lock:
            if self._lm_decoder is None:
                # при первом создании качает LM с HF / берёт OCCULAR_LM_DIR
                from .decoder_lm import LMDecoder
                self._lm_decoder = LMDecoder(self.vocab)
        return self._lm_decoder

    def _decode(self, logits_1tc: np.ndarray) -> Tuple[str, float]:
        """beam+LM если включён, иначе greedy. LM строится лениво при первом вызове."""
        if self.lm:
            return self._ensure_lm().decode(logits_1tc)
        return self._ctc_decode(logits_1tc)

    def logits_per_line(self, image: np.ndarray, quads: List[np.ndarray]) -> List[np.ndarray]:
        """Логиты по каждой строке (batch-инференс с бакетингом по ширине), выровнено к quads.
        per_line[i] = [1,T,C] или None для пустого/нулевого кропа. Декод оставлен вызывающему —
        так мультиязычный роутер может декодить каждую строку своей языковой моделью."""
        per_line = [None] * len(quads)
        preprocessed, valid_indices = [], []
        for i, quad in enumerate(quads):
            crop = self._crop_quad(image, quad)
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            preprocessed.append(self._preprocess_single(crop))
            valid_indices.append(i)
        if not preprocessed:
            return per_line
        # Сортируем по ширине → минимум паддинга в бакете
        width_order = sorted(range(len(preprocessed)), key=lambda i: preprocessed[i].shape[2])
        BUCKET_SIZE = 8
        for start in range(0, len(width_order), BUCKET_SIZE):
            bucket = width_order[start:start + BUCKET_SIZE]
            batch = self._create_batch([preprocessed[i] for i in bucket])
            logits = self._infer(batch)
            for bi, prep_idx in enumerate(bucket):
                per_line[valid_indices[prep_idx]] = logits[bi:bi + 1]
        return per_line

    def recognize(self, image: np.ndarray, quads: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Recognize text in given regions (batch inference with width bucketing)."""
        if not quads:
            return []
        per_line = self.logits_per_line(image, quads)
        valid = [i for i, lg in enumerate(per_line) if lg is not None]
        results = [("", 0.0)] * len(quads)
        if not valid:
            return results
        # Инференс всех строк уже сделан; декод всей страницы одним вызовом. Нативный декодер
        # считает строки параллельно (GIL отпущен); на чистом Python это тот же цикл.
        if self.lm:
            decoded = self._ensure_lm().decode_many([per_line[i] for i in valid])
        else:
            decoded = [self._ctc_decode(per_line[i]) for i in valid]
        for i, dec in zip(valid, decoded):
            results[i] = dec
        return results

    def _preprocess_single(self, image: np.ndarray) -> np.ndarray:
        """RGB-кроп (из пайплайна) → BGR (модель училась на cv2-BGR), ресайз H48 var-cap640, норма [-1,1]."""
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        new_w = max(8, int(round(w * INPUT_HEIGHT / h)))
        new_w = min(new_w, MAX_WIDTH)
        img = cv2.resize(img, (new_w, INPUT_HEIGHT))
        x = ((img.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)  # [-1,1], CHW
        return x

    def _create_batch(self, tensors: List[np.ndarray]) -> np.ndarray:
        """Паддинг по ширине до max-в-батче значением 1.0 (белый = (255/255-0.5)/0.5)."""
        max_w = max(t.shape[2] for t in tensors)
        batch = []
        for t in tensors:
            c, h, w = t.shape
            if w < max_w:
                padded = np.full((c, h, max_w), 1.0, dtype=np.float32)
                padded[:, :, :w] = t
                batch.append(padded)
            else:
                batch.append(t)
        return np.stack(batch, axis=0)

    def _crop_quad(self, image: np.ndarray, quad: np.ndarray) -> np.ndarray:
        quad = np.array(quad)
        x_min, x_max = int(quad[:, 0].min()), int(quad[:, 0].max())
        y_min, y_max = int(quad[:, 1].min()), int(quad[:, 1].max())
        h, w = image.shape[:2]
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        return image[y_min:y_max, x_min:x_max]

    def _ctc_decode(self, logits: np.ndarray) -> Tuple[str, float]:
        """CTC best-path: argmax → схлопнуть повторы → убрать blank(0). Символ k → vocab[k-1]."""
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)

        pred_indices = logits.argmax(axis=-1)[0]  # (T,)
        # confidence = геом. среднее max-prob по emitting-кадрам (не blank): убирает blank-инфляцию
        # и смещение по длине старого min-по-кадрам (AUC 0.91 vs 0.90).
        maxp = probs.max(axis=-1)[0]              # (T,)
        nb = pred_indices != 0
        if nb.any():
            confidence = float(np.exp(np.log(np.clip(maxp[nb], 1e-6, 1.0)).mean()))
        else:
            confidence = float(maxp.mean())

        chars = []
        for k, _ in groupby(pred_indices.tolist()):
            if k != 0 and 0 <= k - 1 < len(self.vocab):
                chars.append(self.vocab[k - 1])
        return ''.join(chars), confidence
