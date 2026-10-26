"""Опциональный нейросетевой UNWARP — расправление ИЗГИБА сфотографированных/смятых страниц (UVDoc).

Отличие от occular.deskew: deskew правит только НАКЛОН скана (поворот, Hough, без весов).
UVDoc правит геометрический ИЗГИБ страницы (фото телефоном, сгиб, волна) через нейросетевую
2D-сетку деформации (grid-based document unwarping). Тяжёлый, поэтому ОПЦИОНАЛЬНЫЙ и НЕ входит
в дефолтный пайплайн — вызывается явно как препроцесс.

Исполняется через onnxruntime (CPU или GPU), отдельных зависимостей не нужно. Один граф делает всё:
уменьшение до входа сети → сеть → растягивание сетки деформации до размера страницы → grid_sample
по оригиналу (экспорт: tools/export/export_uvdoc_onnx.py). Веса — файл model.onnx в репозитории
Hugging Face (model_id) или в локальной папке; дефолта в пакете нет — model_id задаётся явно.

Использование:
    from occular.unwarp_uvdoc import UVDocUnwarper
    uw = UVDocUnwarper(model_id="Shivin11/occular-uvdoc", gpu=True)
    flat_bgr = uw.unwarp(image_bgr)        # np.ndarray BGR -> BGR (размер сохраняется)
"""
from __future__ import annotations
import os
import numpy as np

ONNX_NAME = "model.onnx"


class UVDocUnwarper:
    """Обёртка над ONNX-графом UVDoc."""

    def __init__(self, model_id: str = None, gpu: bool = False, num_threads: int = 4, device: str = None):
        if not model_id:
            raise ValueError(
                "UVDocUnwarper: pass model_id = a Hugging Face repo or a local folder with the UVDoc model.onnx "
                "(or the path to the .onnx itself). occular has no built-in default: the weights are optional."
            )
        if device is not None:                          # совместимость со старым API (device="cuda"/"cpu")
            gpu = str(device).startswith("cuda")
        from ._runtime import make_session
        self.session = make_session(self._resolve(model_id), num_threads, gpu)

    @staticmethod
    def _resolve(model_id: str) -> str:
        """Путь к .onnx: файл, локальная папка с model.onnx или репозиторий Hugging Face."""
        if os.path.isfile(model_id):
            return model_id
        if os.path.isdir(model_id):
            p = os.path.join(model_id, ONNX_NAME)
            if not os.path.exists(p):
                raise FileNotFoundError(f"No {ONNX_NAME} in the folder {model_id}.")
            return p
        from huggingface_hub import hf_hub_download
        return hf_hub_download(model_id, ONNX_NAME)

    def unwarp(self, image_bgr: np.ndarray) -> np.ndarray:
        """Расправить изгиб страницы. Вход/выход — np.ndarray BGR (как cv2.imread); размер сохраняется."""
        import cv2
        if image_bgr is None or image_bgr.ndim not in (2, 3):
            return image_bgr
        # приводим к 3-канальному BGR: серые -> BGR, BGRA -> BGR
        if image_bgr.ndim == 2 or image_bgr.shape[2] == 1:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
        elif image_bgr.shape[2] != 3:
            return image_bgr
        x = (np.ascontiguousarray(image_bgr).transpose(2, 0, 1)[None].astype(np.float32) / np.float32(255.0))
        rect = self.session.run(None, {"image": x})[0][0]       # (3,H,W) в [0,1]
        return (np.clip(rect * np.float32(255.0), 0, 255)).astype(np.uint8).transpose(1, 2, 0).copy()
