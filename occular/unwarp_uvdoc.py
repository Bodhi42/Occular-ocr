"""Опциональный нейросетевой UNWARP — расправление ИЗГИБА сфотографированных/смятых страниц (UVDoc).

Отличие от occular.deskew: deskew правит только НАКЛОН скана (поворот, Hough, без весов).
UVDoc правит геометрический ИЗГИБ страницы (фото телефоном, сгиб, волна) через нейросетевую
2D-сетку деформации (grid-based document unwarping). Тяжёлый, поэтому ОПЦИОНАЛЬНЫЙ и НЕ входит
в дефолтный пайплайн — вызывается явно как препроцесс.

Зависимости (опциональный extra):
    pip install "occular-ocr[unwarp]"      # тянет torch + transformers (UVDoc появился в transformers>=5.17)
Плюс нужен чекпойнт UVDoc с Hugging Face — общепринятого дефолта в пакете НЕТ, задаётся model_id.

Использование:
    from occular.unwarp_uvdoc import UVDocUnwarper
    uw = UVDocUnwarper(model_id="<hf-repo-с-весами-uvdoc>")   # грузится один раз
    flat_bgr = uw.unwarp(image_bgr)                            # np.ndarray BGR -> BGR

⚠️ ЭКСПЕРИМЕНТАЛЬНО: интеграция написана по API transformers UVDoc (UVDocModel +
UVDocImageProcessor.post_process_document_rectification), локально без весов не прогонялась —
проверить на реальном чекпойнте перед продакшеном.
"""
from __future__ import annotations
import numpy as np


class UVDocUnwarper:
    """Ленивая обёртка над UVDoc из transformers. torch/transformers импортируются только при создании."""

    def __init__(self, model_id: str = None, device: str = "cpu"):
        if not model_id:
            raise ValueError(
                "UVDocUnwarper: укажи model_id = HF-репозиторий с весами UVDoc "
                "(в occular нет вшитого дефолта — веса опциональны и качаются отдельно)."
            )
        try:
            import torch  # noqa
            from transformers import UVDocModel, UVDocImageProcessor  # transformers>=5.17
        except Exception as e:  # noqa
            raise ImportError(
                "UVDoc требует torch + transformers>=5.17. Установи опциональный extra: "
                "pip install 'occular-ocr[unwarp]'"
            ) from e
        self._torch = torch
        self.device = device
        self.processor = UVDocImageProcessor.from_pretrained(model_id)
        self.model = UVDocModel.from_pretrained(model_id).to(device).eval()

    def unwarp(self, image_bgr: np.ndarray) -> np.ndarray:
        """Расправить изгиб страницы. Вход/выход — np.ndarray BGR (как cv2.imread)."""
        import cv2
        torch = self._torch
        if image_bgr is None or image_bgr.ndim != 3:
            return image_bgr
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=[rgb], return_tensors="pt").to(self.device)
        # оригинал для warp: CHW, RGB, [0,1] (post_process домножит на scale=255 и вернёт BGR uint8)
        orig = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
        with torch.no_grad():
            out = self.model(inputs["pixel_values"])
            mesh = getattr(out, "last_hidden_state", None)
            if mesh is None:                                   # на случай иного контракта выхода
                mesh = out[0] if isinstance(out, (tuple, list)) else out
            rectified = self.processor.post_process_document_rectification(mesh, [orig], scale=255.0)
        return rectified[0]["images"].cpu().numpy()            # HWC uint8 BGR
