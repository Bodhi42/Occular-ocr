"""Опциональный нейросетевой UNWARP — расправление ИЗГИБА сфотографированных/смятых страниц (UVDoc).

Отличие от occular.deskew: deskew правит только НАКЛОН скана (поворот, Hough, без весов).
UVDoc правит геометрический ИЗГИБ страницы (фото телефоном, сгиб, волна) через нейросетевую
2D-сетку деформации (grid-based document unwarping). Тяжёлый, поэтому ОПЦИОНАЛЬНЫЙ и НЕ входит
в дефолтный пайплайн — вызывается явно как препроцесс.

Реализация самодостаточна: сеть портирована в occular._uvdoc_model (чистый torch), веса грузятся
напрямую из .safetensors. Зависимости минимальны — БЕЗ transformers и torchvision:
    pip install "occular-ocr[unwarp]"      # тянет torch + safetensors + huggingface_hub
Веса UVDoc качаются с Hugging Face (model_id) или берутся из локальной папки; общепринятого
дефолта в пакете нет — model_id задаётся явно.

Использование:
    from occular.unwarp_uvdoc import UVDocUnwarper
    uw = UVDocUnwarper(model_id="<hf-repo-или-локальная-папка-с-весами-uvdoc>", device="cuda")
    flat_bgr = uw.unwarp(image_bgr)        # np.ndarray BGR -> BGR (размер сохраняется)
"""
from __future__ import annotations
import os
import json
import numpy as np

# вход модели (из preprocessor_config UVDoc); переопределяется значением из чекпойнта, если оно есть
_DEFAULT_IN_H, _DEFAULT_IN_W = 712, 488


class UVDocUnwarper:
    """Обёртка над вендор-сетью UVDoc. torch/safetensors/huggingface_hub импортируются при создании."""

    def __init__(self, model_id: str = None, device: str = "cpu"):
        if not model_id:
            raise ValueError(
                "UVDocUnwarper: укажи model_id = HF-репозиторий или локальная папка с весами UVDoc "
                "(config.json + model.safetensors). Вшитого дефолта в occular нет — веса опциональны."
            )
        try:
            import torch  # noqa
            from safetensors.torch import load_file
        except Exception as e:  # noqa
            raise ImportError(
                "UVDoc требует torch + safetensors (+ huggingface_hub для скачивания весов). "
                "Установи опциональный extra: pip install 'occular-ocr[unwarp]'"
            ) from e
        from ._uvdoc_model import UVDocNet

        self._torch = torch
        self.device = device

        cfg_path, w_path, prep_path = self._resolve_files(model_id)
        cfg = json.load(open(cfg_path))
        cfg.setdefault("kernel_size", 5)
        cfg.setdefault("padding_mode", "reflect")
        cfg.setdefault("hidden_act", "prelu")

        # целевой размер входа — из preprocessor_config (если есть), иначе дефолт UVDoc
        self.in_h, self.in_w = _DEFAULT_IN_H, _DEFAULT_IN_W
        if prep_path and os.path.exists(prep_path):
            try:
                size = json.load(open(prep_path)).get("size", {})
                self.in_h = int(size.get("height", self.in_h))
                self.in_w = int(size.get("width", self.in_w))
            except Exception:
                pass

        self.net = UVDocNet(cfg).eval().to(device)
        self.net.load_state_dict(load_file(w_path), strict=True)

    @staticmethod
    def _resolve_files(model_id: str):
        """Локальная папка -> используем файлы из неё; иначе -> скачиваем с Hugging Face."""
        if os.path.isdir(model_id):
            cfg = os.path.join(model_id, "config.json")
            wt = os.path.join(model_id, "model.safetensors")
            prep = os.path.join(model_id, "preprocessor_config.json")
            if not (os.path.exists(cfg) and os.path.exists(wt)):
                raise FileNotFoundError(
                    f"В папке {model_id} нет config.json и/или model.safetensors."
                )
            return cfg, wt, prep
        try:
            from huggingface_hub import hf_hub_download
        except Exception as e:  # noqa
            raise ImportError(
                "Для скачивания весов UVDoc с Hugging Face нужен huggingface_hub "
                "(входит в extra: pip install 'occular-ocr[unwarp]'), либо укажи локальную папку."
            ) from e
        cfg = hf_hub_download(model_id, "config.json")
        wt = hf_hub_download(model_id, "model.safetensors")
        try:
            prep = hf_hub_download(model_id, "preprocessor_config.json")
        except Exception:
            prep = None
        return cfg, wt, prep

    def unwarp(self, image_bgr: np.ndarray) -> np.ndarray:
        """Расправить изгиб страницы. Вход/выход — np.ndarray BGR (как cv2.imread); размер сохраняется.

        Пайплайн (совпадает с референс-реализацией UVDoc бит-в-бит):
          BGR/255 -> resize(in_h,in_w, bilinear align_corners) -> сеть -> mesh(1,2,h',w')
          -> upscale mesh до (H,W) -> grid_sample по BGR-оригиналу -> BGR uint8.
        """
        torch = self._torch
        if image_bgr is None or image_bgr.ndim != 3:
            return image_bgr
        H, W = image_bgr.shape[:2]
        # BGR [0,1], CHW, batched — cv2 уже даёт BGR (модель обучена на BGR)
        orig = (torch.from_numpy(np.ascontiguousarray(image_bgr)).to(self.device)
                .permute(2, 0, 1).float().div(255.0).unsqueeze(0))
        with torch.no_grad():
            pixel_values = torch.nn.functional.interpolate(
                orig, size=(self.in_h, self.in_w), mode="bilinear", align_corners=True)
            mesh = self.net(pixel_values)                      # (1,2,h',w') в [-1,1]
            up = torch.nn.functional.interpolate(
                mesh, size=(H, W), mode="bilinear", align_corners=True)
            grid = up.permute(0, 2, 3, 1)                      # (1,H,W,2)
            rect = torch.nn.functional.grid_sample(orig, grid, align_corners=True)
        out = (rect[0].permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8)
        return out.cpu().numpy()                               # HWC uint8 BGR
