"""GPU-бэкенд на PyTorch/CUDA. Грузится ЛЕНИВО только при gpu=True.
Причина существования: onnxruntime-gpu капризен к версиям CUDA/cuDNN; нативный torch на CUDA надёжнее.
Модели те же (веса .pth, из которых экспортировались ONNX) → числовой паритет с CPU-ONNX путём.

Веса .pth ищутся: сначала в папке из env OCCULAR_TORCH_DIR (локальный оверрайд),
иначе качаются с HuggingFace (тот же репо, что и ONNX)."""
import os
import warnings
from pathlib import Path

# torch-веса распознавателя по архитектуре (для GPU-пути; на CPU исполняется ONNX)
REC_PTH = "recognizer_svtr_fp32.pth"              # svtr_t, ru/en (историческое имя)
REC_PTH_BY_ARCH = {
    "svtr_t":     {"ru": "recognizer_svtr_fp32.pth"},
    "svtr_lcnet": {"ru": "recognizer_svtr_lcnet_fp32.pth",
                   "cyr": "recognizer_svtr_lcnet_cyr12_fp32.pth"},
}
DET_PTH = "detector_dbnet_fp32.pth"


def cuda_available() -> bool:
    """torch установлен И видна CUDA-карта."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def torch_missing_reason() -> str:
    try:
        import torch  # noqa
    except Exception:
        return "не установлен PyTorch (pip install occular-ocr[gpu])"
    try:
        import torch
        if not torch.cuda.is_available():
            return "PyTorch есть, но CUDA недоступна (нет GPU-драйвера/сборки CUDA)"
    except Exception as e:
        return f"ошибка инициализации CUDA: {e}"
    return ""


def _weight(rel: str) -> str:
    """Путь к .pth: сначала OCCULAR_TORCH_DIR, иначе с HuggingFace (как ensure_weight)."""
    d = os.environ.get("OCCULAR_TORCH_DIR")
    if d:
        p = Path(d) / rel
        if not p.exists():
            raise FileNotFoundError(f"OCCULAR_TORCH_DIR задан, но нет {p}")
        return str(p)
    from .model_files import ensure_weight
    return ensure_weight(rel)


def _safe_load(path):
    """Загрузка .pth с weights_only=True (безопасно для недоверенных checkpoint'ов);
    фолбэк на обычную загрузку только для torch < 1.13, где аргумента ещё нет."""
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_recognizer(nclass: int, device="cuda", arch: str = "svtr_t", family: str = "ru"):
    """Распознаватель на CUDA (fp32, eval). nclass = len(charset)+1 (blank).

    arch: 'svtr_lcnet' (лёгкая, по умолчанию в библиотеке) или 'svtr_t' (крупная).
    family: 'ru' (ru/en) или 'cyr' (12 кир-языков)."""
    try:
        rel = REC_PTH_BY_ARCH[arch][family]
    except KeyError:
        raise ValueError(f"нет torch-весов для arch={arch!r}, family={family!r}; "
                         f"доступно: {REC_PTH_BY_ARCH}")
    if arch == "svtr_lcnet":
        from ._svtr_lcnet import build_svtr_lcnet
        m = build_svtr_lcnet(nclass)
    else:
        from ._svtr import build_svtr
        m = build_svtr(nclass, "svtr_t", 48, 640, "none")
    sd = _safe_load(_weight(rel))
    m.load_state_dict(sd["model"] if "model" in sd else sd)
    return m.to(device).eval()


def load_detector(device="cuda"):
    """DBNet(resnet50, db, inner=256) на CUDA (fp32, eval)."""
    from ._dbnet_model import DBNet
    m = DBNet(backbone="resnet50", head="db", inner=256)
    sd = _safe_load(_weight(DET_PTH))
    m.load_state_dict(sd["model"] if "model" in sd else sd)
    return m.to(device).eval()


def infer(model, batch_np, device="cuda"):
    """batch numpy [B,C,H,W] fp32 -> выход модели numpy (fp32, no_grad). Без autocast — паритет с ONNX FP32."""
    import torch
    with torch.no_grad():
        xb = torch.from_numpy(batch_np).to(device).float()
        out = model(xb)
    return out
