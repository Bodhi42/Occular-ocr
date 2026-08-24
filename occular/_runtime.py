"""Выбор ONNX-провайдера: CPU по умолчанию, GPU (CUDA) опционально.

GPU не требует отдельных весов — те же .onnx исполняются на CUDA через провайдер.
Нужен пакет onnxruntime-gpu (ставится отдельно: pip install onnxruntime-gpu) и CUDA.
Если GPU запрошен, но недоступен — тихо (с одним предупреждением) откатываемся на CPU.
"""
import onnxruntime as ort

_warned = False


def resolve_providers(gpu: bool = False):
    """Список провайдеров для ort.InferenceSession.

    gpu=False -> ['CPUExecutionProvider'] (по умолчанию).
    gpu=True  -> ['CUDAExecutionProvider', 'CPUExecutionProvider'], если CUDA доступна;
                 иначе — CPU с одноразовым предупреждением.
    """
    global _warned
    if not gpu:
        return ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']   # CUDA + фолбэк на CPU для неподдерж. опов
    if not _warned:
        print("[occular] gpu=True, но CUDAExecutionProvider недоступен. "
              "Поставьте GPU-рантайм:  pip install onnxruntime-gpu  (и нужен CUDA). Пока работаю на CPU.")
        _warned = True
    return ['CPUExecutionProvider']
