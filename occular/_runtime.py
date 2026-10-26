"""ONNX Runtime: создание сессий на CPU (по умолчанию) или GPU (CUDA). Все модели occular — ONNX.

GPU не требует отдельных весов — те же .onnx исполняются на CUDA через провайдер. Пакет ставит
onnxruntime-gpu (Linux/Windows; на macOS — onnxruntime), он умеет и CPU. Библиотеки CUDA/cuDNN —
системные или из pip (`pip install "occular-ocr[gpu]"`, подгружаются через onnxruntime.preload_dlls).
Если GPU запрошен, но не поднялся — одно предупреждение и работа на CPU.
"""
import sys
import onnxruntime as ort

_warned = False
_preloaded = False


def _preload_cuda_libs():
    """Подгрузить CUDA/cuDNN из pip-пакетов nvidia-* (extra [gpu]), если они есть. Один раз на процесс."""
    global _preloaded
    if _preloaded:
        return
    _preloaded = True
    preload = getattr(ort, "preload_dlls", None)       # onnxruntime-gpu >= 1.21
    import importlib.util
    # Только если CUDA/cuDNN действительно пришли из pip (пакеты nvidia-*): иначе preload_dlls
    # печатает «Failed to load libcublas…» на каждую библиотеку, а CUDA может быть системной.
    if preload is not None and importlib.util.find_spec("nvidia") is not None:
        try:
            preload()
        except Exception:
            pass                                       # нет pip-библиотек -> полагаемся на системную CUDA


def _warn_cpu_fallback(reason: str):
    global _warned
    if not _warned:
        from ._envcheck import ort_gpu_hint
        hint = ort_gpu_hint() or 'For GPU: pip install "occular-ocr[gpu]" (needs an NVIDIA driver).'
        print(f"[occular] gpu=True, but CUDA is unavailable ({reason}). Running on CPU. {hint}", file=sys.stderr)
        _warned = True


def _set_default_severity(level: int):
    try:
        ort.set_default_logger_severity(level)
    except Exception:
        pass


def resolve_providers(gpu: bool = False):
    """Список провайдеров: ['CPUExecutionProvider'] или CUDA + CPU-фолбэк для неподдержанных операций."""
    if not gpu:
        return ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        _warn_cpu_fallback("the installed onnxruntime has no CUDAExecutionProvider")
        return ['CPUExecutionProvider']
    _preload_cuda_libs()
    # TF32 выключен: иначе GPU расходится с CPU (UVDoc: 3,9 % пикселей против 0,02 %), а occular
    # обещает одинаковый результат на CPU и GPU. Цена — ~1,3× ко времени на Ampere+.
    # cudnn_conv_algo_search=HEURISTIC: EXHAUSTIVE заново перебирает алгоритмы на КАЖДУЮ новую форму входа
    # (у распознавателя ширина строки всякий раз своя), а DEFAULT на большинстве моделей уходит в медленный
    # фолбэк (svtr_lcnet ×2, детектор ×1,5). HEURISTIC — лучший или почти лучший на всех шести моделях.
    return [('CUDAExecutionProvider', {'use_tf32': '0', 'cudnn_conv_algo_search': 'HEURISTIC'}),
            'CPUExecutionProvider']


# Арена памяти CPU у ONNX Runtime растёт до пика входа и назад не отдаёт: у нас размеры входов всё время
# разные (страницы, ширина строк), и процесс после нескольких страниц держит ~1.6 ГБ вместо ~0.65.
# Без арены время страницы +4 % (1 поток) … +9 % (8 потоков). Выключаем там, где память умножается
# на число процессов, — в воркерах occular.parallel; в обычном процессе арена остаётся (скорость).
LOW_MEMORY = False


def make_session(path, num_threads: int = 4, gpu: bool = False) -> "ort.InferenceSession":
    """InferenceSession для файла модели. При gpu=True проверяет, что CUDA реально поднялась
    (onnxruntime-gpu перечисляет CUDA в провайдерах даже без библиотек CUDA и молча уходит на CPU)."""
    so = ort.SessionOptions()
    if LOW_MEMORY:
        so.enable_cpu_mem_arena = False
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = max(1, int(num_threads))
    so.inter_op_num_threads = 1
    providers = resolve_providers(gpu)
    if providers == ['CPUExecutionProvider']:
        return ort.InferenceSession(str(path), sess_options=so, providers=providers)
    # Без красной простыни ORT (ошибка загрузки CUDA-провайдера идёт в ГЛОБАЛЬНЫЙ логгер, а не в
    # логгер сессии) — причину скажем сами одной строкой. Уровень глобального логгера возвращаем.
    so.log_severity_level = 3
    _set_default_severity(4)                           # ошибка загрузки провайдера — уровня ERROR (3)
    try:
        sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
    finally:
        _set_default_severity(2)
    if 'CUDAExecutionProvider' not in sess.get_providers():
        _warn_cpu_fallback("CUDA/cuDNN libraries failed to load, or there is no GPU")
    return sess


def session_on_gpu(sess) -> bool:
    return 'CUDAExecutionProvider' in sess.get_providers()
