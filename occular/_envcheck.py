"""Проверка окружения при импорте: пакеты, которые pip ставит рядом, но которые пишут в ОДИН модуль.

onnxruntime / onnxruntime-gpu / … — все кладут файлы в `onnxruntime/`; opencv-python / -headless /
-contrib… — все в `cv2/`. Такие пары появляются сами: occular тянет onnxruntime-gpu и opencv-headless,
а chromadb, fastembed, rapidocr — onnxruntime, ultralytics — opencv-python. pip этого не видит:
последний установленный пакет перезаписывает общие файлы.

Удалять «лишний» пакет нельзя — сломается тот, кто от него зависит. Чинится переустановкой нужного
варианта поверх (`--force-reinstall --no-deps`): файлы его, метаданные обоих на месте, `pip check` чист.
Поэтому молчим, пока всё работает: одна версия onnxruntime в двух пакетах — те же файлы; потерю CUDA
сообщает _runtime при gpu=True. Предупреждаем, только если версии onnxruntime расходятся или cv2 не
импортируется.
"""
import sys
from importlib import metadata

_ORT = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-openvino", "onnxruntime-qnn")
_CV2 = ("opencv-python-headless", "opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless")


def _installed(names):
    out = []
    for n in names:
        try:
            out.append((n, metadata.version(n)))
        except metadata.PackageNotFoundError:
            pass
    return out


def reinstall_cmd(name, version):
    """Команда, возвращающая модулю файлы пакета name, не трогая остальные пакеты."""
    return f'pip install --force-reinstall --no-deps "{name}=={version}"'


def ort_gpu_hint():
    """Если CUDA пропала из-за CPU-сборки onnxruntime, поставленной поверх onnxruntime-gpu, — как вернуть."""
    ort = dict(_installed(_ORT))
    if "onnxruntime-gpu" in ort and len(ort) > 1:
        others = ", ".join(n for n in ort if n != "onnxruntime-gpu")
        return (f"{others} from another package overwrote the onnxruntime-gpu files. To restore GPU: "
                f"{reinstall_cmd('onnxruntime-gpu', ort['onnxruntime-gpu'])}")
    return None


VCREDIST_URL = "https://aka.ms/vc14/vc_redist.x64.exe"


def _msvc_runtime():
    """Windows без Visual C++ Redistributable: pyclipper, onnxruntime и pymupdf не грузятся с невнятным
    «DLL load failed» (им нужен msvcp140.dll, его нет ни в чистой Windows, ни в Python с python.org).
    Колесо win_amd64 везёт рантайм с собой (_winrt.preload) — сюда попадаем только с универсальным
    колесом (Windows на ARM, сборка из исходников) или если подгрузка не удалась."""
    if sys.platform != "win32":
        return
    from . import _winrt
    _winrt.preload()
    try:
        import pyclipper  # noqa: F401
    except ImportError as e:
        if "DLL load failed" in str(e):
            raise ImportError(
                "Microsoft Visual C++ Redistributable (msvcp140.dll) not found — pyclipper, onnxruntime "
                f"and pymupdf cannot load without it. Install it from Microsoft: {VCREDIST_URL} "
                "(once per computer) and restart Python."
            ) from e


def check():
    _msvc_runtime()
    msgs = []
    ort = _installed(_ORT)
    if len({v for _, v in ort}) > 1:
        keep = "onnxruntime-gpu" if "onnxruntime-gpu" in dict(ort) else ort[0][0]
        try:
            import onnxruntime
            loaded = onnxruntime.__version__
        except Exception:
            loaded = None
        if loaded != dict(ort)[keep]:                     # после переустановки keep поверх — уже в порядке
            msgs.append(f"the onnxruntime module mixes files of different versions ({', '.join(f'{n} {v}' for n, v in ort)}).\n"
                        f"  To fix: {reinstall_cmd(keep, dict(ort)[keep])}")
    cv = _installed(_CV2)
    if len(cv) > 1:
        try:
            import cv2  # noqa: F401
        except Exception as e:
            n, v = cv[0]                                  # приоритет — headless (зависимость occular)
            msgs.append(f"cv2 fails to import ({e}): the module mixes files of "
                        f"{', '.join(f'{n} {v}' for n, v in cv)}.\n  To fix: {reinstall_cmd(n, v)}")
    for m in msgs:
        print(f"[occular] WARNING: {m}", file=sys.stderr)
    return msgs
