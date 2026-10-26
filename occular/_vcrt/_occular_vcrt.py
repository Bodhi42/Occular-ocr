"""C++-рантайм Microsoft для onnxruntime/pyclipper/pymupdf на Windows x64 без Visual C++ Redistributable.

Колесо occular-ocr для win_amd64 кладёт этот модуль и occular-vcrt.pth в site-packages; .pth выполняется
при старте Python, поэтому рантайм на месте, даже если onnxruntime или fitz импортируют раньше occular.
Если все 5 файлов есть в System32 (Redistributable стоит — так почти везде), ничего не делаем. Иначе
грузим недостающие: сначала копию, которую находит Windows (например, рядом с python.exe), и только
если её нет — нашу из occular/_vcrt. Ошибки глотаем: .pth не должен мешать запуску Python.
"""
import os
import sys

DLLS = ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll",
        "msvcp140_atomic_wait.dll")    # порядок важен: msvcp140 зависит от vcruntime140
loaded_from_package = []


def preload():
    if sys.platform != "win32":
        return
    sys32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    if all(os.path.exists(os.path.join(sys32, n)) for n in DLLS):
        return
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "occular", "_vcrt")
    if not os.path.isdir(d):
        return
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetModuleHandleW.restype = ctypes.c_void_p
    for name in DLLS:
        if name in loaded_from_package or k32.GetModuleHandleW(name):
            continue
        try:
            ctypes.WinDLL(name)
            continue
        except OSError:
            pass
        try:
            ctypes.WinDLL(os.path.join(d, name))
            loaded_from_package.append(name)
        except OSError:
            pass


try:
    preload()
except Exception:
    pass
