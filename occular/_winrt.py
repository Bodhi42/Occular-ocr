"""Windows x64: C++-рантайм Microsoft для зависимостей (onnxruntime, pyclipper, pymupdf).

Основная работа — в _occular_vcrt (верхний модуль колеса win_amd64, подключается через .pth при старте
Python). Здесь — запасной путь при import occular: если .pth не выполнялся (python -S, встраивание).
"""
loaded_from_package = []


def preload():
    try:
        import _occular_vcrt
    except ImportError:          # универсальное колесо (не win_amd64) — рантайма в пакете нет
        return
    _occular_vcrt.preload()      # повторный вызов дешёвый: уже загруженное пропускается
    loaded_from_package[:] = _occular_vcrt.loaded_from_package
