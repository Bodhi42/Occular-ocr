"""Back-compat alias: the package was renamed `ocr_skel` → `occular` in 0.3.0.

`import occular` is the new name; this shim keeps old imports working
(`from ocr_skel import ocr`, `from ocr_skel.registry import Registry`, `python -m ocr_skel`).
"""
import sys as _sys
import warnings as _warnings
import pkgutil as _pkgutil
import importlib as _importlib

import occular as _occular

_warnings.warn(
    "`ocr_skel` was renamed to `occular` in 0.3.0 — import `occular` instead.",
    DeprecationWarning, stacklevel=2,
)

# Re-export every public name from `occular` into this module (same objects, so
# `ocr_skel.ocr is occular.ocr`). Keep `ocr_skel` a *real* package (not a sys.modules
# alias) so `python -m ocr_skel` can still load this package's own `__main__.py`.
for _k, _v in vars(_occular).items():
    if not _k.startswith("__"):
        globals()[_k] = _v
__version__ = _occular.__version__
__all__ = getattr(_occular, "__all__", None)

# Pre-alias every submodule so `import ocr_skel.<x>` and `from ocr_skel.<x> import ...` keep working.
# Skip dunder modules (`__main__`): aliasing `__main__` would shadow this package's own
# `__main__.py` and break `python -m ocr_skel` (runpy must load the real ocr_skel.__main__).
for _m in _pkgutil.iter_modules(_occular.__path__):
    if _m.name.startswith("__"):
        continue
    try:
        _sys.modules[f"{__name__}.{_m.name}"] = _importlib.import_module(f"occular.{_m.name}")
    except Exception:
        pass
