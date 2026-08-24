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

# make `ocr_skel` resolve to the real package, and pre-alias every submodule so
# `import ocr_skel.<x>` and `from ocr_skel.<x> import ...` keep working.
_sys.modules[__name__] = _occular
for _m in _pkgutil.iter_modules(_occular.__path__):
    try:
        _sys.modules[f"{__name__}.{_m.name}"] = _importlib.import_module(f"occular.{_m.name}")
    except Exception:
        pass
