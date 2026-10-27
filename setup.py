import os
from pathlib import Path
from setuptools import setup, find_packages

import shutil
from setuptools.command.build_py import build_py as _build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:                       # setuptools < 70.1
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

_here = Path(__file__).parent
# Где собран onnxruntime-gpu (PyPI): Linux x86_64 / Windows x64 с Python 3.10+. На Linux aarch64 колёса
# onnxruntime-gpu требуют glibc 2.34+ (на Ubuntu 20.04 / Debian 11 установка падала) — там обычный onnxruntime.
_GPU_OK = "(sys_platform == 'linux' and platform_machine == 'x86_64' and python_version >= '3.10') or (sys_platform == 'win32' and platform_machine == 'AMD64' and python_version >= '3.10')"
# Дополнение к _GPU_OK (в маркерах PEP 508 нет оператора not).
_NO_GPU = "(sys_platform != 'linux' or platform_machine != 'x86_64' or python_version < '3.10') and (sys_platform != 'win32' or platform_machine != 'AMD64' or python_version < '3.10')"
# Windows x64: OCCULAR_WHEEL_WIN_AMD64=1 собирает колесо py3-none-win_amd64 с C++-рантаймом Microsoft
# в occular/_vcrt (tools/fetch_vcrt.py кладёт туда 5 DLL, tools/build_wheels.sh собирает оба колеса).
# pip на Windows x64 предпочитает его универсальному — на чистой Windows не нужен VC++ Redistributable.
_WIN_AMD64 = os.environ.get("OCCULAR_WHEEL_WIN_AMD64") == "1"
_package_data = ["*.json"]   # cyr_decode_config.json (per-язык декод кириллицы)
if _WIN_AMD64:
    _dlls = sorted((_here / "occular" / "_vcrt").glob("*.dll"))
    if len(_dlls) != 5:
        raise SystemExit("occular/_vcrt: нужно 5 DLL — запустите python tools/fetch_vcrt.py")
    _package_data += ["_vcrt/*.dll", "_vcrt/*.txt"]


class build_py(_build_py):
    """win_amd64: _occular_vcrt.py и occular-vcrt.pth — в корень site-packages (.pth работает только там)."""
    def run(self):
        super().run()
        if _WIN_AMD64:
            for f in ("_occular_vcrt.py", "occular-vcrt.pth"):
                shutil.copyfile(_here / "occular" / "_vcrt" / f, Path(self.build_lib) / f)


class bdist_wheel(_bdist_wheel):
    def get_tag(self):
        return ("py3", "none", "win_amd64") if _WIN_AMD64 else super().get_tag()


_long = (_here / "README.md").read_text(encoding="utf-8") if (_here / "README.md").exists() else ""

setup(
    name="occular-ocr",
    version="0.5.1",
    description="State-of-the-art OCR for Russian documents, with a zero-compilation install.",
    long_description=_long,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    license_files=["LICENSE", "LICENSE-pyctcdecode"],   # _beam.py — адаптация pyctcdecode
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    package_data={"occular": _package_data},
    exclude_package_data={"occular": [] if _WIN_AMD64 else ["_vcrt/*"]},
    cmdclass={"build_py": build_py, "bdist_wheel": bdist_wheel},
    install_requires=[
        # версии синхронизированы с requirements.txt (единый источник границ совместимости)
        "numpy>=1.24.0",
        # headless: occular не открывает окон, а GUI-сборка требует libGL и не импортируется в slim-Docker.
        # <4.12: 4.12+ требует numpy>=2 и заставляет pip менять numpy в чужом окружении.
        # >=4.10.0.84: первая сборка под numpy 2 (старые с numpy 2 не импортируются: _ARRAY_API not found).
        # macOS: ровно 4.10.0.84 — у 4.11 колёса только под macOS 13+, а pip не откатывается на версию
        # с колесом, а собирает новейшую из исходников (~45 мин на M1, issue #4). У 4.10.0.84 колёса
        # под Apple silicon с macOS 11 и Intel с macOS 12.
        "opencv-python-headless>=4.10.0.84,<4.12; sys_platform != 'darwin'",
        # Intel-Mac: у 4.10.0.84 колесо под Intel только с macOS 12, а 4.9.0.80 — с macOS 10.16 (11). Версию
        # macOS маркером не узнать (platform_release на Linux — «6.8.0-…», не версия: pip < 26 на таком
        # маркере падает с InvalidVersion), поэтому всем Intel-Mac с Python ≤ 3.12 — 4.9.0.80 и numpy<2
        # (4.9 собран под numpy 1). С Python 3.13+ onnxruntime под Intel всё равно требует macOS 13+ — там 4.10.
        "opencv-python-headless==4.10.0.84; sys_platform == 'darwin' and (platform_machine != 'x86_64' or python_version >= '3.13')",
        "opencv-python-headless==4.9.0.80; sys_platform == 'darwin' and platform_machine == 'x86_64' and python_version < '3.13'",
        "numpy<2; sys_platform == 'darwin' and platform_machine == 'x86_64' and python_version < '3.13'",
        "Pillow>=10.0.0",
        "pyclipper>=1.3.0",
        # Все модели — ONNX. GPU-сборка onnxruntime умеет и CPU (тот же результат и скорость), поэтому
        # ставится везде, где она собрана; иначе (macOS, старый Python, прочие платформы) — обычная.
        # >=1.19: раньше нет CPU-ядра Trilu (порядок чтения не грузится) и нет поддержки numpy 2.
        f"onnxruntime-gpu>=1.19.0; {_GPU_OK}",
        f"onnxruntime>=1.19.0; {_NO_GPU}",
        # 1.23.0 на Linux требует libcrypt.so.2 и не импортируется. <2.0: pymupdf 2.0 тянет onnxruntime
        # (CPU-сборку) — рядом с нашим onnxruntime-gpu это две сборки одного модуля, перезаписывающие
        # файлы друг друга; плюс networkx/psutil/pymupdf4llm, которые occular не нужны. Под Intel-Mac
        # у 2.0 к тому же нет колёс, и pip собирал бы её из исходников.
        "pymupdf>=1.23.1,<2.0",
        "huggingface_hub>=0.20.0",   # загрузка весов/LM с HuggingFace
    ],
    extras_require={
        # GPU: библиотеки CUDA и cuDNN из pip (без системной CUDA; нужен только драйвер NVIDIA).
        # Подгружаются через onnxruntime.preload_dlls() в occular._runtime.
        "gpu": [f"onnxruntime-gpu[cuda,cudnn]>=1.21; {_GPU_OK}"],
        # UVDoc теперь ONNX и отдельных зависимостей не требует; extra оставлен для старых команд установки.
        "unwarp": [],
    },
    entry_points={
        "console_scripts": [
            "ocr=occular.cli:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Operating System :: OS Independent",
    ],
)
