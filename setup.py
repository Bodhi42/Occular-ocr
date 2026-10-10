from pathlib import Path
from setuptools import setup, find_packages

_here = Path(__file__).parent
# Где собран onnxruntime-gpu (PyPI): Linux x86_64 / Windows x64 с Python 3.10+, Linux aarch64 с 3.11+.
_GPU_OK = "(sys_platform == 'linux' and platform_machine == 'x86_64' and python_version >= '3.10') or (sys_platform == 'win32' and platform_machine == 'AMD64' and python_version >= '3.10') or (sys_platform == 'linux' and platform_machine == 'aarch64' and python_version >= '3.11')"
# Дополнение к _GPU_OK (в маркерах PEP 508 нет оператора not).
_NO_GPU = "(sys_platform != 'linux' or platform_machine != 'x86_64' or python_version < '3.10') and (sys_platform != 'win32' or platform_machine != 'AMD64' or python_version < '3.10') and (sys_platform != 'linux' or platform_machine != 'aarch64' or python_version < '3.11')"
_long = (_here / "README.md").read_text(encoding="utf-8") if (_here / "README.md").exists() else ""

setup(
    name="occular-ocr",
    version="0.4.6",
    description="State-of-the-art OCR for Russian documents, with a zero-compilation install.",
    long_description=_long,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    license_files=["LICENSE", "LICENSE-pyctcdecode"],   # _beam.py — адаптация pyctcdecode
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    package_data={"occular": ["*.json"]},   # cyr_decode_config.json (per-язык декод кириллицы)
    install_requires=[
        # версии синхронизированы с requirements.txt (единый источник границ совместимости)
        "numpy>=1.24.0",
        # headless: occular не открывает окон, а GUI-сборка требует libGL и не импортируется в slim-Docker.
        # <4.12: 4.12+ требует numpy>=2 и заставляет pip менять numpy в чужом окружении.
        # >=4.10.0.84: первая сборка под numpy 2 (старые с numpy 2 не импортируются: _ARRAY_API not found).
        "opencv-python-headless>=4.10.0.84,<4.12",
        "Pillow>=10.0.0",
        "pyclipper>=1.3.0",
        # Все модели — ONNX. GPU-сборка onnxruntime умеет и CPU (тот же результат и скорость), поэтому
        # ставится везде, где она собрана; иначе (macOS, старый Python, прочие платформы) — обычная.
        # >=1.19: раньше нет CPU-ядра Trilu (порядок чтения не грузится) и нет поддержки numpy 2.
        f"onnxruntime-gpu>=1.19.0; {_GPU_OK}",
        f"onnxruntime>=1.19.0; {_NO_GPU}",
        "pymupdf>=1.23.1",           # 1.23.0 на Linux требует libcrypt.so.2 и не импортируется
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
