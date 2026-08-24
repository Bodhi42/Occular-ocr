from pathlib import Path
from setuptools import setup, find_packages

_here = Path(__file__).parent
_long = (_here / "README.md").read_text(encoding="utf-8") if (_here / "README.md").exists() else ""

setup(
    name="occular-ocr",
    version="0.3.0",
    description="State-of-the-art OCR for Russian documents, with a zero-compilation install.",
    long_description=_long,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        # версии синхронизированы с requirements.txt (единый источник границ совместимости)
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "pyclipper>=1.3.0",
        "onnxruntime>=1.16.0",
        "pymupdf>=1.23.0",
        "huggingface_hub>=0.20.0",   # загрузка весов/LM с HuggingFace
        "pyctcdecode>=0.5.0",        # beam-CTC (чистый Python)
    ],
    extras_require={
        # GPU-путь = нативный PyTorch на CUDA (надёжнее onnxruntime-gpu).
        # Ставит torch+torchvision; веса .pth качаются с HuggingFace при gpu=True.
        # torch>=2.0 — есть weights_only-загрузка .pth (безопасность supply-chain).
        "gpu": ["torch>=2.0", "torchvision>=0.15"],
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
