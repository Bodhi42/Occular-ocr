from pathlib import Path
from setuptools import setup, find_packages

_here = Path(__file__).parent
_long = (_here / "README.md").read_text(encoding="utf-8") if (_here / "README.md").exists() else ""

setup(
    name="occular-ocr",
    version="0.2.1",
    description="State-of-the-art OCR for Russian documents, with a zero-compilation install.",
    long_description=_long,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "numpy",
        "opencv-python",
        "Pillow",
        "pyclipper",
        "shapely",
        "onnxruntime",
        "pymupdf",
        "huggingface_hub",   # загрузка весов/LM с HuggingFace
        "pyctcdecode",       # beam-CTC (чистый Python)
    ],
    extras_require={
        "gpu": ["onnxruntime-gpu"],
    },
    entry_points={
        "console_scripts": [
            "ocr=ocr_skel.cli:main",
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
