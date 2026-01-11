from setuptools import setup, find_packages

setup(
    name="ocr_skel",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "Pillow",
        "torch",
        "pyclipper",
        "shapely",
        "onnxruntime",
        "pymupdf",
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
)
