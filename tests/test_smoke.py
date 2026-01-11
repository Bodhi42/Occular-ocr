"""Smoke test for OCR"""

import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from ocr_skel.registry import Registry
from ocr_skel.dbnet_detector import DBNetDetector
from ocr_skel.recognizer import CRNNRecognizer
from ocr_skel.pipeline import OCRPipeline


@pytest.fixture
def test_image():
    """Создать синтетическое изображение с текстом для теста"""
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
    except Exception:
        font = ImageFont.load_default()

    draw.text((50, 80), "HELLO", fill='black', font=font)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img.save(f.name)
        yield f.name

    Path(f.name).unlink(missing_ok=True)


def test_registry():
    """Тест реестра детекторов и распознавателей"""
    Registry.register_detector("dbnet", DBNetDetector)
    Registry.register_recognizer("crnn", CRNNRecognizer)

    assert "dbnet" in Registry.list_detectors()
    assert "crnn" in Registry.list_recognizers()

    detector = Registry.get_detector("dbnet", gpu=False)
    assert isinstance(detector, DBNetDetector)

    recognizer = Registry.get_recognizer("crnn", gpu=False)
    assert isinstance(recognizer, CRNNRecognizer)


def test_pipeline_smoke(test_image):
    """Smoke test: pipeline не падает и возвращает валидный JSON"""
    Registry.register_detector("dbnet", DBNetDetector)
    Registry.register_recognizer("crnn", CRNNRecognizer)

    pipeline = OCRPipeline(
        detector_name="dbnet",
        recognizer_name="crnn",
        detector_kwargs={"gpu": False},
        recognizer_kwargs={"gpu": False}
    )

    results = pipeline.process_image(test_image)

    assert isinstance(results, list)

    for item in results:
        assert "quad" in item
        assert "text" in item
        assert "confidence" in item
        assert isinstance(item["quad"], list)
        assert isinstance(item["text"], str)
        assert isinstance(item["confidence"], (int, float))
        assert 0.0 <= item["confidence"] <= 1.0

    json_str = json.dumps(results)
    assert json_str is not None
    assert len(json_str) > 0


def test_cli_smoke(test_image):
    """Smoke test: CLI не падает и создаёт валидный JSON файл"""
    import subprocess
    import sys

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        out_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, "-m", "ocr_skel.cli",
             "--image", test_image,
             "--out", out_path],
            capture_output=True,
            text=True,
            timeout=60
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert Path(out_path).exists()

        with open(out_path, 'r') as f:
            results = json.load(f)

        assert isinstance(results, list)

        for item in results:
            assert "quad" in item
            assert "text" in item
            assert "confidence" in item

    finally:
        Path(out_path).unlink(missing_ok=True)
