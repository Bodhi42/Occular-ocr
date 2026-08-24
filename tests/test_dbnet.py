"""Тесты детектора DBNet (актуальный класс DBNetDetectorONNX / реестр 'dbnet-onnx').

Регистрация — лёгкая проверка. Реальная детекция ПРОПУСКАЕТСЯ, если веса недоступны.
"""
import numpy as np
import pytest
from PIL import Image, ImageDraw

import occular  # noqa: F401  — регистрирует dbnet-onnx при импорте
from occular.registry import Registry
from occular.dbnet_detector_onnx import DBNetDetectorONNX


def test_dbnet_registration():
    """Детектор зарегистрирован под актуальным именем."""
    assert "dbnet-onnx" in Registry.list_detectors()


def _detector_or_skip():
    try:
        return Registry.get_detector("dbnet-onnx", gpu=False)
    except Exception as e:
        pytest.skip(f"веса детектора недоступны: {e}")


def test_dbnet_detect():
    """Детекция на синтетическом изображении: список квадов формы (4,2) в границах кадра."""
    detector = _detector_or_skip()
    img = Image.new('RGB', (400, 200), color='white')
    ImageDraw.Draw(img).rectangle([50, 50, 350, 150], fill='black')
    quads = detector.detect(np.array(img))
    assert isinstance(quads, list)
    for quad in quads:
        assert isinstance(quad, np.ndarray) and quad.shape == (4, 2)
        assert np.all(quad[:, 0] >= 0) and np.all(quad[:, 0] <= 400)
        assert np.all(quad[:, 1] >= 0) and np.all(quad[:, 1] <= 200)


def test_dbnet_empty_image():
    """Пустое изображение — валидный (обычно пустой) список квадов, без падения."""
    detector = _detector_or_skip()
    quads = detector.detect(np.ones((200, 400, 3), dtype=np.uint8) * 255)
    assert isinstance(quads, list)
