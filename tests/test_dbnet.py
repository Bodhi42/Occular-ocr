"""Tests for DBNet detector"""

import numpy as np
from PIL import Image, ImageDraw
import pytest
from ocr_skel.registry import Registry
from ocr_skel.dbnet_detector import DBNetDetector


def test_dbnet_registration():
    """Тест регистрации DBNet детектора"""
    assert "dbnet" in Registry.list_detectors()

    detector = Registry.get_detector("dbnet", gpu=False)
    assert isinstance(detector, DBNetDetector)


def test_dbnet_detect():
    """Тест детекции DBNet на синтетическом изображении"""
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 350, 150], fill='black')

    img_array = np.array(img)

    detector = Registry.get_detector("dbnet", gpu=False)
    quads = detector.detect(img_array)

    assert isinstance(quads, list)

    for quad in quads:
        assert isinstance(quad, np.ndarray)
        assert quad.shape == (4, 2), f"Expected shape (4, 2), got {quad.shape}"
        assert np.all(quad[:, 0] >= 0) and np.all(quad[:, 0] <= 400)
        assert np.all(quad[:, 1] >= 0) and np.all(quad[:, 1] <= 200)


def test_dbnet_empty_image():
    """Тест детекции на пустом изображении"""
    img_array = np.ones((200, 400, 3), dtype=np.uint8) * 255

    detector = Registry.get_detector("dbnet", gpu=False)
    quads = detector.detect(img_array)

    assert isinstance(quads, list)
