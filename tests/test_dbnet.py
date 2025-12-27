"""Smoke test for DBNet detector"""

import numpy as np
from PIL import Image, ImageDraw
import pytest
from ocr_skel.registry import Registry
from ocr_skel.dbnet_detector import DBNetDetector


def test_dbnet_registration():
    """Тест регистрации DBNet детектора"""
    # Проверка что DBNet зарегистрирован
    assert "dbnet" in Registry.list_detectors()
    
    # Получение экземпляра
    detector = Registry.get_detector("dbnet", gpu=False)
    assert isinstance(detector, DBNetDetector)


def test_dbnet_detect():
    """Тест детекции DBNet на синтетическом изображении"""
    # Создаём синтетическое изображение с текстом
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 350, 150], fill='black')
    
    # Конвертируем в numpy array
    img_array = np.array(img)
    
    # Создаём детектор
    detector = Registry.get_detector("dbnet", gpu=False)
    
    # Детектируем
    quads = detector.detect(img_array)
    
    # Проверки
    assert isinstance(quads, list)
    
    # Проверка формата квадов
    for quad in quads:
        assert isinstance(quad, np.ndarray)
        assert quad.shape == (4, 2), f"Expected shape (4, 2), got {quad.shape}"
        # Координаты должны быть в пределах изображения
        assert np.all(quad[:, 0] >= 0) and np.all(quad[:, 0] <= 400)
        assert np.all(quad[:, 1] >= 0) and np.all(quad[:, 1] <= 200)


def test_dbnet_empty_image():
    """Тест детекции на пустом изображении"""
    # Пустое белое изображение
    img_array = np.ones((200, 400, 3), dtype=np.uint8) * 255
    
    detector = Registry.get_detector("dbnet", gpu=False)
    quads = detector.detect(img_array)
    
    # Должен вернуть список (может быть пустым)
    assert isinstance(quads, list)


def test_dbnet_with_craft_compatibility():
    """Тест что DBNet и CRAFT возвращают совместимый формат"""
    from ocr_skel.detector import CRAFTDetector
    
    # Создаём тестовое изображение
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 350, 150], fill='black')
    img_array = np.array(img)
    
    # Тестируем оба детектора
    dbnet = Registry.get_detector("dbnet", gpu=False)
    craft = Registry.get_detector("craft", gpu=False)
    
    quads_dbnet = dbnet.detect(img_array)
    quads_craft = craft.detect(img_array)
    
    # Оба должны возвращать списки квадов
    assert isinstance(quads_dbnet, list)
    assert isinstance(quads_craft, list)
    
    # Формат должен быть одинаковым
    if len(quads_dbnet) > 0:
        assert quads_dbnet[0].shape == (4, 2)
    if len(quads_craft) > 0:
        assert quads_craft[0].shape == (4, 2)
