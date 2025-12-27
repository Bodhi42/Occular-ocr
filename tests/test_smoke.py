"""Smoke test for OCR skeleton"""

import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from ocr_skel.registry import Registry
from ocr_skel.detector import CRAFTDetector
from ocr_skel.recognizer import CRNNRecognizer
from ocr_skel.pipeline import OCRPipeline


@pytest.fixture
def test_image():
    """Создать синтетическое изображение с текстом для теста"""
    # Создаём изображение 400x200 с белым фоном
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)

    # Рисуем простой текст (используем дефолтный шрифт)
    # PIL может не найти TrueType шрифты, используем базовый
    try:
        # Пытаемся загрузить системный шрифт
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except Exception:
        # Fallback на дефолтный шрифт
        font = ImageFont.load_default()

    draw.text((50, 80), "HELLO", fill='black', font=font)

    # Сохраняем во временный файл
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img.save(f.name)
        yield f.name

    # Cleanup
    Path(f.name).unlink(missing_ok=True)


def test_registry():
    """Тест реестра детекторов и распознавателей"""
    # Регистрация
    Registry.register_detector("craft", CRAFTDetector)
    Registry.register_recognizer("crnn", CRNNRecognizer)

    # Проверка списков
    assert "craft" in Registry.list_detectors()
    assert "crnn" in Registry.list_recognizers()

    # Получение экземпляров
    detector = Registry.get_detector("craft", gpu=False)
    assert isinstance(detector, CRAFTDetector)

    recognizer = Registry.get_recognizer("crnn", gpu=False)
    assert isinstance(recognizer, CRNNRecognizer)


def test_pipeline_smoke(test_image):
    """Smoke test: pipeline не падает и возвращает валидный JSON"""
    # Регистрация компонентов
    Registry.register_detector("craft", CRAFTDetector)
    Registry.register_recognizer("crnn", CRNNRecognizer)

    # Создание pipeline (без GPU для CI)
    pipeline = OCRPipeline(
        detector_name="craft",
        recognizer_name="crnn",
        detector_kwargs={"gpu": False},
        recognizer_kwargs={"gpu": False}
    )

    # Обработка изображения
    results = pipeline.process_image(test_image)

    # Проверки
    assert isinstance(results, list)

    # Валидация структуры каждого элемента
    for item in results:
        assert "quad" in item
        assert "text" in item
        assert "confidence" in item

        # quad должен быть списком координат
        assert isinstance(item["quad"], list)
        if len(item["quad"]) > 0:
            assert len(item["quad"]) >= 3  # минимум 3 точки для полигона

        # text должен быть строкой
        assert isinstance(item["text"], str)

        # confidence должен быть числом от 0 до 1
        assert isinstance(item["confidence"], (int, float))
        assert 0.0 <= item["confidence"] <= 1.0

    # Проверка сериализации в JSON
    json_str = json.dumps(results)
    assert json_str is not None
    assert len(json_str) > 0


def test_cli_smoke(test_image):
    """Smoke test: CLI не падает и создаёт валидный JSON файл"""
    import subprocess
    import sys

    # Временный файл для вывода
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        out_path = f.name

    try:
        # Запуск CLI
        result = subprocess.run(
            [sys.executable, "-m", "ocr_skel.cli",
             "--image", test_image,
             "--out", out_path],
            capture_output=True,
            text=True,
            timeout=60
        )

        # CLI не должен падать
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Проверка выходного файла
        assert Path(out_path).exists()

        # Валидация JSON
        with open(out_path, 'r') as f:
            results = json.load(f)

        assert isinstance(results, list)

        # Проверка структуры
        for item in results:
            assert "quad" in item
            assert "text" in item
            assert "confidence" in item

    finally:
        # Cleanup
        Path(out_path).unlink(missing_ok=True)
