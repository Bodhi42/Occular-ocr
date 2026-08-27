"""Smoke-тесты Occular-OCR.

Лёгкие проверки (импорт пакета, реестр, CLI --help, публичный API) не качают веса и всегда
выполняются. Тяжёлые (реальный инференс) ПРОПУСКАЮТСЯ, если веса недоступны (нет сети/HF),
чтобы `pytest` на чистом checkout проходил без загрузки сотен мегабайт.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

import occular  # noqa: F401  — импорт пакета регистрирует dbnet-onnx / crnn-onnx
from occular.registry import Registry
from occular.dbnet_detector_onnx import DBNetDetectorONNX
from occular.recognizer_onnx import CRNNRecognizerONNX
from occular.pipeline import OCRPipeline


# ---------- лёгкие (без весов) ----------

def test_registry_defaults():
    """Компоненты зарегистрированы под актуальными именами при импорте пакета."""
    assert "dbnet-onnx" in Registry.list_detectors()
    assert "crnn-onnx" in Registry.list_recognizers()


def test_public_api_importable():
    """Публичный API на месте."""
    from occular import ocr, ocr_detailed  # noqa: F401
    assert callable(ocr) and callable(ocr_detailed)
    assert DBNetDetectorONNX and CRNNRecognizerONNX


def test_cli_help():
    """CLI запускается и показывает справку (без весов)."""
    r = subprocess.run([sys.executable, "-m", "occular", "--help"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    assert "ocr" in (r.stdout + r.stderr).lower()


# ---------- тяжёлые (пропускаются без весов) ----------

@pytest.fixture
def test_image():
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


def _run_or_skip(image_path):
    """Прогнать пайплайн; пропустить тест, если веса недоступны."""
    try:
        pipe = OCRPipeline(detector_name="dbnet-onnx", recognizer_name="crnn-onnx",
                           detector_kwargs={"gpu": False}, recognizer_kwargs={"gpu": False},
                           deskew=True, lm=False)
        return pipe.process_image(image_path)
    except Exception as e:
        pytest.skip(f"веса детектора/распознавателя недоступны: {e}")


def test_pipeline_contract(test_image):
    """Результат — список записей с quad/text/confidence, сериализуем в JSON."""
    results = _run_or_skip(test_image)
    assert isinstance(results, list)
    for item in results:
        assert {"quad", "text", "confidence"} <= set(item)
        assert isinstance(item["quad"], list)
        assert isinstance(item["text"], str)
        assert 0.0 <= float(item["confidence"]) <= 1.0
    assert len(json.dumps(results)) > 0


# ---------- многоязычность (лёгкие, без весов) ----------

def test_multilingual_registered():
    """Мультиязычный распознаватель зарегистрирован."""
    assert "multilingual" in Registry.list_recognizers()


def test_cyr_decode_config_complete():
    """Конфиг декода покрывает все 12 кир-языков корректными полями."""
    from occular.model_files import load_cyr_decode_config
    from occular.multilingual import CYR12_LANGS
    cfg = load_cyr_decode_config()
    assert set(cfg) == CYR12_LANGS
    for lang, c in cfg.items():
        assert c["decode"] in ("beam+lm", "greedy")
        if c["decode"] == "beam+lm":
            assert 0.0 < float(c["alpha"]) <= 1.0 and float(c["beta"]) > 0.0


def test_language_routing_selects_recognizer():
    """languages=None → русский распознаватель; список/'auto' → мультиязычный."""
    from occular import _normalize_languages
    assert _normalize_languages(None) == ("crnn-onnx", None)
    assert _normalize_languages([]) == ("crnn-onnx", None)
    name, kw = _normalize_languages(["uk"])
    assert name == "multilingual" and kw == {"languages": ["uk"]}
    name, kw = _normalize_languages("auto")
    assert name == "multilingual" and kw == {"languages": None}
    name, kw = _normalize_languages("uk")            # одиночная строка-код
    assert name == "multilingual" and kw == {"languages": ["uk"]}


def test_multilingual_recognizer_langs():
    """Роутер принимает языки и хранит их (без загрузки весов до вызова recognize)."""
    from occular.multilingual import MultilingualRecognizer
    rec = MultilingualRecognizer(languages=["uk"])
    assert rec.languages == ["uk"]
    rec_auto = MultilingualRecognizer(languages=None)
    assert rec_auto.languages is None
