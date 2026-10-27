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
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)  # CLI пишет UTF-8 и в трубу
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


def test_unknown_language_rejected():
    """Неизвестный код языка — понятная ошибка, а не молчаливый уход в кир-модель; регистр не важен."""
    from occular import _normalize_languages
    with pytest.raises(ValueError, match="Unsupported language"):
        _normalize_languages(["ru", "xx"])
    with pytest.raises(ValueError, match="Unsupported language"):
        _normalize_languages("xx")
    assert _normalize_languages(["UK"]) == ("multilingual", {"languages": ["uk"]})


def test_registry_cache_keyed_by_all_args():
    """Экземпляр переиспользуется только при тех же аргументах: раньше ключ (name, threads, gpu, lm)
    отдавал распознаватель, созданный для других languages/arch."""
    made = []

    class Fake:
        def __init__(self, **kw):
            made.append(kw)

    Registry.register_recognizer("_fake_test", Fake)
    try:
        a = Registry.get_recognizer("_fake_test", languages=["uk"], num_threads=2)
        b = Registry.get_recognizer("_fake_test", languages=["kk"], num_threads=2)
        c = Registry.get_recognizer("_fake_test", languages=["uk"], num_threads=2)
        d = Registry.get_recognizer("_fake_test", arch="svtr_t", num_threads=2)
        assert a is not b and a is c and d is not a and len(made) == 3
    finally:
        Registry._recognizers.pop("_fake_test", None)
        for k in [k for k in Registry._recognizer_cache if k[0] == "_fake_test"]:
            del Registry._recognizer_cache[k]


def test_locate_weight_sees_hf_cache(tmp_path, monkeypatch):
    """model_info/layout_ready видят веса в кэше HF, а не только в weights/ (раньше обязательные
    модели в кэше показывались как MISSING)."""
    hf = pytest.importorskip("huggingface_hub")
    import occular.model_files as mf
    monkeypatch.setattr(mf, "WEIGHTS_DIR", tmp_path / "weights")
    cached = tmp_path / "cached.onnx"
    cached.write_bytes(b"x")
    monkeypatch.setattr(hf, "try_to_load_from_cache",
                        lambda repo, f, revision=None: str(cached) if f == "a.onnx" else None)
    assert mf.locate_weight("a.onnx") == cached
    assert mf.locate_weight("b.onnx") is None
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "b.onnx").write_bytes(b"y")
    assert mf.locate_weight("b.onnx") == tmp_path / "weights" / "b.onnx"


def test_multilingual_recognizer_langs():
    """Роутер принимает языки и хранит их (без загрузки весов до вызова recognize)."""
    from occular.multilingual import MultilingualRecognizer
    rec = MultilingualRecognizer(languages=["uk"])
    assert rec.languages == ["uk"]
    rec_auto = MultilingualRecognizer(languages=None)
    assert rec_auto.languages is None


def test_plain_text_keeps_reading_order(monkeypatch, tmp_path):
    """ocr() и CLI печатают строки в порядке конвейера (порядке чтения), а не пересортировывают сверху вниз."""
    import occular
    from occular import cli
    lines = [{"quad": [[0, 50], [9, 50], [9, 60], [0, 60]], "text": "first"},     # ниже на странице, но первая
             {"quad": [[0, 10], [9, 10], [9, 20], [0, 20]], "text": "second"}]

    class Fake:
        def process_image(self, path):
            return list(lines)

    monkeypatch.setattr(occular, "_build_pipeline", lambda *a, **k: Fake())
    img = tmp_path / "x.png"
    img.write_bytes(b"")
    assert occular.ocr(str(img)) == "first\nsecond"
    assert cli._plain_text(Fake(), img, None) == "first\nsecond"


def test_unwarp_single_setting():
    """unwarp: False — выкл, True — с проверкой читаемости, "always" — без; unwarp_gate/unwarp_margin убраны."""
    from occular.pipeline import _unwarp_mode, UNWARP_MODEL
    from occular import Settings, OCRPipeline
    assert _unwarp_mode(False) == (None, False) and _unwarp_mode(None) == (None, False)
    assert _unwarp_mode(True) == (UNWARP_MODEL, False)
    assert _unwarp_mode("always") == (UNWARP_MODEL, True)
    assert _unwarp_mode("my/uvdoc") == ("my/uvdoc", False)
    with pytest.raises(ValueError):
        _unwarp_mode(1.1)
    assert Settings().unwarp is False
    for gone in ("unwarp_gate", "unwarp_margin"):
        assert not hasattr(Settings(), gone)
        with pytest.raises(TypeError):
            OCRPipeline(**{gone: True})
