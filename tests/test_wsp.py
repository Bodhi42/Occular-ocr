"""Opt-in routing, integrity, caching and failure behavior; no network needed."""
import hashlib
import sys
from types import SimpleNamespace

import pytest

import occular
from occular import model_files as mf
from occular.recognizer_onnx import (CRNNRecognizerONNX, DEFAULT_ARCH,
                                     RECOGNIZER_ARCHS, arch_files)
from occular.registry import Registry


def test_original_default_and_file_mapping():
    assert DEFAULT_ARCH == "svtr_lcnet"
    assert occular.Settings().recognizer is None
    assert occular._normalize_languages(None) == ("crnn-onnx", None)
    assert arch_files("svtr_lcnet") == ("recognizer_svtr_lcnet_fp32.onnx", "recognizer_charset.txt")
    assert arch_files("svtr_t", "cyr") == ("recognizer_svtr_cyr12_fp32.onnx", "recognizer_charset_cyr12.txt")
    assert arch_files("svtr_lcnet_wsp") == (mf.WSP_RECOGNIZER_FILE, "recognizer_charset.txt")
    assert mf.WSP_RECOGNIZER_SHA256 == "24c5db4e64ad0a9bb8859b501670eff68969264ff0bee8af7130e3eb50f565f6"


@pytest.mark.parametrize("languages", [None, [], "ru", "EN", ["ru", "en"]])
def test_wsp_ru_en_routing(languages):
    name, kwargs = occular._normalize_languages(languages, "svtr_lcnet_wsp")
    assert name == "crnn-onnx" and kwargs["arch"] == "svtr_lcnet_wsp"


@pytest.mark.parametrize("languages", ["auto", "uk", ["ru", "uk"], ["xx"]])
def test_wsp_languages_rejected_without_model_loading(languages, monkeypatch):
    monkeypatch.setattr(mf, "ensure_weight", lambda *a: pytest.fail("Must not load weights"))
    with pytest.raises(ValueError, match="RU/EN|ru.*en"):
        occular.ocr("unused.png", recognizer="svtr_lcnet_wsp", languages=languages)


def test_gpu_rejected_before_detector_and_download(monkeypatch):
    monkeypatch.setattr(Registry, "get_detector", lambda *a, **k: pytest.fail("Must fail before detector"))
    with pytest.raises(ValueError, match="CPU-only.*CUDA"):
        occular.ocr("unused.png", recognizer="svtr_lcnet_wsp", gpu=True)
    with pytest.raises(ValueError, match="CPU-only"):
        CRNNRecognizerONNX(arch="svtr_lcnet_wsp", gpu=True)
    with pytest.raises(ValueError, match="CPU-only"):
        occular.OCRPipeline(occular.Settings(recognizer="svtr_lcnet_wsp", gpu=True))


def test_no_multilingual_or_custom_checkpoint_fallback():
    from occular.multilingual import MultilingualRecognizer
    with pytest.raises(ValueError, match="RU/EN"):
        arch_files("svtr_lcnet_wsp", "cyr")
    with pytest.raises(ValueError, match="RU/EN"):
        MultilingualRecognizer(arch="svtr_lcnet_wsp")
    with pytest.raises(ValueError, match="exact pinned"):
        CRNNRecognizerONNX(arch="svtr_lcnet_wsp", onnx_file="recognizer_svtr_lcnet_fp32.onnx")
    with pytest.raises(ValueError, match="charset"):
        CRNNRecognizerONNX(arch="svtr_lcnet_wsp", charset_file="recognizer_charset_cyr12.txt")


def test_public_api_passes_default_original_and_wsp(monkeypatch):
    seen = []
    class FakePipeline:
        def __init__(self, **kwargs):seen.append(kwargs)
        def process_image(self, path):return []
    monkeypatch.setattr(occular, "_OCRPipelineBase", FakePipeline)
    for arch in [None, "svtr_lcnet", "svtr_lcnet_wsp"]:
        kwargs = {} if arch is None else {"recognizer":arch}
        assert occular.ocr("unused.png", **kwargs) == ""
    assert seen[0]["recognizer_kwargs"] is None
    assert seen[1]["recognizer_kwargs"]["arch"] == "svtr_lcnet"
    assert seen[2]["recognizer_kwargs"]["arch"] == "svtr_lcnet_wsp"


def test_real_recognizer_cache_separates_architecture(tmp_path, monkeypatch):
    import occular._runtime as runtime
    charset = tmp_path/"charset.txt";charset.write_text("a\nb", encoding="utf-8")
    monkeypatch.setattr(mf, "WSP_CHARSET_SHA256", hashlib.sha256(charset.read_bytes()).hexdigest())
    files = []
    def weight(name):
        files.append(name)
        return str(charset) if name.endswith(".txt") else str(tmp_path/name)
    monkeypatch.setattr(mf, "ensure_weight", weight)
    class FakeSession:
        def get_inputs(self):return [SimpleNamespace(name="x")]
        def get_outputs(self):return [SimpleNamespace(shape=[None,None,3])]
    monkeypatch.setattr(runtime, "make_session", lambda *a, **k: FakeSession())
    monkeypatch.setattr(runtime, "session_on_gpu", lambda s: False)
    old = Registry._recognizer_cache
    monkeypatch.setattr(Registry, "_recognizer_cache", {})
    a = Registry.get_recognizer("crnn-onnx", arch="svtr_lcnet", lm=False)
    b = Registry.get_recognizer("crnn-onnx", arch="svtr_lcnet_wsp", lm=False)
    c = Registry.get_recognizer("crnn-onnx", arch="svtr_lcnet", lm=False)
    assert a is c and a is not b
    assert a._onnx_file == "recognizer_svtr_lcnet_fp32.onnx"
    assert b._onnx_file == mf.WSP_RECOGNIZER_FILE
    assert files.count(mf.WSP_RECOGNIZER_FILE) == 1


def test_local_priority_and_on_demand_download(tmp_path, monkeypatch):
    import huggingface_hub as hf
    monkeypatch.setattr(mf, "WEIGHTS_DIR", tmp_path)
    original = tmp_path/"recognizer_svtr_lcnet_fp32.onnx";original.write_bytes(b"original")
    downloaded = tmp_path/"downloaded.onnx";downloaded.write_bytes(b"test-only dummy")
    monkeypatch.setattr(mf, "WSP_RECOGNIZER_SHA256", hashlib.sha256(downloaded.read_bytes()).hexdigest())
    calls = []
    def download(repo, filename, revision=None):calls.append((repo,filename,revision));return str(downloaded)
    monkeypatch.setattr(hf, "hf_hub_download", download)
    assert mf.ensure_weight(original.name) == str(original) and calls == []
    assert mf.ensure_weight(mf.WSP_RECOGNIZER_FILE) == str(downloaded)
    assert calls == [(mf.WEIGHTS_HF_REPO,mf.WSP_RECOGNIZER_FILE,mf.WEIGHTS_REVISION)]
    (tmp_path/mf.WSP_RECOGNIZER_FILE).write_bytes(downloaded.read_bytes());calls.clear()
    assert mf.ensure_weight(mf.WSP_RECOGNIZER_FILE) == str(tmp_path/mf.WSP_RECOGNIZER_FILE)
    assert calls == []


def test_bad_hash_and_unavailable_weights_are_explicit(tmp_path, monkeypatch):
    import huggingface_hub as hf
    monkeypatch.setattr(mf, "WEIGHTS_DIR", tmp_path)
    target = tmp_path/mf.WSP_RECOGNIZER_FILE;target.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        mf.ensure_weight(mf.WSP_RECOGNIZER_FILE)
    target.unlink()
    def unavailable(*a, **k):raise OSError("not published")
    monkeypatch.setattr(hf, "hf_hub_download", unavailable)
    with pytest.raises(RuntimeError, match="weights are unavailable.*svtr_lcnet"):
        mf.ensure_weight(mf.WSP_RECOGNIZER_FILE)


def test_model_info_optional_without_network(monkeypatch, capsys):
    monkeypatch.setattr(mf, "locate_weight", lambda *a, **k: None)
    mf.model_info();output = capsys.readouterr().out
    assert "svtr_lcnet_wsp" in output and mf.WSP_RECOGNIZER_FILE in output
    row = next(v for k,v in mf.MODELS.items() if "svtr_lcnet_wsp" in k)
    assert row["required"] is False


@pytest.mark.parametrize("model", ["svtr_lcnet", "svtr_lcnet_wsp"])
def test_cli_model_option(tmp_path, monkeypatch, model):
    from occular import cli
    source=tmp_path/"source.png";source.write_bytes(b"not decoded in this routing test")
    seen=[]
    def pipeline(**kw):seen.append(kw);return object()
    monkeypatch.setattr(occular,"OCRPipeline",pipeline)
    monkeypatch.setattr(cli,"_run_single",lambda *a, **k: None)
    monkeypatch.setattr(sys,"argv",["ocr",str(source),"--model",model])
    cli.main();assert seen[0]["recognizer"] == model
