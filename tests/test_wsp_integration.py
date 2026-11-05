"""Opt-in real CPU checks. Require verified models; never silently skip failures.

OCCULAR_WSP_INTEGRATION=1 python -m pytest tests/test_wsp_integration.py -q
The scientific replay is separate: research/weight_space_programming/reproduce.py.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("OCCULAR_WSP_INTEGRATION") != "1",
                               reason="Explicit opt-in requires local original/WSP/DBNet/LM artifacts")
ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "research/weight_space_programming/padding_example.png"
ORIGINAL_SHA = "3de3ad346181af4e4446e6c365d5d9a8bfb187ad803d6f419dd61ad3509a65f2"
WSP_SHA = "24c5db4e64ad0a9bb8859b501670eff68969264ff0bee8af7130e3eb50f565f6"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def models():
    from occular.recognizer_onnx import CRNNRecognizerONNX
    return tuple(CRNNRecognizerONNX(arch=arch, lm=False, gpu=False, num_threads=2)
                 for arch in ["svtr_lcnet", "svtr_lcnet_wsp"])


@pytest.fixture(scope="module")
def raster_pdf(tmp_path_factory):
    import pymupdf
    path = tmp_path_factory.mktemp("wsp-pdf") / "synthetic.pdf"
    with pymupdf.open() as doc:
        image = Image.open(EXAMPLE)
        page = doc.new_page(width=image.width, height=image.height)
        page.insert_image(page.rect, filename=str(EXAMPLE))
        doc.save(str(path))
    return path


def test_exact_checkpoint_and_charset_hashes(models):
    from occular.model_files import ensure_weight, WSP_CHARSET_SHA256
    for model, expected in zip(models, [ORIGINAL_SHA, WSP_SHA]):
        assert sha(ensure_weight(model._onnx_file)) == expected
        assert len(model.vocab) == 185
        assert model.session.get_providers() == ["CPUExecutionProvider"]
    assert sha(ensure_weight("recognizer_charset.txt")) == WSP_CHARSET_SHA256


def test_actual_padding_intervention(models):
    from examples.compare_wsp import scores
    image = np.asarray(Image.open(EXAMPLE).convert("RGB"))
    x = models[0]._preprocess_single(image)
    assert x.shape == (3, 48, 357)
    predictions = []
    for model in models:
        natural = model._ctc_decode(model._infer(x[None]))[0]
        padded = np.ones((1, 3, 48, 640), np.float32)
        padded[:, :, :, :357] = x
        prediction = model._ctc_decode(model._infer(padded))[0]
        predictions.append((natural, prediction))
    # Direction check on an existing public illustrative example, not a benchmark.
    reference = json.loads(EXAMPLE.with_suffix(".json").read_text())["reference"]
    assert predictions[0][0] == predictions[1][0]
    assert scores(reference, predictions[1][1])["character_errors"] < scores(reference, predictions[0][1])["character_errors"]


def test_shared_crops_and_batches(models):
    from examples.compare_wsp import compare_page
    from occular.dbnet_detector_onnx import DBNetDetectorONNX
    image = np.asarray(Image.open(EXAMPLE).convert("RGB"))
    quads = DBNetDetectorONNX(gpu=False, num_threads=2).detect(image)
    assert len(quads) > 0
    original, wsp, batches = compare_page(image, quads, *models)
    assert len(original) == len(wsp) == len(quads)
    assert batches and all(row["dtype"] == "float32" for row in batches)


def test_public_api_default_explicit_and_switching(raster_pdf):
    import occular
    from occular.registry import Registry
    options = {"lm": False, "reading_order": False, "deskew": False, "num_threads": 2}
    default = occular.ocr(str(EXAMPLE), **options)
    original = occular.ocr(str(EXAMPLE), recognizer="svtr_lcnet", **options)
    wsp = occular.ocr(str(EXAMPLE), recognizer="svtr_lcnet_wsp", **options)
    original_again = occular.ocr(str(EXAMPLE), recognizer="svtr_lcnet", **options)
    assert default and default == original == original_again
    assert isinstance(wsp, str) and wsp
    a = Registry.get_recognizer("crnn-onnx", arch="svtr_lcnet", lm=False, gpu=False, num_threads=2)
    b = Registry.get_recognizer("crnn-onnx", arch="svtr_lcnet_wsp", lm=False, gpu=False, num_threads=2)
    assert a is not b and a._onnx_file != b._onnx_file
    for arch in ["svtr_lcnet", "svtr_lcnet_wsp"]:
        pages = occular.ocr(str(raster_pdf), recognizer=arch, **options)
        assert len(pages) == 1 and pages[0]
        pipe = occular.OCRPipeline(recognizer=arch, **options)
        forced = pipe.process_pdf(str(raster_pdf), dpi=72, force_ocr=True, workers=1)
        assert len(forced) == 1 and forced[0]["results"]


@pytest.mark.parametrize("arch", ["svtr_lcnet", "svtr_lcnet_wsp"])
def test_cli_beam_image_and_pdf(arch, raster_pdf):
    # Existing CLI uses beam; no alternate decoding settings are injected.
    env = dict(os.environ, PYTHONPATH=str(ROOT), RAYON_NUM_THREADS="2")
    for path in [EXAMPLE, raster_pdf]:
        result = subprocess.run([sys.executable, "-m", "occular", str(path), "--model", arch,
                                 "--no-reading-order", "--workers", "1"], env=env,
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip()
