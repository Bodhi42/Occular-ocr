# Experimental Weight-Space Programming recognizer

Occular-OCR 0.5.2 adds `svtr_lcnet_wsp`, an optional derivative of the original
RU/EN SVTR-LCNet. An AI coding agent traced padding-induced recognition errors
to squeeze-and-excitation gates and compiled a conditional compensator into
384 existing weights and one bias. There was no gradient-based training or
addition of operations. DBNet has no accepted WSP edit and is unchanged.

## Select the alternative

```python
from occular import ocr, OCRPipeline

original = ocr("document.png")  # original SVTR-LCNet remains the default
original = ocr("document.png", recognizer="svtr_lcnet")
modified = ocr("document.png", recognizer="svtr_lcnet_wsp", gpu=False)
# Also supported by OCRPipeline, ocr_detailed, and image/PDF inputs.
```

```bash
ocr document.png --model svtr_lcnet
ocr document.pdf --model svtr_lcnet_wsp
```

The alternative supports the original Russian/English vocabulary only. Leave
`languages=None`, or explicitly choose `ru`, `en`, or both. Other languages,
`languages="auto"`, Cyrillic-12 models, and multilingual routing raise a clear
error. `gpu=True` / `--gpu` also raises an error for this variant: the research
CUDA numerical check failed on 18 of 48 pairs. Use CPU. These restrictions do
not affect existing models.

## Weights and integrity

The original file remains `recognizer_svtr_lcnet_fp32.onnx`. The new optional
file is `recognizer_svtr_lcnet_wsp_fp32.onnx`, using the same
`recognizer_charset.txt` and CTC blank index 0 (186 output classes).

| Artifact | SHA-256 |
|---|---|
| Original ONNX | `3de3ad346181af4e4446e6c365d5d9a8bfb187ad803d6f419dd61ad3509a65f2` |
| WSP ONNX | `24c5db4e64ad0a9bb8859b501670eff68969264ff0bee8af7130e3eb50f565f6` |
| RU/EN charset | `b4943823e5f46eb162fb1bce8ba341da89729cb62ed0617ee1c1f27ea0497303` |

Only three tensors differ: `bb.body.3.se.f.1.weight` (128 values),
`bb.body.3.se.f.1.bias` (one), and `bb.body.3.se.f.3.weight` (256).
The graph, all other parameters, and metadata are identical.

A local file in the installed package's `occular/weights/` takes priority.
Otherwise the existing Hugging Face loader uses `Shivin11/occular-ocr`.
Only choosing WSP requests its checkpoint. Ordinary use never requests or
switches to it. `occular.model_info()` lists it as optional. WSP verifies the
checkpoint and charset hashes; a mismatch or missing download raises an
explicit error, with no fallback to a different model. An environment pinned
to an older `OCCULAR_WEIGHTS_REVISION` may not contain this new file.

The checkpoint is not included in the wheel or sdist. Its public Hugging Face
revision is `4848df59dd8f10bee7360366e8f2f067678bcda8`; the frozen research
compiler also reconstructs its exact bytes. See the release instructions. Derivative weights
retain [the original model-weights terms](../WEIGHTS_LICENSE.md), including
commercial-use conditions. Apache-2.0 covers source code, not these weights.

## What has been measured

The strongest confirmed benefit is robustness to white right padding while
retaining normal batch processing. This is not evidence of general OCR
improvement on arbitrary inputs. Natural-width inference provides comparable
accuracy and is an alternative when padding can be avoided.

The frozen scientific beam harness measured the following character error
rates, with identical inputs and decoding for each model:

| Condition | Original CER | WSP CER |
|---|---:|---:|
| Real-document page buckets | 8.619% | 8.212% |
| Synthetic natural width | 2.844% | 2.844% |
| Synthetic 2× padding | 3.279% | 2.927% |
| Synthetic 4× padding | 4.863% | 2.915% |

All 2,048 natural-width synthetic beam predictions were preserved. Some
individual padded predictions regress. The real-document result is a
replication on previously used documents with AI-produced reference
transcriptions, not a blind human-annotated test. Those private images and
transcriptions are not redistributed.

These are measurements of the fixed research protocol, not a promise that the
current library runtime reproduces the same aggregate numbers. Its later beam
wrapper and width buckets differ from the scientific per-line padding sweep.
[The public reproduction harness](../research/weight_space_programming/README.md)
keeps that evaluation separate from release API checks.

## Compare your own inputs

From a source checkout with Occular 0.5.2 installed:

```bash
python examples/compare_wsp.py document.pdf --html comparison.html --json comparison.json
python examples/compare_wsp.py document.png --greedy --html comparison-greedy.html
```

One detector pass supplies both recognizers with the same crops. The script
checks equality of every preprocessed input batch. It writes a standalone
HTML report and optional JSON, with changed characters highlighted. A changed
answer is not labelled an improvement without reference text. Optional
`--ground-truth refs.json` accepts `{"p1-l1": "reference", ...}` keyed by the
report's one-based page/line identifiers and adds CER/WER and regression labels.
Reports contain your input crops and predictions; choose their destination
and sharing accordingly.
