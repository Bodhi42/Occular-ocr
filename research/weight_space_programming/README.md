# Frozen Weight-Space Programming reproduction

This directory supplies the fixed synthetic inputs, reference strings,
recorded predictions, exact weight recipe, and evaluated beam wrapper for
**Weight-Space Programming for OCR: Repairing Padding-Induced Errors by Editing
385 Parameters**. It is separate from the 0.5.2 API integration. No private
page images or reference transcriptions are included. No public preprint URL
is claimed yet.

## Artifacts and provenance

- `synthetic_2048.tar.gz`: the existing 2,048 renderer-generated records
  (2,046 unique PNG hashes), original text references and recorded answers.
  Manifest schema was normalized for portable paths after the experiment;
  image bytes, references and predictions were not changed. The original
  freeze and provenance are included. Check `fixture_sha256.json`.
- `recipe.json` and `compile_guard.py`: dataset-free deterministic compiler
  for the fixed guard and ablation controls, with full output SHA checks.
- `frozen_decoder/`: evaluated wrapper, code-lock helper and loader under an
  isolated import namespace. Source license and notice are included.
- `vendor/occular_decode-0.1.0-cp38-abi3-manylinux_2_34_x86_64.whl`: the exact
  original native decoder, Apache-2.0, Linux x86-64 / glibc >=2.34 / Python
  >=3.8. SHA-256:
  `9a17459661049a16b3d04971590e076bf3b61388578020e0e3e999589afeddf2`.
  This is a scientific dependency only; the library's current optional native
  decoder is not replaced. Other platforms need a separate validated build.
- `padding_example.png` and `.json`: an existing public synthetic illustration
  from the separate 576-render cohort. The example was selected post hoc for
  illustration, not used to change the intervention or success criteria.
- `real_replication_counts.json`: anonymous sufficient counts from the real
  replication. Private images and reference strings are unavailable here, so
  real-document inference is not publicly reproducible from this package.

The exact charset includes a literal space at line 129; do not strip its lines.

The fixture contains generated fictitious strings, not extracted bank or
personal documents. Rendered examples are released under the repository's
Apache-2.0 terms; fonts used to render the images are not bundled. Model/LM
weights retain their own existing terms in `WEIGHTS_LICENSE.md`.

## Run the scientific beam replay

Use a dedicated Python 3.10–3.12 environment on Linux x86-64; the pinned NumPy
release needs a compatible Python version. From the repository root:

```bash
python3.12 -m venv /tmp/occular-wsp-replay
/tmp/occular-wsp-replay/bin/pip install -r research/weight_space_programming/requirements.txt
/tmp/occular-wsp-replay/bin/pip install research/weight_space_programming/vendor/occular_decode-0.1.0-cp38-abi3-manylinux_2_34_x86_64.whl
/tmp/occular-wsp-replay/bin/pip install 'huggingface_hub>=0.20'
```

Obtain the exact original and WSP ONNX files from `Shivin11/occular-ocr` at
revision `4848df59dd8f10bee7360366e8f2f067678bcda8`, or use verified local
files. Expected hashes:

```
original  3de3ad346181af4e4446e6c365d5d9a8bfb187ad803d6f419dd61ad3509a65f2
wsp       24c5db4e64ad0a9bb8859b501670eff68969264ff0bee8af7130e3eb50f565f6
```

Download `compact_lm.npz` and `unigrams.txt` from `Shivin11/occular-lm-ru` at
revision `d6f666e1180c190c8ef79ac340036d5445ee0360`, using `hf_hub_download`
or an existing verified cache. Pass the directory containing both files.
Their SHA-256 values are checked before decoding:

```
compact_lm.npz b2f60561802930b7bf5a1926253a4e3279e7507f701d8612bc631b6070e49e9d
unigrams.txt   290deae673303ca03b067153a4d9d04c52d316bb6487931744c6656e11c08109
```

```bash
/tmp/occular-wsp-replay/bin/python research/weight_space_programming/reproduce.py \
  --original /path/to/recognizer_svtr_lcnet_fp32.onnx \
  --wsp /path/to/recognizer_svtr_lcnet_wsp_fp32.onnx \
  --lm-dir /path/to/pinned-lm \
  --output /tmp/wsp-scientific-replay.json
```

The script verifies every image hash and runs both CPU models on identical
FP32 BGR inputs: height 48, normalization to [-1,1], natural width and white
right padding at 2×/4×, capped at 640. Each view is a separate batch of one.
The frozen beam uses width 100, alpha 0.5, beta 1 and code locking. JSON reports
error counts, CER/WER, all predictions and mismatches against recorded answers.
It never silently substitutes the installed library's later beam wrapper.
`--greedy` replays the recorded greedy comparison. Explicit
`--allow-runtime-drift` permits an alternate ORT version and records it; such
results do not replace the scientific ones.

Expected beam CER:

| Synthetic view | Original | WSP |
|---|---:|---:|
| Natural width | 2.844% | 2.844% |
| 2× padding, cap 640 | 3.279% | 2.927% |
| 4× padding, cap 640 | 4.863% | 2.915% |

The release-preparation replay matched every recorded prediction, with all
2,048 natural-width answers identical across models. Real-document CER
8.619% → 8.212% and WER 18.741% → 18.025% remain historical replication
measurements, not a new blind test. References there were produced by AI
visual annotation, not certified human ground truth.

## Rebuild the exact edited model if necessary

Use a separate development environment with ONNX 1.22.0 and NumPy 2.5.3 (the
recorded compiler versions), leaving the source file untouched:

```bash
python research/weight_space_programming/compile_guard.py \
  --source /path/to/recognizer_svtr_lcnet_fp32.onnx \
  --recipe research/weight_space_programming/recipe.json \
  --output /tmp/wsp-compiled
```

The compiler refuses existing output files and rejects a differing full-file
hash. It applies the already frozen recipe; it does not fit or search for a
new intervention. `lcnet_guard.onnx` must have the WSP hash above.

## Verify the release runtime separately

After installing the built 0.5.2 wheel and making the exact models available:

```bash
OCCULAR_WSP_INTEGRATION=1 python -m pytest tests/test_wsp_integration.py -q
python examples/compare_wsp.py research/weight_space_programming/padding_example.png \
  --html /tmp/wsp-api.html --json /tmp/wsp-api.json
```

These checks use the current public API, crop processing, width buckets and
beam wrapper. They verify integration; the scientific sweep bypasses DBNet
and explicitly controls padding. Do not label their metrics as interchangeable.
Neither CUDA validation nor a generic recognition improvement is claimed.
