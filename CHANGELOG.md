# Changelog

## 0.3.1

- **`--json` output is now clean JSON.** Model-loading progress messages ("Loaded ...", language-model
  info) now go to stderr instead of stdout, so `occular img.png --json > out.json` produces valid,
  parseable JSON.
- **`python -m occular` compatibility fix.** The deprecated `ocr_skel` alias could be imported but
  `python -m ocr_skel` failed; the compatibility shim now keeps the alias a real package so both the
  import and the `-m` form work. (`python -m occular` was already fine.)

## 0.3.0

- **Package renamed `ocr_skel` → `occular`.** Import `occular` now (`from occular import ocr`).
  The old `ocr_skel` name keeps working as a deprecated alias, so existing code doesn't break.
- **Table recognition (new `TableRecognizer`).** Detects tables on a page and reconstructs their
  structure — the row/column grid plus merged cells (colspan/rowspan). Detection and the grid run
  on ONNX (CPU, torch-free); merged-cell reconstruction uses a small PyTorch model on CPU when
  `torch` is installed, and otherwise falls back to grid-only. See `occular.tables.TableRecognizer`.
- **Optional native (Rust) decoder.** An optional `occular-decode` module accelerates beam+LM
  decoding 5–13× per line / 17–48× per page with **byte-identical** output; picked up automatically
  when installed, pure Python otherwise. See `native/`.
- Weights continue to download from the Hub on first use (not bundled in the wheel);
  `model_info()` lists what's present locally.

## 0.2.2

- **GPU now runs on PyTorch/CUDA** instead of onnxruntime-gpu (which was fragile across CUDA/cuDNN
  versions). `pip install occular-ocr[gpu]` pulls in torch + torchvision; the PyTorch weights
  download from the Hub on first GPU use, and the output matches the CPU (ONNX) path bit-for-bit.
  If PyTorch/CUDA is unavailable, `gpu=True` warns and falls back to CPU. The default CPU path is
  unchanged and stays torch-free.
- **Better confidence scores.** Per-line confidence now reflects the chosen beam+LM hypothesis
  (length-normalized acoustic score) instead of the old worst-frame heuristic, so it separates
  correct from incorrect lines far better. Recognized text is unchanged.
- Trimmed the benchmarks/methodology section from the README.

## 0.2.1

- **CLI batch mode (folder → `.txt`).** `ocr ./scans ./out` now processes every image/PDF in a
  folder and writes one `.txt` per file (output folder optional — defaults to alongside the
  sources). Previously only a single file was accepted; the documented folder usage now works.
- **Lazy language model.** The ~270 MB LM is built on first *recognition*, not at pipeline
  construction, so vector PDFs (text layer) and empty inputs no longer pay for it. Thread-safe.
- **PDF memory fix.** In parallel mode pages are rendered one at a time inside each worker instead
  of rasterizing the whole document into RAM up front — no more OOM risk on large scans.
- **Friendlier errors.** `ocr()` / `ocr_detailed()` now raise a clear `FileNotFoundError` /
  `ValueError` on a missing path or a non-image file instead of a raw library traceback.
- Removed a dead `--onnx` no-op flag and an unused pipeline method; `tests/` no longer ships in the
  source distribution; docstrings document the `lm` option.

## 0.2.0

- **Beam search + language model, on by default.** A 4-gram Russian language model rescoring the
  CTC beam cuts word errors ~18–25 % over greedy decoding, with no model retraining. Turn it off
  with `OCRPipeline(lm=False)`.
- **Pure-Python decoding stack — zero native dependencies.** The language model and beam search are
  implemented entirely in Python, so `pip install` works on every platform with no compiler and
  nothing to build. Decoding quality and speed are unchanged.
- **Compact language model** (`compact_lm.npz`, ~270 MB) that loads in seconds and streams from the
  weights host on first use; a local override is available via `OCCULAR_LM_DIR`.
- **Upgraded text detector** — fuller line boxes (less clipping), which markedly improves recognition
  on dense documents.
- **Weights fetched on demand** from the Hugging Face Hub and cached locally; the optional
  reading-order model downloads only when enabled.
- **CPU thread fix:** inference threads are bounded (`num_threads`, default 4) instead of grabbing
  every core, so batch/parallel runs no longer saturate the machine.
- **Lazy ONNX components:** `import occular` succeeds without heavy optional dependencies present.

## 0.1.0

- First release: text detector + text recognizer on ONNX Runtime.
