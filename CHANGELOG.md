# Changelog

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
- **Lazy ONNX components:** `import ocr_skel` succeeds without heavy optional dependencies present.

## 0.1.0

- First release: text detector + text recognizer on ONNX Runtime.
