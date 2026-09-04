# Changelog

## 0.4.1

- **New default recognizer `svtr_lcnet` — about 5× faster on CPU.** A lighter architecture
  (LCNet stem + 6 global blocks) replaces `svtr_t` as the default: recognition of a dense page
  drops from ~2.4 s to ~0.45 s at 4 threads, at 99.6 % of the old model's accuracy
  (greedy CER 1.19 % for Russian/English, 0.53 % for the 12 Cyrillic languages); with the default
  beam+LM decoding the two are on par. Both the Russian/English and the Cyrillic-12 recognizers
  ship in the new architecture.
- **The previous model is still available**: `ocr(img, recognizer="svtr_t")`,
  `Settings(recognizer="svtr_t")` or `occular img.png --model svtr_t`. Models are now selected by
  architecture name (`svtr_lcnet`, `svtr_t`) instead of an internal registry name; the old registry
  names keep working. The GPU path supports `svtr_t` only — asking for `svtr_lcnet` with `gpu=True`
  warns and recognizes on CPU.
- **Page orientation detection (new, off by default).** `ocr(img, orientation=True)`,
  `Settings(orientation=True)` or `occular img.png --orientation` detects a 0/90/180/270° rotation
  and straightens the page before detection — for phone photos and bulk scans that arrive sideways.
  A 1.2 M-parameter model (~4.8 MB, ~28 ms at 8 threads) votes over all four rotations. A rotation is applied
  only when confidence is at least 0.8: below that the page is left untouched, because on inputs
  that are not documents (logos, photographs of scenes) the model can be confidently wrong.
  Runs before deskew, which handles the remaining few degrees of skew.
- **GPU support for the new default recognizer.** `gpu=True` now runs `svtr_lcnet` natively on
  PyTorch/CUDA for both Russian/English and the 12 Cyrillic languages — verified to produce the
  same text as the CPU (ONNX) path line for line. Previously only `svtr_t` had CUDA weights and
  asking for the default with `gpu=True` fell back to CPU.
- **Smaller detector graph.** The exported detector kept two training-only branches (`thr`, `binr`)
  that inference never reads; they are gone (240 → 222 nodes). Detector output is bit-identical —
  verified on 20 pages and 1100 boxes, `max|Δ| = 0`.

## 0.3.2

- **Multilingual recognition (12 more languages).** Pass `languages=` to read 12 additional
  Cyrillic-script languages beyond the default Russian/English: Bashkir, Belarusian, Bulgarian,
  Chuvash, Kazakh, Kyrgyz, Macedonian, Mongolian, Serbian, Tajik, Tatar, Ukrainian.
  `ocr(img, languages=["uk"])` for a single language (fastest), `languages=["ru", "kk"]` to detect
  the language per line, or `languages="auto"` to auto-detect across all supported languages.
  Also on `OCRPipeline(...)`, `Settings(languages=...)`, and the CLI (`--languages` / `--lang`).
- Each language decodes with its own tuned settings (a per-language language model where it helps,
  fast greedy decoding where it doesn't). The recognizer and only the requested language's weights
  download from the Hub on first use. More languages will be added in future minor releases.
- The default Russian/English path is unchanged — same model, same output, no extra downloads.

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
