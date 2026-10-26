# Changelog

## 0.5.0

- **Layout regions (new, optional): `LayoutDetector`.** Returns the regions of a page — box in
  pixels of the input image, score, one of 18 labels — plus the page size the boxes refer to.
  D-FINE-S detector, ONNX on CPU (~0.3 s per page at 4 threads), weights `layout_dfine_s144k_fp32.onnx`
  (~42 MB) downloaded on first use. Default score threshold 0.6. Labels are informational only:
  nothing is removed or skipped because of a label, and `TableRecognizer` is unchanged.
- **Experimental reading order from layout regions (off by default).**
  `OCRPipeline(experimental_layout_order=True)` / `Settings(experimental_layout_order=True)` reorders
  the lines of the default mode by layout regions. The objects to order are the text regions plus
  every line that no region covers by more than 30 % (instead of moving such lines to the end); inside
  a region lines go top to bottom. The objects are ordered by a small trained block-order model
  (`layout_order_method="model"`, default; 3.3M parameters, ONNX on CPU, ~5 ms per page, weights
  `layout_order_blocks_fp32.onnx` ~13 MB downloaded on first use) or by a recursive XY-cut
  (`layout_order_method="xycut"`). On held-out pages the two agree on most pages and are of similar
  quality. Lines and text are identical to the default mode; each line gains `layout_region` and
  `layout_label`. Mutually exclusive with `reading_order=True`, which is unchanged.
- **Removed the `ocr_skel` alias.** The package was renamed `ocr_skel` → `occular` in 0.3.0 and the old
  name has warned as deprecated since. Use `import occular` and `python -m occular`.
- **Native decoder `occular-decode` 0.2.0 scales under load** (optional package, wheel in
  `native/wheels/`). 0.1.0 copied every line's logits element by element while holding the GIL and the
  code lock ran a per-line numpy argmax under the GIL, so concurrent requests queued up: 16 parallel
  requests decoded 13k lines/s and 64 requests only 9.9k. Now the logits are read in place, lines are
  handed out one per task with the heaviest first, and the acoustic greedy reading comes from the same
  parallel pass: 49k lines/s at 16 requests, 47k at 64; one 77-line page at 16 threads 104 → 70 ms.
  Same text, byte for byte. `occular` uses the new path automatically and still works with 0.1.0.
  A macOS arm64 wheel (macOS 11+) is built as well: same text as pure Python on 599 lines, 15–50× faster
  decoding.
- **No PyMuPDF deprecation warning.** occular imported PyMuPDF as `fitz`, and PyMuPDF 1.28 prints
  `The fitz API is deprecated` on that import — once per run, and once per worker process with
  `workers`. It now imports `pymupdf` (falls back to `fitz` on PyMuPDF older than 1.24.3).
- **All console output is in English** (#5): warnings, errors, `model_info()`, `print(Settings())` and the
  CLI help and messages used to mix Russian and English (e.g. the LM line `[occular] LM (чисто-Python): …`).
  A test now keeps Russian text out of the messages.
- **Multi-page work scales across cores.** `process_pdf(workers=N)`, `ocr(...)` on PDFs and the CLI
  (`--workers`, now also for a whole folder) run pages in worker *processes* taking pages from one queue,
  instead of threads of one process. Threads were limited by Python's GIL: with the default pure-Python
  decoder, PDF throughput peaked at 8 threads (0.66 pages/s on a 64-core machine) and then dropped
  (0.44 at 128); a folder in the CLI was processed one file at a time whatever `--workers` said. Now:
  1.57 pages/s at 8 workers, ~3 pages/s on a 512-page PDF with 16–128 workers. Each worker holds its
  own models (~1.1 GB RAM; ONNX Runtime's memory arena is off in workers and freed memory is returned
  after every page — with them a worker grew to ~1.9 GB); auto `workers` is limited by cores, free memory and 16. Processes need the
  `if __name__ == "__main__":` guard in the calling script; without it occular uses threads (up to 8)
  and warns once. Default `num_threads` (threads per page) is now `min(cores, 8)` instead of 4 (one
  page: 2.35 → 1.64 s). The cores of the machine are split between workers by *physical* cores, read from
  the CPU topology on Linux and macOS (on Apple Silicon all cores are physical: 6 workers on a 12-core M-series
  chip get 2 threads each, 2.19 → 2.42 pages/s). The CLI folder mode now also reads `.gif`.
- **Tables in the OCR pipeline: `Settings(tables=True)`** (also `OCRPipeline(tables=True)`,
  `ocr(..., tables=True)`, `ocr_detailed(..., tables=True)`, CLI `--tables`; off by default). Tables are
  found on the same page image the lines are read from (after deskew/rotation/dewarp), and the
  recognized lines are put into cells: lines inside a table gain `table` and `cell`
  (`r`, `c`, `rowspan`, `colspan`); `process_image` returns the usual list of lines with a `.tables`
  attribute, PDF pages gain a `tables` key (on text-layer pages, in PDF points). Each table has its box,
  row/column bands and cells with text. Plain text (`ocr`, `get_text`, CLI) prints tables as Markdown
  tables in place. A line the detector glued across a cell border is divided at the whitespace by that border
  (marked `"split": True`); the parts keep the original recognized text, cut at a space or a
  letter/digit boundary located from the line's CTC alignment, never inside a word. `TableRecognizer` is unchanged.
- **`TableRecognizer` on GPU: tables with a single row or a single column** crashed the merged-cell
  step with a CUDA error (`BiasGelu: cudaErrorInvalidConfiguration` — the step has no cell edges along
  one axis, and CUDA rejects the empty tensors), which failed the whole page. Such tables now run that
  small step (2 MB) on CPU; results match the CPU path. On our test set 47 of 1830 pages had such a table.
- **New table models in `TableRecognizer`.** Trained on our own visually annotated corpus of real
  documents (forms, statements, invoices, books, scans). Measured on its held-out part:
  - **Detection** is now a D-FINE-S box detector (`table_detect_dfine_v4_fp32.onnx`, 640×640, default
    threshold 0.75): one box and score per table, so neighbouring tables no longer merge. F1 0.82
    (0.43 for the previous segmentation detector).
  - **Structure** (`table_struct_v4_main/merge_fp32.onnx`): row/column grid F1 0.92 (0.80 before),
    merged cells F1 0.83 (0.58 before). On public benchmarks the grid F1 is 0.97 (PubTables-1M,
    PubTabNet) and 0.88 on FinTabNet, which the model never saw.
  - **Fixed: empty and missing rows/columns.** The number of rows/columns is now taken from the
    detected separators instead of a separate count prediction. When the two disagreed, the grid got
    zero-height rows or lost the last rows of the table. Merged cells are now computed on the final
    grid.
  - **Tall tables** (higher than half their width) are no longer squashed into a square: their rows
    are found on overlapping horizontal strips and stitched by row lines; columns and merged cells
    use the whole table. Row F1 on tables with more than 50 rows: 0.89 (was close to 0).
  - `detect(..., min_frac=...)` still works (drops boxes smaller than this fraction of the page; default 0).
- Without `TableRecognizer` and the new flag the output is byte-identical to 0.4.6.
- **Installation never compiles on supported platforms** (issue #4). pip takes the newest allowed version
  of a dependency and, if it has no wheel for your system, builds it from source instead of falling back
  to an older wheel. On macOS 11–12 this compiled OpenCV 4.11 for ~45 minutes, and on Intel Macs it would
  build PyMuPDF 2.0. macOS now gets OpenCV 4.10.0.84 (wheels for Apple silicon on macOS 11+ and Intel on
  12+); Intel Macs with Python ≤ 3.12 get OpenCV 4.9.0.80 with NumPy < 2, which also covers macOS 11. On Linux aarch64 the plain
  `onnxruntime` is installed instead of `onnxruntime-gpu`, whose ARM wheels need glibc 2.34+ (installation
  failed on Ubuntu 20.04 / Debian 11).
- **Windows: CLI output redirected to a file or pipe** (`ocr scan.pdf > out.txt`, `ocr --help | more`)
  failed with `UnicodeEncodeError`: Python wrote Cyrillic in the locale code page (cp1252). The CLI now
  writes UTF-8 when its output is not already UTF-8; the console is unchanged.
- **Windows without the Visual C++ Redistributable** (a clean install) could not load pyclipper, ONNX
  Runtime and PyMuPDF and failed with a bare "DLL load failed". There is now a separate
  `py3-none-win_amd64` wheel that carries the five Microsoft C++ runtime DLLs these modules need
  (unmodified, from the official redistributable, 14.51); `import occular` loads them only when Windows
  has no copy of its own. pip on 64-bit Windows picks this wheel automatically. Elsewhere (e.g. Windows
  on ARM) `import occular` still says what is missing and links Microsoft's installer.
- **PyMuPDF < 2.0 everywhere.** PyMuPDF 2.0 (released 2026-10-19) depends on the CPU build of
  `onnxruntime`, which collides with `onnxruntime-gpu` (two builds of one module overwriting each other's
  files), and pulls in packages occular does not use; it also has no Intel-Mac wheels.
  `tools/check_wheels.py` replays pip's choice for every platform and Python version against PyPI; run
  it before each release.

## 0.4.6

- **No PyTorch anywhere — every model is ONNX, on CPU and on GPU.** `gpu=True` now runs the same
  `.onnx` files on CUDA through ONNX Runtime instead of a separate PyTorch backend with `.pth`
  weights. TF32 is disabled on GPU so the text matches the CPU path.
- **Table merged cells and UVDoc dewarping no longer need torch.** The table structure model
  decides how many rows/columns to process from its own outputs, so it cannot be traced into one
  graph (tracing silently froze the grid of the example). It is now split into two ONNX graphs
  with the data-dependent steps in numpy between them (`table_struct_v3_main_fp32.onnx`,
  `table_struct_v3_merge_fp32.onnx`); on 60 real tables the grid and merged cells are identical to
  the torch model, at ~4× the speed. UVDoc is one ONNX graph (`model.onnx` in the model repo).
  The `[unwarp]` extra is kept as an empty alias.
- **Installation no longer touches your numpy.** `pyctcdecode` (pins `numpy<2`) is gone: its beam
  search is vendored as `occular/_beam.py` (Apache-2.0, see `LICENSE-pyctcdecode`) with
  bit-identical output and ~25 % faster pure-Python decoding. `opencv` is capped below 4.12, whose
  `numpy>=2` requirement made pip upgrade numpy and break packages built for numpy 1. Together this
  removes the source build of numpy 1.26 on Python 3.13/3.14 that failed without a compiler.
- **`opencv-python-headless` instead of `opencv-python`** — the GUI build needs `libGL` and failed to
  import in slim Docker images.
- **ONNX Runtime build chosen per platform.** `onnxruntime-gpu` (also runs on CPU, same speed and
  output) on Linux x86_64 / Windows x64 with Python 3.10+ and Linux aarch64 with 3.11+; plain
  `onnxruntime` elsewhere. `occular-ocr[gpu]` adds CUDA and cuDNN from pip, loaded via
  `onnxruntime.preload_dlls()` — only an NVIDIA driver is needed.
- **Tables are returned in reading order** (top to bottom, then left to right) instead of by detector
  confidence: confidences of tables on one page often agree to the 4th digit, so the order flipped
  on float noise (e.g. between CPU and GPU).
- **Clashing installs are handled without breaking neighbours.** Packages such as chromadb or
  fastembed pull plain `onnxruntime`, ultralytics pulls `opencv-python`; pip installs them into the
  same module as Occular's builds. Occular stays quiet while that works. If `gpu=True` finds the GPU
  build overwritten, the fallback message gives `pip install --force-reinstall --no-deps
  onnxruntime-gpu==<version>`, which restores CUDA and keeps every package's requirements satisfied.
  A warning is printed on import only when onnxruntime versions differ or `cv2` fails to import.
- **Minimum versions raised to ones that actually work** (checked by installing the lowest allowed
  versions on Python 3.8–3.14): `onnxruntime>=1.19` (older builds lack the CPU `Trilu` kernel, so
  reading order failed to load, and do not support numpy 2), `opencv-python-headless>=4.10.0.84`
  (older builds do not import under numpy 2), `pymupdf>=1.23.1` (1.23.0 needs `libcrypt.so.2`).
  A PyMuPDF import failure now shows the real cause instead of "install pymupdf".

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
