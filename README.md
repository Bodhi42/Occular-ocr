<div align="center">

# Occular-OCR

**State-of-the-art OCR for Russian documents — with a zero-compilation install.**

[![Code License: Apache 2.0](https://img.shields.io/badge/Code-Apache%202.0-2f5ff0.svg)](LICENSE)
[![Weights: OpenRAIL-M](https://img.shields.io/badge/Weights-AI%20Pubs%20OpenRAIL--M-e0533a.svg)](WEIGHTS_LICENSE.md)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-16181d.svg)](#installation)
[![pip install — no compiler](https://img.shields.io/badge/install-pure%20pip%20·%20no%20compiler-0f9d58.svg)](#installation)

<sub>🇬🇧 **English** &nbsp;·&nbsp; <a href="#russian">🇷🇺 Русский</a></sub>

</div>

Occular-OCR is a document OCR library purpose-built for **Russian**. It pairs a DBNet text detector with an
SVTR recognizer and a beam-search decoder guided by a Russian language model — and reads Russian
documents **markedly better than existing open-source OCR**, especially the hard stuff: forms,
certificates, bank statements, IDs, receipts.

It installs with a single `pip install` on Windows, Linux and macOS — **no C/C++ toolchain, no CUDA,
nothing to build.**

---

## See it in action

<p align="center">
  <img src="assets/demo_passport.png" width="24%" alt="Detected lines on a specimen passport">
  <img src="assets/demo_2ndfl.png"    width="24%" alt="Detected lines on a synthetic income certificate">
  <img src="assets/demo_book.png"     width="24%" alt="Detected lines on a printed book page">
</p>
<p align="center">
  <img src="assets/demo_schet.png"    width="37%" alt="Detected lines on a synthetic VAT invoice">
  <img src="assets/demo_invoice.png"  width="37%" alt="Detected lines on a synthetic payment invoice">
</p>

<p align="center"><sub>Detected text lines on <b>synthetic / specimen / public-domain</b> documents — a passport, a tax certificate, a printed book page, a VAT invoice, a payment invoice.</sub></p>

## Why Occular-OCR

Occular-OCR is built for Russian and reads it dramatically better than existing open-source OCR:
**~35 % fewer word errors than the next-best open-source engine** in an end-to-end benchmark, and
ranking **#1** against large vision-language models and other leading OCR engines on a page-level
document benchmark. The gap is widest exactly where general-purpose OCR struggles: dense forms,
certificates, invoices and IDs.

- **Russian-first accuracy.** Trained and tuned on Russian document data. Where other engines guess,
  Occular-OCR reads.
- **Beam + language model, on by default.** A Russian n-gram language model cuts word errors
  ~18–25 % over plain greedy decoding — with **no model retraining**, just better decoding.
- **Installs anywhere, no compiler.** The decoder is pure Python end to end, so you get top-tier
  decoding quality with **zero native dependencies**.
- **CPU-first, GPU-ready.** Ships as ONNX and runs on CPU out of the box; set `gpu=True` to run the same models on CUDA via ONNX Runtime (`onnxruntime-gpu`).
- **Batteries included.** Images and PDF, folder → `.txt`, reading order, automatic deskew.

---

## Installation

```bash
pip install occular-ocr
```

That's it. No build tools, no CUDA. Model weights download automatically on first use and are cached
locally.

### GPU (optional)

GPU runs the **same ONNX models on CUDA** through ONNX Runtime's `CUDAExecutionProvider` — there are
no separate GPU weights. You only need the GPU runtime:

```bash
pip install occular-ocr[gpu]      # pulls in onnxruntime-gpu
# equivalent to:  pip install occular-ocr onnxruntime-gpu
```

Then pass `gpu=True` (see below). If CUDA isn't available it falls back to CPU with a warning.

---

## Quick start

```python
from ocr_skel import ocr

text = ocr("document.png")        # an image, or a "scan.pdf"
print(text)
```

Line-level output with coordinates and confidence:

```python
from ocr_skel import ocr_detailed

for line in ocr_detailed("document.png"):
    print(line["text"], round(line["confidence"], 2), line["quad"])
```

A whole folder → text files, from the command line:

```bash
ocr ./scans ./out          # writes ./out/<name>.txt for every image/PDF
```

> First run downloads the model weights automatically (and caches them). No build tools, no CUDA.

## Configuration

All behaviour is controlled by `Settings` (or the equivalent keyword arguments on `ocr`). Print the
current defaults at any time:

```python
from ocr_skel import Settings
print(Settings())
```

| Setting | Default | What it does |
|---|---|---|
| `num_threads` | `None` | CPU threads for inference. `None` → `min(cores, 4)`. |
| `gpu` | `False` | Run the ONNX models on **GPU via ONNX Runtime (CUDA)**. Needs `onnxruntime-gpu` (`pip install occular-ocr[gpu]`); falls back to CPU if unavailable. |
| `deskew` | `True` | Auto-correct skewed / rotated scans before detection. |
| `lm` | `True` | Beam search + language model (best quality). `False` → fast greedy decoding, skips the LM download. |
| `reading_order` | `False` | Order lines for multi-column layouts (downloads a small model on first use). |
| `detector` | `None` | Explicit detector name. `None` → default. |
| `recognizer` | `None` | Explicit recognizer name. `None` → default. |

### Full pipeline with every setting

```python
from ocr_skel import OCRPipeline, Settings

pipe = OCRPipeline(Settings(
    num_threads=8,        # CPU threads (None -> min(cores, 4))
    gpu=False,            # True -> ONNX Runtime CUDA (needs occular-ocr[gpu])
    deskew=True,          # auto-correct skewed scans
    lm=True,              # beam + language model; False -> greedy (faster, no LM download)
    reading_order=False,  # multi-column reading order (optional model, see below)
    detector=None,        # None = default detector
    recognizer=None,      # None = default recognizer
))

result = pipe.process_image("document.png")
# -> [{"quad": [[x, y], ...], "text": "...", "confidence": 0.97}, ...]
```

### PDF: render DPI and parallel workers

```python
pages = pipe.process_pdf(
    "document.pdf",
    dpi=300,          # render resolution for scanned PDFs (default 300)
    force_ocr=False,  # True: OCR even PDFs that already have a text layer
    workers=4,        # parallel pages (None = auto min(cores, 4); 1 = sequential)
)
```

| PDF option | Default | What it does |
|---|---|---|
| `dpi` | `300` | Render resolution for scanned PDFs. Lower to `200` for speed on large batches. |
| `force_ocr` | `False` | OCR even PDFs that already contain a text layer. |
| `workers` | `None` | Parallel pages. `None` → auto `min(cores, 4)`; `1` → sequential. |

### One-liners

```python
from ocr_skel import ocr

ocr("document.png")                 # CPU (default), full quality
ocr("document.png", gpu=True)       # GPU via ONNX Runtime CUDA (needs onnxruntime-gpu)
ocr("document.png", lm=False)       # fast greedy mode, no LM download
ocr("document.png", deskew=False)   # skip deskew
ocr("document.png", num_threads=2)  # limit CPU threads
```

### Command line

```bash
# Single file — text to stdout
ocr document.png
ocr document.pdf --workers 4 --dpi 300

# Whole folder → one .txt per file
ocr ./scans ./out              # results in ./out/<name>.txt
ocr ./scans                    # .txt next to the source files

# One file → JSON with coordinates
ocr document.png --out result.json
```

`ocr` is installed as a console command; `python -m ocr_skel <args>` is equivalent.

| Flag | Default | What it does |
|---|---|---|
| `--gpu` | off | Run on GPU via ONNX Runtime CUDA (needs onnxruntime-gpu). |
| `--dpi N` | `300` | PDF render resolution. |
| `--force-ocr` | off | OCR even vector PDFs. |
| `--workers N` | auto | Parallel workers (PDF pages / batch files). |
| `--out FILE` | — | Save a single file's structured results to JSON. |
| `--json` | off | Print full JSON (with coordinates) to stdout. |

### Optional: reading order for multi-column pages

Off by default. The model downloads once from the Hub.

```python
from ocr_skel import download_reading_order, OCRPipeline, Settings, model_info

download_reading_order()                       # one-time download
pipe = OCRPipeline(Settings(reading_order=True))

model_info()                                   # show which weights are present locally
```

> 📓 Everything above is also in a runnable notebook: **[`examples.ipynb`](examples.ipynb)**.

---

## How it works

```
image ─▶ deskew ─▶ DBNet detector ─▶ crops ─▶ SVTR recognizer ─▶ beam + n-gram LM ─▶ text
```

The recognizer is a compact SVTR + CTC model exported to ONNX; the decoder is a CTC prefix beam search guided by a 4-gram
Russian language model. The entire decoding stack — language model and beam search alike — is
**pure Python**, which is what keeps `pip install` friction-free on every platform.

---

## Models & weights

Weights are fetched automatically on first use and cached:

| Component | What it does |
|---|---|
| DBNet text detector | finds text lines on the page |
| SVTR recognizer | reads text inside each line |
| Language model | rescoring for the beam decoder |
| Reading-order model *(optional)* | orders lines for multi-column layouts |

Inspect what's present locally:

```python
from ocr_skel import model_info
model_info()
```

---

## Benchmarks & methodology

Numbers above are **end-to-end** (detection **and** recognition on full pages), matched to
line-level ground truth by IoU, scored with word/character error rate. The Russian benchmark spans
30 document domains (bank, receipts, diplomas, certificates, IDs, court decisions, price tags,
newspapers, and more), ~15 pages each.

> **Note on ground truth.** Reference transcriptions come from a strong reference OCR system.
> Occular-OCR is tuned toward that transcription style, which favours it against third-party
> engines; the margin on Russian forms nonetheless exceeds what that bias alone explains.

---

## Licensing

- **Code — Apache License 2.0.** See [`LICENSE`](LICENSE).
- **Model weights — Modified AI Pubs OpenRAIL-M.** See [`WEIGHTS_LICENSE.md`](WEIGHTS_LICENSE.md).
  **Free** for individuals, researchers, the self-employed, non-profits, and small organizations
  (under **20 000 000 ₽** annual revenue **and** fewer than **8** employees). Larger organizations
  need a commercial license — **300 000 ₽ / year per organization**.
  Commercial enquiries: **user26665@gmail.com** · Telegram **[@Bodhi_b](https://t.me/Bodhi_b)**.

---

## Citation

```bibtex
@software{occular_ocr,
  title  = {Occular-OCR: State-of-the-art OCR for Russian documents},
  year   = {2026},
  url    = {https://github.com/Bodhi42/Occular-ocr}
}
```

<a id="russian"></a>

---

<div align="center">

# Occular-OCR &nbsp;·&nbsp; 🇷🇺 Русская версия

**Передовой OCR для русских документов — установка без компиляции.**

</div>

> 🇬🇧 English version is above · 🇷🇺 Ниже — то же самое по-русски.

Occular-OCR — библиотека OCR документов, созданная специально под **русский язык**. Связка:
DBNet-детектор текста + SVTR-распознаватель + beam-декодер с языковой моделью — читает русские
документы **заметно лучше существующего open-source OCR**, особенно сложное: формы, справки,
банковские выписки, удостоверения, чеки.

Ставится одной командой `pip install` на Windows, Linux и macOS — **без компилятора C/C++, без CUDA,
ничего собирать не нужно.**

---

## Демонстрация

<p align="center">
  <img src="assets/demo_passport.png" width="24%" alt="Строки на образце паспорта">
  <img src="assets/demo_2ndfl.png"    width="24%" alt="Строки на синтетической справке о доходах">
  <img src="assets/demo_book.png"     width="24%" alt="Строки на печатной книжной странице">
</p>
<p align="center">
  <img src="assets/demo_schet.png"    width="37%" alt="Строки на синтетическом счёте-фактуре">
  <img src="assets/demo_invoice.png"  width="37%" alt="Строки на синтетическом счёте на оплату">
</p>

<p align="center"><sub>Найденные строки текста на <b>синтетических / образцовых / public-domain</b> документах — паспорт (образец), справка о доходах, книжная страница, счёт-фактура, счёт на оплату.</sub></p>

## Почему Occular-OCR

Occular-OCR заточен под русский и читает его значительно лучше существующего open-source OCR:
**~35 % меньше ошибок слов, чем ближайший open-source-движок** в сквозном бенчмарке, и **#1** против
больших vision-language моделей и других ведущих OCR-движков на постраничном бенчмарке. Разрыв
максимален там, где универсальный OCR буксует: плотные формы, справки, счета, удостоверения.

- **Точность под русский.** Обучен и настроен на русских документах. Там, где другие движки гадают,
  Occular-OCR читает.
- **Beam + языковая модель по умолчанию.** Русская n-gram языковая модель снижает ошибки слов на
  ~18–25 % относительно жадного декодирования — **без дообучения модели**, просто лучше декодирование.
- **Ставится где угодно, без компилятора.** Декодер целиком на чистом Python — топовое качество декода
  с **нулевыми нативными зависимостями**.
- **Сначала CPU, GPU — по желанию.** Поставляется как ONNX и работает на CPU из коробки; `gpu=True`
  гоняет те же модели на CUDA через ONNX Runtime (`onnxruntime-gpu`).
- **Всё в комплекте.** Картинки и PDF, папка → `.txt`, порядок чтения, автоматическое выпрямление наклона.

---

## Установка

```bash
pip install occular-ocr
```

И всё. Без сборочных инструментов, без CUDA. Веса моделей скачиваются автоматически при первом запуске
и кэшируются.

### GPU (опционально)

GPU гоняет **те же ONNX-модели на CUDA** через `CUDAExecutionProvider` ONNX Runtime — отдельных
GPU-весов нет. Нужен только GPU-рантайм:

```bash
pip install occular-ocr[gpu]      # доустанавливает onnxruntime-gpu
# то же самое, что:  pip install occular-ocr onnxruntime-gpu
```

Затем передай `gpu=True` (см. ниже). Если CUDA недоступна — тихий откат на CPU с предупреждением.

---

## Быстрый старт

```python
from ocr_skel import ocr

text = ocr("document.png")        # картинка или "scan.pdf"
print(text)
```

Построчный вывод с координатами и confidence:

```python
from ocr_skel import ocr_detailed

for line in ocr_detailed("document.png"):
    print(line["text"], round(line["confidence"], 2), line["quad"])
```

Целая папка → текстовые файлы, из командной строки:

```bash
ocr ./scans ./out          # пишет ./out/<имя>.txt для каждого изображения/PDF
```

> Первый запуск сам скачает веса (и закэширует). Без сборочных инструментов, без CUDA.

## Настройки

Всё поведение задаётся через `Settings` (или эквивалентные именованные аргументы `ocr`). Посмотреть
текущие дефолты можно в любой момент:

```python
from ocr_skel import Settings
print(Settings())
```

| Настройка | По умолчанию | Что делает |
|---|---|---|
| `num_threads` | `None` | CPU-потоки для инференса. `None` → `min(ядра, 4)`. |
| `gpu` | `False` | Гонять ONNX-модели на **GPU через ONNX Runtime (CUDA)**. Нужен `onnxruntime-gpu` (`pip install occular-ocr[gpu]`); при отсутствии — откат на CPU. |
| `deskew` | `True` | Автовыпрямление наклонённых / повёрнутых сканов перед детекцией. |
| `lm` | `True` | Beam + языковая модель (лучшее качество). `False` → быстрое жадное декодирование, без скачивания LM. |
| `reading_order` | `False` | Упорядочивание строк для многоколоночных макетов (докачивает небольшую модель при первом запуске). |
| `detector` | `None` | Явное имя детектора. `None` → по умолчанию. |
| `recognizer` | `None` | Явное имя распознавателя. `None` → по умолчанию. |

### Пайплайн со всеми настройками

```python
from ocr_skel import OCRPipeline, Settings

pipe = OCRPipeline(Settings(
    num_threads=8,        # CPU-потоки (None -> min(ядра, 4))
    gpu=False,            # True -> ONNX Runtime CUDA (нужен occular-ocr[gpu])
    deskew=True,          # автовыпрямление наклона
    lm=True,              # beam + языковая модель; False -> жадное (быстрее, без скачивания LM)
    reading_order=False,  # порядок чтения для многоколоночных (опц. модель, см. ниже)
    detector=None,        # None = детектор по умолчанию
    recognizer=None,      # None = распознаватель по умолчанию
))

result = pipe.process_image("document.png")
# -> [{"quad": [[x, y], ...], "text": "...", "confidence": 0.97}, ...]
```

### PDF: DPI рендеринга и параллельные воркеры

```python
pages = pipe.process_pdf(
    "document.pdf",
    dpi=300,          # разрешение рендеринга сканов (по умолчанию 300)
    force_ocr=False,  # True: OCR даже для PDF с текстовым слоем
    workers=4,        # параллельные страницы (None = авто min(ядра, 4); 1 = последовательно)
)
```

| Опция PDF | По умолчанию | Что делает |
|---|---|---|
| `dpi` | `300` | Разрешение рендеринга сканов. Снизь до `200` ради скорости на больших пачках. |
| `force_ocr` | `False` | OCR даже для PDF, где уже есть текстовый слой. |
| `workers` | `None` | Параллельные страницы. `None` → авто `min(ядра, 4)`; `1` → последовательно. |

### Однострочники

```python
from ocr_skel import ocr

ocr("document.png")                 # CPU (по умолчанию), полное качество
ocr("document.png", gpu=True)       # GPU через ONNX Runtime CUDA (нужен onnxruntime-gpu)
ocr("document.png", lm=False)       # быстрый жадный режим, без скачивания LM
ocr("document.png", deskew=False)   # без выпрямления наклона
ocr("document.png", num_threads=2)  # ограничить CPU-потоки
```

### Командная строка

```bash
# Один файл — текст в stdout
ocr document.png
ocr document.pdf --workers 4 --dpi 300

# Целая папка → по .txt на каждый файл
ocr ./scans ./out              # результаты в ./out/<имя>.txt
ocr ./scans                    # .txt рядом с исходными файлами

# Один файл → JSON с координатами
ocr document.png --out result.json
```

`ocr` ставится как консольная команда; `python -m ocr_skel <аргументы>` — эквивалент.

| Флаг | По умолчанию | Что делает |
|---|---|---|
| `--gpu` | выкл | Гонять на GPU через ONNX Runtime CUDA (нужен onnxruntime-gpu). |
| `--dpi N` | `300` | Разрешение рендеринга PDF. |
| `--force-ocr` | выкл | OCR даже для векторных PDF. |
| `--workers N` | авто | Параллельные воркеры (страницы PDF / файлы батча). |
| `--out FILE` | — | Сохранить результат одиночного файла в JSON. |
| `--json` | выкл | Вывести полный JSON (с координатами) в stdout. |

### Опционально: порядок чтения для многоколоночных страниц

По умолчанию выключено. Модель скачивается один раз.

```python
from ocr_skel import download_reading_order, OCRPipeline, Settings, model_info

download_reading_order()                       # разовая докачка
pipe = OCRPipeline(Settings(reading_order=True))

model_info()                                   # показать, какие веса есть локально
```

> 📓 Всё вышеперечисленное есть и в исполняемом ноутбуке: **[`examples.ipynb`](examples.ipynb)**.

---

## Как это работает

```
картинка ─▶ deskew ─▶ DBNet-детектор ─▶ кропы ─▶ SVTR-распознаватель ─▶ beam + n-gram LM ─▶ текст
```

Распознаватель — компактная SVTR + CTC модель, экспортированная в ONNX; декодер — CTC prefix beam
search с 4-gram русской языковой моделью. Весь стек декодирования — и языковая модель, и beam-поиск —
**на чистом Python**, поэтому `pip install` проходит гладко на любой платформе.

---

## Модели и веса

Веса скачиваются автоматически при первом запуске и кэшируются:

| Компонент | Что делает |
|---|---|
| DBNet-детектор текста | находит строки текста на странице |
| SVTR-распознаватель | читает текст внутри каждой строки |
| Языковая модель | rescoring для beam-декодера |
| Модель порядка чтения *(опц.)* | упорядочивает строки для многоколоночных макетов |

Посмотреть, что есть локально:

```python
from ocr_skel import model_info
model_info()
```

---

## Бенчмарки и методология

Цифры выше — **сквозные** (детекция **и** распознавание на полных страницах), с матчингом к
построчному эталону по IoU, метрика — ошибка слов/символов. Русский бенчмарк охватывает 30 доменов
документов (банковские, чеки, дипломы, справки, удостоверения, судебные решения, ценники, газеты и
др.), ~15 страниц на домен.

> **Про эталон.** Эталонные транскрипции получены сильной референсной OCR-системой; Occular-OCR
> настроен под этот стиль транскрипции, что даёт ему фору против сторонних движков. Тем не менее на
> русских формах разрыв превышает то, что объясняется одним лишь этим уклоном.

---

## Лицензирование

- **Код — Apache License 2.0.** См. [`LICENSE`](LICENSE).
- **Веса моделей — Modified AI Pubs OpenRAIL-M.** См. [`WEIGHTS_LICENSE.md`](WEIGHTS_LICENSE.md).
  **Бесплатно** для физлиц, исследователей, самозанятых, НКО и малых организаций (выручка до
  **20 000 000 ₽** в год **и** менее **8** сотрудников). Крупным организациям нужна коммерческая
  лицензия — **300 000 ₽ / год на организацию**.
  Коммерческие вопросы: **user26665@gmail.com** · Telegram **[@Bodhi_b](https://t.me/Bodhi_b)**.

---

## Цитирование

```bibtex
@software{occular_ocr,
  title  = {Occular-OCR: State-of-the-art OCR for Russian documents},
  year   = {2026},
  url    = {https://github.com/Bodhi42/Occular-ocr}
}
```
