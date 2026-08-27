"""
Где лежат модели Occular-OCR — простыми словами.

Для чайников (1 строка):
    from occular import model_info
    model_info()          # покажет все модели, их файлы, размер и статус

Веса берутся так: сначала локальный файл в occular/weights/ (если положен), иначе — загрузка
с HuggingFace в кэш. В sdist/wheel сами веса НЕ упакованы (MANIFEST.in исключает *.onnx/*.pth).
  • Детектор и распознаватель — ONNX FP32, качаются с HuggingFace при первом запуске (или локально).
  • Порядок чтения (layout) — опционально, качается с HuggingFace, по умолчанию ВЫКЛ.
"""
from pathlib import Path

# ⬇⬇⬇ ВСЕ МОДЕЛИ ЗДЕСЬ ⬇⬇⬇
WEIGHTS_DIR = Path(__file__).parent / "weights"
READING_ORDER_DIR = WEIGHTS_DIR / "reading_order"        # сюда качается layout-модель

# Все веса моделей на HuggingFace (детектор, рекогнайзер, charset, reading_order/*).
# Локальный файл в weights/ (если есть) имеет приоритет; иначе качается с HF в кэш.
WEIGHTS_HF_REPO = "Shivin11/occular-ocr"
READING_ORDER_HF_REPO = WEIGHTS_HF_REPO                  # reading_order/encoder.onnx, decoder.onnx
# Закрепление версии весов для воспроизводимости: HF commit SHA релиза. None = ветка по умолчанию
# (последняя ревизия). Для production выставьте конкретный commit, чтобы веса не менялись со временем.
WEIGHTS_REVISION = None
# Переопределение через окружение (не редактируя пакет): OCCULAR_WEIGHTS_REVISION=<sha>
import os as _os
WEIGHTS_REVISION = _os.environ.get("OCCULAR_WEIGHTS_REVISION", WEIGHTS_REVISION)


def ensure_weight(rel_path: str) -> str:
    """Путь к файлу веса: сначала локально в weights/, иначе качаем с HF (WEIGHTS_HF_REPO) в кэш.
    Ревизия закрепляется через WEIGHTS_REVISION (commit SHA) для воспроизводимости."""
    local = WEIGHTS_DIR / rel_path
    if local.exists():
        return str(local)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            f"Веса '{rel_path}' нет локально, а huggingface_hub не установлен. "
            f"pip install huggingface_hub  (или положите файл в {local})"
        )
    return hf_hub_download(WEIGHTS_HF_REPO, rel_path, revision=WEIGHTS_REVISION)


def ensure_cyr_lm(lang: str) -> tuple:
    """Файлы пер-язычной кир-LM: (compact_lm.npz, unigrams.txt). Сначала локально
    (OCCULAR_CYR_LM_DIR/<lang>/), иначе качаем нужный язык с HF (CYR_LM_HF_REPO)."""
    d = _os.environ.get("OCCULAR_CYR_LM_DIR")
    if d:
        npz = Path(d) / lang / NPZ_NAME
        uni = Path(d) / lang / UNI_NAME
        if not npz.exists() or not uni.exists():
            raise FileNotFoundError(f"OCCULAR_CYR_LM_DIR задан, но нет {npz} или {uni}")
        return str(npz), str(uni)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            f"Для кир-LM '{lang}' нужен huggingface_hub (pip install huggingface_hub) "
            f"или задайте OCCULAR_CYR_LM_DIR с папками языков.")
    npz = hf_hub_download(CYR_LM_HF_REPO, f"{lang}/{NPZ_NAME}", revision=WEIGHTS_REVISION)
    uni = hf_hub_download(CYR_LM_HF_REPO, f"{lang}/{UNI_NAME}", revision=WEIGHTS_REVISION)
    return npz, uni


def load_cyr_decode_config() -> dict:
    """Конфиг декода кир-языков (per-язык beam+lm/greedy + alpha/beta), поставляется в пакете
    (cyr_decode_config.json) — залочен с этой версией модели. Пусто, если файла нет."""
    import json
    p = Path(__file__).parent / "cyr_decode_config.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

# Языковая модель для beam-CTC (декод по умолчанию). Качается с HF в кэш при первом запуске
# (или берётся из папки OCCULAR_LM_DIR с файлами compact_lm.npz + unigrams.txt).
LM_HF_REPO = "Shivin11/occular-lm-ru"   # n-gram LM + униграммы (beam-декодер, русская модель ru/en)

# Пер-язычные кир-LM (12 языков, beam-декодер мультиязычного роутера). В репо файлы разложены
# по подпапкам языка: <lang>/compact_lm.npz + <lang>/unigrams.txt. Качается только нужный язык.
# Локальный оверрайд: OCCULAR_CYR_LM_DIR/<lang>/{compact_lm.npz,unigrams.txt}.
CYR_LM_HF_REPO = "Shivin11/occular-lm-cyr"


NPZ_NAME = "compact_lm.npz"
UNI_NAME = "unigrams.txt"

# Реестр: человекочитаемое имя -> файлы + описание + обязательна ли
MODELS = {
    "Детектор текста (находит строки на странице)": {
        "files": ["detector_dbnet_fp32.onnx"],
        "desc": "Детектор текста, ONNX FP32 — локально в weights/ или с HuggingFace",
        "required": True,
    },
    "Распознаватель (читает текст в строках)": {
        "files": ["recognizer_svtr_fp32.onnx", "recognizer_charset.txt"],
        "desc": "Распознаватель, ONNX FP32 — локально в weights/ или с HuggingFace",
        "required": True,
    },
    "Кир-распознаватель 12 языков (опционально)": {
        "files": ["recognizer_svtr_cyr12_fp32.onnx", "recognizer_charset_cyr12.txt"],
        "desc": "Распознаватель 12 кир-языков (ba be bg cv kk ky mk mn sr tg tt uk), ONNX FP32 — "
                "для ocr(languages=[...]); локально в weights/ или с HuggingFace",
        "required": False,
    },
    "Определитель языка (опц., для авто-роутинга)": {
        "files": ["langid_models.pkl"],
        "desc": "char-n-gram определитель (12 кир + ru) для авто-режима languages=None; "
                "локально в weights/ или с HuggingFace",
        "required": False,
    },
    "Детектор таблиц (опционально)": {
        "files": ["table_detect_v3_fp32.onnx"],
        "desc": "Находит таблицы на странице (карта таблица/фон → рамки), ONNX FP32 — для TableRecognizer; "
                "локально в weights/ или с HuggingFace",
        "required": False,
    },
    "Структура таблиц: split (ONNX, опц.)": {
        "files": ["table_struct_split_v2_fp32.onnx"],
        "desc": "Сетка строк/столбцов таблицы, ONNX FP32 (без объединённых ячеек) — фолбэк без torch; "
                "локально в weights/ или с HuggingFace",
        "required": False,
    },
    "Структура таблиц: split+merge (torch CPU, опц.)": {
        "files": ["table_struct_split_merge_v2.pt"],
        "desc": "Полная структура таблицы с объединёнными ячейками (colspan/rowspan) — PyTorch на CPU "
                "(нужен пакет torch); локально в weights/ или с HuggingFace",
        "required": False,
    },
    "Порядок чтения / layout (опционально)": {
        "files": ["reading_order/encoder.onnx", "reading_order/decoder.onnx"],
        "desc": f"AR-layout, ONNX FP32 — качается с HuggingFace ({READING_ORDER_HF_REPO}), по умолчанию ВЫКЛ",
        "required": False,
    },
    "Языковая модель beam-CTC (по умолчанию ВКЛ)": {
        "files": [],   # в кэше HF, не в weights/; см. decoder_lm._resolve_lm_files
        "desc": f"n-gram LM (compact_lm.npz ~270МБ + unigrams.txt) — чистый Python, без C-зависимостей; "
                f"качается с HuggingFace ({LM_HF_REPO}) при 1-м запуске или из OCCULAR_LM_DIR. "
                f"Отключить: OCRPipeline(lm=False)",
        "required": False,
    },
}


def model_info():
    """Показать все модели: где лежат, размер, статус. Для чайников."""
    print(f"\nМодели Occular-OCR лежат в:\n  {WEIGHTS_DIR}\n")
    print(f"{'МОДЕЛЬ':48} {'РАЗМЕР':>8}  СТАТУС")
    print("-" * 78)
    for name, m in MODELS.items():
        present = all((WEIGHTS_DIR / f).exists() for f in m["files"])
        total = sum((WEIGHTS_DIR / f).stat().st_size for f in m["files"] if (WEIGHTS_DIR / f).exists())
        size = f"{total/1e6:.0f} МБ" if total else "—"
        if present:
            status = "✅ на месте"
        elif m["required"]:
            status = "❌ НЕТ (обязательна!)"
        else:
            status = "⬇ не скачана (опц., см. download_reading_order())"
        print(f"{name[:48]:48} {size:>8}  {status}")
        for f in m["files"]:
            print(f"    └─ {f}")
    print("-" * 78)
    print("Порядок чтения выключен по умолчанию. Включить: OCRPipeline(reading_order=True)")
    print("Скачать порядок чтения:  from occular import download_reading_order; download_reading_order()\n")


def reading_order_ready() -> bool:
    """Скачана ли layout-модель (порядок чтения)?"""
    return (READING_ORDER_DIR / "encoder.onnx").exists() and (READING_ORDER_DIR / "decoder.onnx").exists()


def download_reading_order():
    """Скачать опциональную модель порядка чтения (layout) с HuggingFace в occular/weights/reading_order/."""
    if reading_order_ready():
        print(f"✅ Уже скачана: {READING_ORDER_DIR}")
        return str(READING_ORDER_DIR)
    READING_ORDER_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("Нужен huggingface_hub: pip install huggingface_hub")
    import shutil
    print(f"Скачиваю порядок чтения с HuggingFace ({READING_ORDER_HF_REPO}) в {READING_ORDER_DIR} ...")
    # в репо файлы плоские (reading_order_*.onnx) — раскладываем в weights/reading_order/{encoder,decoder}.onnx
    for repo_name, local_name in [("reading_order_encoder.onnx", "encoder.onnx"),
                                  ("reading_order_decoder.onnx", "decoder.onnx")]:
        p = hf_hub_download(READING_ORDER_HF_REPO, repo_name, revision=WEIGHTS_REVISION)
        shutil.copy(p, READING_ORDER_DIR / local_name)
    print(f"✅ Готово: {READING_ORDER_DIR}")
    return str(READING_ORDER_DIR)
