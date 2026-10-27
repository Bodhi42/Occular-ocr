"""
Где лежат модели Occular-OCR — простыми словами.

Для чайников (1 строка):
    from occular import model_info
    model_info()          # покажет все модели, их файлы, размер и статус

Веса берутся так: сначала локальный файл в occular/weights/ (если положен), иначе — загрузка
с HuggingFace в кэш. В sdist/wheel сами веса НЕ упакованы (MANIFEST.in исключает *.onnx).
  • Детектор и распознаватель — ONNX FP32, качаются с HuggingFace при первом запуске (или локально).
  • Порядок чтения (layout) — опционально, качается с HuggingFace, по умолчанию ВЫКЛ.
"""
from pathlib import Path

# ⬇⬇⬇ ВСЕ МОДЕЛИ ЗДЕСЬ ⬇⬇⬇
WEIGHTS_DIR = Path(__file__).parent / "weights"

# Все веса моделей на HuggingFace (детектор, рекогнайзер, charset, layout, таблицы).
# Локальный файл в weights/ (если есть) имеет приоритет; иначе качается с HF в кэш.
WEIGHTS_HF_REPO = "Shivin11/occular-ocr"
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
            f"Weights '{rel_path}' are not available locally and huggingface_hub is not installed. "
            f"pip install huggingface_hub  (or put the file at {local})"
        )
    return hf_hub_download(WEIGHTS_HF_REPO, rel_path, revision=WEIGHTS_REVISION)


def locate_weight(rel_path: str, repo: str = None) -> "Path | None":
    """Где файл уже лежит, без сети: weights/, затем кэш HF (для LM — ещё OCCULAR_LM_DIR). None — не скачан."""
    repo = repo or WEIGHTS_HF_REPO
    if repo == WEIGHTS_HF_REPO and (WEIGHTS_DIR / rel_path).exists():
        return WEIGHTS_DIR / rel_path
    if repo == LM_HF_REPO and _os.environ.get("OCCULAR_LM_DIR"):
        p = Path(_os.environ["OCCULAR_LM_DIR"]) / rel_path
        return p if p.exists() else None
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    hit = try_to_load_from_cache(repo, rel_path, revision=WEIGHTS_REVISION if repo == WEIGHTS_HF_REPO else None)
    return Path(hit) if isinstance(hit, str) and Path(hit).exists() else None


def ensure_cyr_lm(lang: str) -> tuple:
    """Файлы пер-язычной кир-LM: (compact_lm.npz, unigrams.txt). Сначала локально
    (OCCULAR_CYR_LM_DIR/<lang>/), иначе качаем нужный язык с HF (CYR_LM_HF_REPO)."""
    d = _os.environ.get("OCCULAR_CYR_LM_DIR")
    if d:
        npz = Path(d) / lang / NPZ_NAME
        uni = Path(d) / lang / UNI_NAME
        if not npz.exists() or not uni.exists():
            raise FileNotFoundError(f"OCCULAR_CYR_LM_DIR is set, but {npz} or {uni} does not exist")
        return str(npz), str(uni)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            f"The Cyrillic LM for '{lang}' needs huggingface_hub (pip install huggingface_hub), "
            f"or set OCCULAR_CYR_LM_DIR to a folder with per-language subfolders.")
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
    "Text detector (finds lines on the page)": {
        "files": ["detector_dbnet_fp32.onnx"],
        "desc": "Text detector, ONNX FP32 — local in weights/ or from Hugging Face",
        "required": True,
    },
    "Recognizer svtr_lcnet (default)": {
        "files": ["recognizer_svtr_lcnet_fp32.onnx", "recognizer_charset.txt"],
        "desc": "Light ru/en recognizer (LCNet stem), ONNX FP32 — the default model, "
                "~4x faster than svtr_t at 99.6% of its quality; local in weights/ or from Hugging Face",
        "required": True,
    },
    "Recognizer svtr_t (larger, optional)": {
        "files": ["recognizer_svtr_fp32.onnx", "recognizer_charset.txt"],
        "desc": "Larger ru/en recognizer — for ocr(recognizer='svtr_t'); "
                "local in weights/ or from Hugging Face",
        "required": False,
    },
    "Cyrillic recognizer, 12 languages, svtr_lcnet (opt.)": {
        "files": ["recognizer_svtr_lcnet_cyr12_fp32.onnx", "recognizer_charset_cyr12.txt"],
        "desc": "Light recognizer for 12 Cyrillic-script languages (ba be bg cv kk ky mk mn sr tg tt uk), ONNX FP32 — "
                "default for ocr(languages=[...]); local in weights/ or from Hugging Face",
        "required": False,
    },
    "Cyrillic recognizer, 12 languages, svtr_t (opt.)": {
        "files": ["recognizer_svtr_cyr12_fp32.onnx", "recognizer_charset_cyr12.txt"],
        "desc": "Recognizer for 12 Cyrillic-script languages (ba be bg cv kk ky mk mn sr tg tt uk), ONNX FP32 — "
                "for ocr(languages=[...]); local in weights/ or from Hugging Face",
        "required": False,
    },
    "Layout regions + reading order (on by default)": {
        "files": ["layout_dfine_s144k_fp32.onnx", "layout_order_blocks_fp32.onnx"],
        "desc": "Page regions D-FINE-S (box, score, one of 18 labels; also LayoutDetector) and the block-order "
                "model (3.3M) — for reading_order (on by default); local in weights/ or from Hugging Face",
        "required": False,
    },
    "Page orientation (optional)": {
        "files": ["orientation_orinet_fp32.onnx"],
        "desc": "Detects a 0/90/180/270° page rotation — for ocr(orientation=True); "
                "local in weights/ or from Hugging Face",
        "required": False,
    },
    "Language identifier (opt., for auto routing)": {
        "files": ["langid_models.pkl"],
        "desc": "char-n-gram language identifier (12 Cyrillic + ru) for languages='auto'; "
                "local in weights/ or from Hugging Face",
        "required": False,
    },
    "Table detector (optional)": {
        "files": ["table_detect_dfine_v4_fp32.onnx"],
        "desc": "Finds tables on a page (D-FINE-S, a box and score per table), ONNX FP32 — for tables=True "
                "and TableRecognizer; local in weights/ or from Hugging Face",
        "required": False,
    },
    "Table structure (optional)": {
        "files": ["table_struct_v4_main_fp32.onnx", "table_struct_v4_merge_fp32.onnx"],
        "desc": "Row/column grid + merged cells (colspan/rowspan), two ONNX FP32 graphs (tall tables are "
                "read in strips) — for tables=True and TableRecognizer; local in weights/ or from Hugging Face",
        "required": False,
    },
    "Beam-CTC language model (on by default)": {
        "files": [NPZ_NAME, UNI_NAME],   # в кэше HF (или OCCULAR_LM_DIR), не в weights/
        "repo": LM_HF_REPO,
        "desc": f"n-gram LM (compact_lm.npz ~270 MB + unigrams.txt) — pure Python, no C dependencies "
                f"(optional native decoder: occular-decode); downloaded from Hugging Face ({LM_HF_REPO}) "
                f"on first run, or from OCCULAR_LM_DIR. Turn off: OCRPipeline(lm=False)",
        "required": False,
    },
}


def model_info():
    """Показать все модели: где лежат, размер, статус. Для чайников."""
    print(f"\nOccular-OCR models are looked up in:\n  {WEIGHTS_DIR}\n  then the Hugging Face cache "
          f"(HF_HOME, downloaded on first use)\n")
    print(f"{'MODEL':48} {'SIZE':>8}  STATUS")
    print("-" * 78)
    for name, m in MODELS.items():
        found = [locate_weight(f, m.get("repo", WEIGHTS_HF_REPO)) for f in m["files"]]
        present = all(found)
        total = sum(f.stat().st_size for f in found) if present else 0
        size = f"{total/1e6:.0f} MB" if total else "—"
        if present:
            status = "✅ present"
        elif m["required"]:
            status = "❌ MISSING (required!)"
        else:
            # опциональные модели качаются сами при первом использовании своей функции
            status = "⬇ not downloaded (optional, downloaded on first use)"
        print(f"{name[:48]:48} {size:>8}  {status}")
        for f in m["files"]:
            print(f"    └─ {f}")
    print("-" * 78)
    print("Optional models are downloaded on first use. Reading order is on by default; turn it off: "
          "OCRPipeline(reading_order=False)\n")


def layout_ready() -> bool:
    """Есть ли layout-модель (D-FINE-S) и модель порядка блоков — в weights/ или в кэше HF? (Иначе скачаются при первом использовании.)"""
    return all(locate_weight(f) for f in ("layout_dfine_s144k_fp32.onnx", "layout_order_blocks_fp32.onnx"))


def download_layout() -> str:
    """Скачать layout-модель D-FINE-S и модель порядка блоков с HuggingFace в кэш (или вернуть локальный
    путь). Возвращает путь к layout-модели."""
    ensure_weight("layout_order_blocks_fp32.onnx")
    return ensure_weight("layout_dfine_s144k_fp32.onnx")


def reading_order_ready() -> bool:
    """Есть ли модели порядка чтения (layout + порядок блоков) локально или в кэше HF? То же, что layout_ready()."""
    return layout_ready()


def download_reading_order() -> str:
    """Заранее скачать модели порядка чтения (layout + порядок блоков). То же, что download_layout();
    без этого вызова они скачиваются при первом использовании."""
    return download_layout()
