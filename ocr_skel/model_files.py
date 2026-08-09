"""
Где лежат модели Occular-OCR — простыми словами.

Для чайников (1 строка):
    from ocr_skel import model_info
    model_info()          # покажет все модели, их файлы, размер и статус

Все модели лежат в ОДНОЙ папке: ocr_skel/weights/
  • Детектор и распознаватель — в комплекте (ONNX, всегда на месте).
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


def ensure_weight(rel_path: str) -> str:
    """Путь к файлу веса: сначала локально в weights/, иначе качаем с HF (WEIGHTS_HF_REPO) в кэш."""
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
    return hf_hub_download(WEIGHTS_HF_REPO, rel_path)

# Языковая модель для beam-CTC (декод по умолчанию). Качается с HF в кэш при первом запуске
# (или берётся из папки OCCULAR_LM_DIR с файлами compact_lm.npz + unigrams.txt).
LM_HF_REPO = "Shivin11/occular-lm-ru"   # n-gram LM + униграммы (beam-декодер)

# Реестр: человекочитаемое имя -> файлы + описание + обязательна ли
MODELS = {
    "Детектор текста (находит строки на странице)": {
        "files": ["detector_dbnet_fp32.onnx"],
        "desc": "Детектор текста, ONNX FP32 — в комплекте",
        "required": True,
    },
    "Распознаватель (читает текст в строках)": {
        "files": ["recognizer_svtr_fp32.onnx", "recognizer_charset.txt"],
        "desc": "Распознаватель, ONNX FP32 — в комплекте",
        "required": True,
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
    print("Скачать порядок чтения:  from ocr_skel import download_reading_order; download_reading_order()\n")


def reading_order_ready() -> bool:
    """Скачана ли layout-модель (порядок чтения)?"""
    return (READING_ORDER_DIR / "encoder.onnx").exists() and (READING_ORDER_DIR / "decoder.onnx").exists()


def download_reading_order():
    """Скачать опциональную модель порядка чтения (layout) с HuggingFace в ocr_skel/weights/reading_order/."""
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
        p = hf_hub_download(READING_ORDER_HF_REPO, repo_name)
        shutil.copy(p, READING_ORDER_DIR / local_name)
    print(f"✅ Готово: {READING_ORDER_DIR}")
    return str(READING_ORDER_DIR)
