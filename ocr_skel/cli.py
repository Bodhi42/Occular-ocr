"""
CLI для OCR

Использование:
    # Один файл (текст в stdout)
    ocr photo.png
    ocr document.pdf

    # Папка → .txt на каждый файл (батч)
    ocr ./scans ./out            # результаты в ./out/<имя>.txt
    ocr ./scans                  # .txt рядом с исходными файлами
    ocr ./scans ./out --workers 4 --dpi 300

    # С опциями
    ocr document.pdf --out result.json   # один файл → JSON с координатами
    ocr photo.png --gpu                  # на GPU (нужен onnxruntime-gpu)

    # Через python -m
    python -m ocr_skel photo.png
"""

import argparse
import json
import sys
from pathlib import Path

# Расширения, которые батч-режим берёт из папки
BATCH_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}


def main():
    """Точка входа CLI"""

    parser = argparse.ArgumentParser(
        description="Occular OCR - распознавание текста",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  ocr photo.png                      # Распознать изображение (текст в stdout)
  ocr document.pdf                   # Распознать PDF
  ocr ./scans ./out                  # Папка → ./out/<имя>.txt на каждый файл
  ocr ./scans --workers 8            # Батч папки в 8 воркеров (.txt рядом с файлами)
  ocr photo.png --gpu                # На GPU/CUDA (нужен onnxruntime-gpu)
  ocr photo.png --out result.json    # Один файл → JSON с координатами
        """
    )

    # Позиционные: вход (файл или папка) и опциональный выход
    parser.add_argument(
        "input",
        nargs="?",
        help="Путь к изображению/PDF или папке с ними"
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Папка для .txt (только когда вход — папка). Если не указана — рядом с исходниками"
    )

    # Для обратной совместимости
    parser.add_argument("--image", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--pdf", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--gpu", action="store_true", help="Исполнять на GPU/CUDA (нужен пакет onnxruntime-gpu)")

    # PDF опции
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI для рендеринга PDF (default: 300)"
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Принудительно OCR даже для векторных PDF"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Количество воркеров для PDF/батча (default: auto, max 4)"
    )

    # Вывод (для одиночного файла)
    parser.add_argument(
        "--out", "-o",
        type=str,
        help="Сохранить результат одиночного файла в JSON"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Вывести полный JSON (с координатами)"
    )

    # Продвинутые опции
    parser.add_argument("--detector", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--recognizer", type=str, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Определяем вход
    input_file = args.input or args.image or args.pdf
    if not input_file:
        parser.print_help()
        sys.exit(1)

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Ошибка: путь не найден: {input_file}", file=sys.stderr)
        sys.exit(1)

    # Импортируем и регистрируем компоненты
    from . import _ensure_registered, OCRPipeline
    _ensure_registered()

    detector = args.detector or "dbnet-onnx"
    recognizer = args.recognizer or "crnn-onnx"

    try:
        pipeline = OCRPipeline(
            detector=detector,
            recognizer=recognizer,
            gpu=args.gpu
        )
    except Exception as e:
        print(f"Ошибка инициализации: {e}", file=sys.stderr)
        sys.exit(1)

    # Папка → батч в .txt
    if input_path.is_dir():
        _run_batch(pipeline, input_path, args)
        return

    # Одиночный файл
    _run_single(pipeline, input_path, args)


def _plain_text(pipeline, file_path: Path, args) -> str:
    """Распознать один файл и вернуть только текст (страницы PDF разделены пустой строкой)."""
    is_pdf = file_path.suffix.lower() == ".pdf"
    if is_pdf:
        results = pipeline.process_pdf(
            str(file_path), dpi=args.dpi, force_ocr=args.force_ocr, workers=args.workers
        )
        pages = []
        for page in results:
            lines = sorted(page["results"], key=lambda r: r["quad"][0][1])
            pages.append("\n".join(item["text"] for item in lines))
        return "\n\n".join(pages)
    results = pipeline.process_image(str(file_path))
    lines = sorted(results, key=lambda r: r["quad"][0][1])
    return "\n".join(item["text"] for item in lines)


def _run_batch(pipeline, input_dir: Path, args):
    """Папка → .txt на каждый файл. Выход: args.output или рядом с исходниками."""
    files = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in BATCH_EXTS)
    if not files:
        print(f"В папке {input_dir} нет изображений/PDF ({', '.join(sorted(BATCH_EXTS))})",
              file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output) if args.output else input_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for f in files:
        try:
            text = _plain_text(pipeline, f, args)
        except Exception as e:
            print(f"  [пропуск] {f.name}: {e}", file=sys.stderr)
            continue
        out_path = out_dir / (f.stem + ".txt")
        out_path.write_text(text, encoding="utf-8")
        ok += 1
        print(f"  {f.name} → {out_path}")
    print(f"Готово: {ok}/{len(files)} файлов, .txt в {out_dir}")


def _run_single(pipeline, input_path: Path, args):
    """Один файл: stdout-текст, либо --json / --out."""
    is_pdf = input_path.suffix.lower() == ".pdf"
    try:
        if is_pdf:
            results = pipeline.process_pdf(
                str(input_path), dpi=args.dpi, force_ocr=args.force_ocr, workers=args.workers
            )
        else:
            results = pipeline.process_image(str(input_path))
    except Exception as e:
        print(f"Ошибка обработки: {e}", file=sys.stderr)
        sys.exit(1)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Сохранено: {args.out}")
    elif args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        # Простой вывод текста (сортируем по Y сверху вниз)
        if is_pdf:
            for page in results:
                print(f"--- Страница {page['page']} ---")
                for item in sorted(page["results"], key=lambda r: r["quad"][0][1]):
                    print(item["text"])
        else:
            for item in sorted(results, key=lambda r: r["quad"][0][1]):
                print(item["text"])


if __name__ == "__main__":
    main()
