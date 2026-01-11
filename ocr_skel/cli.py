"""
CLI для OCR

Использование:
    # Простое (1-2 команды)
    ocr photo.png
    ocr document.pdf

    # С опциями
    ocr document.pdf --onnx --workers 4 --out result.json

    # Через python -m
    python -m ocr_skel photo.png
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    """Точка входа CLI"""

    parser = argparse.ArgumentParser(
        description="Occular OCR - распознавание текста",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  ocr photo.png                      # Распознать изображение
  ocr document.pdf                   # Распознать PDF
  ocr scan.pdf --onnx                # Использовать ONNX (быстрее на AMD)
  ocr book.pdf --workers 8           # 8 воркеров для PDF
  ocr photo.png --out result.json    # Сохранить в JSON
        """
    )

    # Позиционный аргумент - путь к файлу
    parser.add_argument(
        "file",
        nargs="?",
        help="Путь к изображению или PDF"
    )

    # Для обратной совместимости
    parser.add_argument("--image", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--pdf", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--onnx", action="store_true", help=argparse.SUPPRESS)  # теперь по умолчанию

    # Основные опции
    parser.add_argument(
        "--no-onnx",
        action="store_true",
        help="Использовать PyTorch вместо ONNX (медленнее)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Использовать GPU"
    )

    # PDF опции
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI для рендеринга PDF (default: 200)"
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
        help="Количество воркеров для PDF (default: auto, max 4)"
    )

    # Вывод
    parser.add_argument(
        "--out", "-o",
        type=str,
        help="Сохранить результат в JSON файл"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Вывести полный JSON (с координатами)"
    )

    # Продвинутые опции
    parser.add_argument(
        "--detector",
        type=str,
        help=argparse.SUPPRESS  # dbnet, dbnet-onnx
    )
    parser.add_argument(
        "--recognizer",
        type=str,
        help=argparse.SUPPRESS  # crnn, crnn-onnx
    )

    args = parser.parse_args()

    # Определяем входной файл
    input_file = args.file or args.image or args.pdf
    if not input_file:
        parser.print_help()
        sys.exit(1)

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Ошибка: файл не найден: {input_file}", file=sys.stderr)
        sys.exit(1)

    is_pdf = input_path.suffix.lower() == '.pdf'

    # Импортируем и регистрируем компоненты
    from . import _ensure_registered, OCRPipeline
    _ensure_registered()

    # Определяем компоненты (ONNX по умолчанию)
    detector = args.detector
    recognizer = args.recognizer
    use_onnx = not args.no_onnx

    if detector is None:
        detector = "dbnet-onnx" if use_onnx else "dbnet"
    if recognizer is None:
        recognizer = "crnn-onnx" if use_onnx else "crnn"

    # Создаём pipeline
    try:
        pipeline = OCRPipeline(
            detector=detector,
            recognizer=recognizer,
            gpu=args.gpu
        )
    except Exception as e:
        print(f"Ошибка инициализации: {e}", file=sys.stderr)
        sys.exit(1)

    # Обработка
    try:
        if is_pdf:
            results = pipeline.process_pdf(
                input_file,
                dpi=args.dpi,
                force_ocr=args.force_ocr,
                workers=args.workers
            )
        else:
            results = pipeline.process_image(input_file)
    except Exception as e:
        print(f"Ошибка обработки: {e}", file=sys.stderr)
        sys.exit(1)

    # Вывод результатов
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Сохранено: {args.out}")

    elif args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))

    else:
        # Простой вывод текста (сортируем по Y сверху вниз)
        if is_pdf:
            for page in results:
                print(f"--- Страница {page['page']} ---")
                sorted_results = sorted(page["results"], key=lambda r: r["quad"][0][1])
                for item in sorted_results:
                    print(item["text"])
        else:
            sorted_results = sorted(results, key=lambda r: r["quad"][0][1])
            for item in sorted_results:
                print(item["text"])


if __name__ == "__main__":
    main()
