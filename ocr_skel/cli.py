"""CLI для OCR skeleton"""

import argparse
import json
import sys
from pathlib import Path

from .pipeline import OCRPipeline
from .registry import Registry
from .detector import CRAFTDetector
from .recognizer import CRNNRecognizer


def main():
    """Точка входа CLI"""

    # Регистрация компонентов
    Registry.register_detector("craft", CRAFTDetector)
    Registry.register_recognizer("crnn", CRNNRecognizer)

    parser = argparse.ArgumentParser(description="OCR Skeleton CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--detector", type=str, default="craft", help="Detector name (default: craft)")
    parser.add_argument("--recognizer", type=str, default="crnn", help="Recognizer name (default: crnn)")
    parser.add_argument("--out", type=str, help="Output JSON file path")
    parser.add_argument("--print-text", action="store_true", help="Print recognized text to stdout")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available (default: CPU)")

    args = parser.parse_args()

    # Проверка существования файла
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Создание pipeline
    try:
        detector_kwargs = {"gpu": args.gpu}
        recognizer_kwargs = {"gpu": args.gpu}

        pipeline = OCRPipeline(
            detector_name=args.detector,
            recognizer_name=args.recognizer,
            detector_kwargs=detector_kwargs,
            recognizer_kwargs=recognizer_kwargs
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Обработка изображения
    try:
        results = pipeline.process_image(args.image)
    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        sys.exit(1)

    # Вывод результатов
    if args.out:
        # Сохранение в JSON файл
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.out}")

    if args.print_text:
        # Вывод текста в stdout
        for item in results:
            print(item["text"])

    # Если не указаны флаги вывода, выводим JSON в stdout
    if not args.out and not args.print_text:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
