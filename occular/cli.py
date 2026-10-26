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
    ocr photo.png --gpu                  # на GPU/CUDA (CUDA-библиотеки: pip install "occular-ocr[gpu]")
    ocr invoice.png --tables             # таблицы — Markdown-таблицами на своём месте

    # Через python -m
    python -m occular photo.png
"""

import argparse
import json
import sys
from pathlib import Path

# Расширения, которые батч-режим берёт из папки
BATCH_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".pdf"}


def _utf8_stdio():
    """Вывод в UTF-8, если stdout/stderr не в нём. На Windows при перенаправлении (`ocr scan.pdf > out.txt`,
    pipe) Python пишет в кодировке локали (cp1252), и русский текст или справка падали с UnicodeEncodeError.
    В консоли Windows и на Linux/macOS вывод уже UTF-8 — там ничего не меняется."""
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if enc != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main():
    """Точка входа CLI"""
    _utf8_stdio()

    parser = argparse.ArgumentParser(
        description="Occular OCR - text recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ocr photo.png                      # Recognize an image (text to stdout)
  ocr document.pdf                   # Recognize a PDF
  ocr ./scans ./out                  # Folder -> ./out/<name>.txt for every file
  ocr ./scans --workers 8            # Folder with 8 workers (.txt next to the files)
  ocr photo.png --gpu                # On GPU/CUDA (CUDA libraries: pip install "occular-ocr[gpu]")
  ocr photo.png --out result.json    # One file -> JSON with coordinates
  ocr invoice.png --tables           # Tables as Markdown tables in place
        """
    )

    # Позиционные: вход (файл или папка) и опциональный выход
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to an image/PDF or a folder of them"
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Folder for the .txt files (only when the input is a folder). Default: next to the source files"
    )

    # Для обратной совместимости
    parser.add_argument("--image", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--pdf", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--gpu", action="store_true", help="Run on GPU/CUDA (CUDA libraries: pip install 'occular-ocr[gpu]')")
    parser.add_argument(
        "--languages", "--lang",
        type=str,
        default=None,
        help="Text language(s), comma-separated: default ru/en; e.g. 'uk' or 'ru,kk,uk'; "
             "'auto' = detect automatically. Cyrillic-script languages: ba be bg cv kk ky mk mn sr tg tt uk",
    )

    # PDF опции
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PDF render DPI (default: 300)"
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="OCR even PDFs that have a text layer"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Pages in parallel (worker processes, ~1.1 GB each) for a PDF or a folder; "
             "default: auto by cores and memory, up to 16"
    )

    # Вывод (для одиночного файла)
    parser.add_argument(
        "--out", "-o",
        type=str,
        help="Save the result for a single file as JSON"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON (with coordinates)"
    )

    parser.add_argument(
        "--orientation",
        action="store_true",
        help="Detect page rotation (0/90/180/270°) and straighten before OCR"
    )
    parser.add_argument(
        "--tables",
        action="store_true",
        help="Find tables: Markdown tables in the text output, lines by cell and the structure in JSON"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["svtr_lcnet", "svtr_t"],
        default=None,
        help="Recognizer architecture: svtr_lcnet (default, fast) or svtr_t (larger)"
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
        print(f"Error: path not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # Импортируем и регистрируем компоненты
    from . import _ensure_registered, OCRPipeline
    _ensure_registered()

    detector = args.detector or "dbnet-onnx"
    # --languages: 'auto' или список кодов через запятую. Если задан явный --recognizer, он в приоритете.
    languages = None
    if args.languages:
        languages = args.languages.strip() if args.languages.strip().lower() == "auto" \
            else [c.strip() for c in args.languages.split(",") if c.strip()]

    try:
        pipeline = OCRPipeline(
            detector=detector,
            # --model задаёт архитектуру (svtr_lcnet по умолчанию); --recognizer — скрытая
            # опция для имени из реестра, она имеет приоритет.
            recognizer=args.recognizer or args.model,
            languages=languages,
            orientation=args.orientation,
            gpu=args.gpu,
            tables=args.tables
        )
    except Exception as e:
        print(f"Initialization error: {e}", file=sys.stderr)
        sys.exit(1)

    # Папка → батч в .txt
    if input_path.is_dir():
        _run_batch(pipeline, input_path, args)
        return

    # Одиночный файл
    _run_single(pipeline, input_path, args)


def _by_y(lines):
    return sorted(lines, key=lambda r: r["quad"][0][1])


def _plain_text(pipeline, file_path: Path, args) -> str:
    """Распознать один файл и вернуть только текст (страницы PDF разделены пустой строкой)."""
    from .tables import compose_text
    is_pdf = file_path.suffix.lower() == ".pdf"
    if is_pdf:
        results = pipeline.process_pdf(
            str(file_path), dpi=args.dpi, force_ocr=args.force_ocr, workers=args.workers
        )
        return "\n\n".join(compose_text(_by_y(page["results"]), page.get("tables")) for page in results)
    results = pipeline.process_image(str(file_path))
    return compose_text(_by_y(results), getattr(results, "tables", None))


def _run_batch(pipeline, input_dir: Path, args):
    """Папка → .txt на каждый файл. Выход: args.output или рядом с исходниками.
    Все картинки и все страницы всех PDF — одна очередь задач для процессов-воркеров (--workers,
    по умолчанию авто): страницы не ждут друг друга в пределах файла, .txt пишется, как только готов
    весь файл. Если процессы нельзя (см. occular.parallel) — по одному файлу, как раньше."""
    files = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in BATCH_EXTS)
    if not files:
        print(f"No images/PDFs in {input_dir} ({', '.join(sorted(BATCH_EXTS))})",
              file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output) if args.output else input_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from .parallel import auto_workers, can_use_processes, processes_safe, run_tasks, threads_per_worker
    base = pipeline._pipeline
    tasks, owner, bad = [], [], {}
    for f in files:
        if f.suffix.lower() == ".pdf":
            try:
                from .pipeline import pymupdf
                fitz = pymupdf()
                with fitz.open(str(f)) as d:
                    n = len(d)
            except Exception as e:
                bad[f] = e
                continue
            for p in range(n):
                tasks.append(("pdf", str(f), p, args.dpi, args.force_ocr)); owner.append(f)
        else:
            tasks.append(("image", str(f))); owner.append(f)
    workers = auto_workers(len(tasks)) if args.workers is None else max(1, min(args.workers, len(tasks)))

    ok = 0
    for f, e in bad.items():
        print(f"  [skipped] {f.name}: {e}", file=sys.stderr)
    if workers > 1 and can_use_processes(base) and processes_safe():
        from .tables import compose_text
        left = {f: owner.count(f) for f in set(owner)}
        parts = {f: {} for f in left}

        def on_done(i, res):
            nonlocal ok
            f = owner[i]
            parts[f][i] = res
            left[f] -= 1
            if left[f]:
                return
            got = [parts[f][k] for k in sorted(parts[f])]
            err = next((r for r in got if isinstance(r, Exception)), None)
            if err is not None:
                print(f"  [skipped] {f.name}: {err}", file=sys.stderr)
                return
            if tasks[i][0] == "image":
                lines, tables = got[0]
                text = compose_text(_by_y(lines), tables)
            else:
                text = "\n\n".join(compose_text(_by_y(pg["results"]), pg.get("tables")) for pg in got)
            out_path = out_dir / (f.stem + ".txt")
            out_path.write_text(text, encoding="utf-8")
            ok += 1
            print(f"  {f.name} → {out_path}")

        tpw = threads_per_worker(workers, base.num_threads)
        run_tasks(base, tasks, workers, tpw, on_done=on_done)
    else:
        for f in files:
            if f in bad:
                continue
            try:
                text = _plain_text(pipeline, f, args)
            except Exception as e:
                print(f"  [skipped] {f.name}: {e}", file=sys.stderr)
                continue
            out_path = out_dir / (f.stem + ".txt")
            out_path.write_text(text, encoding="utf-8")
            ok += 1
            print(f"  {f.name} → {out_path}")
    print(f"Done: {ok}/{len(files)} files, .txt in {out_dir}")


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
        print(f"Processing error: {e}", file=sys.stderr)
        sys.exit(1)

    from .tables import compose_text
    # изображение с --tables: строки и таблицы рядом (у PDF таблицы уже в ключе 'tables' страницы)
    data = results
    if not is_pdf and args.tables:
        data = {"results": list(results), "tables": getattr(results, "tables", [])}

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved: {args.out}")
    elif args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        # Простой вывод текста (сверху вниз; таблицы — Markdown на своём месте)
        if is_pdf:
            for page in results:
                print(f"--- Page {page['page']} ---")
                print(compose_text(_by_y(page["results"]), page.get("tables")))
        else:
            print(compose_text(_by_y(results), getattr(results, "tables", None)))


if __name__ == "__main__":
    main()
