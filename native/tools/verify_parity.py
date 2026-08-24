#!/usr/bin/env python3
"""Проверка, что нативный декодер выдаёт ровно тот же текст, что и чистый Python.

Гоняет обе реализации на одних и тех же логитах и сравнивает построчно, заодно меряя время.

    python verify_parity.py <папка с картинками или файл> [ещё файлы...]

Ничего не меняет и никуда не пишет. Если нативного модуля нет — честно скажет об этом.
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect(args):
    files = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            files += sorted(x for x in p.iterdir() if x.suffix.lower() in EXTS)
        elif p.suffix.lower() in EXTS:
            files.append(p)
    return files


def crop(img, quad):
    q = np.asarray(quad, np.int32)
    return img[max(0, q[:, 1].min()):q[:, 1].max(), max(0, q[:, 0].min()):q[:, 0].max()]


def main():
    files = collect(sys.argv[1:])
    if not files:
        print(__doc__)
        sys.exit(1)

    from occular import OCRPipeline, Settings
    from occular.decoder_lm import LMDecoder, _resolve_lm_files, ALPHA, BETA, BEAM_WIDTH

    pipe = OCRPipeline(Settings(lm=True))
    rec, det = pipe._pipeline.recognizer, pipe._pipeline.detector
    native = rec._ensure_lm()
    if native.native is None:
        print("Нативный декодер не подхватился — сравнивать не с чем.\n"
              "Проверьте: pip install occular_decode-*.whl  и переменную OCCULAR_NATIVE_DECODER.")
        sys.exit(2)

    # вторая, чисто питоновская реализация — для сравнения
    from occular import _pylm
    npz_path, uni_path = _resolve_lm_files()
    t = time.time()
    unigrams = [l.rstrip("\n") for l in open(uni_path, encoding="utf-8") if l.strip()]
    pure = _pylm.build_decoder([""] + list(rec.vocab), npz_path, unigrams, alpha=ALPHA, beta=BETA)
    print(f"питоновский декодер собран за {time.time() - t:.2f} с\n")

    print(f"{'файл':28} {'строк':>5} {'python':>9} {'rust':>9} {'ускорение':>10}  расхождений")
    print("-" * 78)
    total = mismatch = 0
    for f in files:
        img = np.array(Image.open(f).convert("RGB"))
        logits = []
        for q in det.detect(img):
            c = crop(img, q)
            if c.size:
                logits.append(rec._infer(rec._create_batch([rec._preprocess_single(c)])))
        if not logits:
            continue

        lp = [LMDecoder._log_softmax(x[0].astype(np.float32)) for x in logits]
        t = time.time()
        py_txt = [pure.decode_beams(l, beam_width=BEAM_WIDTH)[0][0] for l in lp]
        t_py = time.time() - t

        t = time.time()
        rs_txt = [txt for txt, _ in native.decode_many(logits)]
        t_rs = time.time() - t

        bad = [(a, b) for a, b in zip(py_txt, rs_txt) if a != b]
        total += len(py_txt)
        mismatch += len(bad)
        print(f"{f.name[:28]:28} {len(py_txt):5} {t_py:8.3f}с {t_rs:8.3f}с "
              f"{t_py / max(t_rs, 1e-9):9.1f}× {len(bad):>12}")
        for a, b in bad[:5]:
            print(f"    python: {a!r}")
            print(f"    rust  : {b!r}")

    print("-" * 78)
    print(f"строк {total}, расхождений {mismatch} "
          f"({0 if not total else mismatch / total * 100:.2f} %)")
    sys.exit(0 if mismatch == 0 else 3)


if __name__ == "__main__":
    main()
