"""Beam-CTC + языковая модель — БЕЗ единой C-зависимости (чистый Python).

Зачем: greedy-argmax теряет качество. beam + n-gram LM (обучена на 5М строк русского текста)
+ словарь униграмм даёт заметно меньше ошибок БЕЗ дообучения рекогнайзера. Всё на чистом Python,
ставится одним pip (backend LM — см. _pylm.py).

Файлы LM (compact-формат):
  • compact_lm.npz  — n-граммы (uint64-хеши + logprob/backoff), ~270 МБ, грузится за секунды
  • unigrams.txt    — словарь (по слову на строку), ~25 МБ, для partial-word scoring

Откуда берутся (в порядке приоритета):
  1) переменная окружения OCCULAR_LM_DIR — папка с обоими файлами (локально);
  2) HuggingFace-репо LM_HF_REPO — качается один раз в кэш.

Параметры декода (подобраны абляцией): alpha=0.5, beta=1.0, beam_width=100.
"""
import os
import logging
import warnings
import numpy as np
from pathlib import Path

from .model_files import LM_HF_REPO

# доброкачественные предупреждения декодера — глушим.
logging.getLogger("pyctcdecode").setLevel(logging.ERROR)

NPZ = "compact_lm.npz"
UNIGRAMS = "unigrams.txt"
ALPHA = 0.5
BETA = 1.0
BEAM_WIDTH = 100


def _resolve_lm_files() -> tuple:
    """Вернуть (путь_к_compact_lm.npz, путь_к_unigrams.txt). Сначала OCCULAR_LM_DIR, потом HuggingFace."""
    d = os.environ.get("OCCULAR_LM_DIR")
    if d:
        npz = Path(d) / NPZ
        uni = Path(d) / UNIGRAMS
        if not npz.exists():
            raise FileNotFoundError(f"OCCULAR_LM_DIR задан, но нет {npz}")
        if not uni.exists():
            raise FileNotFoundError(f"OCCULAR_LM_DIR задан, но нет {uni}")
        return str(npz), str(uni)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "Для beam+LM нужен huggingface_hub (pip install huggingface_hub) "
            "или задайте OCCULAR_LM_DIR с локальными compact_lm.npz и unigrams.txt."
        )
    npz = hf_hub_download(LM_HF_REPO, NPZ)
    uni = hf_hub_download(LM_HF_REPO, UNIGRAMS)
    return npz, uni


def _build_native(labels, npz_path, uni_path, alpha, beta, beam_width):
    """Собрать нативный декодер (occular_decode, Rust), если он установлен.

    Пакет опционален: колёса собираются заранее под каждую платформу, компилятор пользователю
    не нужен. Если колеса под платформу нет — возвращаем None и работаем на чистом Python,
    результат совпадает построчно. Отключить принудительно: OCCULAR_NATIVE_DECODER=0.
    """
    if os.environ.get("OCCULAR_NATIVE_DECODER", "1") == "0":
        return None
    try:
        import occular_decode
    except ImportError:
        return None

    # Быстрый путь: файлы отображаются в память на стороне Rust, ничего не копируется.
    try:
        return occular_decode.Decoder.from_npz(
            labels, str(npz_path), str(uni_path),
            alpha=alpha, beta=beta, beam_width=int(beam_width),
        )
    except Exception as e:
        # npz со сжатием, необычный формат, ФС без mmap — грузим как раньше, через numpy.
        warnings.warn(f"occular: mmap-загрузка LM не удалась ({e}); читаю через numpy")

    d = np.load(npz_path)
    orders = (1, 2, 3, 4)
    return occular_decode.Decoder(
        labels,
        [d[f"k{o}"] for o in orders],
        [d[f"lp{o}"] for o in orders],
        [d[f"bo{o}"] for o in orders],
        str(uni_path),
        alpha=alpha, beta=beta, beam_width=int(beam_width),
    )


class LMDecoder:
    """Beam-CTC + n-gram LM. Считает нативный декодер (occular_decode), если он установлен;
    иначе — чистый Python. Результат совпадает построчно. Один инстанс на процесс."""

    def __init__(self, vocab, alpha: float = ALPHA, beta: float = BETA,
                 beam_width: int = BEAM_WIDTH, lm_files: tuple = None):
        """lm_files=(npz_path, uni_path) — явные файлы LM (напр. пер-язычная кир-LM);
        None = русская LM по умолчанию (_resolve_lm_files)."""
        self.beam_width = int(beam_width)
        labels = [""] + list(vocab)                 # index 0 = CTC blank
        npz_path, uni_path = lm_files if lm_files else _resolve_lm_files()
        size_mb = os.path.getsize(npz_path) / 1e6

        # Нативный (Rust) декодер, если установлен пакет occular_decode: страница считается
        # параллельно за один вызов (GIL отпущен), файлы LM отображаются в память (общие меж воркерами).
        self.native = _build_native(labels, npz_path, uni_path, alpha, beta, self.beam_width)
        if self.native is not None:
            self.decoder = None
            print(f"[occular] LM (нативный декодер): {Path(npz_path).name} ({size_mb:.0f} МБ), "
                  f"униграмм={self.native.vocab_size()}, alpha={alpha} beta={beta} "
                  f"beam={self.beam_width}")
            return

        from . import _pylm
        unigrams = [l.rstrip("\n") for l in open(uni_path, encoding="utf-8") if l.strip()]
        self.decoder = _pylm.build_decoder(labels, npz_path, unigrams, alpha=alpha, beta=beta)
        print(f"[occular] LM (чисто-Python): {Path(npz_path).name} "
              f"({size_mb:.0f} МБ), униграмм={len(unigrams)}, "
              f"alpha={alpha} beta={beta} beam={self.beam_width}")

    @staticmethod
    def _log_softmax(logits: np.ndarray) -> np.ndarray:
        m = logits.max(axis=-1, keepdims=True)
        e = np.exp(logits - m)
        return (logits - m) - np.log(e.sum(axis=-1, keepdims=True))

    @staticmethod
    def _confidence(text: str, logit_score: float) -> float:
        """Confidence = exp(acoustic_score / len(text)) верхней beam-гипотезы: привязан к выбранному
        тексту и нормирован на длину (AUC 0.97 vs 0.90 у старого min-по-кадрам)."""
        return min(1.0, float(np.exp(logit_score / max(1, len(text)))))

    def decode(self, logits_1tc: np.ndarray) -> tuple:
        """logits [1,T,C] -> (text, confidence)."""
        if self.native is not None:
            text, logit_score = self.native.decode(np.ascontiguousarray(logits_1tc[0], np.float32))
            return text, self._confidence(text, logit_score)

        lp = self._log_softmax(logits_1tc[0].astype(np.float32))
        beams = self.decoder.decode_beams(lp, beam_width=self.beam_width)
        if not beams:
            return "", 0.0
        top = beams[0]
        text = top[0]                                    # (text, last_word, frames, logit_score, lm_score)
        logit_score = float(top[-2])                     # акустический score пути (без LM), ≤ 0
        return text, self._confidence(text, logit_score)

    def decode_many(self, logits_list) -> list:
        """Список [1,T,C] -> список (text, confidence).

        Нативный декодер считает строки страницы параллельно за один вызов, поэтому целую
        страницу выгоднее отдавать сюда, а не по строке. Без нативного декодера — обычный цикл.
        """
        if self.native is None:
            return [self.decode(lg) for lg in logits_list]
        batch = [np.ascontiguousarray(lg[0], np.float32) for lg in logits_list]
        return [(t, self._confidence(t, s)) for t, s in self.native.decode_batch(batch)]
