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


class LMDecoder:
    """Beam-CTC + n-gram LM на чистом Python. Один инстанс на процесс (LM держим в памяти)."""

    def __init__(self, vocab, alpha: float = ALPHA, beta: float = BETA,
                 beam_width: int = BEAM_WIDTH):
        from . import _pylm
        self.beam_width = int(beam_width)
        labels = [""] + list(vocab)                 # index 0 = CTC blank
        npz_path, uni_path = _resolve_lm_files()
        unigrams = [l.rstrip("\n") for l in open(uni_path, encoding="utf-8") if l.strip()]
        self.decoder = _pylm.build_decoder(labels, npz_path, unigrams, alpha=alpha, beta=beta)
        print(f"[occular] LM (чисто-Python): {Path(npz_path).name} "
              f"({os.path.getsize(npz_path) / 1e6:.0f} МБ), униграмм={len(unigrams)}, "
              f"alpha={alpha} beta={beta} beam={self.beam_width}")

    @staticmethod
    def _log_softmax(logits: np.ndarray) -> np.ndarray:
        m = logits.max(axis=-1, keepdims=True)
        e = np.exp(logits - m)
        return (logits - m) - np.log(e.sum(axis=-1, keepdims=True))

    def decode(self, logits_1tc: np.ndarray) -> tuple:
        """logits [1,T,C] -> (text, confidence). Confidence — как в greedy (min по кадрам max-prob)."""
        lp = self._log_softmax(logits_1tc[0].astype(np.float32))
        text = self.decoder.decode(lp, beam_width=self.beam_width)
        conf = float(np.exp(lp).max(axis=-1).min())
        return text, conf
