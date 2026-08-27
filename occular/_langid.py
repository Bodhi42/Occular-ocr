"""Лёгкий определитель языка: char-n-gram (order 4, stupid backoff). ЧИСТЫЙ Python — без sklearn/numpy/scipy.
Классы: 12 кириллических языков + русский. Модель — langid_models.pkl (14 МБ, счётчики n-грамм)."""
import math
import pickle

ORDER = 4
LAM = 0.4


def _norm(s):
    return ''.join(c.lower() for c in s if c.isalpha() or c == ' ')


def _logscore(text, model, V):
    counts, tot = model
    s = '^' + _norm(text) + '$'
    lp = 0.0
    n = 0
    for i in range(1, len(s)):
        got = False
        for k in range(ORDER, 0, -1):
            if i - (k - 1) < 0:
                continue
            ng = s[i - (k - 1):i + 1]
            ctx = s[i - (k - 1):i]
            c = counts[k][ng]
            if c > 0:
                cc = counts[k - 1][ctx] if k > 1 else tot
                lp += math.log((c / cc) * (LAM ** (ORDER - k)))
                n += 1
                got = True
                break
        if not got:
            lp += math.log((1.0 / V) * (LAM ** ORDER))
            n += 1
    return lp / max(n, 1)


class LangID:
    """char-n-gram определитель. identify(text, candidates) → язык среди кандидатов."""

    def __init__(self, pkl_path):
        self.models, self.V = pickle.load(open(pkl_path, 'rb'))
        self.langs = list(self.models)

    def identify(self, text, candidates=None):
        cand = [c for c in (candidates or self.langs) if c in self.models]
        if not cand:
            return None
        return max(cand, key=lambda lg: _logscore(text, self.models[lg], self.V))
