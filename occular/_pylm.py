"""Self-contained n-gram language-model backend for the beam decoder — no native dependencies.

The beam search (``_beam.py``) is pure Python; it only needs a scoring backend. This module provides that backend
on top of a compact numpy n-gram model, plus a fast prefix checker, so decoding quality and speed
match a native implementation while everything installs with a plain ``pip``.

The n-gram model lives in ``compact_lm.npz``: for each order, sorted uint64 hashes of the n-grams
(FNV-1a over the words) together with log-probabilities and back-off weights (float32). No word→id
table is needed — lookups compare hashes directly.
"""
import bisect
import numpy as np

_MASK = (1 << 64) - 1
_OFF = 1469598103934665603
_PRIME = 1099511628211


def _h64(words):
    """FNV-1a 64-bit over a tuple of words (with a separator). Must match the model builder."""
    h = _OFF
    for w in words:
        for b in w.encode("utf-8"):
            h = ((h ^ b) * _PRIME) & _MASK
        h = ((h ^ 255) * _PRIME) & _MASK
    return h


class State:
    """Decoder LM state — holds the context (up to 3 preceding words)."""
    __slots__ = ("ctx",)

    def __init__(self):
        self.ctx = ()


class Model:
    """N-gram model over the compact numpy store (``compact_lm.npz``)."""

    def __init__(self, path):
        d = np.load(path)
        self.K = {o: d[f"k{o}"] for o in (1, 2, 3, 4)}
        self.LP = {o: d[f"lp{o}"] for o in (1, 2, 3, 4)}
        self.BO = {o: d[f"bo{o}"] for o in (1, 2, 3, 4)}
        self.order = 4
        self._uni = set(int(x) for x in self.K[1])          # unigram hashes — fast membership
        u = self._look(("<unk>",))
        self._unk = u[0] if u else -7.0

    def _look(self, words):
        o = len(words)
        if o > 4:
            return None
        k = self.K[o]
        key = np.uint64(_h64(words))
        i = int(np.searchsorted(k, key))
        if i < len(k) and k[i] == key:
            return (float(self.LP[o][i]), float(self.BO[o][i]))
        return None

    def __contains__(self, word):
        return _h64((word,)) in self._uni

    def BeginSentenceWrite(self, st):
        st.ctx = ("<s>",)

    def NullContextWrite(self, st):
        st.ctx = ()

    def _score(self, word, ctx):
        """log10 P(word | ctx) with Katz back-off."""
        ctx = ctx[-3:]
        for start in range(len(ctx) + 1):
            g = self._look(ctx[start:] + (word,))
            if g is not None:
                back = 0.0
                for s in range(start):
                    gb = self._look(ctx[s:])
                    if gb:
                        back += gb[1]
                return g[0] + back
        return self._unk

    def BaseScore(self, in_st, word, out_st):
        out_st.ctx = (in_st.ctx + (word,))[-3:]
        return self._score(word, in_st.ctx)


class _PrefixChecker:
    """``has_node(p)`` returns 1 if ``p`` is a prefix of some vocabulary word.

    The beam only asks whether the current partial word can still form a known word — a binary
    search over the sorted vocabulary, so it loads instantly and gives identical answers.
    """

    def __init__(self, sorted_words):
        self.w = sorted_words

    def has_node(self, p):
        i = bisect.bisect_left(self.w, p)
        return 1 if (i < len(self.w) and self.w[i].startswith(p)) else 0


def build_decoder(labels, npz_path, unigrams, alpha=0.5, beta=1.0):
    """Build a beam decoder over the pure-Python LM backend.

    Args:
        labels: [""] + list of characters (blank first)
        npz_path: path to compact_lm.npz
        unigrams: full vocabulary word list
    """
    from ._beam import BeamSearchDecoder, LanguageModel
    lm = LanguageModel(Model(str(npz_path)), State, unigrams, _PrefixChecker(sorted(unigrams)),
                       alpha=alpha, beta=beta)
    return BeamSearchDecoder(labels, lm)
