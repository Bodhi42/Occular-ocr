# Beam search adapted from pyctcdecode 0.5.0 (decoder.py, language_model.py), Copyright 2021-present
# Kensho Technologies, LLC, licensed under the Apache License 2.0; see LICENSE-pyctcdecode.
# Changes: only the configuration occular uses is kept (character alphabet, n-gram LM, no hotwords,
# no history pruning, no multiprocessing); word-frame bookkeeping is dropped (it never affects scores).
"""CTC beam search with shallow n-gram LM fusion — pure Python + numpy, no third-party decoder.

Scores, tie-breaking and output are identical to ``pyctcdecode.BeamSearchDecoderCTC.decode_beams``
with the same model; kept in-tree so the package does not inherit pyctcdecode's ``numpy<2`` pin.
"""
import heapq
import math

import numpy as np

# pyctcdecode defaults (decoder parameters occular relies on)
UNK_LOGP_OFFSET = -10.0
BEAM_PRUNE_LOGP = -10.0
TOKEN_MIN_LOGP = -5.0
AVG_TOKEN_LEN = 6                          # expected characters per word (for OOV prefix scoring)
MIN_TOKEN_CLIP_P = 1e-15
LOG_BASE_CHANGE_FACTOR = 1.0 / math.log10(math.e)    # n-gram scores are log10, beam works in ln


def _log_softmax(x, axis):
    """Logarithm of softmax (as in scipy.special / pyctcdecode)."""
    x_max = np.amax(x, axis=axis, keepdims=True)
    if x_max.ndim > 0:
        x_max[~np.isfinite(x_max)] = 0
    elif not np.isfinite(x_max):
        x_max = 0
    tmp = x - x_max
    exp_tmp = np.exp(tmp)
    with np.errstate(divide="ignore"):
        s = np.sum(exp_tmp, axis=axis, keepdims=True)
        out = np.log(s)
    return tmp - out


def _sum_log_scores(s1, s2):
    if s1 >= s2:
        return s1 + math.log(1 + math.exp(s2 - s1))
    return s2 + math.log(1 + math.exp(s1 - s2))


def _merge_tokens(token_1, token_2):
    if len(token_2) == 0:
        return token_1
    if len(token_1) == 0:
        return token_2
    return token_1 + " " + token_2


# beam: (text, next_word, word_part, last_char, logit_score)
def _merge_beams(beams):
    """Merge beams that read the same (text, partial word, last char); sum their probabilities."""
    beam_dict = {}
    for text, next_word, word_part, last_char, logit_score in beams:
        key = (_merge_tokens(text, next_word), word_part, last_char)
        prev = beam_dict.get(key)
        if prev is None:
            beam_dict[key] = (text, next_word, word_part, last_char, logit_score)
        else:
            beam_dict[key] = (text, next_word, word_part, last_char,
                              _sum_log_scores(prev[-1], logit_score))
    return list(beam_dict.values())


class LanguageModel:
    """Shallow-fusion scorer over an n-gram model with the kenlm-like interface of ``_pylm.Model``."""

    def __init__(self, model, state_cls, unigrams, prefix_checker, alpha, beta,
                 unk_score_offset=UNK_LOGP_OFFSET, score_boundary=True):
        self._model = model
        self._state_cls = state_cls
        self._unigram_set = set(unigrams)
        self._char_trie = prefix_checker
        self.alpha = alpha
        self.beta = beta
        self.unk_score_offset = unk_score_offset
        self.score_boundary = score_boundary

    @property
    def order(self):
        return self._model.order

    def get_start_state(self):
        st = self._state_cls()
        if self.score_boundary:
            self._model.BeginSentenceWrite(st)
        else:
            self._model.NullContextWrite(st)
        return st

    def _get_raw_end_score(self, start_state):
        if self.score_boundary:
            return self._model.BaseScore(start_state, "</s>", self._state_cls())
        return 0.0

    def score_partial_token(self, partial_token):
        is_oov = int(self._char_trie.has_node(partial_token) == 0)
        unk_score = self.unk_score_offset * is_oov
        if len(partial_token) > AVG_TOKEN_LEN:
            unk_score = unk_score * len(partial_token) / AVG_TOKEN_LEN
        return unk_score

    def score(self, prev_state, word, is_last_word=False):
        end_state = self._state_cls()
        lm_score = self._model.BaseScore(prev_state, word, end_state)
        # same precedence as pyctcdecode: (known-set non-empty and word unknown) or word not in model
        if len(self._unigram_set) > 0 and word not in self._unigram_set or word not in self._model:
            lm_score += self.unk_score_offset
        if is_last_word:
            lm_score = lm_score + self._get_raw_end_score(end_state)
        lm_score = self.alpha * lm_score * LOG_BASE_CHANGE_FACTOR + self.beta
        return lm_score, end_state


class BeamSearchDecoder:
    """CTC beam search over a character alphabet (index 0 = blank ``""``, word separator ``" "``)."""

    def __init__(self, labels, language_model):
        labels = list(labels)
        if "" not in labels:
            raise ValueError("alphabet must contain the CTC blank ''")
        if len(labels) != len(set(labels)):
            raise ValueError("alphabet contains duplicate entries")
        self._idx2vocab = dict(enumerate(labels))
        self._lm = language_model

    def _get_lm_beams(self, beams, cached_lm_scores, cached_partial_token_scores, is_eos=False):
        """Beams with combined score appended: (text, '', word_part, last_char, logit, combined)."""
        lm = self._lm
        new_beams = []
        for text, next_word, word_part, last_char, logit_score in beams:
            new_text = _merge_tokens(text, next_word)
            cache_key = (new_text, is_eos)
            if cache_key not in cached_lm_scores:
                prev_raw_lm_score, start_state = cached_lm_scores[(text, False)]
                score, end_state = lm.score(start_state, next_word, is_last_word=is_eos)
                cached_lm_scores[cache_key] = (prev_raw_lm_score + score, end_state)
            lm_score = cached_lm_scores[cache_key][0]
            if len(word_part) > 0:
                if word_part not in cached_partial_token_scores:
                    cached_partial_token_scores[word_part] = lm.score_partial_token(word_part)
                lm_score += cached_partial_token_scores[word_part]
            new_beams.append((new_text, "", word_part, last_char, logit_score, logit_score + lm_score))
        return new_beams

    def _decode_logits(self, logits, beam_width, beam_prune_logp, token_min_logp):
        cached_lm_scores = {("", False): (0.0, self._lm.get_start_state())}
        cached_p_lm_scores = {}
        beams = [("", "", "", None, 0.0)]
        idx2vocab = self._idx2vocab
        for logit_col in logits:
            max_idx = logit_col.argmax()
            # set iteration order matters for merge/tie-breaking — keep pyctcdecode's construction
            idx_list = set(np.where(logit_col >= token_min_logp)[0]) | {max_idx}
            new_beams = []
            for idx_char in idx_list:
                p_char = logit_col[idx_char]
                char = idx2vocab[idx_char]
                for text, next_word, word_part, last_char, logit_score in beams:
                    if char == "" or last_char == char:          # blank or repeated char
                        new_beams.append((text, next_word, word_part, char, logit_score + p_char))
                    elif char == " ":                            # word boundary
                        new_beams.append((text, word_part, "", char, logit_score + p_char))
                    else:                                        # continue the current word
                        new_beams.append((text, next_word, word_part + char, char, logit_score + p_char))
            new_beams = _merge_beams(new_beams)
            scored = self._get_lm_beams(new_beams, cached_lm_scores, cached_p_lm_scores)
            max_score = max([b[-1] for b in scored])
            scored = [b for b in scored if b[-1] >= max_score + beam_prune_logp]
            beams = [b[:-1] for b in heapq.nlargest(beam_width, scored, key=lambda x: x[-1])]

        # close the last word, score end of sentence
        new_beams = [(text, word_part, "", None, logit_score)
                     for text, _, word_part, _, logit_score in beams]
        new_beams = _merge_beams(new_beams)
        scored = self._get_lm_beams(new_beams, cached_lm_scores, cached_p_lm_scores, is_eos=True)
        max_score = max([b[-1] for b in scored])
        scored = [b for b in scored if b[-1] >= max_score + beam_prune_logp]
        trimmed = heapq.nlargest(beam_width, scored, key=lambda x: x[-1])
        return [(" ".join(text.split()), logit_score, combined_score)
                for text, _, _, _, logit_score, combined_score in trimmed]

    def decode_beams(self, logits, beam_width=100, beam_prune_logp=BEAM_PRUNE_LOGP,
                     token_min_logp=TOKEN_MIN_LOGP):
        """logits [T, C] -> beams sorted best-first: (text, acoustic_logit_score, combined_score)."""
        if logits.ndim != 2 or logits.shape[-1] != len(self._idx2vocab):
            raise ValueError(f"logits must be (time, {len(self._idx2vocab)}), got {logits.shape}")
        if math.isclose(logits.sum(axis=1).mean(), 1):          # probabilities -> log probs
            logits = np.log(np.clip(logits, MIN_TOKEN_CLIP_P, 1))
        else:
            logits = np.clip(_log_softmax(logits, axis=1), np.log(MIN_TOKEN_CLIP_P), 0)
        return self._decode_logits(logits, beam_width, beam_prune_logp, token_min_logp)
