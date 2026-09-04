"""Мультиязычный роутер: русская модель (ru/en +LM) + кириллическая-12 + per-язык декод по конфигу.

Один кир-распознаватель покрывает 12 языков: ba be bg cv kk ky mk mn sr tg tt uk. Как декодить
каждый язык — задаёт `cyr_decode_config.json` (подобран абляцией под текущую модель):
  • 8 языков — beam+CTC со СВОЕЙ пер-язычной n-gram LM и своими alpha/beta (репо occular-lm-cyr);
  • 4 языка (be cv mn tt) — greedy: LM их не улучшает (распознаватель у потолка / нет доменного корпуса).

Два сценария, один механизм:
  1) юзер задаёт языки, напр. languages=['ru','en','uk'] — если все в одной модели, грузим только её;
     если охватывают обе — по-строчный langid СРЕДИ запрошенных, каждую строку её моделью и декодом;
  2) авто (languages=None) — langid среди всех 14 (ru/en + 12 кир.), маршрутизация по-строчно.

Модели грузятся ЛЕНИВО: работа с одним кир-языком не тянет русскую модель, и наоборот.
lm=False → greedy для всех (LM игнорируется). Пер-язычная LM грузится по требованию; если её нет —
откат на greedy для этого языка (с предупреждением).
"""
import warnings
import numpy as np
from typing import List, Tuple, Optional, Dict

RU_MODEL_LANGS = {'ru', 'en'}                                   # русская модель (+beam+LM)
CYR12_LANGS = {'ba', 'be', 'bg', 'cv', 'kk', 'ky', 'mk', 'mn', 'sr', 'tg', 'tt', 'uk'}
ALL_LANGS = RU_MODEL_LANGS | CYR12_LANGS


class MultilingualRouter:
    def __init__(self, num_threads: int = 4, gpu: bool = False, lm: bool = True,
                 arch: str = None):
        from .recognizer_onnx import DEFAULT_ARCH
        self._arch = arch or DEFAULT_ARCH                       # одна архитектура на обе модели
        self._num_threads = num_threads
        self._gpu = gpu
        self._use_lm = bool(lm)
        self._ru_rec = None                                     # ленивая русская модель (ru/en)
        self._cyr_rec = None                                    # ленивая кир-модель (12 языков)
        self._cyr_lms: Dict[str, object] = {}                   # lang -> LMDecoder | None (ленивый кэш)
        self._langid = None
        self._langid_tried = False
        from .model_files import load_cyr_decode_config
        self._cfg = load_cyr_decode_config()                    # {lang: {decode, alpha, beta, ...}}

    # --- ленивые модели -----------------------------------------------------
    @property
    def _ru(self):
        if self._ru_rec is None:
            from .recognizer_onnx import CRNNRecognizerONNX
            self._ru_rec = CRNNRecognizerONNX(languages=['ru', 'en'], num_threads=self._num_threads,
                                              gpu=self._gpu, lm=self._use_lm, arch=self._arch)
        return self._ru_rec

    @property
    def _cyr(self):
        if self._cyr_rec is None:
            from .recognizer_onnx import CRNNRecognizerONNX
            # Кир-модель — только CPU/ONNX (torch-бэкенда .pth для неё нет). lm=False: декодом рулит
            # роутер (каждую кир-строку — её пер-язычной LM или greedy по конфигу).
            from .recognizer_onnx import arch_files
            cyr_onnx, cyr_charset = arch_files(self._arch, "cyr")
            # GPU для кир-модели есть только у svtr_lcnet (torch-веса залиты с 0.4.1)
            self._cyr_rec = CRNNRecognizerONNX(languages=sorted(CYR12_LANGS), num_threads=self._num_threads,
                                               gpu=self._gpu, lm=False, arch=self._arch, family="cyr",
                                               onnx_file=cyr_onnx, charset_file=cyr_charset)
        return self._cyr_rec

    def _get_langid(self):
        if self._langid is None and not self._langid_tried:
            self._langid_tried = True
            try:
                from ._langid import LangID
                from .model_files import ensure_weight
                self._langid = LangID(ensure_weight("langid_models.pkl"))
            except Exception as e:
                warnings.warn(f"langid недоступен ({e}); авто-роутинг по языкам выключен — "
                              f"задайте языки одной модели явно (languages=[...]).")
        return self._langid

    # --- декод кир-строки по конфигу ---------------------------------------
    def _cyr_lm(self, lang: str):
        """Пер-язычная LM кир-языка (ленивая загрузка + кэш) с alpha/beta из конфига. None → greedy."""
        if lang not in self._cyr_lms:
            cfg = self._cfg.get(lang, {})
            if cfg.get('decode') != 'beam+lm':
                self._cyr_lms[lang] = None
                return None
            from .decoder_lm import LMDecoder
            from .model_files import ensure_cyr_lm
            try:
                self._cyr_lms[lang] = LMDecoder(
                    self._cyr.vocab, lm_files=ensure_cyr_lm(lang),
                    alpha=float(cfg.get('alpha', 0.5)), beta=float(cfg.get('beta', 1.0)))
            except Exception as e:
                warnings.warn(f"LM языка '{lang}' недоступна ({e}); строки '{lang}' декодятся greedy.")
                self._cyr_lms[lang] = None
        return self._cyr_lms[lang]

    def _decode_cyr(self, logits: np.ndarray, lang: Optional[str]) -> Tuple[str, float]:
        """Декод кир-строки: beam+пер-язычная LM если (lm вкл, язык известен, конфиг=beam+lm, LM есть),
        иначе greedy. Пороги/alpha/beta — из `cyr_decode_config.json` (абляция под текущую модель)."""
        if self._use_lm and lang:
            lm = self._cyr_lm(lang)
            if lm is not None:
                return lm.decode(logits)
        return self._cyr._ctc_decode(logits)

    def _detect(self, text: str, cand: set) -> Optional[str]:
        """Язык строки среди кандидатов. Латиница → en; кириллица → langid среди кир-кандидатов."""
        letters = [c for c in text if c.isalpha()]
        if letters and sum(0x400 <= ord(c) <= 0x52f for c in letters) / len(letters) < 0.5:
            return 'en' if 'en' in cand else None                # латинское письмо
        lid = self._get_langid()
        if lid is None:
            return None
        cyr_cand = [c for c in cand if c in lid.models]
        return lid.identify(text, cyr_cand) if cyr_cand else None

    def recognize(self, image: np.ndarray, quads: List[np.ndarray],
                  languages: Optional[List[str]] = None) -> List[Tuple[str, float, str]]:
        """→ [(текст, уверенность, язык)] на каждую строку. languages=None → авто."""
        if not quads:
            return []
        cand = set(languages) if languages else set(ALL_LANGS)

        # быстрый путь: весь запрос — русская модель (ru/en +LM), кир-модель не грузим
        if cand <= RU_MODEL_LANGS:
            return [(t, c, 'ru/en') for t, c in self._ru.recognize(image, quads)]

        logits = self._cyr.logits_per_line(image, quads)         # логиты кир-модели, выровнены к quads

        # быстрый путь: один кир-язык — без langid, вся страница его декодом (LM или greedy по конфигу)
        cyr_cand = cand & CYR12_LANGS
        if cand <= CYR12_LANGS and len(cyr_cand) == 1:
            lang = next(iter(cyr_cand))
            return [self._decode_cyr(lg, lang) + (lang,) if lg is not None else ("", 0.0, lang)
                    for lg in logits]

        # смешанный / авто: greedy-проход для langid → маршрут (кир-строки — свой декод; ru/en — рус. модель)
        greedy = [self._cyr._ctc_decode(lg) if lg is not None else ("", 0.0) for lg in logits]
        langs = [self._detect(t, cand) if lg is not None else None
                 for (t, _), lg in zip(greedy, logits)]
        out: List[Optional[Tuple[str, float, str]]] = [None] * len(quads)
        ru_idx = []
        for i, (lg, lang) in enumerate(zip(logits, langs)):
            if lg is None:
                out[i] = ("", 0.0, '?')
            elif lang in RU_MODEL_LANGS:
                ru_idx.append(i)                                 # соберём и прогоним русской моделью пачкой
            else:
                t, c = self._decode_cyr(lg, lang)                # кир-строка своим декодом (LM/greedy)
                out[i] = (t, c, lang or '?')
        if ru_idx:
            ru_res = self._ru.recognize(image, [quads[i] for i in ru_idx])
            for j, i in enumerate(ru_idx):
                out[i] = (ru_res[j][0], ru_res[j][1], langs[i])
        return out


class MultilingualRecognizer:
    """Адаптер MultilingualRouter под интерфейс распознавателя occular.

    Совместим с CRNNRecognizerONNX: recognize(image, quads) → [(текст, уверенность)]. Языки фиксируются
    при создании (recognizer_kwargs={'languages': [...]}); None = авто-определение по строкам.
    Регистрируется как распознаватель 'multilingual'; выбирается публичным ocr(..., languages=...).
    """

    def __init__(self, languages: Optional[List[str]] = None, num_threads: int = 4,
                 gpu: bool = False, lm: bool = True, arch: str = None, **_ignore):
        self.languages = list(languages) if languages else None
        self._router = MultilingualRouter(num_threads=num_threads, gpu=gpu, lm=lm, arch=arch)

    def recognize(self, image: np.ndarray, quads: List[np.ndarray]) -> List[Tuple[str, float]]:
        return [(t, c) for t, c, _lg in self._router.recognize(image, quads, self.languages)]
