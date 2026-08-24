//! CTC prefix beam search + n-gram LM для Occular-OCR — порт чистого Python-пути на Rust.
//!
//! Воспроизводит связку `pyctcdecode.BeamSearchDecoderCTC._decode_logits` +
//! `LanguageModel.score` + `ocr_skel._pylm.Model` так, чтобы выдавать тот же текст,
//! но без Python-объектов в горячем цикле.
//!
//! Отличия от Python-пути, влияющие только на производительность:
//!   * состояние LM — это не кортеж слов, а четыре 64-битных хеша суффиксов контекста,
//!     поэтому n-граммный ключ считается инкрементально, без пересборки строк;
//!   * словарь униграмм лежит одним отсортированным блобом (`Vec<u8>` + смещения),
//!     а не списком из двух миллионов строк;
//!   * строки в лучах разделяются через `Rc<str>`, копируется только изменившийся кусок;
//!   * `decode_batch` раскладывает строки страницы по потокам через rayon.

mod npz;

use std::rc::Rc;
use std::sync::Arc;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

// ─────────────────────────────────────────────────────────── константы pyctcdecode

const FNV_OFFSET: u64 = 1469598103934665603;
const FNV_PRIME: u64 = 1099511628211;
/// kenlm отдаёт log10, декодер работает в натуральных логарифмах
const LOG_BASE_CHANGE: f64 = std::f64::consts::LN_10;
const AVG_TOKEN_LEN: f64 = 6.0;
/// ln(1e-15) — нижний клип логарифмических вероятностей
const MIN_TOKEN_LOGP_CLIP: f32 = -34.538_776_394_910_684;

#[inline(always)]
fn fold_word(h: u64, w: &[u8]) -> u64 {
    let mut h = h;
    for b in w {
        h = (h ^ (*b as u64)).wrapping_mul(FNV_PRIME);
    }
    (h ^ 255).wrapping_mul(FNV_PRIME)
}

// ─────────────────────────────────────────────────────── n-граммная модель

/// Массив u64: либо своя копия, либо окно в отображённом npz.
enum U64Src {
    Owned(Vec<u64>),
    Mapped(npz::Col),
}

impl U64Src {
    #[inline(always)]
    fn get(&self, i: usize) -> u64 {
        match self {
            U64Src::Owned(v) => v[i],
            U64Src::Mapped(c) => c.u64_at(i),
        }
    }
    #[inline(always)]
    fn len(&self) -> usize {
        match self {
            U64Src::Owned(v) => v.len(),
            U64Src::Mapped(c) => c.len(),
        }
    }
}

/// То же для f32.
enum F32Src {
    Owned(Vec<f32>),
    Mapped(npz::Col),
}

impl F32Src {
    #[inline(always)]
    fn get(&self, i: usize) -> f32 {
        match self {
            F32Src::Owned(v) => v[i],
            F32Src::Mapped(c) => c.f32_at(i),
        }
    }
}

struct Order {
    keys: U64Src,
    logp: F32Src,
    backoff: F32Src,
}

impl Order {
    /// Первое место, куда можно вставить key (как np.searchsorted слева).
    #[inline(always)]
    fn lower_bound(&self, key: u64) -> usize {
        let (mut lo, mut hi) = (0usize, self.keys.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.keys.get(mid) < key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    #[inline(always)]
    fn find(&self, key: u64) -> Option<(f32, f32)> {
        let i = self.lower_bound(key);
        if i < self.keys.len() && self.keys.get(i) == key {
            Some((self.logp.get(i), self.backoff.get(i)))
        } else {
            None
        }
    }

    #[inline(always)]
    fn contains(&self, key: u64) -> bool {
        let i = self.lower_bound(key);
        i < self.keys.len() && self.keys.get(i) == key
    }
}

/// Состояние LM: `h[i]` — FNV-хеш суффикса контекста, начинающегося с i-го слова.
/// `n` — длина контекста (максимум 3 слова, как у 4-граммной модели).
#[derive(Clone, Copy, PartialEq, Eq)]
struct LmState {
    h: [u64; 4],
    n: usize,
}

impl LmState {
    fn empty() -> Self {
        LmState { h: [FNV_OFFSET; 4], n: 0 }
    }

    fn begin_sentence() -> Self {
        let mut s = LmState::empty();
        s.h[0] = fold_word(FNV_OFFSET, b"<s>");
        s.h[1] = FNV_OFFSET;
        s.n = 1;
        s
    }

    /// Контекст после дописывания слова (с усечением до трёх слов).
    #[inline]
    fn advance(&self, word: &[u8]) -> LmState {
        let mut ext = [FNV_OFFSET; 5];
        for i in 0..=self.n {
            ext[i] = fold_word(self.h[i], word);
        }
        ext[self.n + 1] = FNV_OFFSET;

        let mut out = LmState::empty();
        if self.n + 1 <= 3 {
            out.n = self.n + 1;
            out.h[..=out.n].copy_from_slice(&ext[..=self.n + 1]);
        } else {
            out.n = 3;
            out.h[..4].copy_from_slice(&ext[1..5]);
        }
        out
    }
}

struct Lm {
    orders: [Order; 4],
    unk: f32,
}

impl Lm {
    /// log10 P(word | ctx) с откатом по Катцу — порт `_pylm.Model._score`.
    fn raw_score(&self, st: &LmState, word: &[u8]) -> f64 {
        for start in 0..=st.n {
            let order = st.n - start + 1; // длина искомой n-граммы
            let key = fold_word(st.h[start], word);
            if let Some((lp, _)) = self.orders[order - 1].find(key) {
                let mut back = 0.0f64;
                for s in 0..start {
                    let bo_order = st.n - s;
                    if let Some((_, bo)) = self.orders[bo_order - 1].find(st.h[s]) {
                        back += bo as f64;
                    }
                }
                return lp as f64 + back;
            }
        }
        self.unk as f64
    }

    #[inline]
    fn has_unigram(&self, word: &[u8]) -> bool {
        self.orders[0].contains(fold_word(FNV_OFFSET, word))
    }
}

// ─────────────────────────────────────────────────────── словарь для partial-token

/// Отсортированный словарь одним блобом: проверка «является ли p префиксом слова из словаря».
struct Vocab {
    blob: Vec<u8>,
    offsets: Vec<u32>,
}

impl Vocab {
    fn build(mut words: Vec<Vec<u8>>) -> Self {
        words.sort_unstable();
        words.dedup();
        let total: usize = words.iter().map(|w| w.len()).sum();
        let mut blob = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(words.len() + 1);
        offsets.push(0u32);
        for w in &words {
            blob.extend_from_slice(w);
            offsets.push(blob.len() as u32);
        }
        Vocab { blob, offsets }
    }

    #[inline]
    fn word(&self, i: usize) -> &[u8] {
        &self.blob[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }

    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Аналог `pygtrie.CharTrie.has_node(p) > 0`: p — префикс какого-нибудь слова словаря.
    fn has_prefix(&self, p: &[u8]) -> bool {
        let (mut lo, mut hi) = (0usize, self.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.word(mid) < p {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo < self.len() && self.word(lo).starts_with(p)
    }
}

// ─────────────────────────────────────────────────────── лучи

#[derive(Clone)]
struct Beam {
    text: Rc<str>,
    next_word: Rc<str>,
    word_part: Rc<str>,
    last_char: u32, // индекс метки; u32::MAX — стартовый луч (None в Python)
    logit: f64,
}

#[derive(Clone)]
struct ScoredBeam {
    text: Rc<str>,       // уже слитый text + next_word
    prev_text: Rc<str>,  // левая часть разбиения — от неё берётся состояние LM
    next_word: Rc<str>,  // слово, дописанное на этом кадре
    word_part: Rc<str>,
    last_char: u32,
    logit: f64,
    combined: f64,
}

#[inline]
fn sum_log_scores(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + (1.0 + (lo - hi).exp()).ln()
}

fn merge_tokens(a: &Rc<str>, b: &Rc<str>) -> Rc<str> {
    if b.is_empty() {
        a.clone()
    } else if a.is_empty() {
        b.clone()
    } else {
        let mut s = String::with_capacity(a.len() + 1 + b.len());
        s.push_str(a);
        s.push(' ');
        s.push_str(b);
        Rc::from(s.as_str())
    }
}

fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ─────────────────────────────────────────────────────── сам декодер

struct Params {
    /// false — как в pyctcdecode (при коллизии остаётся последнее разбиение);
    /// true — оставлять первое (иногда находит гипотезу лучше, но это уже не паритет)
    keep_first_split: bool,
    alpha: f64,
    beta: f64,
    unk_offset: f64,
    beam_width: usize,
    beam_prune_logp: f64,
    token_min_logp: f32,
}

enum VocabSrc {
    Owned(Vocab),
    Mapped(npz::MappedVocab),
}

impl VocabSrc {
    #[inline(always)]
    fn has_prefix(&self, p: &[u8]) -> bool {
        match self {
            VocabSrc::Owned(v) => v.has_prefix(p),
            VocabSrc::Mapped(v) => v.has_prefix(p),
        }
    }
    fn len(&self) -> usize {
        match self {
            VocabSrc::Owned(v) => v.len(),
            VocabSrc::Mapped(v) => v.len(),
        }
    }
}

struct Inner {
    lm: Lm,
    vocab: VocabSrc,
    labels: Vec<String>,
    /// индекс метки-пробела (обычно есть в алфавите); None — если пробела нет
    space_idx: Option<u32>,
    params: Params,
}

impl Inner {
    /// `LanguageModel.score_partial_token`
    fn partial_score(&self, part: &str) -> f64 {
        let is_oov = if self.vocab.has_prefix(part.as_bytes()) { 0.0 } else { 1.0 };
        let mut unk = self.params.unk_offset * is_oov;
        let n_chars = part.chars().count() as f64;
        if n_chars > AVG_TOKEN_LEN {
            unk = unk * n_chars / AVG_TOKEN_LEN;
        }
        unk
    }

    /// `LanguageModel.score`: сырой скор в log10 → масштаб alpha/beta
    fn word_score(&self, st: &LmState, word: &str, is_eos: bool) -> (f64, LmState) {
        let wb = word.as_bytes();
        let mut raw = self.lm.raw_score(st, wb);
        if !self.lm.has_unigram(wb) {
            raw += self.params.unk_offset;
        }
        let end = st.advance(wb);
        if is_eos {
            raw += self.lm.raw_score(&end, b"</s>");
        }
        (self.params.alpha * raw * LOG_BASE_CHANGE + self.params.beta, end)
    }

    fn decode_line(&self, logits: &[f32], t: usize, c: usize, beam_width: usize) -> (String, f64) {
        let beams = self.decode_line_full(logits, t, c, beam_width);
        match beams.into_iter().next() {
            Some((text, logit, _)) => (text, logit),
            None => (String::new(), 0.0),
        }
    }

    /// Полный список финальных гипотез: (текст, акустический score, score с LM), по убыванию.
    fn decode_line_full(&self, logits: &[f32], t: usize, c: usize, beam_width: usize) -> Vec<(String, f64, f64)> {
        // log-softmax по каждому кадру + клип, как в decode_beams
        let mut lp = vec![0f32; t * c];
        for i in 0..t {
            let row = &logits[i * c..(i + 1) * c];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f64;
            for v in row {
                sum += ((*v - m) as f64).exp();
            }
            let logsum = sum.ln() as f32;
            let out = &mut lp[i * c..(i + 1) * c];
            for (o, v) in out.iter_mut().zip(row) {
                let x = *v - m - logsum;
                *o = if x < MIN_TOKEN_LOGP_CLIP { MIN_TOKEN_LOGP_CLIP } else if x > 0.0 { 0.0 } else { x };
            }
        }

        let empty: Rc<str> = Rc::from("");
        let mut beams: Vec<Beam> = vec![Beam {
            text: empty.clone(),
            next_word: empty.clone(),
            word_part: empty.clone(),
            last_char: u32::MAX,
            logit: 0.0,
        }];

        // кеши на строку: текст -> (сырой lm-скор, состояние); частичное слово -> штраф
        let mut cached_lm: FxHashMap<Rc<str>, (f64, LmState)> = FxHashMap::default();
        cached_lm.insert(empty.clone(), (0.0, LmState::begin_sentence()));
        let mut cached_partial: FxHashMap<Rc<str>, f64> = FxHashMap::default();

        let mut cand: Vec<u32> = Vec::with_capacity(16);
        let mut expanded: Vec<Beam> = Vec::with_capacity(1024);
        let mut merge_map: FxHashMap<(Rc<str>, Rc<str>, u32), usize> = FxHashMap::default();
        let mut merged: Vec<ScoredBeam> = Vec::with_capacity(1024);

        for frame in 0..t {
            let col = &lp[frame * c..(frame + 1) * c];

            // кандидаты: всё выше порога плюс обязательный argmax
            cand.clear();
            let mut max_idx = 0usize;
            let mut max_val = f32::NEG_INFINITY;
            for (i, v) in col.iter().enumerate() {
                if *v > max_val {
                    max_val = *v;
                    max_idx = i;
                }
                if *v >= self.params.token_min_logp {
                    cand.push(i as u32);
                }
            }
            if !cand.contains(&(max_idx as u32)) {
                cand.push(max_idx as u32);
                cand.sort_unstable();
            }

            expanded.clear();
            for &idx in cand.iter() {
                let p = col[idx as usize] as f64;
                let ch = self.labels[idx as usize].as_str();
                let is_blank = ch.is_empty();
                let is_space = Some(idx) == self.space_idx;

                for b in beams.iter() {
                    if is_blank || b.last_char == idx {
                        // blank или повтор символа: префикс не растёт
                        expanded.push(Beam {
                            text: b.text.clone(),
                            next_word: b.next_word.clone(),
                            word_part: b.word_part.clone(),
                            last_char: idx,
                            logit: b.logit + p,
                        });
                    } else if is_space {
                        // граница слова: частичное слово уезжает в next_word
                        expanded.push(Beam {
                            text: b.text.clone(),
                            next_word: b.word_part.clone(),
                            word_part: empty.clone(),
                            last_char: idx,
                            logit: b.logit + p,
                        });
                    } else {
                        let mut wp = String::with_capacity(b.word_part.len() + ch.len());
                        wp.push_str(&b.word_part);
                        wp.push_str(ch);
                        expanded.push(Beam {
                            text: b.text.clone(),
                            next_word: b.next_word.clone(),
                            word_part: Rc::from(wp.as_str()),
                            last_char: idx,
                            logit: b.logit + p,
                        });
                    }
                }
            }

            // Слияние одинаковых префиксов. Важная деталь pyctcdecode: при коллизии ключа
            // сохраняются поля ПОСЛЕДНЕГО луча (разбиение text/next_word), а складывается
            // только score. Разбиение влияет на финальный скоринг конца строки, поэтому
            // повторяем поведение буквально.
            merge_map.clear();
            merged.clear();
            for b in expanded.iter() {
                let new_text = merge_tokens(&b.text, &b.next_word);
                let key = (new_text.clone(), b.word_part.clone(), b.last_char);
                match merge_map.get(&key) {
                    Some(&pos) => {
                        merged[pos].logit = sum_log_scores(merged[pos].logit, b.logit);
                        if !self.params.keep_first_split {
                            merged[pos].prev_text = b.text.clone();
                            merged[pos].next_word = b.next_word.clone();
                        }
                    }
                    None => {
                        merge_map.insert(key, merged.len());
                        merged.push(ScoredBeam {
                            text: new_text,
                            prev_text: b.text.clone(),
                            next_word: b.next_word.clone(),
                            word_part: b.word_part.clone(),
                            last_char: b.last_char,
                            logit: b.logit,
                            combined: 0.0,
                        });
                    }
                }
            }

            // скоринг языковой моделью — уже после того, как все коллизии разрешены
            for i in 0..merged.len() {
                let (text, prev_text, next_word, word_part) = (
                    merged[i].text.clone(),
                    merged[i].prev_text.clone(),
                    merged[i].next_word.clone(),
                    merged[i].word_part.clone(),
                );
                if !cached_lm.contains_key(&text) {
                    let (prev_raw, st) = cached_lm[&prev_text];
                    let (add, end) = self.word_score(&st, &next_word, false);
                    cached_lm.insert(text.clone(), (prev_raw + add, end));
                }
                let mut lm_score = cached_lm[&text].0;
                if !word_part.is_empty() {
                    let ps = match cached_partial.get(&word_part) {
                        Some(v) => *v,
                        None => {
                            let v = self.partial_score(&word_part);
                            cached_partial.insert(word_part.clone(), v);
                            v
                        }
                    };
                    lm_score += ps;
                }
                merged[i].combined = merged[i].logit + lm_score;
            }

            // отсечение выбросов и обрезка по ширине луча
            let max_comb = merged.iter().fold(f64::NEG_INFINITY, |m, b| m.max(b.combined));
            let cutoff = max_comb + self.params.beam_prune_logp;
            let mut kept: Vec<&ScoredBeam> = merged.iter().filter(|b| b.combined >= cutoff).collect();
            kept.sort_by(|a, b| b.combined.partial_cmp(&a.combined).unwrap());
            kept.truncate(beam_width);

            beams.clear();
            for sb in kept {
                beams.push(Beam {
                    text: sb.text.clone(),
                    next_word: empty.clone(),
                    word_part: sb.word_part.clone(),
                    last_char: sb.last_char,
                    logit: sb.logit,
                });
            }
        }

        // финальный скоринг с концом предложения: word_part становится последним словом
        merge_map.clear();
        merged.clear();
        for b in beams.iter() {
            let new_text = merge_tokens(&b.text, &b.word_part);
            let key = (new_text.clone(), empty.clone(), u32::MAX);
            match merge_map.get(&key) {
                Some(&pos) => {
                    merged[pos].logit = sum_log_scores(merged[pos].logit, b.logit);
                    if !self.params.keep_first_split {
                        merged[pos].prev_text = b.text.clone();
                        merged[pos].next_word = b.word_part.clone();
                    }
                }
                None => {
                    merge_map.insert(key, merged.len());
                    merged.push(ScoredBeam {
                        text: new_text,
                        prev_text: b.text.clone(),
                        next_word: b.word_part.clone(),
                        word_part: empty.clone(),
                        last_char: u32::MAX,
                        logit: b.logit,
                        combined: 0.0,
                    });
                }
            }
        }
        for i in 0..merged.len() {
            let (prev_raw, st) = cached_lm[&merged[i].prev_text];
            let nw = merged[i].next_word.clone();
            let (add, _) = self.word_score(&st, &nw, true);
            merged[i].combined = merged[i].logit + prev_raw + add;
        }

        if merged.is_empty() {
            return Vec::new();
        }
        let max_comb = merged.iter().fold(f64::NEG_INFINITY, |m, b| m.max(b.combined));
        let cutoff = max_comb + self.params.beam_prune_logp;
        let mut out: Vec<(String, f64, f64)> = merged
            .iter()
            .filter(|b| b.combined >= cutoff)
            .map(|b| (normalize_whitespace(&b.text), b.logit, b.combined))
            .collect();
        out.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        out.truncate(beam_width);
        out
    }
}

#[pyclass]
struct Decoder {
    inner: Arc<Inner>,
}

#[pymethods]
impl Decoder {
    /// Собрать декодер. Массивы n-грамм приходят из `compact_lm.npz`, словарь — путём
    /// к `unigrams.txt` (читается и сортируется здесь, чтобы не плодить python-строки).
    #[new]
    #[pyo3(signature = (labels, keys, logps, backoffs, unigrams_path, alpha=0.5, beta=1.0,
                        unk_offset=-10.0, beam_width=100, beam_prune_logp=-10.0, token_min_logp=-5.0,
                        keep_first_split=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        labels: Vec<String>,
        keys: &Bound<'_, PyList>,
        logps: &Bound<'_, PyList>,
        backoffs: &Bound<'_, PyList>,
        unigrams_path: &str,
        alpha: f64,
        beta: f64,
        unk_offset: f64,
        beam_width: usize,
        beam_prune_logp: f64,
        token_min_logp: f32,
        keep_first_split: bool,
    ) -> PyResult<Self> {
        if keys.len() != 4 || logps.len() != 4 || backoffs.len() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "нужны массивы для порядков 1..4",
            ));
        }
        let mut built: Vec<Order> = Vec::with_capacity(4);
        for i in 0..4 {
            let k: PyReadonlyArray1<u64> = keys.get_item(i)?.extract()?;
            let l: PyReadonlyArray1<f32> = logps.get_item(i)?.extract()?;
            let b: PyReadonlyArray1<f32> = backoffs.get_item(i)?.extract()?;
            built.push(Order {
                keys: U64Src::Owned(k.as_array().to_vec()),
                logp: F32Src::Owned(l.as_array().to_vec()),
                backoff: F32Src::Owned(b.as_array().to_vec()),
            });
        }
        let orders: [Order; 4] = built.try_into().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("не удалось собрать 4 порядка n-грамм")
        })?;
        let unk = orders[0]
            .find(fold_word(FNV_OFFSET, b"<unk>"))
            .map(|(lp, _)| lp)
            .unwrap_or(-7.0);

        let text = std::fs::read(unigrams_path).map_err(|e| {
            pyo3::exceptions::PyOSError::new_err(format!("не читается {unigrams_path}: {e}"))
        })?;
        let words: Vec<Vec<u8>> = text
            .split(|b| *b == b'\n')
            .filter(|l| !l.is_empty() && l.iter().any(|b| !b.is_ascii_whitespace()))
            .map(|l| {
                let end = l.iter().rposition(|b| !b.is_ascii_whitespace()).map_or(0, |i| i + 1);
                l[..end].to_vec()
            })
            .collect();
        let vocab = VocabSrc::Owned(Vocab::build(words));

        let space_idx = labels.iter().position(|l| l == " ").map(|i| i as u32);

        Ok(Decoder {
            inner: Arc::new(Inner {
                lm: Lm { orders, unk },
                vocab,
                labels,
                space_idx,
                params: Params {
                    keep_first_split,
                    alpha,
                    beta,
                    unk_offset,
                    beam_width,
                    beam_prune_logp,
                    token_min_logp,
                },
            }),
        })
    }

    /// Собрать декодер прямо из файлов: `compact_lm.npz` отображается в память, словарь тоже.
    /// Ничего не копируется, поэтому загрузка почти мгновенная, а страницы файлов общие
    /// между процессами и вытесняемые.
    #[staticmethod]
    #[pyo3(signature = (labels, npz_path, unigrams_path, alpha=0.5, beta=1.0, unk_offset=-10.0,
                        beam_width=100, beam_prune_logp=-10.0, token_min_logp=-5.0,
                        keep_first_split=false))]
    #[allow(clippy::too_many_arguments)]
    fn from_npz(
        labels: Vec<String>,
        npz_path: &str,
        unigrams_path: &str,
        alpha: f64,
        beta: f64,
        unk_offset: f64,
        beam_width: usize,
        beam_prune_logp: f64,
        token_min_logp: f32,
        keep_first_split: bool,
    ) -> PyResult<Self> {
        let err = |e: String| pyo3::exceptions::PyValueError::new_err(e);
        let arch = npz::Npz::open(npz_path).map_err(err)?;
        let mut built: Vec<Order> = Vec::with_capacity(4);
        for o in 1..=4 {
            built.push(Order {
                keys: U64Src::Mapped(arch.col(&format!("k{o}"), "<u8").map_err(err)?),
                logp: F32Src::Mapped(arch.col(&format!("lp{o}"), "<f4").map_err(err)?),
                backoff: F32Src::Mapped(arch.col(&format!("bo{o}"), "<f4").map_err(err)?),
            });
        }
        let orders: [Order; 4] = built
            .try_into()
            .map_err(|_| err("не удалось собрать 4 порядка n-грамм".to_string()))?;
        let unk = orders[0]
            .find(fold_word(FNV_OFFSET, b"<unk>"))
            .map(|(lp, _)| lp)
            .unwrap_or(-7.0);
        let vocab = VocabSrc::Mapped(npz::MappedVocab::open(unigrams_path).map_err(err)?);
        let space_idx = labels.iter().position(|l| l == " ").map(|i| i as u32);

        Ok(Decoder {
            inner: Arc::new(Inner {
                lm: Lm { orders, unk },
                vocab,
                labels,
                space_idx,
                params: Params {
                    keep_first_split,
                    alpha,
                    beta,
                    unk_offset,
                    beam_width,
                    beam_prune_logp,
                    token_min_logp,
                },
            }),
        })
    }

    /// Одна строка: логиты [T, C] → (текст, акустический score выбранной гипотезы).
    #[pyo3(signature = (logits, beam_width=None))]
    fn decode(
        &self,
        py: Python<'_>,
        logits: &Bound<'_, PyAny>,
        beam_width: Option<usize>,
    ) -> PyResult<(String, f64)> {
        let bw = beam_width.unwrap_or(self.inner.params.beam_width);
        let (data, t, c) = {
            let arr: PyReadonlyArray2<f32> = logits.extract()?;
            let shape = arr.shape();
            let a = arr.as_array();
            (a.iter().cloned().collect::<Vec<f32>>(), shape[0], shape[1])
        };
        let inner = self.inner.clone();
        Ok(py.allow_threads(move || inner.decode_line(&data, t, c, bw)))
    }

    /// Вся страница разом: список массивов [T, C] → список (текст, score). Строки считаются
    /// параллельно, поэтому один вызов на страницу выгоднее, чем вызов на строку.
    #[pyo3(signature = (batch, beam_width=None))]
    fn decode_batch(
        &self,
        py: Python<'_>,
        batch: &Bound<'_, PyList>,
        beam_width: Option<usize>,
    ) -> PyResult<Vec<(String, f64)>> {
        let bw = beam_width.unwrap_or(self.inner.params.beam_width);
        let mut lines: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(batch.len());
        for item in batch.iter() {
            let arr: PyReadonlyArray2<f32> = item.extract()?;
            let shape = arr.shape();
            let a = arr.as_array();
            lines.push((a.iter().cloned().collect::<Vec<f32>>(), shape[0], shape[1]));
        }
        let inner = self.inner.clone();
        Ok(py.allow_threads(move || {
            lines
                .par_iter()
                .map(|(data, t, c)| inner.decode_line(data, *t, *c, bw))
                .collect()
        }))
    }

    /// Топ-k гипотез строки: (текст, акустический score, score с языковой моделью).
    #[pyo3(signature = (logits, k=5, beam_width=None))]
    fn decode_topk(
        &self,
        py: Python<'_>,
        logits: &Bound<'_, PyAny>,
        k: usize,
        beam_width: Option<usize>,
    ) -> PyResult<Vec<(String, f64, f64)>> {
        let bw = beam_width.unwrap_or(self.inner.params.beam_width);
        let (data, t, c) = {
            let arr: PyReadonlyArray2<f32> = logits.extract()?;
            let shape = arr.shape();
            let a = arr.as_array();
            (a.iter().cloned().collect::<Vec<f32>>(), shape[0], shape[1])
        };
        let inner = self.inner.clone();
        let mut v = py.allow_threads(move || inner.decode_line_full(&data, t, c, bw));
        v.truncate(k);
        Ok(v)
    }

    /// Сколько слов в словаре префиксов (для отладки).
    fn vocab_size(&self) -> usize {
        self.inner.vocab.len()
    }

    // ── отладочные хуки: по ним сверяется каждый слой с Python-реализацией ──

    fn dbg_has_prefix(&self, p: &str) -> bool {
        self.inner.vocab.has_prefix(p.as_bytes())
    }

    fn dbg_has_unigram(&self, w: &str) -> bool {
        self.inner.lm.has_unigram(w.as_bytes())
    }

    /// Сырой log10-скор слова в контексте (аналог `_pylm.Model.BaseScore`).
    fn dbg_raw_score(&self, ctx: Vec<String>, word: &str) -> f64 {
        let mut st = LmState::begin_sentence();
        if !ctx.is_empty() {
            st = LmState::empty();
            for w in &ctx {
                st = st.advance(w.as_bytes());
            }
        }
        self.inner.lm.raw_score(&st, word.as_bytes())
    }

    /// Полный скор слова со шкалой alpha/beta (аналог `LanguageModel.score`).
    #[pyo3(signature = (ctx, word, is_eos=false, from_start=true))]
    fn dbg_word_score(&self, ctx: Vec<String>, word: &str, is_eos: bool, from_start: bool) -> f64 {
        let mut st = if from_start { LmState::begin_sentence() } else { LmState::empty() };
        for w in &ctx {
            st = st.advance(w.as_bytes());
        }
        self.inner.word_score(&st, word, is_eos).0
    }

    /// Штраф за незаконченное слово (аналог `LanguageModel.score_partial_token`).
    fn dbg_partial_score(&self, part: &str) -> f64 {
        self.inner.partial_score(part)
    }
}

#[pymodule]
fn occular_decode(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Decoder>()?;
    Ok(())
}
