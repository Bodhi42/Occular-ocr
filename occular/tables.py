"""Распознавание таблиц: детектор таблиц на странице + структура (сетка строк/столбцов + объединённые ячейки).

Два шага, всё на ONNX (CPU или GPU):
  • Детектор: страница 640×640 → рамки таблиц со score (D-FINE, по рамке на таблицу — соседние таблицы
    не слипаются).
  • Структура (split+merge): кроп таблицы 768×768 → сетка полос (строки/столбцы) и объединённые ячейки
    (colspan/rowspan). Высокую таблицу (высота > STRIP_RATIO·ширины) сжатие в квадрат сплющивает и строки
    теряются, поэтому её строки ищутся по перекрывающимся горизонтальным полосам высотой ≤ STRIP_RATIO·ширины,
    а границы полос сшиваются по линиям; столбцы и объединения — по всему кропу. Сеть принимает решения по своим выходам (число полос, границы клеток), поэтому
    разрезана на два графа, а динамическая часть между ними — numpy:
      main  (статический): логиты счётчиков и сепараторов + карта merge-признаков;
      numpy: пиковый декод границ, усреднение карты по клеткам, выборка признаков на границах клеток;
      merge (оси r, c динамические): Grid-CNN по сетке клеток + MLP рёбер «слить / не слить».

Быстрый старт:
    from occular.tables import TableRecognizer
    tr = TableRecognizer()                 # gpu=True — на CUDA
    import cv2; page = cv2.imread("scan.png")
    for t in tr(page):                     # список таблиц страницы
        print(t["bbox"], "строк:", len(t["rows"]), "столбцов:", len(t["cols"]), "ячеек:", len(t["cells"]))

Только детекция:  tr.detect(page)
Только структура:  tr.structure(table_crop)
"""
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple

INPUT = 768                              # вход модели структуры
DET_INPUT = 640                          # вход детектора
DET_THRESHOLD = 0.75                     # прод-порог детектора (F1 на валидации)
STRIP_RATIO = 0.5                        # полоса для высоких таблиц: высота ≤ 0.5·ширины (как при обучении)
_STRIP_OVERLAP = 0.15                    # перекрытие соседних полос (доля высоты полосы)
_LINE_MERGE_PX = 3.0                     # линии соседних полос ближе этого — одна линия
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _img_tensor(bgr: np.ndarray) -> np.ndarray:
    """BGR (cv2) → нормализованный тензор [1,3,768,768] float32 (ImageNet-норма, RGB)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    a = cv2.resize(rgb, (INPUT, INPUT), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    a = (a - _MEAN) / _STD
    return a.transpose(2, 0, 1)[None].copy()


def _det_tensor(bgr: np.ndarray) -> np.ndarray:
    """BGR (cv2) → [1,3,640,640] float32 RGB 0..1 (растяжение PIL-bilinear, как при обучении детектора)."""
    from PIL import Image
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    a = np.asarray(Image.fromarray(rgb).resize((DET_INPUT, DET_INPUT), Image.BILINEAR), np.float32) / 255.0
    return a.transpose(2, 0, 1)[None].copy()


def _bands_from(cb: np.ndarray, size: float) -> List[Tuple[float, float]]:
    """Нормированные границы (из _sep_to_bounds) → полосы (a,b) в пикселях кропа, от 0 до size.
    Число полос = найденные разделители + 1. Счётчик полос модели не используется: он расходится с
    разделителями, и тогда либо появляются пустые полосы, либо отрезаются нижние/правые строки
    (на валидации сетка F1 0.901 → 0.921, объединённые ячейки 0.784 → 0.831)."""
    L = [0.0] + sorted(float(v) * size for v in cb if v < 1 - 1e-6) + [float(size)]
    out = [(L[k], L[k + 1]) for k in range(len(L) - 1) if L[k + 1] - L[k] >= size / INPUT]
    return out or [(0.0, float(size))]


def _cells_from_edges(hor: np.ndarray, ver: np.ndarray, R: int, C: int) -> List[Dict]:
    """Union-find по рёбрам сетки → логические ячейки (r,c,rowspan,colspan). hor[R,C-1], ver[R-1,C]."""
    par = list(range(R * C))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a

    def uni(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            par[rb] = ra

    for i in range(R):
        for j in range(C - 1):
            if hor[i, j] > 0.5:
                uni(i * C + j, i * C + j + 1)
    for i in range(R - 1):
        for j in range(C):
            if ver[i, j] > 0.5:
                uni(i * C + j, (i + 1) * C + j)
    groups: Dict[int, list] = {}
    for i in range(R):
        for j in range(C):
            groups.setdefault(find(i * C + j), []).append((i, j))
    out = []
    for g in groups.values():
        rs = [a for a, _ in g]; cs = [b for _, b in g]
        r0, r1 = min(rs), max(rs); c0, c1 = min(cs), max(cs)
        out.append(dict(r=r0, c=c0, rowspan=r1 - r0 + 1, colspan=c1 - c0 + 1))
    return sorted(out, key=lambda q: (q["r"], q["c"]))


def _sep_to_bounds(logit: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Логиты сепараторов [Lp] → нормированные внутренние границы по возрастанию + завершающая 1.0.
    Декод как при обучении и валидации модели v4: порог 0.5 → связные участки → центр масс каждого = граница.
    (Пиковый декод, подобранный под модель v3, на v4 хуже: struct-F1 на валидации 0.881 против 0.901.)"""
    p = (1.0 / (1.0 + np.exp(-logit.astype(np.float32)))).astype(np.float32)
    Lp = len(p)
    cent = []
    i = 0
    while i < Lp:
        if p[i] > thr:
            j = i
            while j + 1 < Lp and p[j + 1] > thr:
                j += 1
            w = p[i:j + 1]
            c = (float((w * np.arange(i, j + 1)).sum() / max(float(w.sum()), 1e-6)) + 0.5) / Lp
            if 1e-4 < c < 1 - 1e-4:
                cent.append(c)
            i = j + 1
        else:
            i += 1
    return np.array(cent + [1.0], np.float32)


def _pool_cells(P: np.ndarray, y: np.ndarray, x: np.ndarray, r: int, c: int) -> np.ndarray:
    """Признак клетки = среднее карты P [d,Hf,Wf] по её боксу (границы y [r+1], x [c+1] в [0,1]) → [d,r,c]."""
    d, Hf, Wf = P.shape
    yi = np.clip(y * np.float32(Hf), 0, Hf); xi = np.clip(x * np.float32(Wf), 0, Wf)
    out = np.zeros((d, r, c), np.float32)
    for i in range(r):
        a = int(np.floor(yi[i])); b = min(max(a + 1, int(np.ceil(yi[i + 1]))), Hf)
        if a >= b:
            a = max(0, b - 1)
        for j in range(c):
            u = int(np.floor(xi[j])); v = min(max(u + 1, int(np.ceil(xi[j + 1]))), Wf)
            if u >= v:
                u = max(0, v - 1)
            out[:, i, j] = P[:, a:b, u:v].mean(axis=(1, 2))
    return out


def _sample(P: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Билинейная выборка карты P [d,Hf,Wf] в точках (gx, gy) из [0,1] → [..., d].
    Та же семантика, что torch grid_sample(align_corners=False, padding_mode='zeros')."""
    d, Hf, Wf = P.shape
    ix = gx.astype(np.float32) * Wf - 0.5; iy = gy.astype(np.float32) * Hf - 0.5
    x0 = np.floor(ix).astype(np.int64); y0 = np.floor(iy).astype(np.int64)
    wx1 = ix - x0; wy1 = iy - y0; wx0 = 1 - wx1; wy0 = 1 - wy1
    out = np.zeros(gx.shape + (d,), np.float32)
    for yy, wy in ((y0, wy0), (y0 + 1, wy1)):
        for xx, wx in ((x0, wx0), (x0 + 1, wx1)):
            ok = (xx >= 0) & (xx < Wf) & (yy >= 0) & (yy < Hf)
            v = P[:, np.clip(yy, 0, Hf - 1), np.clip(xx, 0, Wf - 1)]          # [d, ...]
            out += np.moveaxis(v, 0, -1) * (wx * wy * ok)[..., None]
    return out


def _merge_inputs(P: np.ndarray, rb: np.ndarray, cb: np.ndarray, r: int, c: int) -> Dict[str, np.ndarray]:
    """Входы merge-графа для сетки r×c: признаки клеток, признаки на границах и геометрия пар."""
    y = np.concatenate([np.zeros(1, np.float32), rb[:r]]).astype(np.float32)
    x = np.concatenate([np.zeros(1, np.float32), cb[:c]]).astype(np.float32)
    f = _pool_cells(P, y, x, r, c)[None]
    hw = np.broadcast_to((y[1:] - y[:-1])[:, None], (r, c))
    ww = np.broadcast_to((x[1:] - x[:-1])[None, :], (r, c))
    cy = np.broadcast_to(((y[:-1] + y[1:]) * 0.5)[:, None], (r, c))
    cx = np.broadcast_to(((x[:-1] + x[1:]) * 0.5)[None, :], (r, c))
    fb_h = _sample(P, np.broadcast_to(x[1:-1][None, :], (r, c - 1)), cy[:, :-1])
    lij_h = np.stack([ww[:, :-1], ww[:, 1:], hw[:, :-1], cx[:, 1:] - cx[:, :-1], cy[:, 1:] - cy[:, :-1]], -1)
    fb_v = _sample(P, cx[:-1], np.broadcast_to(y[1:-1][:, None], (r - 1, c)))
    lij_v = np.stack([hw[:-1], hw[1:], ww[:-1], cy[1:] - cy[:-1], cx[1:] - cx[:-1]], -1)
    return {"f": f, "fb_h": fb_h, "lij_h": lij_h.astype(np.float32),
            "fb_v": fb_v, "lij_v": lij_v.astype(np.float32)}


class TableRecognizer:
    """Детекция таблиц + структура (сетка + объединённые ячейки). Всё — ONNX, CPU или GPU (gpu=True)."""

    def __init__(self, num_threads: int = 4, merge: bool = True, gpu: bool = False):
        from .model_files import ensure_weight
        from ._runtime import make_session

        self._det = make_session(ensure_weight("table_detect_dfine_v4_fp32.onnx"), num_threads, gpu)
        self._main = make_session(ensure_weight("table_struct_v4_main_fp32.onnx"), num_threads, gpu)
        self._merge = (make_session(ensure_weight("table_struct_v4_merge_fp32.onnx"), num_threads, gpu)
                       if merge else None)
        # Таблица в одну строку или один столбец: у merge-графа пустые входы/выходы (рёбер по одной оси
        # нет). CPU их считает, CUDA падает (BiasGelu: cudaErrorInvalidConfiguration) — на GPU такие
        # таблицы идут через CPU-копию графа (2 МБ, создаётся при первой такой таблице).
        self._merge_cpu = None if (merge and gpu) else self._merge
        self._threads = num_threads

    # ---- детекция ----
    def detect(self, image: np.ndarray, thr: float = DET_THRESHOLD,
               min_frac: float = 0.0) -> List[Tuple[int, int, int, int, float]]:
        """Страница (BGR) → список рамок таблиц (x0,y0,x1,y1,conf) в ПИКСЕЛЯХ исходного изображения,
        сверху вниз, затем слева направо. min_frac — отбросить рамки меньше этой доли площади страницы."""
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return []
        _, boxes, scores = self._det.run(None, {"images": _det_tensor(image),
                                                "orig_target_sizes": np.array([[w, h]], np.int64)})
        out = []
        for (x0, y0, x1, y1), sc in zip(boxes[0], scores[0]):
            if sc < thr:
                continue
            x0, x1 = int(round(max(0.0, min(x0, w)))), int(round(max(0.0, min(x1, w))))
            y0, y1 = int(round(max(0.0, min(y0, h)))), int(round(max(0.0, min(y1, h))))
            if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) < min_frac * w * h:
                continue
            out.append((x0, y0, x1, y1, float(sc)))
        return sorted(out, key=lambda b: (b[1], b[0]))

    # ---- структура ----
    def _split(self, crop: np.ndarray):
        """Один прогон main-графа: логиты счётчиков, границы (норм. [0,1]) строк/столбцов и карта merge-признаков."""
        rcnt, rsep, ccnt, csep, pmap = self._main.run(None, {"x": _img_tensor(crop)})
        return rcnt[0], _sep_to_bounds(rsep[0]), ccnt[0], _sep_to_bounds(csep[0]), pmap[0]

    def _strip_rows(self, crop: np.ndarray) -> List[Tuple[float, float]]:
        """Строки высокого кропа по перекрывающимся полосам высотой STRIP_RATIO·ширины. Сшивка по ЛИНИЯМ:
        внутренние границы строк каждой полосы, попавшие в её ядро (без половины перекрытия с соседями),
        сливаются, если ближе _LINE_MERGE_PX; крайние линии — края кропа. Полосы в пикселях кропа."""
        import math
        Hc, Wc = crop.shape[:2]
        sh = STRIP_RATIO * Wc; ov = _STRIP_OVERLAP
        k = max(2, math.ceil((Hc - sh * ov) / (sh * (1 - ov))))
        step = (Hc - sh) / (k - 1)
        lines = []
        for j in range(k):
            y0 = int(j * step); y1 = min(Hc, int(math.ceil(j * step + sh)))
            c0 = y0 + (sh * ov / 2 if j else 0); c1 = y1 - (sh * ov / 2 if j < k - 1 else 0)
            _, rcb, _, _, _ = self._split(crop[y0:y1])
            b = [(a, e) for a, e in _bands_from(rcb, float(y1 - y0)) if e - a >= 1.0]
            lines += [y0 + (b1 + a2) / 2 for (_, b1), (a2, _) in zip(b, b[1:]) if c0 <= y0 + (b1 + a2) / 2 < c1]
        merged: List[float] = []
        for y in sorted(lines):
            if merged and y - merged[-1] < _LINE_MERGE_PX:
                merged[-1] = (merged[-1] + y) / 2
            else:
                merged.append(y)
        L = [0.0] + merged + [float(Hc)]
        return [(L[i], L[i + 1]) for i in range(len(L) - 1)]

    def structure(self, table_crop: np.ndarray) -> Dict:
        """Кроп таблицы (BGR) → {'rows':[(y0,y1)...], 'cols':[(x0,x1)...], 'cells':[{r,c,rowspan,colspan}...]}.
        Координаты полос — в пикселях переданного кропа. cells пуст при merge=False."""
        h, w = table_crop.shape[:2]
        _, rcb, _, ccb, pmap = self._split(table_crop)
        cols = _bands_from(ccb, w)
        rows = self._strip_rows(table_crop) if h > STRIP_RATIO * w else _bands_from(rcb, h)
        cells = []
        if self._merge is not None and rows and cols:
            # вход merge — ИТОГОВАЯ сетка (после сшивки полос и чистки), граф merge динамический по r, c
            R, C = len(rows), len(cols)
            rb = np.array([b for _, b in rows], np.float32) / h
            cb = np.array([b for _, b in cols], np.float32) / w
            sess = self._merge
            if R == 1 or C == 1:
                if self._merge_cpu is None:
                    from .model_files import ensure_weight
                    from ._runtime import make_session
                    self._merge_cpu = make_session(ensure_weight("table_struct_v4_merge_fp32.onnx"),
                                                   self._threads, False)
                sess = self._merge_cpu
            hor, ver = sess.run(None, _merge_inputs(pmap, rb, cb, R, C))
            sig = lambda a: 1.0 / (1.0 + np.exp(-a))
            horm = sig(hor) if C > 1 else np.zeros((R, 0)); verm = sig(ver) if R > 1 else np.zeros((0, C))
            cells = _cells_from_edges(horm, verm, R, C)
        return {"rows": rows, "cols": cols, "cells": cells}

    def __call__(self, image: np.ndarray, thr: float = DET_THRESHOLD) -> List[Dict]:
        """Страница (BGR) → список таблиц: {'bbox':(x0,y0,x1,y1,conf), 'rows','cols','cells'} (координаты полос
        приведены к пикселям всей страницы)."""
        out = []
        for (x0, y0, x1, y1, conf) in self.detect(image, thr=thr):
            crop = image[max(0, y0):y1, max(0, x0):x1]
            if crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            st = self.structure(crop)
            rows = [(y0 + a, y0 + b) for a, b in st["rows"]]
            cols = [(x0 + a, x0 + b) for a, b in st["cols"]]
            out.append({"bbox": (x0, y0, x1, y1, conf), "rows": rows, "cols": cols, "cells": st["cells"]})
        return out


# ---- таблицы внутри OCR-конвейера (Settings(tables=True)) ----

def _band_index(bands: List[Tuple[float, float]], v: float) -> int:
    """Номер полосы, в которую попадает координата v (вне всех — ближайшая)."""
    for i, (a, b) in enumerate(bands):
        if a <= v < b:
            return i
    return min(range(len(bands)), key=lambda i: min(abs(v - bands[i][0]), abs(v - bands[i][1])))


def cell_boundaries(table: Dict, y: float) -> List[float]:
    """x-границы между РАЗНЫМИ ячейками в строке таблицы на высоте y (пиксели страницы): границы сетки
    внутри объединённой ячейки не считаются. table — элемент TableRecognizer(image)."""
    rows, cols = table["rows"], table["cols"]
    if not rows or len(cols) < 2:
        return []
    i = _band_index(rows, y)
    owner = {}
    for k, c in enumerate(table["cells"] or [dict(r=a, c=b, rowspan=1, colspan=1)
                                             for a in range(len(rows)) for b in range(len(cols))]):
        for a in range(c["r"], c["r"] + c["rowspan"]):
            for b in range(c["c"], c["c"] + c["colspan"]):
                owner[(a, b)] = k
    return [float(cols[j][0]) for j in range(1, len(cols)) if owner.get((i, j)) != owner.get((i, j - 1))]


def _gap_cut(image: np.ndarray, x0: float, x1: float, y0: float, y1: float,
             b: float) -> Optional[Tuple[float, float]]:
    """Пробел у границы ячейки b, по которому можно резать строку: (конец левого куска, начало правого).
    Пробел — участок без текста не уже межсловного (0.3 высоты строки, не меньше 3 px) в пределах ±0.6
    высоты от b; вертикальная линейка таблицы (столбец, залитый почти на всю высоту строки) внутри него
    допустима. Сам пробел с линейкой из кусков выбрасывается: каждый кусок заходит в него не дальше
    0.2 высоты строки и не до линейки — иначе линейка на краю кропа читается как буква («02» → «T»).
    Нет пробела — None: текст переходит через линию (вылез из ячейки), резать нельзя."""
    H, W = image.shape[:2]
    h = y1 - y0
    r = 0.6 * h
    a, c = int(max(0, x0, b - r)), int(min(W, x1, b + r)) + 1
    ya, yb = int(max(0, y0)), int(min(H, y1)) + 1
    if c - a < 3 or yb - ya < 3:
        return None
    crop = image[ya:yb, a:c]
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thr <= 0:                                             # чисто чёрно-белое: Оцу вырождается в 0
        thr = 128
    ink = (g < min(thr, 200)).mean(axis=0)                  # доля тёмных пикселей по столбцам
    rule = ink > 0.85
    empty = (ink < 0.02) | rule                              # пусто или линейка
    need = max(3, int(round(0.3 * h)))
    pad = max(1, int(round(0.2 * h)))
    best, i = None, 0
    while i < len(empty):
        if empty[i]:
            j = i
            while j + 1 < len(empty) and empty[j + 1]:
                j += 1
            if j - i + 1 >= need and (best is None or abs(a + (i + j + 1) / 2 - b) < abs(best[2] - b)):
                rules = [k for k in range(i, j + 1) if rule[k]]
                left_end = min(i + pad, (rules[0] if rules else j + 1))       # не дальше линейки
                right_start = max(j + 1 - pad, (rules[-1] + 1 if rules else i))
                best = (a + left_end, a + right_start, a + (i + j + 1) / 2)
            i = j + 1
        else:
            i += 1
    return (best[0], best[1]) if best else None


def _text_cuts(text: str, fracs: List[float], logits: Optional[np.ndarray], vocab) -> List[int]:
    """Позиции разреза в тексте строки для разрезов на долях ширины fracs (0..1 слева направо).
    Позиция символа — из выравнивания CTC той же строки (best path: кадр t ↔ доля (t+0.5)/T ширины
    кропа): k непробельных символов левее разреза → позиция после k-го непробельного символа исходного
    текста (если best path и текст разошлись по длине — пропорционально), затем к ближайшему пробелу
    или стыку со знаком в ±3 символах (внутри слова и кода не режем; нет такого места — []).
    Без логитов — пропорционально длине текста."""
    nonsp = [i for i, ch in enumerate(text) if not ch.isspace()]
    if not nonsp:
        return []
    ks = []
    if logits is not None and vocab is not None:
        idx = logits.argmax(axis=-1)[0]
        T = len(idx)
        em, prev = [], 0
        for t, k in enumerate(idx.tolist()):
            if k != 0 and k != prev and 0 <= k - 1 < len(vocab) and not vocab[k - 1].isspace():
                em.append((t + 0.5) / T)
            prev = k
        scale = len(nonsp) / max(1, len(em))
        ks = [int(round(sum(e < f for e in em) * scale)) for f in fracs] if em else []
    if not ks:
        ks = [int(round(f * len(nonsp))) for f in fracs]
    def kind(ch):
        return "L" if ch.isalpha() else "D" if ch.isdigit() else "S" if ch.isspace() else "P"

    cuts = []
    for k in ks:
        if k <= 0 or k >= len(nonsp):
            continue
        pos = nonsp[k - 1] + 1                               # сразу после k-го непробельного
        # резать можно по пробелу или на стыке со знаком («документа:|28»), но не внутри слова («К|абель»)
        # и не на стыке буквы с цифрой (коды «CK001», «g1491»): выравнивание CTC ошибается на символ;
        # ближайшее такое место в ±3, пробел — первым
        cand = [j for j in range(max(1, pos - 3), min(len(text), pos + 4))
                if text[j].isspace() or (not text[j - 1].isspace() and kind(text[j - 1]) != kind(text[j])
                                         and "P" in (kind(text[j - 1]), kind(text[j])))]
        if not cand:
            return []                                        # некуда резать — строку не трогаем
        cuts.append(min(cand, key=lambda j: (abs(j - pos), not text[j].isspace())))
    return sorted(set(cuts))


def split_crossing_lines(found: List[Dict], lines: List[Dict], image: np.ndarray, recognizer) -> List[Dict]:
    """Строка, которую детектор склеил через границу ячеек (узкий зазор между столбцами), делится на
    куски по пробелу у этой границы. Режем, только если граница отступает от краёв строки хотя бы на
    половину её высоты И на картинке у границы есть пробел (_gap_cut): текст, вылезший из ячейки
    через линию, не трогаем. Текст кусков — части ИСХОДНОГО текста строки (место разреза — по
    выравниванию CTC, _text_cuts): перераспознавать короткие куски нельзя — сеть теряет в них пробелы.
    Повёрнутые/вертикальные строки не режем. image — RGB."""
    out: List[Dict] = []
    for line in lines:
        q = np.asarray(line["quad"], dtype=float)
        x0, x1, y0, y1 = q[:, 0].min(), q[:, 0].max(), q[:, 1].min(), q[:, 1].max()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        t = next((t for t in found if t["bbox"][0] <= cx <= t["bbox"][2] and t["bbox"][1] <= cy <= t["bbox"][3]), None)
        m = 0.5 * (y1 - y0)
        horizontal = not line.get("orientation_deg") and line.get("layout", "horizontal") in (None, "horizontal")
        seps = [b for b in cell_boundaries(t, cy) if x0 + m < b < x1 - m] if (t and horizontal) else []
        gaps = sorted({g for g in (_gap_cut(image, x0, x1, y0, y1, b) for b in seps)
                       if g is not None and x0 + m < g[0] and g[1] < x1 - m})
        text = line.get("text", "")
        if not gaps or not text.strip():
            out.append(line)
            continue
        logits = None
        if hasattr(recognizer, "logits_per_line"):
            try:
                logits = recognizer.logits_per_line(image, [q.astype(np.float32)])[0]
            except Exception:
                logits = None
        mids = [((le + rs) / 2 - x0) / max(1e-6, x1 - x0) for le, rs in gaps]
        cuts = _text_cuts(text, mids, logits, getattr(recognizer, "vocab", None))
        bounds = [0] + cuts + [len(text)]
        texts = [text[a:b].strip() for a, b in zip(bounds, bounds[1:])]
        if len(cuts) != len(gaps) or not all(texts):        # не сошлось — строку не трогаем
            out.append(line)
            continue
        xs = [x0] + [(le + rs) / 2 for le, rs in gaps] + [x1]
        for (a, b), txt in zip(zip(xs, xs[1:]), texts):
            quad = [[a, y0], [b, y0], [b, y1], [a, y1]]
            out.append({**{k: v for k, v in line.items() if k not in ("quad", "text")},
                        "quad": quad, "text": txt, "split": True})
    return out


def attach_lines(found: List[Dict], lines: List[Dict], scale: float = 1.0) -> List[Dict]:
    """Разложить распознанные строки по ячейкам найденных таблиц.

    found — результат TableRecognizer(image) в пикселях картинки; scale переводит их в координаты строк
    (для текстового слоя PDF: точки = пиксели / zoom). Строке внутри таблицы (по центру её рамки)
    добавляются ключи 'table' (номер) и 'cell' ({r, c, rowspan, colspan}). Возвращает таблицы:
    {'bbox', 'confidence', 'n_rows', 'n_cols', 'rows', 'cols', 'cells': [{r, c, rowspan, colspan, text}]},
    координаты — в системе строк."""
    tables = []
    for t in found:
        x0, y0, x1, y1, conf = t["bbox"]
        rows = [(a * scale, b * scale) for a, b in t["rows"]]
        cols = [(a * scale, b * scale) for a, b in t["cols"]]
        R, C = len(rows), len(cols)
        cells = [dict(c) for c in t["cells"]] or [dict(r=i, c=j, rowspan=1, colspan=1)
                                                    for i in range(R) for j in range(C)]
        owner = {}
        for k, c in enumerate(cells):
            for i in range(c["r"], c["r"] + c["rowspan"]):
                for j in range(c["c"], c["c"] + c["colspan"]):
                    owner[(i, j)] = k
        tables.append(dict(bbox=[x0 * scale, y0 * scale, x1 * scale, y1 * scale], confidence=float(conf),
                           n_rows=R, n_cols=C, rows=rows, cols=cols, cells=cells, _owner=owner, _lines=[]))
    if tables:
        for line in lines:
            q = np.asarray(line["quad"], dtype=float)
            cx, cy = float(q[:, 0].mean()), float(q[:, 1].mean())
            hits = [ti for ti, t in enumerate(tables)
                    if t["bbox"][0] <= cx <= t["bbox"][2] and t["bbox"][1] <= cy <= t["bbox"][3]]
            if not hits or not tables[hits[0]]["rows"] or not tables[hits[0]]["cols"]:
                continue
            ti = min(hits, key=lambda i: (tables[i]["bbox"][2] - tables[i]["bbox"][0]) *
                                         (tables[i]["bbox"][3] - tables[i]["bbox"][1]))   # вложенные — меньшая
            t = tables[ti]
            k = t["_owner"].get((_band_index(t["rows"], cy), _band_index(t["cols"], cx)))
            if k is None:
                continue
            c = t["cells"][k]
            line["table"] = ti
            line["cell"] = {key: c[key] for key in ("r", "c", "rowspan", "colspan")}
            t["_lines"].append((k, float(q[:, 1].min()), float(q[:, 1].max()), float(q[:, 0].min()),
                                line.get("text", "")))
    for t in tables:
        per_cell: Dict[int, list] = {}
        for k, ytop, ybot, xl, txt in t.pop("_lines"):
            per_cell.setdefault(k, []).append((ytop, ybot, xl, txt))
        for k, c in enumerate(t["cells"]):
            # внутри ячейки: сверху вниз; строки, чьи верхние края ближе половины высоты, — одна строка,
            # её куски слева направо
            ordered, row = [], []
            for ln in sorted(per_cell.get(k, []), key=lambda z: z[0]):
                if row and ln[0] - row[0][0] >= 0.5 * (row[0][1] - row[0][0]):
                    ordered += sorted(row, key=lambda z: z[2]); row = []
                row.append(ln)
            ordered += sorted(row, key=lambda z: z[2])
            c["text"] = " ".join(z[3].strip() for z in ordered if z[3].strip())
        del t["_owner"]
    return tables


def table_markdown(table: Dict) -> str:
    """Таблица → Markdown. Текст объединённой ячейки — в её левой верхней клетке, остальные пустые."""
    R, C = table["n_rows"], table["n_cols"]
    if R == 0 or C == 0:
        return ""
    grid = [[""] * C for _ in range(R)]
    for c in table["cells"]:
        grid[c["r"]][c["c"]] = c.get("text", "").replace("|", "\\|")
    out = ["| " + " | ".join(grid[0]) + " |", "|" + "---|" * C]
    out += ["| " + " | ".join(row) + " |" for row in grid[1:]]
    return "\n".join(out)


def compose_text(lines: List[Dict], tables: Optional[List[Dict]]) -> str:
    """Текст страницы: строки в переданном порядке, таблица — Markdown на месте её первой строки
    (с пустыми строками вокруг). Таблицы без текста не печатаются."""
    if not tables:
        return "\n".join(item["text"] for item in lines)
    blocks: List[str] = []
    done = set()
    for item in lines:
        ti = item.get("table")
        if ti is None:
            blocks.append(item["text"])
        elif ti not in done:
            done.add(ti)
            md = table_markdown(tables[ti])
            if md:
                if blocks and blocks[-1] != "":
                    blocks.append("")
                blocks += [md, ""]
    while blocks and blocks[-1] == "":
        blocks.pop()
    return "\n".join(blocks)
