"""Распознавание таблиц: детектор таблиц на странице + структура (сетка строк/столбцов + объединённые ячейки).

Два шага, всё на ONNX (CPU или GPU):
  • Детектор: страница 768×768 → карта «таблица/фон» → рамки таблиц (связные компоненты).
  • Структура (split+merge): кроп таблицы 768×768 → сетка полос (строки/столбцы) и объединённые ячейки
    (colspan/rowspan). Сеть принимает решения по своим выходам (число полос, границы клеток), поэтому
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

INPUT = 768                              # вход обеих моделей
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _img_tensor(bgr: np.ndarray) -> np.ndarray:
    """BGR (cv2) → нормализованный тензор [1,3,768,768] float32 (ImageNet-норма, RGB)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    a = cv2.resize(rgb, (INPUT, INPUT), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    a = (a - _MEAN) / _STD
    return a.transpose(2, 0, 1)[None].copy()


def _boxes_from_map(prob: np.ndarray, thr=0.65, min_frac=0.01) -> List[Tuple[float, float, float, float, float]]:
    """Карта вероятностей [H,W] → нормированные рамки (x0,y0,x1,y1,conf) через связные компоненты (cv2).
    Порядок — как читают: сверху вниз, затем слева направо. (Раньше — по убыванию conf: у таблиц одной
    страницы она часто равна до 4-го знака, и порядок менялся от шума float, напр. между CPU и GPU.)"""
    m = (prob > thr).astype(np.uint8)
    if not m.any():
        return []
    hf, wf = m.shape
    n, lab = cv2.connectedComponents(m, connectivity=8)
    out = []
    for q in range(1, n):
        ys, xs = np.where(lab == q)
        if len(ys) / (hf * wf) < min_frac:
            continue
        conf = float(prob[lab == q].mean())
        out.append((xs.min() / wf, ys.min() / hf, (xs.max() + 1) / wf, (ys.max() + 1) / hf, conf))
    return sorted(out, key=lambda b: (b[1], b[0]))


def _bands_from(cnt_logits: np.ndarray, cb: np.ndarray, size: float) -> List[Tuple[float, float]]:
    """Логиты числа полос + кумулятивные границы → список полос (a,b) в пикселях кропа."""
    n = int(np.argmax(cnt_logits)) + 1
    b = np.asarray(cb[:n], np.float64) * size
    lines = sorted(min(size, max(0.0, v)) for v in ([0.0] + [float(v) for v in b]))
    return [(lines[k], lines[k + 1]) for k in range(len(lines) - 1)]


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


def _sep_to_bounds(logit: np.ndarray, nmax: int, thr: float = 0.70, min_dist: int = 3) -> np.ndarray:
    """Логиты сепараторов [Lp] → нормированные кумулятивные границы [nmax+1].
    Пиковый декод: локальные максимумы выше thr, прореживание ближе min_dist, параболическое уточнение.
    (На длинных таблицах порог-0.5 + связные компоненты слипает соседние сепараторы и теряет строки.)"""
    p = (1.0 / (1.0 + np.exp(-logit.astype(np.float32)))).astype(np.float32)
    Lp = len(p)
    cand = [i for i in range(Lp)
            if p[i] > thr and p[i] >= p[max(0, i - 1)] and p[i] >= p[min(Lp - 1, i + 1)]]
    keep = []
    for i in cand:
        if not keep or i - keep[-1] >= min_dist:
            keep.append(i)
        elif p[i] > p[keep[-1]]:
            keep[-1] = i
    cent = []
    for i in keep:
        if 0 < i < Lp - 1:
            y0, y1, y2 = p[i - 1], p[i], p[i + 1]
            den = (y0 - 2 * y1 + y2)
            off = 0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0
            off = float(np.clip(off, -0.5, 0.5))
            c = (i + off + 0.5) / Lp
            if 1e-4 < c < 1 - 1e-4:
                cent.append(c)
    cent = sorted(cent)[:nmax - 1]
    out = np.ones(nmax + 1, np.float32)
    if cent:
        out[:len(cent)] = cent
    return out


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

        self._det = make_session(ensure_weight("table_detect_v3_fp32.onnx"), num_threads, gpu)
        self._det_in = self._det.get_inputs()[0].name
        self._main = make_session(ensure_weight("table_struct_v3_main_fp32.onnx"), num_threads, gpu)
        self._merge = (make_session(ensure_weight("table_struct_v3_merge_fp32.onnx"), num_threads, gpu)
                       if merge else None)
        out = {o.name: o.shape for o in self._main.get_outputs()}
        self._nmax_r = out["rcnt"][-1]; self._nmax_c = out["ccnt"][-1]

    # ---- детекция ----
    def detect(self, image: np.ndarray, thr: float = 0.65, min_frac: float = 0.01) -> List[Tuple[int, int, int, int, float]]:
        """Страница (BGR) → список рамок таблиц (x0,y0,x1,y1,conf) в ПИКСЕЛЯХ исходного изображения."""
        h, w = image.shape[:2]
        prob = self._det.run(None, {self._det_in: _img_tensor(image)})[0][0, 0]   # [MP,MP]
        return [(int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h), c)
                for x0, y0, x1, y1, c in _boxes_from_map(prob, thr, min_frac)]

    # ---- структура ----
    def structure(self, table_crop: np.ndarray) -> Dict:
        """Кроп таблицы (BGR) → {'rows':[(y0,y1)...], 'cols':[(x0,x1)...], 'cells':[{r,c,rowspan,colspan}...]}.
        Координаты полос — в пикселях переданного кропа. cells пуст при merge=False."""
        h, w = table_crop.shape[:2]
        rcnt, rsep, ccnt, csep, pmap = self._main.run(None, {"x": _img_tensor(table_crop)})
        rcb = _sep_to_bounds(rsep[0], self._nmax_r); ccb = _sep_to_bounds(csep[0], self._nmax_c)
        rows = _bands_from(rcnt[0], rcb, h); cols = _bands_from(ccnt[0], ccb, w)
        cells = []
        if self._merge is not None:
            R = int(np.argmax(rcnt[0])) + 1; C = int(np.argmax(ccnt[0])) + 1
            hor, ver = self._merge.run(None, _merge_inputs(pmap[0], rcb, ccb, R, C))
            sig = lambda a: 1.0 / (1.0 + np.exp(-a))
            R, C = len(rows), len(cols)
            horm = sig(hor) if C > 1 else np.zeros((R, 0)); verm = sig(ver) if R > 1 else np.zeros((0, C))
            if R >= 1 and C >= 1:
                cells = _cells_from_edges(horm, verm, R, C)
        return {"rows": rows, "cols": cols, "cells": cells}

    def __call__(self, image: np.ndarray, thr: float = 0.65) -> List[Dict]:
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
