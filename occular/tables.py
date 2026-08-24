"""Распознавание таблиц: детектор таблиц на странице + структура (сетка строк/столбцов + объединённые ячейки).

Два шага, две модели:
  • Детектор (ONNX FP32, CPU): страница 768×768 → карта «таблица/фон» → рамки таблиц (связные компоненты).
  • Структура (split[+merge]): кроп таблицы 768×768 → сетка полос (строки/столбцы) и, опционально,
    объединённые ячейки (colspan/rowspan). Split — feed-forward (есть ONNX-путь без torch);
    merge (объединения) — на карте признаков переменного размера, поэтому только PyTorch на CPU.

Быстрый старт:
    from occular.tables import TableRecognizer
    tr = TableRecognizer()                 # merge включается автоматически, если установлен torch
    import cv2; page = cv2.imread("scan.png")
    for t in tr(page):                     # список таблиц страницы
        print(t["bbox"], "строк:", len(t["rows"]), "столбцов:", len(t["cols"]), "ячеек:", len(t["cells"]))

Только детекция:  tr.detect(page)
Только структура:  tr.structure(table_crop)
"""
import warnings
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
    """Карта вероятностей [H,W] → нормированные рамки (x0,y0,x1,y1,conf) через связные компоненты (cv2)."""
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
    return sorted(out, key=lambda b: -b[4])


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


class TableRecognizer:
    """Детекция таблиц + структура (сетка + объединённые ячейки).

    Детектор — ONNX (CPU). Структура: если установлен PyTorch — полная модель split+merge на CPU
    (даёт объединённые ячейки); иначе фолбэк на ONNX split (только сетка строк/столбцов, без спанов).
    """

    def __init__(self, num_threads: int = 4, merge: bool = True):
        from .model_files import ensure_weight
        import onnxruntime as ort

        def _sess(path):
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = max(1, int(num_threads))
            so.inter_op_num_threads = 1
            return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])

        self._det = _sess(ensure_weight("table_detect_v3_fp32.onnx"))
        self._det_in = self._det.get_inputs()[0].name

        # Структура: PyTorch split+merge (объединённые ячейки), иначе — ONNX split (только сетка).
        self._torch_model = None
        self._split_sess = None
        self._nmax_r = self._nmax_c = None
        want_merge = merge
        if want_merge:
            try:
                import torch
                from ._tables_model import FastTabSplit
                sd = torch.load(ensure_weight("table_struct_split_merge_v2.pt"), map_location="cpu",
                                weights_only=False)
                cfg = sd["cfg"]
                m = FastTabSplit(sd["dmodel"], T=cfg["T"], L=cfg["L"], vdown=cfg["vdown"],
                                 pool=cfg["pool"], head=cfg["head"], upax=cfg["upax"],
                                 merge=cfg.get("merge", False), bedge=bool(cfg.get("bedge", 0)),
                                 mgrid=bool(cfg.get("mgrid", 0)))
                if cfg.get("upc"):
                    m.upax_col = cfg["upc"]
                m.load_state_dict(sd["model"]); m.eval()
                self._torch = torch
                self._torch_model = m
                self._nmax_r = m.row.nmax; self._nmax_c = m.col.nmax
            except Exception as e:
                warnings.warn(f"PyTorch-путь структуры недоступен ({e}); откат на ONNX split "
                              f"(сетка строк/столбцов без объединённых ячеек). Для спанов: pip install occular-ocr[gpu]")
        if self._torch_model is None:
            self._split_sess = _sess(ensure_weight("table_struct_split_v2_fp32.onnx"))
            self._split_in = self._split_sess.get_inputs()[0].name

    # ---- детекция ----
    def detect(self, image: np.ndarray, thr: float = 0.65, min_frac: float = 0.01) -> List[Tuple[int, int, int, int, float]]:
        """Страница (BGR) → список рамок таблиц (x0,y0,x1,y1,conf) в ПИКСЕЛЯХ исходного изображения."""
        h, w = image.shape[:2]
        prob = self._det.run(None, {self._det_in: _img_tensor(image)})[0][0, 0]   # [MP,MP]
        return [(int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h), c)
                for x0, y0, x1, y1, c in _boxes_from_map(prob, thr, min_frac)]

    # ---- структура ----
    def _sep_to_bounds_np(self, sep_logit: np.ndarray, nmax: int) -> np.ndarray:
        """ONNX-фолбэк: логиты сепараторов [Lp] → кумулятивные границы [nmax+1] (как sep_to_bounds)."""
        p = 1.0 / (1.0 + np.exp(-sep_logit)); Lp = len(p); on = p > 0.5
        cent, i = [], 0
        while i < Lp:
            if on[i]:
                j = i
                while j + 1 < Lp and on[j + 1]:
                    j += 1
                w = p[i:j + 1]; idx = np.arange(i, j + 1)
                cent.append(float(((w * idx).sum() / max(w.sum(), 1e-6) + 0.5) / Lp)); i = j + 1
            else:
                i += 1
        cent = [c for c in cent if 1e-4 < c < 1 - 1e-4][:nmax - 1]
        out = np.ones(nmax + 1, np.float32)
        if cent:
            out[:len(cent)] = cent
        return out

    def structure(self, table_crop: np.ndarray) -> Dict:
        """Кроп таблицы (BGR) → {'rows':[(y0,y1)...], 'cols':[(x0,x1)...], 'cells':[{r,c,rowspan,colspan}...]}.
        Координаты полос — в пикселях переданного кропа. cells пуст, если merge недоступен (ONNX-фолбэк)."""
        h, w = table_crop.shape[:2]
        x = _img_tensor(table_crop)
        if self._torch_model is not None:
            with self._torch.no_grad():
                o = self._torch_model(self._torch.from_numpy(x))
            rcnt = o["rcnt"][0].numpy(); rcb = o["rcb"][0].numpy()
            ccnt = o["ccnt"][0].numpy(); ccb = o["ccb"][0].numpy()
            rows = _bands_from(rcnt, rcb, h); cols = _bands_from(ccnt, ccb, w)
            cells = []
            if "medges" in o and o["medges"]:
                hor, ver = o["medges"][0]
                R, C = len(rows), len(cols)
                horm = (self._torch.sigmoid(hor).numpy() if hor is not None else np.zeros((R, max(C - 1, 0))))
                verm = (self._torch.sigmoid(ver).numpy() if ver is not None else np.zeros((max(R - 1, 0), C)))
                if R >= 1 and C >= 1:
                    cells = _cells_from_edges(horm, verm, R, C)
            return {"rows": rows, "cols": cols, "cells": cells}
        # ONNX-фолбэк: split → сетка (без объединённых ячеек)
        rcnt, rsep, ccnt, csep = self._split_sess.run(None, {self._split_in: x})
        rcb = self._sep_to_bounds_np(rsep[0], self._nmax_r or (len(rsep[0]) + 1))
        ccb = self._sep_to_bounds_np(csep[0], self._nmax_c or (len(csep[0]) + 1))
        rows = _bands_from(rcnt[0], rcb, h); cols = _bands_from(ccnt[0], ccb, w)
        return {"rows": rows, "cols": cols, "cells": []}

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
