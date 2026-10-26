"""ЭКСПЕРИМЕНТАЛЬНЫЙ обученный порядок чтения по блокам layout (ONNX, CPU).

Те же объекты, что у XY-cut в layout_order.py:
  1. блоки = текстовые регионы LayoutDetector (text_blocks: без stamp/signature, handwritten_text
     внутри другого региона поглощается);
  2. строка, не покрытая ни одним блоком больше чем на 30% площади, становится своим объектом;
  3. ПОРЯДОК ОБЪЕКТОВ задаёт модель (3.3M параметров): по геометрии, метке и уверенности каждого
     объекта она даёт попарные логиты «i раньше j» и логиты «j сразу после i»; декодер жадно берёт
     следующий объект по сумме обоих (alpha = вес второго, подобран на валидации);
  4. строка привязывается к объекту с максимальным покрытием > 0.3 (при равенстве — к более раннему
     по списку объектов); внутри объекта строки идут сверху вниз по центру.
Каждая входная строка попадает в выход ровно один раз. Модель обучена на ~76 тыс. страниц и по
качеству сопоставима с XY-cut; правильный порядок на произвольной странице она не гарантирует.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .layout import LAYOUT_LABELS
from .layout_order import ASSIGN_THRESHOLD, COVER_THRESHOLD, _cov, _region_fields, text_blocks

MODEL_FILE = "layout_order_blocks_fp32.onnx"
DEFAULT_ALPHA = 3.0
_NAME2ID = {n: i for i, n in LAYOUT_LABELS.items()}
_ORPHAN_ID = 19          # метка «строка вне блоков»
_UNKNOWN_ID = 14         # неизвестная метка считается Text (как при обучении)


def _region_score(r) -> float:
    return float(r.get("score", 1.0)) if isinstance(r, dict) else float(r.score)


def unit_features(boxes: Sequence[Sequence[float]], scores: Sequence[float], width: float,
                  height: float) -> np.ndarray:
    """Признаки объектов [U, 12]: нормированный бокс, размеры, центр, форма, уверенность,
    форма страницы и число объектов (ровно как при обучении)."""
    W, H = float(width), float(height)
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    n = len(b)
    x1, y1, x2, y2 = b[:, 0] / W, b[:, 1] / H, b[:, 2] / W, b[:, 3] / H
    w, h = np.maximum(x2 - x1, 1e-4), np.maximum(y2 - y1, 1e-4)
    cols = [x1, y1, x2, y2, w, h, (x1 + x2) / 2, (y1 + y2) / 2,
            np.clip(np.log(h * H / (w * W)), -5, 5) / 5,
            np.asarray(scores, np.float32), np.full(n, np.log(H / W), np.float32),
            np.full(n, n / 100.0, np.float32)]
    return np.stack(cols, 1).astype(np.float32)


def hybrid_decode(pair: np.ndarray, succ: np.ndarray, alpha: float = DEFAULT_ALPHA) -> List[int]:
    """Порядок объектов: следующий = argmax_j [sum_{k в оставшихся} log P(j раньше k)
    + alpha * log softmax_{оставшиеся} succ(текущий, j)]. Детерминирован (ничьи — меньший индекс)."""
    n = pair.shape[0]
    if n == 0:
        return []
    lp = -np.logaddexp(0.0, -pair.astype(np.float64))      # log sigmoid
    np.fill_diagonal(lp, 0.0)
    rem = np.ones(n, bool)
    ssum = lp.sum(1)
    cur, out = None, []
    for _ in range(n):
        sc = ssum.copy()
        if cur is not None and alpha > 0:
            z = np.where(rem, succ[cur].astype(np.float64), -np.inf)
            z = z - z.max()
            sc = sc + alpha * (z - np.log(np.exp(z).sum()))
        sc[~rem] = -np.inf
        j = int(np.argmax(sc))
        out.append(j)
        rem[j] = False
        ssum = ssum - lp[:, j]
        cur = j
    return out


class BlockOrderModel:
    """Обученный порядок чтения по блокам layout. Интерфейс как у layout_order.order_lines,
    плюс размер страницы (признаки нормируются на него)."""

    def __init__(self, num_threads: int = 4, gpu: bool = False, model_path: str = None,
                 alpha: float = DEFAULT_ALPHA):
        import onnxruntime as ort
        if model_path is None:
            from .model_files import ensure_weight
            try:
                model_path = ensure_weight(MODEL_FILE)
            except Exception as e:
                raise RuntimeError(
                    f"Could not get the reading-order weights '{MODEL_FILE}': {e}\n"
                    f"Put the file in occular/weights/ or check access to Hugging Face."
                ) from e
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(num_threads))
        so.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        if gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(str(model_path), so, providers=providers)
        self.alpha = float(alpha)

    def order_units(self, feats: np.ndarray, labels: np.ndarray) -> List[int]:
        if len(labels) <= 1:
            return list(range(len(labels)))
        pair, succ = self.session.run(None, {"feats": feats[None].astype(np.float32),
                                             "labels": labels[None].astype(np.int64)})
        return hybrid_decode(pair[0], succ[0], self.alpha)

    def order_lines(self, line_boxes: Sequence[Sequence[float]], regions: Sequence, width: float,
                    height: float) -> Tuple[List[int], List[int], Dict]:
        """Порядок строк по блокам layout.

        Args:
            line_boxes: прямоугольники строк [x1,y1,x2,y2] (в тех же координатах, что регионы).
            regions: LayoutRegion или dict {"bbox": [...], "label": str, "score": float (по умолч. 1.0)}.
            width, height: размер страницы в пикселях этих координат.
        Returns:
            (order, line_region, stats) — как у layout_order.order_lines.
        """
        lines = [tuple(float(v) for v in b) for b in line_boxes]
        n = len(lines)
        stats = {"lines": n, "regions": len(regions), "text_blocks": 0, "coverage_80": 0.0, "unassigned_30": 0.0}
        if n == 0:
            return [], [], stats
        blk_idx = text_blocks(regions)
        stats["text_blocks"] = len(blk_idx)
        rf = [_region_fields(regions[i]) for i in blk_idx]
        rb = [b for b, _ in rf]
        covered = sum(any(_cov(l, b) >= COVER_THRESHOLD for b in rb) for l in lines)
        orphans = [k for k, l in enumerate(lines) if not any(_cov(l, b) > ASSIGN_THRESHOLD for b in rb)]
        stats["coverage_80"] = covered / n
        stats["unassigned_30"] = len(orphans) / n
        # объекты: (bbox, исходный индекс региона или -1, метка, уверенность)
        units = [(b, i, _NAME2ID.get(lab, _UNKNOWN_ID), _region_score(regions[i]))
                 for (b, lab), i in zip(rf, blk_idx)]
        units += [(lines[k], -1, _ORPHAN_ID, 0.0) for k in orphans]
        feats = unit_features([u[0] for u in units], [u[3] for u in units], width, height)
        labels = np.array([u[2] for u in units], np.int64)
        rank = {u: r for r, u in enumerate(self.order_units(feats, labels))}
        keyed, line_region = [], []
        for k, l in enumerate(lines):
            best, best_ov = -1, ASSIGN_THRESHOLD
            for j, u in enumerate(units):
                ov = _cov(l, u[0])
                if ov > best_ov:
                    best_ov, best = ov, j
            line_region.append(units[best][1] if best >= 0 else -1)
            # best == -1 только у вырожденной строки нулевой площади → в конец
            keyed.append((rank[best] if best >= 0 else len(units), (l[1] + l[3]) / 2, k))
        keyed.sort()
        return [k for _, _, k in keyed], line_region, stats
