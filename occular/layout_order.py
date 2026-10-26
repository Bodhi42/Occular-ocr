"""ЭКСПЕРИМЕНТАЛЬНЫЙ порядок чтения по layout-регионам: XY-cut по регионам + строки-сироты как мини-блоки.

Чистая геометрия: на входе только прямоугольники строк (детектор строк) и регионы LayoutDetector
той же страницы. Никакой модели порядка, никаких меток в решении (кроме того, что stamp/signature
не считаются текстовыми блоками). Перенесено без изменений из проверенного варианта:
  1. блоки = регионы, кроме stamp/signature; handwritten_text, лежащий на ≥80% внутри другого
     региона, считается наложением и отдельным блоком не является;
  2. строка, не покрытая ни одним блоком больше чем на 30% площади, становится своим мини-блоком;
  3. блоки упорядочиваются рекурсивным XY-cut (разрез по наибольшему просвету, при равенстве — по Y);
  4. строка привязывается к блоку с максимальным покрытием > 0.3 (при равенстве — к более раннему),
     как в штатном reading_order; внутри блока строки — сверху вниз по центру.
Каждая входная строка попадает в выход ровно один раз. Это эвристика: она не гарантирует
правильного порядка чтения на произвольной странице.
"""
from typing import Dict, List, Sequence, Tuple

ASSIGN_THRESHOLD = 0.3       # штатное правило привязки строки к региону
COVER_THRESHOLD = 0.8        # «строка покрыта регионом» для статистики coverage
_NON_TEXT = {"stamp", "signature"}


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _cov(line, reg):
    """Доля площади строки внутри региона."""
    return _inter(line, reg) / max(1e-6, _area(line))


def _region_fields(r) -> Tuple[Tuple[float, float, float, float], str]:
    if isinstance(r, dict):
        return tuple(float(v) for v in r["bbox"]), str(r.get("label", "Text"))
    return tuple(float(v) for v in r.bbox), str(r.label)


def text_blocks(regions: Sequence) -> List[int]:
    """Индексы регионов, которые служат блоками чтения (см. п.1 модуля)."""
    rs = [_region_fields(r) for r in regions]
    base = [i for i, (b, lab) in enumerate(rs) if lab not in _NON_TEXT and lab != "handwritten_text"]
    hw = [i for i, (b, lab) in enumerate(rs) if lab == "handwritten_text"]
    free_hw = [i for i in hw
               if not any(_inter(rs[i][0], rs[j][0]) / max(1e-6, _area(rs[i][0])) >= 0.8 for j in base)]
    return base + free_hw


def xy_cut(boxes: Sequence[Sequence[float]], idx: List[int] = None) -> List[int]:
    """Рекурсивный XY-cut: порядок индексов boxes. Детерминирован (ничьи решаются индексом)."""
    if idx is None:
        idx = list(range(len(boxes)))
    if len(idx) <= 1:
        return list(idx)
    best = None
    for ax in (1, 0):                       # сначала Y (горизонтальные разрезы), при равенстве он и остаётся
        iv = sorted((boxes[i][ax], boxes[i][ax + 2], i) for i in idx)
        segs, cur, end, gaps = [], [iv[0][2]], iv[0][1], []
        for s, e, i in iv[1:]:
            if s > end:
                gaps.append(s - end); segs.append(cur); cur = [i]
            else:
                cur.append(i)
            end = max(end, e)
        segs.append(cur)
        if len(segs) > 1:
            g = max(gaps)
            if best is None or g > best[0]:
                best = (g, segs)
    if best is None:                        # неразрезаемая группа (перекрытия) — сверху вниз, слева направо
        return sorted(idx, key=lambda i: (boxes[i][1], boxes[i][0], i))
    out = []
    for seg in best[1]:
        out += xy_cut(boxes, seg)
    return out


def order_lines(line_boxes: Sequence[Sequence[float]], regions: Sequence) -> Tuple[List[int], List[int], Dict]:
    """Порядок строк по layout-регионам.

    Args:
        line_boxes: прямоугольники строк [x1,y1,x2,y2] (в тех же координатах, что регионы).
        regions: LayoutRegion или dict {"bbox": [...], "label": str}.
    Returns:
        order: перестановка индексов строк (каждый ровно один раз);
        line_region: для каждой строки индекс региона в `regions`, к которому она привязана, или -1
            (строка-сирота, встала в порядок своим мини-блоком);
        stats: {"lines", "coverage_80" (доля строк, покрытых ≥80% одним блоком),
                "unassigned_30" (доля строк без привязки по правилу >30% — стали мини-блоками),
                "regions", "text_blocks"}.
    """
    lines = [tuple(float(v) for v in b) for b in line_boxes]
    n = len(lines)
    stats = {"lines": n, "regions": len(regions), "text_blocks": 0, "coverage_80": 0.0, "unassigned_30": 0.0}
    if n == 0:
        return [], [], stats
    blk_idx = text_blocks(regions)
    stats["text_blocks"] = len(blk_idx)
    rb = [_region_fields(regions[i])[0] for i in blk_idx]
    covered = sum(any(_cov(l, b) >= COVER_THRESHOLD for b in rb) for l in lines)
    orphans = [k for k, l in enumerate(lines) if not any(_cov(l, b) > ASSIGN_THRESHOLD for b in rb)]
    stats["coverage_80"] = covered / n
    stats["unassigned_30"] = len(orphans) / n
    # блоки: (bbox, исходный индекс региона или -1 для мини-блока строки)
    units = [(b, i) for b, i in zip(rb, blk_idx)] + [(lines[k], -1) for k in orphans]
    uorder = xy_cut([u[0] for u in units])
    ordered = [units[i] for i in uorder]
    keyed, line_region = [], []
    for k, l in enumerate(lines):
        best, best_ov = len(ordered), ASSIGN_THRESHOLD
        for j, (ub, _) in enumerate(ordered):
            ov = _cov(l, ub)
            if ov > best_ov:
                best_ov, best = ov, j
        line_region.append(ordered[best][1] if best < len(ordered) else -1)
        keyed.append((best, (l[1] + l[3]) / 2, k))   # best==len(ordered) только у вырожденной строки нулевой площади → в конец
    keyed.sort()
    return [k for _, _, k in keyed], line_region, stats
