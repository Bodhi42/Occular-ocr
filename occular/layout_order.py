"""Общая геометрия порядка чтения по layout-регионам: какие регионы считаются текстовыми блоками и как
строка привязывается к блоку. Порядок самих блоков задаёт обученная модель (layout_order_model.py):
  1. блоки = регионы, кроме stamp/signature; handwritten_text, лежащий на ≥80% внутри другого
     региона, считается наложением и отдельным блоком не является; блок, лежащий на ≥80% внутри
     более крупного блока, тоже не блок (строки читаются в составе внешнего);
  2. строка, не покрытая ни одним блоком больше чем на 30% площади, становится своим мини-блоком;
  3. строка привязывается к блоку с максимальным покрытием > 0.3; внутри блока строки — сверху вниз.
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
    blocks = base + free_hw
    # вложенные блоки: блок, лежащий ≥80% внутри более крупного (при равной площади — более раннего), не
    # блок — иначе строки одного абзаца расходятся по «половине» и «целому» и читаются вразнобой.
    # Замер (955 стр., GPU): CER 0.229 → 0.225, лучше на 67 стр., хуже на 26.
    def nested(a):
        return any(b != a and _inter(rs[a][0], rs[b][0]) / max(1e-6, _area(rs[a][0])) >= 0.8
                   and (_area(rs[b][0]) > _area(rs[a][0])
                        or (_area(rs[b][0]) == _area(rs[a][0]) and blocks.index(b) < blocks.index(a)))
                   for b in blocks)
    return [i for i in blocks if not nested(i)]
