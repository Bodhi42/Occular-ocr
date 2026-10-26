"""Тесты layout-регионов и ЭКСПЕРИМЕНТАЛЬНОГО порядка чтения по ним.

Геометрические тесты (order_lines / xy_cut) не требуют весов. Они проверяют инварианты
(каждая строка ровно один раз, детерминизм, сироты не теряются) и несколько синтетических раскладок.
Прохождение синтетики НЕ гарантирует правильный порядок чтения на реальных страницах.
Тесты с реальной моделью пропускаются, если веса недоступны.
"""
import random
from pathlib import Path

import numpy as np
import pytest

from occular.layout_order import order_lines, xy_cut, text_blocks
from occular.layout import LayoutRegion, LayoutResult, LAYOUT_LABELS


def R(x1, y1, x2, y2, label="Text"):
    return {"bbox": [x1, y1, x2, y2], "label": label}


def column_lines(x1, x2, y0, n, h=20, gap=10):
    return [[x1, y0 + k * (h + gap), x2, y0 + k * (h + gap) + h] for k in range(n)]


def assert_permutation(order, n):
    assert sorted(order) == list(range(n)), "каждая строка должна присутствовать ровно один раз"


# ---------- инварианты ----------

def test_empty_page():
    order, reg, st = order_lines([], [R(0, 0, 100, 100)])
    assert order == [] and reg == [] and st["lines"] == 0


def test_no_regions_keeps_all_lines():
    lines = column_lines(10, 200, 10, 5)
    order, reg, st = order_lines(lines, [])
    assert order == [0, 1, 2, 3, 4]
    assert reg == [-1] * 5
    assert st["unassigned_30"] == 1.0 and st["coverage_80"] == 0.0


def test_random_layouts_every_line_once_and_deterministic():
    rng = random.Random(0)
    labels = list(LAYOUT_LABELS.values())
    for _ in range(300):
        n = rng.randint(0, 40)
        lines = []
        for _ in range(n):
            x, y = rng.uniform(0, 900), rng.uniform(0, 1200)
            lines.append([x, y, x + rng.uniform(0, 300), y + rng.uniform(0, 40)])  # включая вырожденные
        regs = []
        for _ in range(rng.randint(0, 12)):
            x, y = rng.uniform(0, 800), rng.uniform(0, 1100)
            regs.append(R(x, y, x + rng.uniform(5, 600), y + rng.uniform(5, 600), rng.choice(labels)))
        o1, r1, _ = order_lines(lines, regs)
        o2, r2, _ = order_lines([list(l) for l in lines], [dict(r) for r in regs])
        assert_permutation(o1, n)
        assert (o1, r1) == (o2, r2)


def test_identical_coordinates_deterministic():
    lines = [[10, 10, 100, 30]] * 4 + [[10, 50, 100, 70]] * 3
    regs = [R(0, 0, 200, 100), R(0, 0, 200, 100)]   # два одинаковых региона
    o1, r1, _ = order_lines(lines, regs)
    o2, r2, _ = order_lines(lines, regs)
    assert o1 == o2 == [0, 1, 2, 3, 4, 5, 6]
    assert r1 == r2 and len(set(r1)) == 1        # все строки в одном (первом по порядку) регионе


def test_xy_cut_ties_deterministic():
    boxes = [[0, 0, 10, 10]] * 5
    assert xy_cut(boxes) == [0, 1, 2, 3, 4]


# ---------- синтетические раскладки ----------

def test_two_columns():
    left = column_lines(50, 450, 100, 6)
    right = column_lines(550, 950, 100, 6)
    lines = [l for pair in zip(left, right) for l in pair]          # вход перемешан построчно
    regs = [R(40, 90, 460, 290), R(540, 90, 960, 290)]
    order, reg, _ = order_lines(lines, regs)
    assert_permutation(order, 12)
    assert order == [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]          # вся левая колонка, затем правая


def test_missing_region_in_middle_is_not_moved_to_end():
    top = column_lines(50, 900, 50, 3)
    mid = column_lines(50, 900, 300, 3)                              # регион пропущен детектором
    bot = column_lines(50, 900, 600, 3)
    lines = top + mid + bot
    regs = [R(40, 40, 910, 150), R(40, 590, 910, 700)]
    order, reg, st = order_lines(lines, regs)
    assert order == list(range(9))                                   # середина осталась в середине
    assert reg[3:6] == [-1, -1, -1]
    assert st["unassigned_30"] == pytest.approx(3 / 9)


def test_nested_regions_no_duplicates():
    lines = column_lines(100, 800, 100, 8)
    regs = [R(50, 50, 900, 400, "Form"), R(90, 90, 810, 200, "Text"), R(90, 90, 810, 200, "Text"),
            R(600, 300, 700, 350, "stamp")]
    order, reg, _ = order_lines(lines, regs)
    assert_permutation(order, 8)
    assert order == list(range(8))


def test_multiple_documents_in_frame():
    # два документа рядом, у каждого заголовок и абзац; второй без регионов (не найден)
    a_title, a_body = [[50, 50, 400, 80]], column_lines(50, 400, 120, 5)
    b_title, b_body = [[550, 60, 900, 90]], column_lines(550, 900, 130, 5)
    lines = a_title + a_body + b_title + b_body
    regs = [R(40, 40, 410, 90, "SectionHeader"), R(40, 110, 410, 280)]
    order, reg, _ = order_lines(lines, regs)
    assert_permutation(order, 12)
    assert order == list(range(12))                                  # документ A целиком, затем B


def test_stamp_signature_are_not_blocks_and_hw_inside_form_absorbed():
    regs = [R(0, 0, 500, 500, "Form"), R(10, 10, 100, 40, "handwritten_text"),
            R(600, 0, 700, 50, "handwritten_text"), R(0, 0, 50, 50, "stamp"), R(0, 0, 60, 60, "signature")]
    assert text_blocks(regs) == [0, 2]


def test_coverage_and_unassigned_are_different_numbers():
    lines = [[0, 0, 100, 20], [0, 100, 100, 120]]
    regs = [R(0, 0, 50, 20)]                       # покрывает первую строку на 50%: привязана, но не «покрыта ≥80%»
    _, reg, st = order_lines(lines, regs)
    assert reg == [0, -1]
    assert st["coverage_80"] == 0.0 and st["unassigned_30"] == 0.5


# ---------- пайплайн (без весов, через подмену моделей) ----------

class _FakeLayout:
    def __init__(self, regions):
        self.regions, self.seen = regions, []

    def detect(self, image):
        self.seen.append(image)
        return LayoutResult(width=image.shape[1], height=image.shape[0], regions=self.regions)


def _bare_pipeline(results, regions, block_order=None):
    from occular.pipeline import OCRPipeline
    p = OCRPipeline.__new__(OCRPipeline)
    p.reading_order, p._ro = False, None
    p.experimental_layout_order = True
    p._layout = _FakeLayout(regions)
    p._block_order, p.layout_order_method = block_order, ("model" if block_order is not None else "xycut")
    p.seen_whole = []
    def whole(image):
        p.seen_whole.append(image)
        return [dict(r) for r in results]
    p._ocr_whole = whole
    return p


def _line(x1, y1, x2, y2, text):
    return {"quad": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "text": text, "confidence": 0.9}


def test_pipeline_picture_does_not_drop_text_and_same_image_used():
    results = [_line(10, 10, 200, 30, "a"), _line(10, 50, 200, 70, "b"), _line(500, 10, 700, 30, "c")]
    regs = [LayoutRegion((0, 0, 800, 100), 0.9, "Picture", 10)]       # Picture поверх всего текста
    p = _bare_pipeline(results, regs)
    img = np.zeros((100, 800, 3), np.uint8)
    out = p._ocr_page(img)
    assert sorted(r["text"] for r in out) == ["a", "b", "c"]          # ничего не выброшено
    assert all(r["layout_label"] == "Picture" for r in out)
    assert p._layout.seen[0] is img and p.seen_whole[0] is img       # строки и регионы с ОДНОЙ картинки


def test_pipeline_no_regions_and_empty_page():
    p = _bare_pipeline([], [])
    assert p._ocr_page(np.zeros((10, 10, 3), np.uint8)) == []
    p = _bare_pipeline([_line(0, 50, 10, 60, "2"), _line(0, 0, 10, 10, "1")], [])
    out = p._ocr_page(np.zeros((100, 100, 3), np.uint8))
    assert [r["text"] for r in out] == ["1", "2"] and all(r["layout_region"] is None for r in out)


def test_modes_are_exclusive_and_defaults_unchanged():
    from occular import Settings
    from occular.pipeline import OCRPipeline
    s = Settings()
    assert s.reading_order is False and s.experimental_layout_order is False
    with pytest.raises(ValueError):
        OCRPipeline(reading_order=True, experimental_layout_order=True)
    assert s.layout_order_method == "model"
    with pytest.raises(ValueError):
        OCRPipeline(experimental_layout_order=True, layout_order_method="magic")


# ---------- обученный порядок блоков ----------

from occular.layout_order_model import hybrid_decode, unit_features


def test_hybrid_decode_permutation_and_follows_pairs():
    rng = np.random.RandomState(0)
    for n in (0, 1, 2, 5, 40):
        a = rng.randn(n, n); pair = a - a.T; succ = rng.randn(n, n)
        o = hybrid_decode(pair, succ)
        assert sorted(o) == list(range(n)) and o == hybrid_decode(pair, succ)
    # сильные попарные логиты задают полный порядок 2 → 0 → 1 при любом succ (alpha=0 — только пары)
    pair = np.zeros((3, 3)); pair[2, 0] = pair[2, 1] = pair[0, 1] = 20; pair = pair - pair.T
    assert hybrid_decode(pair, rng.randn(3, 3), alpha=0.0) == [2, 0, 1]


def test_unit_features_layout():
    f = unit_features([[0, 0, 100, 50], [100, 50, 200, 100]], [0.9, 0.0], 200, 100)
    assert f.shape == (2, 12) and f.dtype == np.float32
    assert np.allclose(f[0, :4], [0, 0, 0.5, 0.5]) and np.allclose(f[:, 9], [0.9, 0.0]) and np.allclose(f[:, 11], 0.02)


class _FakeBlockOrder:
    """Подмена модели: порядок объектов задаётся извне (для проверки обвязки без весов)."""
    def __init__(self, unit_order):
        from occular.layout_order_model import BlockOrderModel
        self.unit_order = unit_order
        self.order_lines = BlockOrderModel.order_lines.__get__(self)

    def order_units(self, feats, labels):
        assert feats.shape == (len(labels), 12)
        return list(self.unit_order)


def test_block_order_every_line_once_orphans_and_inside_block_top_down():
    from occular.layout_order_model import BlockOrderModel
    left, right = column_lines(0, 100, 0, 4), column_lines(200, 300, 0, 4)
    orphan = [[500, 500, 560, 520]]
    lines = right[::-1] + left + orphan
    regs = [R(0, 0, 100, 200), R(200, 0, 300, 200)]
    m = _FakeBlockOrder([1, 0, 2])                 # правая колонка, левая, строка-сирота
    order, reg, st = m.order_lines(lines, regs, 600, 600)
    assert_permutation(order, len(lines))
    assert order == [3, 2, 1, 0, 4, 5, 6, 7, 8]    # внутри блока — сверху вниз
    assert reg[8] == -1 and st["unassigned_30"] == 1 / 9 and st["text_blocks"] == 2
    assert m.order_lines([], regs, 600, 600)[0] == []


def test_pipeline_uses_block_order_model():
    results = [_line(200, 0, 300, 20, "R"), _line(0, 0, 100, 20, "L")]
    regs = [LayoutRegion((0, 0, 100, 50), 0.9, "Text", 14), LayoutRegion((200, 0, 300, 50), 0.9, "Text", 14)]
    p = _bare_pipeline(results, regs, block_order=_FakeBlockOrder([1, 0]))
    out = p._ocr_page(np.zeros((100, 400, 3), np.uint8))
    assert [r["text"] for r in out] == ["R", "L"]  # порядок от модели, а не XY-cut (тот дал бы L, R)


def _block_order_or_skip():
    try:
        from occular.layout_order_model import BlockOrderModel
        return BlockOrderModel(num_threads=1)
    except Exception as e:  # нет весов/сети
        pytest.skip(f"веса порядка блоков недоступны: {e}")


def test_block_order_model_real_weights():
    m = _block_order_or_skip()
    rng = random.Random(0)
    for _ in range(30):
        n = rng.randint(0, 60)
        lines = [[x, y, x + rng.randint(5, 300), y + rng.randint(5, 40)]
                 for x, y in ((rng.uniform(0, 900), rng.uniform(0, 1300)) for _ in range(n))]
        regs = [LayoutRegion((x, y, x + rng.uniform(50, 500), y + rng.uniform(20, 400)), rng.uniform(0.6, 1),
                             LAYOUT_LABELS[k], k) for x, y, k in ((rng.uniform(0, 800), rng.uniform(0, 1200),
                                                                    rng.randint(1, 18)) for _ in range(rng.randint(0, 12)))]
        o1 = m.order_lines(lines, regs, 1000, 1400)
        assert_permutation(o1[0], n)
        assert o1 == m.order_lines(lines, regs, 1000, 1400)
    # простая двухколоночная страница: сначала вся левая колонка, потом правая
    left, right = column_lines(50, 450, 100, 10), column_lines(550, 950, 100, 10)
    regs = [LayoutRegion((40, 90, 460, 400), 0.95, "Text", 14), LayoutRegion((540, 90, 960, 400), 0.95, "Text", 14)]
    order, _, _ = m.order_lines(right + left, regs, 1000, 1400)
    assert order == list(range(10, 20)) + list(range(10))


# ---------- реальная модель (пропуск без весов) ----------

def _layout_or_skip():
    try:
        from occular import LayoutDetector
        return LayoutDetector()
    except Exception as e:  # нет весов/сети
        pytest.skip(f"layout-веса недоступны: {e}")


SAMPLE = Path(__file__).resolve().parents[1] / "sample_page.png"


def test_layout_detector_output_and_scaling():
    det = _layout_or_skip()
    from PIL import Image
    img = np.array(Image.open(SAMPLE).convert("RGB"))
    res = det.detect(img)
    assert (res.width, res.height) == (img.shape[1], img.shape[0])
    assert res.regions, "на образце страницы должны быть регионы"
    for r in res.regions:
        x1, y1, x2, y2 = r.bbox
        assert 0 <= x1 < x2 <= res.width and 0 <= y1 < y2 <= res.height
        assert r.score >= 0.6 and LAYOUT_LABELS[r.label_id] == r.label
    d = res.to_dict()
    assert set(d) == {"width", "height", "regions"} and set(d["regions"][0]) == {"bbox", "score", "label", "label_id"}
    # координаты в системе ПЕРЕДАННОЙ картинки: при 2× масштабе боксы растут ~2×
    big = np.array(Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2), Image.BILINEAR))
    res2 = det.detect(big)
    top1, top2 = res.regions[0], max(res2.regions, key=lambda r: r.score)
    assert top1.label == top2.label
    assert np.allclose(np.array(top2.bbox) / 2, np.array(top1.bbox), atol=0.02 * max(img.shape))
    # путь к файлу и PIL дают тот же результат, что numpy
    assert det.detect(str(SAMPLE)).to_dict() == res.to_dict()
