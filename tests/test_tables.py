"""Тесты распознавания таблиц.

Сшивка полос высоких таблиц проверяется без весов: main-граф подменяется заглушкой, которая
«видит» строки синтетической таблицы с известными границами. Тесты с реальными моделями
пропускаются, если веса недоступны.
"""
import numpy as np
import pytest

import occular.tables as T


class _FakeMain:
    """Заглушка main-графа: строки высотой ROW px по всему кропу, 3 столбца. Сепараторы ставятся на
    границы строк, попавшие внутрь полосы; полоса определяется по высоте кропа через глобальный указатель."""
    ROW = 40

    def __init__(self, nmax=40, lp=192):
        self.nmax, self.lp = nmax, lp
        self.offset = 0                       # верх текущего кропа в координатах всей таблицы

    def run(self, _names, feeds):
        h = feeds["x"].shape[-2]              # всегда 768 — реальную высоту берём из self.cur_h
        hc = self.cur_h
        bounds = [y - self.offset for y in range(self.ROW, 10 ** 6, self.ROW) if 0 < y - self.offset < hc]
        rsep = np.full(self.lp, -9.0, np.float32)
        for b in bounds:
            rsep[min(self.lp - 1, int(b / hc * self.lp))] = 9.0
        rcnt = np.full(self.nmax, -9.0, np.float32); rcnt[min(self.nmax - 1, len(bounds))] = 9.0
        csep = np.full(self.lp, -9.0, np.float32); csep[self.lp // 3] = csep[2 * self.lp // 3] = 9.0
        ccnt = np.full(self.nmax, -9.0, np.float32); ccnt[2] = 9.0
        return rcnt[None], rsep[None], ccnt[None], csep[None], np.zeros((1, 4, 8, 8), np.float32)


def _recognizer_with_fake():
    tr = T.TableRecognizer.__new__(T.TableRecognizer)
    tr._main = _FakeMain(); tr._merge = None; tr._nmax_r = tr._nmax_c = 40
    orig_split = T.TableRecognizer._split

    def split(self, crop):
        self._main.cur_h = crop.shape[0]
        return orig_split(self, crop)
    tr._split = split.__get__(tr)
    return tr


def test_strip_rows_recover_all_rows_of_tall_table():
    tr = _recognizer_with_fake()
    H, W = 2000, 300                          # 50 строк по 40 px, высота ≫ 0.5·ширины → полосы
    crop = np.full((H, W, 3), 255, np.uint8)
    # верх каждой полосы выставляется в _strip_rows через срез; заглушке нужен оффсет — оборачиваем срезы
    real_split = tr._split

    def split_with_offset(c):
        # найти, какой срез кропа передан: по адресу памяти относительно исходного массива
        off = (c.__array_interface__["data"][0] - crop.__array_interface__["data"][0]) // crop.strides[0]
        tr._main.offset = int(off)
        return real_split(c)
    tr._split = split_with_offset
    rows = tr._strip_rows(crop)
    assert len(rows) == H // _FakeMain.ROW
    assert rows[0][0] == 0.0 and rows[-1][1] == float(H)
    inner = np.array([b for _, b in rows[:-1]])
    assert np.abs(inner - np.arange(_FakeMain.ROW, H, _FakeMain.ROW)).max() < 6   # разрешение сепараторов


def test_short_table_does_not_use_strips():
    tr = _recognizer_with_fake()
    called = []
    tr._strip_rows = lambda c: called.append(1) or []
    tr._main.offset = 0
    st = tr.structure(np.full((120, 400, 3), 255, np.uint8))   # высота ≤ 0.5·ширины
    assert not called and len(st["rows"]) == 3 and len(st["cols"]) == 3 and st["cells"] == []


def _real_or_skip():
    try:
        return T.TableRecognizer(num_threads=2)
    except Exception as e:                    # нет весов локально и нет сети
        pytest.skip(f"веса таблиц недоступны: {e}")


def test_real_models_on_synthetic_grid():
    import cv2
    tr = _real_or_skip()
    page = np.full((1400, 1000, 3), 255, np.uint8)
    x0, y0, cw, rh, R, C = 100, 300, 200, 40, 6, 4
    for i in range(R + 1):
        cv2.line(page, (x0, y0 + i * rh), (x0 + C * cw, y0 + i * rh), (0, 0, 0), 2)
    for j in range(C + 1):
        cv2.line(page, (x0 + j * cw, y0), (x0 + j * cw, y0 + R * rh), (0, 0, 0), 2)
    for i in range(R):
        for j in range(C):
            cv2.putText(page, f"{i * C + j + 10}", (x0 + j * cw + 20, y0 + i * rh + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    st = tr.structure(page[y0:y0 + R * rh, x0:x0 + C * cw])
    assert len(st["rows"]) == R and len(st["cols"]) == C
    assert len(st["cells"]) == R * C
    assert isinstance(tr.detect(page), list)


# ---- таблицы в OCR-конвейере (Settings(tables=True)) ----

def _found_2x3():
    """Таблица 2×3 в пикселях страницы: в первой строке ячейки (0,1)+(0,2) объединены."""
    return [{"bbox": (100, 100, 400, 160, 0.9),
             "rows": [(100, 130), (130, 160)], "cols": [(100, 200), (200, 300), (300, 400)],
             "cells": [dict(r=0, c=0, rowspan=1, colspan=1), dict(r=0, c=1, rowspan=1, colspan=2),
                       dict(r=1, c=0, rowspan=1, colspan=1), dict(r=1, c=1, rowspan=1, colspan=1),
                       dict(r=1, c=2, rowspan=1, colspan=1)]}]


def _line(x0, y0, x1, y1, text):
    return {"quad": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "text": text, "confidence": 0.9}


def test_attach_lines_cells_and_markdown():
    lines = [_line(10, 20, 200, 40, "Заголовок"),
             _line(110, 105, 180, 125, "Товар"), _line(210, 105, 390, 125, "Сумма | с НДС"),
             _line(110, 135, 150, 155, "Бумага"), _line(152, 135, 190, 155, "А4"),
             _line(210, 135, 290, 155, "10"), _line(310, 135, 390, 155, "350,00"),
             _line(10, 200, 200, 220, "Итого")]
    tables = T.attach_lines(_found_2x3(), lines)
    assert len(tables) == 1 and tables[0]["n_rows"] == 2 and tables[0]["n_cols"] == 3
    assert "table" not in lines[0] and "table" not in lines[-1]
    assert lines[2]["cell"] == {"r": 0, "c": 1, "rowspan": 1, "colspan": 2}
    texts = {(c["r"], c["c"]): c["text"] for c in tables[0]["cells"]}
    assert texts[(1, 0)] == "Бумага А4"                      # куски одной строки — слева направо
    md = T.table_markdown(tables[0])
    assert md.splitlines() == ["| Товар | Сумма \\| с НДС |  |", "|---|---|---|", "| Бумага А4 | 10 | 350,00 |"]
    text = T.compose_text(lines, tables)
    assert text == "Заголовок\n\n" + md + "\n\nИтого"


def test_attach_lines_scale_for_pdf_points():
    lines = [_line(55, 52.5, 90, 62.5, "x")]                  # точки PDF = пиксели / 2
    tables = T.attach_lines(_found_2x3(), lines, scale=0.5)
    assert tables[0]["bbox"] == [50.0, 50.0, 200.0, 80.0]
    assert lines[0]["cell"]["r"] == 0 and lines[0]["cell"]["c"] == 0


def test_split_line_crossing_cell_boundary():
    class Rec:                                                # без logits_per_line — разрез пропорционально тексту
        def recognize(self, image, quads):
            raise AssertionError("куски не перераспознаются")
    found = _found_2x3()
    img = np.full((300, 500, 3), 255, np.uint8)
    # 2-я строка таблицы: три куска текста с пробелами у границ x=200 и x=300 → два разреза
    for a, b in ((112, 192), (208, 292), (306, 388)):
        img[139:151, a:b] = 0
    # 1-я строка: текст сплошняком через границу x=200 (вылез из ячейки) — пробела нет, не режем
    img[109:121, 112:388] = 0
    img[100:160, 199:201] = 0                                 # вертикальная линейка таблицы
    lines = [_line(110, 135, 390, 155, "Бумага 10шт 350,00"),
             _line(110, 105, 390, 125, "через границу без пробела")]
    out = T.split_crossing_lines(found, lines, img, Rec())
    assert [l["text"] for l in out] == ["Бумага", "10шт", "350,00", "через границу без пробела"]
    assert all(l.get("split") for l in out[:3])
    assert T.cell_boundaries(found[0], 140) == [200.0, 300.0]
    assert T.cell_boundaries(found[0], 110) == [200.0]


def test_text_cuts_never_inside_a_word():
    # «2 Кабель»: оценка на символ правее («2 К|абель») уходит на пробел; внутри слова не режем
    assert T._text_cuts("2 Кабель", [0.35], None, None) == [1]
    # стык знака и цифры — можно («документа:|28»)
    assert T._text_cuts("документа:28", [0.83], None, None) == [10]
    # сплошное слово без места для разреза — не режем вовсе
    assert T._text_cuts("Электротехническое", [0.5], None, None) == []


def test_pipeline_tables_setting(tmp_path):
    import cv2
    _real_or_skip()
    from occular import OCRPipeline, Settings
    page = np.full((1400, 1000, 3), 255, np.uint8)
    x0, y0, cw, rh, R, C = 100, 300, 200, 40, 6, 4
    for i in range(R + 1):
        cv2.line(page, (x0, y0 + i * rh), (x0 + C * cw, y0 + i * rh), (0, 0, 0), 2)
    for j in range(C + 1):
        cv2.line(page, (x0 + j * cw, y0), (x0 + j * cw, y0 + R * rh), (0, 0, 0), 2)
    for i in range(R):
        for j in range(C):
            cv2.putText(page, f"{i * C + j + 10}", (x0 + j * cw + 20, y0 + i * rh + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    path = tmp_path / "grid.png"
    cv2.imwrite(str(path), page)
    try:
        p = OCRPipeline(Settings(tables=True, lm=False))
    except Exception as e:
        pytest.skip(f"модели OCR недоступны: {e}")
    res = p.process_image(str(path))
    assert len(res.tables) == 1
    t = res.tables[0]
    assert (t["n_rows"], t["n_cols"]) == (R, C)
    got = {(c["r"], c["c"]): c["text"].strip() for c in t["cells"]}
    ok = sum(got.get((i, j)) == str(i * C + j + 10) for i in range(R) for j in range(C))
    assert ok >= R * C - 2, got                                # допускаем пару ошибок распознавания
    assert "| 10 | 11 | 12 | 13 |" in p.get_text(str(path))
