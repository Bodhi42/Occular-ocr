"""Тесты замка кодов: коды из акустики сохраняются дословно, слова LM правит, greedy_ctc корректен."""
import re
import importlib.util
from pathlib import Path

# импорт модуля напрямую (без тяжёлого occular.__init__ с onnxruntime)
_p = Path(__file__).resolve().parent.parent / "occular" / "code_lock.py"
_spec = importlib.util.spec_from_file_location("occular_code_lock", _p)
cl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cl)

_CODE = re.compile(r"[0-9A-Za-zА-Яа-яЁё№/\-.]{2,}")


def _codes(s):
    return [m.group(0) for m in _CODE.finditer(s) if any(c.isdigit() for c in m.group(0))]


def _guarantee(beam, greedy):
    out = cl.lock_codes(beam, greedy)
    for c in _codes(greedy):
        assert c in out, f"код {c!r} потерян: greedy={greedy!r} beam={beam!r} -> {out!r}"
    return out


def test_digit_dropped_restored():
    assert _guarantee("сумма 18 руб", "сумма 188 руб") == "сумма 188 руб"


def test_account_number_not_mangled():
    # LM превратил цифры счёта в букву — замок возвращает акустику
    out = _guarantee("р/с:40101810800000004Т, банк", "р/с:4010181080000010041, банк")
    assert "4010181080000010041" in out


def test_letter_in_code_protected():
    # серия документа: буквы+цифры не трогаются
    assert _guarantee("VIII-МЮ № 79639", "VIII-МЮ № 796392") == "VIII-МЮ № 796392"


def test_inn_kpp_digits_kept():
    out = _guarantee("ИНН/КПП организации", "ИНН2/КПП3 организации")
    assert "ИНН2" in out and "КПП3" in out


def test_plain_word_still_corrected():
    # обычное слово без цифр — LM-исправление остаётся (замок не вмешивается)
    assert cl.lock_codes("обычное предложение", "обычнае предложение") == "обычное предложение"


def test_no_codes_returns_beam():
    assert cl.lock_codes("привет мир", "привет мор") == "привет мир"


def test_greedy_ctc_collapse_blank():
    import numpy as np
    chars = list("аб")                      # класс1='а', класс2='б', 0=blank
    # кадры: а, а(повтор), blank, б  -> "аб"
    logits = np.array([[0, 9, 0], [0, 9, 0], [9, 0, 0], [0, 0, 9]], np.float32)
    assert cl.greedy_ctc(logits, chars) == "аб"


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"\n{len(fns)}/{len(fns)} ПРОЙДЕНО")
