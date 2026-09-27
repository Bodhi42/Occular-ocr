"""Замок буквенно-цифровых кодов для beam+LM декода.

Зачем: n-gram языковая модель не знает конкретных чисел и кодов (номера счетов, ИНН/КПП,
серии документов, VIN, артикулы) и может «поправить» их к похожим словам — испортив и цифры,
и буквы внутри кода (напр. расчётный счёт «...10041» → «...004Т», «188» → «18»).

Решение: любой максимальный alnum-ран (буквы/цифры/-/./№/слэш), содержащий хотя бы одну цифру,
берётся ЦЕЛИКОМ из акустического чтения рекогнайзера (greedy-argmax) — LM его не трогает.
Обычные слова (без цифр) LM правит как раньше. На общем CER цена ~0, но код никогда не искажается.

Без внешних зависимостей: выравнивание строк — на стандартном difflib.
"""
from difflib import SequenceMatcher

# символы, из которых состоит «код»: буквы (лат+кир), цифры и типичные разделители кодов
_CODECH = set(
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "-/.№"
)


def _protected_mask(s: str, min_digits: int = 2, min_ratio: float = 0.5) -> list:
    """True для символов код-рана. Код = alnum-ран, который ЦИФРО-ДОМИНАНТНЫЙ:
    ≥min_digits цифр И доля цифр среди букв+цифр ≥min_ratio. Так защищаются реальные
    номера/серии/ИНН/суммы, но НЕ обычные слова со случайной OCR-цифрой (иначе слово
    целиком бралось из greedy и портилось: 'судебного'→'сулебног0', '304'→'Зг0уч')."""
    n = len(s)
    m = [False] * n
    i = 0
    while i < n:
        if s[i] in _CODECH:
            j = i
            while j < n and s[j] in _CODECH:
                j += 1
            run = s[i:j]
            nd = sum(c.isdigit() for c in run)
            nal = sum(c.isalnum() for c in run)
            if nd >= min_digits and nal >= 2 and nd >= nal * min_ratio:
                for k in range(i, j):
                    m[k] = True
            i = j
        else:
            i += 1
    return m


def lock_codes(beam: str, greedy: str) -> str:
    """Вернуть beam, где каждый код-ран заменён на его акустическое (greedy) чтение.

    beam   — текст из beam+LM (буквы-слова уже исправлены LM);
    greedy — акустическое argmax-чтение того же изображения (источник истины для кодов).
    Гарантия: цифро-содержащие alnum-раны из greedy присутствуют в результате дословно.
    """
    pm = _protected_mask(greedy)
    if not any(pm):
        return beam
    out = []
    # opcodes преобразуют greedy (a) -> beam (b); i = позиции в greedy, j = в beam
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, greedy, beam, autojunk=False).get_opcodes():
        gseg, bseg = greedy[i1:i2], beam[j1:j2]
        prot = any(pm[k] for k in range(i1, i2))
        if tag == "equal":
            out.append(bseg)
        elif tag == "replace":
            out.append(gseg if prot else bseg)          # затронут код -> акустика
        elif tag == "delete":
            if prot:                                     # LM выкинул кусок кода -> вернуть
                out.append(gseg)
        elif tag == "insert":
            left = pm[i1 - 1] if i1 - 1 >= 0 else False
            right = pm[i1] if i1 < len(pm) else False
            if not (left and right):                     # вставка внутри кода -> выбросить
                out.append(bseg)
    return "".join(out)


def greedy_ctc(logits_tc, chars) -> str:
    """Акустическое greedy-CTC чтение: argmax по кадрам, схлопнуть повторы, убрать blank(0).

    logits_tc — [T, C] (логиты или лог-вероятности, argmax одинаков); chars — список символов
    словаря (индекс класса k>=1 соответствует chars[k-1], 0 = CTC blank).
    """
    idx = logits_tc.argmax(1)
    out = []
    prev = -1
    for k in idx:
        k = int(k)
        if k != prev and k != 0:
            out.append(chars[k - 1])
        prev = k
    return "".join(out)
