"""Весь вывод пакета (print, предупреждения, исключения, справка CLI) — на английском (issue #5).
Строковые литералы с кириллицей допустимы только в docstring/комментариях и в данных из ALLOWED."""
import ast
import re
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "occular"
CYR = re.compile(r"[А-Яа-яЁё]")
ALLOWED = {"code_lock.py"}            # набор символов кириллицы — данные, не сообщение


def test_no_cyrillic_in_messages():
    bad = []
    for f in sorted(PKG.rglob("*.py")):
        if f.name in ALLOWED:
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        docs = {id(n.body[0].value) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs and CYR.search(n.value):
                bad.append(f"{f.name}:{n.lineno}")
    assert not bad, "русский текст в выводимых строках: " + ", ".join(bad)
