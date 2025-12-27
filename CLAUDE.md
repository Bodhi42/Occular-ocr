- Всегда общаемся по-русски. Английский только в код-блоках и в сырых логах.

# DoD
- CLI не падает: `python -m ocr_skel.cli --image <path> --out out.json`
- out.json валиден: список {quad, text, confidence}
- `pytest -q` проходит
- Нельзя говорить PASS/“готово” без LOG BUNDLE от verifier

# Команды
- Установка: `pip install -r requirements.txt`
- Тесты: `pytest -q`
