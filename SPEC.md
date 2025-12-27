# SPEC: OCR skeleton
Нужно:
- Реестр детекторов и распознавателей (выбор по имени, дефолты)
- Detector: CRAFT (pretrained)
- Recognizer: CRNN (ResNet+LSTM+CTC) через готовую pretrained реализацию
- Pipeline: detect -> recognize -> вывод
- CLI: --image, --detector, --recognizer, --out/--print-text
- 1 smoke test

Ограничения:
- Минимально рабочее решение, без дообучения
- Маленькие патчи, без “переписать всё”
