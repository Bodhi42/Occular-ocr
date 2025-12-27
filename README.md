# OCR Skeleton

Минимальный OCR pipeline с CRAFT детектором и CRNN распознавателем.

## Установка

```bash
pip install -r requirements.txt
```

## Использование

### CLI

```bash
# Базовое использование (вывод JSON в stdout)
python -m ocr_skel.cli --image path/to/image.png

# Сохранение результата в файл
python -m ocr_skel.cli --image path/to/image.png --out result.json

# Вывод только текста
python -m ocr_skel.cli --image path/to/image.png --print-text

# Использование GPU
python -m ocr_skel.cli --image path/to/image.png --gpu --out result.json
```

### Python API

```python
from ocr_skel import Registry, CRAFTDetector, CRNNRecognizer, OCRPipeline

# Регистрация компонентов
Registry.register_detector("craft", CRAFTDetector)
Registry.register_recognizer("crnn", CRNNRecognizer)

# Создание pipeline
pipeline = OCRPipeline(
    detector_name="craft",
    recognizer_name="crnn",
    detector_kwargs={"gpu": False},
    recognizer_kwargs={"gpu": False}
)

# Обработка изображения
results = pipeline.process_image("image.png")

# Результат: [{"quad": [[x,y], ...], "text": "...", "confidence": 0.95}, ...]
```

## Тестирование

```bash
pytest -q
```

## Структура

- `ocr_skel/registry.py` - реестр детекторов и распознавателей
- `ocr_skel/detector.py` - CRAFT детектор (pretrained)
- `ocr_skel/recognizer.py` - CRNN распознаватель (pretrained)
- `ocr_skel/pipeline.py` - OCR pipeline (detect -> recognize)
- `ocr_skel/cli.py` - CLI интерфейс
- `tests/test_smoke.py` - smoke тесты

## Формат вывода

JSON массив объектов:

```json
[
  {
    "quad": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "text": "распознанный текст",
    "confidence": 0.95
  }
]
```
