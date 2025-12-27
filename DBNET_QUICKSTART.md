# DBNet Quick Start

## Быстрый старт

### 1. Базовое использование

```python
from ocr_skel import Registry
import numpy as np
from PIL import Image

# Создать детектор DBNet
detector = Registry.get_detector("dbnet", gpu=False)

# Загрузить изображение
img = Image.open("image.png").convert("RGB")
img_array = np.array(img)

# Детектировать текстовые регионы
quads = detector.detect(img_array)

# Результат: список квадов [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
for quad in quads:
    print(quad)
```

### 2. Использование с Pipeline

```python
from ocr_skel.pipeline import OCRPipeline

# Создать pipeline с DBNet детектором
pipeline = OCRPipeline(
    detector_name="dbnet",
    recognizer_name="crnn",
    detector_kwargs={"gpu": False, "backbone": "resnet18"},
    recognizer_kwargs={"gpu": False}
)

# Обработать изображение
results = pipeline.process_image("image.png")

# Результат: список с распознанным текстом
for item in results:
    print(f"Text: {item['text']}, Confidence: {item['confidence']}")
```

### 3. CLI использование

```bash
# Использовать DBNet вместо CRAFT
python3 -m ocr_skel.cli --image test.png --detector dbnet --out result.json

# Вывести текст в консоль
python3 -m ocr_skel.cli --image test.png --detector dbnet --print-text
```

## Доступные backbone

- `resnet18` (по умолчанию) - быстрее, легче
- `resnet50` - точнее, но медленнее

```python
from ocr_skel.dbnet_detector import DBNetDetector

# ResNet-50 для лучшей точности
detector = DBNetDetector(gpu=False, backbone='resnet50')
```

## Текущее состояние

✅ **Работает:**
- Архитектура DBNet реализована
- Регистрация в Registry
- Интеграция с Pipeline
- Smoke-тесты проходят
- ImageNet pretrained backbone

⚠️ **Ограничения:**
- Без pretrained весов на текстовых датасетах
- Точность ниже оптимальной
- Подходит для прототипирования

📦 **Для production:**
- Скачайте pretrained веса (см. DBNET_INFO.md)
- Сохраните в `ocr_skel/weights/dbnet_resnet18.pth`
- Веса загрузятся автоматически

## Тестирование

```bash
# Запустить smoke-тесты
pytest tests/test_dbnet.py -v

# Проверить все тесты
pytest tests/ -v
```

## Документация

- `DBNET_INFO.md` - полная документация
- `tests/test_dbnet.py` - примеры кода
- `download_dbnet_weights.py` - инструкция по загрузке весов
