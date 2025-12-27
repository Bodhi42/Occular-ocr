# ViTSTR Recognizer Usage

ViTSTR (Vision Transformer for Scene Text Recognition) - это современный распознаватель текста на основе архитектуры Vision Transformer.

## Быстрый старт

### Прямое использование

```python
from ocr_skel import Registry
from ocr_skel.vitstr_recognizer import ViTSTRRecognizer

# Через Registry
recognizer = Registry.get_recognizer("vitstr", gpu=False)

# Или напрямую
recognizer = ViTSTRRecognizer(languages=['en'], gpu=False)

# Распознавание
import numpy as np
image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
quads = [np.array([[10, 10], [200, 10], [200, 50], [10, 50]])]
results = recognizer.recognize(image, quads)

for text, confidence in results:
    print(f"Text: {text}, Confidence: {confidence:.3f}")
```

### Использование с Pipeline

```python
from ocr_skel import OCRPipeline

# Создать pipeline с ViTSTR
pipeline = OCRPipeline(
    detector_name="craft",
    recognizer_name="vitstr",
    detector_kwargs={"gpu": False},
    recognizer_kwargs={"gpu": False}
)

# Обработать изображение
results = pipeline.process_image("image.png")

for result in results:
    print(f"Text: {result['text']}, Confidence: {result['confidence']:.3f}")
```

## Архитектура

- **Model**: Vision Transformer (ViT-Small)
- **Encoder**: 12 transformer blocks
- **Embedding dim**: 384
- **Attention heads**: 6
- **Patch size**: 16x16
- **Input size**: 224x224 (grayscale)
- **Output**: Character sequence (CTC decoding)

## Параметры

- `languages` (List[str]): Список языков (по умолчанию: ['en'])
- `gpu` (bool): Использовать GPU если доступен (по умолчанию: True)

## Веса модели

Веса модели опциональны. Модель будет работать с случайной инициализацией если веса не найдены.

Для загрузки pretrained весов:
- Источник: HuggingFace (roatienza/deep-text-recognition-benchmark)
- Файл: `ocr_skel/weights/vitstr_small.pth`
- Размер: ~85 MB

## Сравнение с CRNN

| Feature | CRNN | ViTSTR |
|---------|------|--------|
| Архитектура | CNN + BiLSTM + CTC | Vision Transformer + CTC |
| Input size | 32 x variable | 224 x 224 (fixed) |
| Параметры | ~8M | ~21M |
| Скорость | Быстрее | Медленнее |
| Точность | Хорошая | Потенциально выше |

## Тестирование

Запустить smoke-тест:
```bash
python test_vitstr_smoke.py
```

Запустить интеграционный тест:
```bash
python test_vitstr_integration.py
```

## Ограничения

- Фиксированный размер входа (224x224), что может привести к потере качества для длинных текстов
- Более ресурсоёмкий чем CRNN
- Требует больше памяти (GPU/CPU)

## Примеры

См. `test_vitstr_smoke.py` и `test_vitstr_integration.py` для детальных примеров использования.
