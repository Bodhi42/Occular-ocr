# DBNet (Differentiable Binarization Network) Detector

## Обзор

DBNet — это современный детектор текста на основе сегментации, который использует дифференцируемую бинаризацию для улучшения точности обнаружения текстовых регионов.

**Статья:** [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)

## Архитектура

1. **Backbone:** ResNet-18 или ResNet-50 (pretrained на ImageNet)
2. **Neck:** Feature Pyramid Network (FPN) для многоуровневых признаков
3. **Head:** DBNet head с тремя картами:
   - Probability map (вероятность текста)
   - Threshold map (адаптивный порог)
   - Binary map (бинарная маска текста)

## Differentiable Binarization

Ключевая идея DBNet — использование дифференцируемой бинаризации:

```
binary_map = 1 / (1 + exp(-k * (prob_map - thresh_map)))
```

Где:
- `k` — коэффициент усиления (по умолчанию 50)
- `prob_map` — карта вероятности текста
- `thresh_map` — адаптивная карта порогов

## Использование

### Базовое использование

```python
from ocr_skel import Registry
import numpy as np
from PIL import Image

# Регистрация (автоматически при импорте)
detector = Registry.get_detector("dbnet", gpu=False)

# Загрузка изображения
img = Image.open("image.png").convert("RGB")
img_array = np.array(img)

# Детекция текста
quads = detector.detect(img_array)

# quads — список 4-точечных полигонов (квадов)
for quad in quads:
    print(quad)  # shape: (4, 2) - [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
```

### Выбор backbone

```python
from ocr_skel.dbnet_detector import DBNetDetector

# ResNet-18 (легче, быстрее)
detector = DBNetDetector(gpu=False, backbone='resnet18')

# ResNet-50 (точнее, медленнее)
detector = DBNetDetector(gpu=False, backbone='resnet50')
```

### Интеграция с Pipeline

```python
from ocr_skel.pipeline import OCRPipeline

# Использование DBNet вместо CRAFT
pipeline = OCRPipeline(
    detector_name="dbnet",
    recognizer_name="crnn",
    detector_kwargs={"gpu": False, "backbone": "resnet18"},
    recognizer_kwargs={"gpu": False}
)

results = pipeline.process_image("image.png")
```

## Файлы

### Новые файлы:
- `ocr_skel/models/dbnet_model.py` — архитектура DBNet на PyTorch
- `ocr_skel/dbnet_detector.py` — обёртка детектора
- `tests/test_dbnet.py` — smoke-тесты
- `download_dbnet_weights.py` — инструкция по загрузке весов

### Изменённые файлы:
- `ocr_skel/models/__init__.py` — экспорт DBNet
- `ocr_skel/models/model_utils.py` — функция load_dbnet_weights
- `ocr_skel/__init__.py` — регистрация DBNetDetector

## Веса (Pretrained Weights)

### Текущее состояние
DBNet работает с ImageNet pretrained ResNet backbone без дополнительных весов. Для production рекомендуется загрузить полные pretrained веса.

### Источники весов:

1. **Original DBNet repo:** https://github.com/MhLiao/DB
   - Официальные веса, обученные на SynthText + MLT + ICDAR2015

2. **mmocr (OpenMMLab):** https://github.com/open-mmlab/mmocr
   - Веса в формате PyTorch, готовые к использованию

3. **PaddleOCR:** https://github.com/PaddlePaddle/PaddleOCR
   - Требуется конвертация из PaddlePaddle в PyTorch

### Установка весов

1. Скачайте веса из одного из источников
2. Сохраните как `ocr_skel/weights/dbnet_resnet18.pth`
3. Веса автоматически загрузятся при инициализации

```bash
# Пример (если бы были прямые ссылки)
cd ocr_skel/weights
curl -L -o dbnet_resnet18.pth "URL_TO_WEIGHTS"
```

## Производительность

### Без pretrained весов (только ImageNet backbone):
- ✅ Работает
- ⚠️ Средняя точность (без обучения на текстовых датасетах)
- ✅ Подходит для тестирования и прототипирования

### С pretrained весами:
- ✅ Высокая точность
- ✅ Production-ready
- ✅ Сопоставимо с CRAFT

## Сравнение с CRAFT

| Характеристика | CRAFT | DBNet |
|---------------|-------|-------|
| Backbone | VGG16 | ResNet-18/50 |
| Метод | Character-level | Segmentation |
| Скорость | Средняя | Быстрая |
| Точность | Высокая | Высокая |
| FPS (GPU) | ~10-15 | ~20-30 |

## Тестирование

Запустить smoke-тесты:

```bash
pytest tests/test_dbnet.py -v
```

Все 4 теста должны пройти:
- ✅ test_dbnet_registration
- ✅ test_dbnet_detect
- ✅ test_dbnet_empty_image
- ✅ test_dbnet_with_craft_compatibility

## Ограничения и TODO

### Текущие ограничения:
1. Без pretrained весов точность ниже оптимальной
2. Постобработка упрощена (базовый cv2.findContours)
3. Параметры детекции зафиксированы (min_area, min_score)

### Возможные улучшения:
1. Добавить полные pretrained веса
2. Улучшить постобработку (polygon refinement)
3. Добавить параметры в detect() (thresholds, min_area, etc.)
4. Поддержка батчевой обработки
5. Экспорт в ONNX для deployment

## Примеры

См. также:
- `demo.ipynb` — интерактивные примеры
- `tests/test_dbnet.py` — примеры использования в коде
- `SPEC.md` — общая спецификация проекта

## Лицензия

DBNet — open source (см. оригинальный репозиторий для деталей лицензии)
