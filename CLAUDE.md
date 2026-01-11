# Occular OCR - Project Context

## Что это
OCR библиотека для русскоязычных документов. Оптимизирована под CPU (ONNX Runtime).

## Архитектура
- **Детектор**: DBNet (ResNet18) → находит текстовые области (quads)
- **Распознаватель**: CRNN (MobileNetV3-Large + BiLSTM) → CTC декодирование

## Ключевые файлы
```
ocr_skel/
├── pipeline.py          # OCRPipeline - главный класс
├── dbnet_detector.py    # PyTorch детектор
├── dbnet_detector_onnx.py  # ONNX детектор (ИДЕНТИЧЕН PyTorch!)
├── recognizer.py        # PyTorch распознаватель
├── recognizer_onnx.py   # ONNX распознаватель (батчинг с паддингом)
├── models/
│   ├── dbnet.py         # DBNet архитектура
│   └── crnn_mobilenet.py # CRNN архитектура
└── weights/
    ├── dbnet.onnx, dbnet_weights.pth
    └── crnn_encoder.onnx, crnn_mobilenet_large.pth
```

## Важные детали

### ONNX vs PyTorch детектор
- Preprocessing: `(h+31)//32*32` - округление ВВЕРХ до кратного 32
- Postprocessing: unclip контур → minAreaRect (не наоборот!)
- Order points: сортировка по Y, потом по X
- После фиксов - 0.000000 px разница

### ONNX распознаватель
- Батчинг с паддингом нулями влияет на LSTM → небольшие различия с PyTorch
- Это нормально, не баг

### Тренировка
Скрипты в `/home/user/dataset/recognition/`:
- `train_recognition_v3.py` - основной
- `train_stage2.py` - второй этап
- Используется Optuna для hyperparameter search

### Датасеты
- `recognition_dataset_surya/` - реальные кропы (важно!)
- `recognition_dataset_synthetic_new/` - синтетика (52GB)
- `recognition_v3/` - финальный датасет (112GB)

Архивы на Яндекс.Диске: `/home/user/Yandex.Disk/ocr_datasets/`

## API
```python
from ocr_skel import ocr, OCRPipeline

# Простой
text = ocr("image.png")

# Продвинутый
pipeline = OCRPipeline(onnx=True, gpu=False)
results = pipeline.process_image("image.png")
```

## Лицензия
- Код: GPL-3.0
- Веса: Custom (бесплатно для <20M RUB выручки и <8 сотрудников)
