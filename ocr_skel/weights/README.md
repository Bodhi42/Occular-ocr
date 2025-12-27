# Model Weights

This directory contains pretrained weights for OCR models.

## Files

- `craft_mlt_25k.pth` - CRAFT text detector weights (79 MB)
- `crnn.pth` - CRNN text recognizer weights (32 MB)
- `dbnet_resnet18.pth` - DBNet text detector weights with ImageNet backbone (63 MB)
- `vitstr_small.pth` - ViTSTR text recognizer weights (ViT-Small, 82 MB)

## Current Status

All weights are available and loaded without warnings:

- ✅ CRAFT: Pretrained on MLT dataset
- ✅ CRNN: Pretrained on text recognition datasets
- ✅ DBNet: ImageNet pretrained ResNet18 backbone + initialized FPN/head
- ✅ ViTSTR: Initialized ViT-Small architecture

## Download Sources

**CRAFT weights:**
```bash
curl -L -o craft_mlt_25k.pth "https://drive.google.com/uc?export=download&id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ"
```

**CRNN weights:**
```bash
curl -L -o crnn.pth "https://www.dropbox.com/s/dboqjk20qjkpta3/crnn.pth?dl=1"
```

**DBNet weights:**
- Created from ImageNet pretrained ResNet18 backbone
- FPN and detection head initialized with Kaiming normal
- For fully pretrained weights, check:
  - https://github.com/MhLiao/DB (original implementation)
  - https://github.com/WenmuZhou/DBNet.pytorch (PyTorch reimplementation)

**ViTSTR weights:**
- Initialized ViT-Small architecture (37 classes: 0-9, a-z, blank)
- For pretrained weights, check:
  - https://github.com/roatienza/deep-text-recognition-benchmark
  - HuggingFace: roatienza models

## Notes

- All models load without warnings or errors
- DBNet uses ImageNet pretrained backbone (suitable for transfer learning)
- ViTSTR uses initialized weights (suitable for fine-tuning)
- Total weights size: ~257 MB
