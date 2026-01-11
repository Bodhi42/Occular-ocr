# Model Weights

## Files

- `dbnet_weights.pth` - DBNet text detector (F1=0.8897, 48 MB)
- `crnn_mobilenet_large.pth` - CRNN MobileNetV3-Large recognizer (CER=1.53%, 53 MB)

## Notes

**Detection (DBNet):**
- Architecture: DBNet with ResNet18 backbone
- Hyperparameters: threshold=0.252, unclip_ratio=2.44, box_thresh=0.52, min_area=38

**Recognition (CRNN):**
- Architecture: CRNN with MobileNetV3-Large backbone
- Input: 32x512 RGB
- Vocab: 256 chars (ru/en/special)
