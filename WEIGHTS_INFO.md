# Pretrained Model Weights

## Overview

This project now includes pretrained weights for CRAFT and CRNN models stored locally in `ocr_skel/weights/`.

## Files

- **CRAFT** (Text Detector): `craft_mlt_25k.pth` (79.3 MB)
  - Source: Original CRAFT-pytorch implementation
  - Architecture: VGG16-based with UNet decoder
  
- **CRNN** (Text Recognizer): `crnn.pth` (31.8 MB)
  - Source: Original CRNN.pytorch implementation
  - Architecture: CNN + BiLSTM + CTC

## Changes Made

### 1. Directory Structure
```
ocr_skel/
├── weights/
│   ├── craft_mlt_25k.pth (79.3 MB)
│   ├── crnn.pth (31.8 MB)
│   ├── README.md
│   └── .gitignore
```

### 2. Updated Files

**ocr_skel/models/model_utils.py**
- Removed: `download_weights()` function and internet URLs
- Changed: `load_craft_weights()` and `load_crnn_weights()` now load from local files
- Added: `get_weights_path()` helper for local weight paths
- Added: Support for PyTorch 2.6+ with `weights_only=False`
- Added: Automatic removal of 'module.' prefix from DataParallel weights

**ocr_skel/models/craft_model.py**
- Updated: Architecture to match pretrained weights exactly
- Changed: VGG backbone split into 5 slices (slice1-slice5)
- Fixed: Decoder dimensions to match pretrained (1536→256, 768→128, 384→64, 192→32)

**ocr_skel/models/__init__.py**
- Updated: Exports to reflect new API (removed `download_weights`)

**ocr_skel/__init__.py**
- Added: Auto-registration of CRAFT and CRNN implementations

### 3. Weight Sources

**CRAFT weights downloaded from:**
```bash
curl -L -o craft_mlt_25k.pth "https://drive.google.com/uc?export=download&id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ"
```

**CRNN weights downloaded from:**
```bash
curl -L -o crnn.pth "https://www.dropbox.com/s/dboqjk20qjkpta3/crnn.pth?dl=1"
```

## Verification

Run the test script to verify weights are loaded correctly:

```bash
python3 test_local_weights.py
```

Expected output:
```
✓ CRAFT weights: .../craft_mlt_25k.pth (79.3 MB)
✓ CRNN weights: .../crnn.pth (31.8 MB)
✓ All models loaded with local pretrained weights
✅ SUCCESS: No internet downloads required!
```

## Benefits

1. **No Internet Required**: Weights are bundled, no download at runtime
2. **Faster Startup**: No waiting for downloads
3. **Reproducible**: Exact same weights every time
4. **Offline Support**: Works without network connection

## Notes

- Weights are excluded from git via `.gitignore` (too large for repositories)
- For distribution, include weights in release packages or provide download script
- Total weights size: ~111 MB
