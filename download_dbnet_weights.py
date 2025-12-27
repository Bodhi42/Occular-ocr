"""Download DBNet pretrained weights"""
import os
import urllib.request
from pathlib import Path

# DBNet ResNet18 weights from PaddleOCR converted to PyTorch
# Alternative sources:
# 1. Original repo: https://github.com/MhLiao/DB
# 2. mmocr: https://github.com/open-mmlab/mmocr
# 3. PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR

# For now, we'll create a placeholder
# Real weights would need to be downloaded from one of these sources

weights_dir = Path(__file__).parent / "ocr_skel" / "weights"
weights_path = weights_dir / "dbnet_resnet18.pth"

print("DBNet weights download script")
print("=" * 60)
print("Note: DBNet pretrained weights need to be manually downloaded")
print("from one of these sources:")
print()
print("1. Original DBNet repo:")
print("   https://github.com/MhLiao/DB")
print()
print("2. mmocr (OpenMMLab):")
print("   https://github.com/open-mmlab/mmocr")
print()
print("3. PaddleOCR (convert PaddlePaddle to PyTorch):")
print("   https://github.com/PaddlePaddle/PaddleOCR")
print()
print(f"Save weights to: {weights_path}")
print()
print("For testing purposes, DBNet works with ImageNet pretrained")
print("ResNet backbone without additional weights.")
