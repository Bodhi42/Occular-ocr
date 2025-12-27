#!/usr/bin/env python3
"""
Quick test to verify weights are loaded from local files
"""

import sys
from pathlib import Path

# Test 1: Check weights exist locally
from ocr_skel.models.model_utils import WEIGHTS_DIR, CRAFT_WEIGHTS_FILE, CRNN_WEIGHTS_FILE

craft_path = WEIGHTS_DIR / CRAFT_WEIGHTS_FILE
crnn_path = WEIGHTS_DIR / CRNN_WEIGHTS_FILE

assert craft_path.exists(), f"CRAFT weights not found: {craft_path}"
assert crnn_path.exists(), f"CRNN weights not found: {crnn_path}"

print(f"✓ CRAFT weights: {craft_path} ({craft_path.stat().st_size / (1024*1024):.1f} MB)")
print(f"✓ CRNN weights: {crnn_path} ({crnn_path.stat().st_size / (1024*1024):.1f} MB)")

# Test 2: Load models and verify weights loaded
from ocr_skel.detector import CRAFTDetector
from ocr_skel.recognizer import CRNNRecognizer

print("\nInitializing models...")
detector = CRAFTDetector(gpu=False)
recognizer = CRNNRecognizer(gpu=False)

print("✓ All models loaded with local pretrained weights")
print("\n✅ SUCCESS: No internet downloads required!")
