"""Integration test for ViTSTR recognizer with Pipeline"""

import sys
import os
import numpy as np
from PIL import Image
from ocr_skel import Registry, OCRPipeline

def test_vitstr_with_recognizer_directly():
    """Test ViTSTR recognizer directly"""
    print("Test: ViTSTR recognizer прямой вызов...")

    # Create recognizer
    recognizer = Registry.get_recognizer("vitstr", gpu=False)

    # Create dummy image
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Create dummy quads
    quads = [
        np.array([[10, 10], [200, 10], [200, 50], [10, 50]]),
        np.array([[10, 60], [200, 60], [200, 100], [10, 100]])
    ]

    # Run recognition
    results = recognizer.recognize(image, quads)

    print(f"  ✓ Recognizer вернул {len(results)} результатов")
    for i, (text, confidence) in enumerate(results):
        print(f"    Result {i}: text='{text}', confidence={confidence:.3f}")

def test_vitstr_with_pipeline():
    """Test ViTSTR with OCRPipeline"""
    print("\nTest: ViTSTR integration with OCRPipeline...")

    # Create a temporary test image
    test_img_path = "/tmp/test_vitstr.png"
    img = Image.new('RGB', (256, 256), color='white')
    img.save(test_img_path)

    try:
        # Create pipeline with ViTSTR
        pipeline = OCRPipeline(
            detector_name="craft",
            recognizer_name="vitstr",
            detector_kwargs={"gpu": False},
            recognizer_kwargs={"gpu": False}
        )

        # Run OCR
        results = pipeline.process_image(test_img_path)

        print(f"  ✓ Pipeline вернул {len(results)} результатов")
        for i, result in enumerate(results):
            print(f"    Result {i}: text='{result['text']}', confidence={result['confidence']:.3f}")

    finally:
        # Cleanup
        if os.path.exists(test_img_path):
            os.remove(test_img_path)

def test_registry_list():
    """Test Registry list methods"""
    print("Test: Registry recognizers list...")

    recognizers = Registry.list_recognizers()
    print(f"  Доступные recognizers: {recognizers}")

    assert "crnn" in recognizers
    assert "vitstr" in recognizers

    print("  ✓ Оба recognizer доступны (crnn, vitstr)\n")

if __name__ == "__main__":
    print("="*60)
    print("ViTSTR Integration Test")
    print("="*60)

    try:
        test_registry_list()
        test_vitstr_with_recognizer_directly()
        test_vitstr_with_pipeline()

        print("\n" + "="*60)
        print("✓ Все интеграционные тесты пройдены!")
        print("="*60)
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ Тест провален: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
