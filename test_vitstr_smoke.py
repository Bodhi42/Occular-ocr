"""Smoke test for ViTSTR recognizer"""

import sys
import numpy as np
from ocr_skel import Registry
from ocr_skel.vitstr_recognizer import ViTSTRRecognizer

def test_vitstr_registration():
    """Test ViTSTR registration in Registry"""
    print("Test 1: Проверка регистрации ViTSTR...")

    # Check if vitstr is registered
    recognizers = Registry.list_recognizers()
    assert "vitstr" in recognizers, f"ViTSTR not registered. Available: {recognizers}"
    print("  ✓ ViTSTR зарегистрирован в Registry")

def test_vitstr_instantiation():
    """Test ViTSTR instantiation"""
    print("\nTest 2: Проверка создания экземпляра ViTSTR...")

    # Create instance via Registry
    recognizer = Registry.get_recognizer("vitstr", gpu=False)
    assert isinstance(recognizer, ViTSTRRecognizer)
    print("  ✓ Экземпляр ViTSTRRecognizer создан")

    # Check attributes
    assert hasattr(recognizer, 'model')
    assert hasattr(recognizer, 'device')
    assert hasattr(recognizer, 'charset')
    print("  ✓ Все необходимые атрибуты присутствуют")

def test_vitstr_recognize():
    """Test ViTSTR recognize method"""
    print("\nTest 3: Проверка метода recognize...")

    recognizer = Registry.get_recognizer("vitstr", gpu=False)

    # Create dummy image (224x224 grayscale)
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Create dummy quad
    quad = np.array([[10, 10], [200, 10], [200, 50], [10, 50]])

    # Run recognition
    results = recognizer.recognize(image, [quad])

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], tuple)
    assert len(results[0]) == 2

    text, confidence = results[0]
    assert isinstance(text, str)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0

    print(f"  ✓ recognize вернул: text='{text}', confidence={confidence:.3f}")

def test_direct_import():
    """Test direct import"""
    print("\nTest 4: Проверка прямого импорта...")

    from ocr_skel import ViTSTRRecognizer
    recognizer = ViTSTRRecognizer(gpu=False)
    assert recognizer is not None
    print("  ✓ Прямой импорт работает")

if __name__ == "__main__":
    print("="*60)
    print("ViTSTR Smoke Test")
    print("="*60)

    try:
        test_vitstr_registration()
        test_vitstr_instantiation()
        test_vitstr_recognize()
        test_direct_import()

        print("\n" + "="*60)
        print("✓ Все тесты пройдены успешно!")
        print("="*60)
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ Тест провален: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
