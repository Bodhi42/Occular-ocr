#!/usr/bin/env python3
"""Тест из спецификации задания"""

from ocr_skel import Registry
from ocr_skel.vitstr_recognizer import ViTSTRRecognizer

# Регистрация (уже автоматически выполнена в __init__.py)
# Registry.register_recognizer("vitstr", ViTSTRRecognizer)

print("="*60)
print("Тест примера из задания")
print("="*60)

# Создание recognizer
print("\n1. Создание recognizer через Registry...")
recognizer = Registry.get_recognizer("vitstr", gpu=False)
print(f"   ✓ Создан: {type(recognizer).__name__}")

# Проверка атрибутов
print("\n2. Проверка атрибутов...")
print(f"   - Device: {recognizer.device}")
print(f"   - Charset: {recognizer.charset}")
print(f"   - Num classes: {recognizer.num_classes}")
print(f"   - Languages: {recognizer.languages}")
print(f"   ✓ Все атрибуты присутствуют")

# Проверка что модель инициализирована
print("\n3. Проверка модели...")
print(f"   - Model type: {type(recognizer.model).__name__}")
print(f"   - Model on device: {next(recognizer.model.parameters()).device}")
print(f"   ✓ Модель инициализирована")

print("\n" + "="*60)
print("✓ Тест из задания пройден успешно!")
print("="*60)
