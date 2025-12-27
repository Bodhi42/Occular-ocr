"""Registry for detectors and recognizers"""

from typing import Dict, Type, Any


class Registry:
    """Реестр детекторов и распознавателей"""

    _detectors: Dict[str, Type] = {}
    _recognizers: Dict[str, Type] = {}
    _default_detector: str = "craft"
    _default_recognizer: str = "crnn"

    @classmethod
    def register_detector(cls, name: str, detector_class: Type):
        """Регистрация детектора"""
        cls._detectors[name] = detector_class

    @classmethod
    def register_recognizer(cls, name: str, recognizer_class: Type):
        """Регистрация распознавателя"""
        cls._recognizers[name] = recognizer_class

    @classmethod
    def get_detector(cls, name: str = None, **kwargs) -> Any:
        """Получить экземпляр детектора по имени"""
        name = name or cls._default_detector
        if name not in cls._detectors:
            raise ValueError(f"Detector '{name}' not found. Available: {list(cls._detectors.keys())}")
        return cls._detectors[name](**kwargs)

    @classmethod
    def get_recognizer(cls, name: str = None, **kwargs) -> Any:
        """Получить экземпляр распознавателя по имени"""
        name = name or cls._default_recognizer
        if name not in cls._recognizers:
            raise ValueError(f"Recognizer '{name}' not found. Available: {list(cls._recognizers.keys())}")
        return cls._recognizers[name](**kwargs)

    @classmethod
    def list_detectors(cls) -> list:
        """Список доступных детекторов"""
        return list(cls._detectors.keys())

    @classmethod
    def list_recognizers(cls) -> list:
        """Список доступных распознавателей"""
        return list(cls._recognizers.keys())
