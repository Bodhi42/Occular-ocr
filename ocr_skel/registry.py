"""Registry for detectors and recognizers"""

from typing import Dict, Type, Any


class Registry:
    """Реестр детекторов и распознавателей"""

    _detectors: Dict[str, Type] = {}
    _recognizers: Dict[str, Type] = {}
    _default_detector: str = "dbnet"
    _default_recognizer: str = "crnn"

    # Кэш инстансов (ключ: (name, gpu))
    _detector_cache: Dict[tuple, Any] = {}
    _recognizer_cache: Dict[tuple, Any] = {}

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
        """Получить экземпляр детектора по имени (с кэшированием)"""
        name = name or cls._default_detector
        if name not in cls._detectors:
            raise ValueError(f"Detector '{name}' not found. Available: {list(cls._detectors.keys())}")

        # Кэш по (name, gpu)
        cache_key = (name, kwargs.get('gpu', False))
        if cache_key not in cls._detector_cache:
            cls._detector_cache[cache_key] = cls._detectors[name](**kwargs)
        return cls._detector_cache[cache_key]

    @classmethod
    def get_recognizer(cls, name: str = None, **kwargs) -> Any:
        """Получить экземпляр распознавателя по имени (с кэшированием)"""
        name = name or cls._default_recognizer
        if name not in cls._recognizers:
            raise ValueError(f"Recognizer '{name}' not found. Available: {list(cls._recognizers.keys())}")

        # Кэш по (name, gpu)
        cache_key = (name, kwargs.get('gpu', False))
        if cache_key not in cls._recognizer_cache:
            cls._recognizer_cache[cache_key] = cls._recognizers[name](**kwargs)
        return cls._recognizer_cache[cache_key]

    @classmethod
    def list_detectors(cls) -> list:
        """Список доступных детекторов"""
        return list(cls._detectors.keys())

    @classmethod
    def list_recognizers(cls) -> list:
        """Список доступных распознавателей"""
        return list(cls._recognizers.keys())
