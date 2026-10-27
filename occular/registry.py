"""Registry for detectors and recognizers"""

from typing import Dict, Type, Any


def _freeze(value):
    """Аргументы конструктора -> хэшируемый ключ кэша (списки/словари -> кортежи)."""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_freeze(v) for v in value]
        return tuple(sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items)
    return value


def _cached(cache: Dict[tuple, Any], name: str, factory: Type, kwargs: dict) -> Any:
    """Один экземпляр на имя И полный набор аргументов: ключ только по (name, num_threads, gpu, lm)
    отдавал распознаватель, созданный для других languages/arch (ocr(languages=['uk']), затем
    ocr(languages=['kk']) работал украинским; recognizer='svtr_t' после дефолтного — дефолтным)."""
    try:
        key = (name, _freeze(kwargs))
        hash(key)
    except TypeError:                     # нехэшируемый аргумент — без кэша
        return factory(**kwargs)
    if key not in cache:
        cache[key] = factory(**kwargs)
    return cache[key]


class Registry:
    """Реестр детекторов и распознавателей"""

    _detectors: Dict[str, Type] = {}
    _recognizers: Dict[str, Type] = {}
    _default_detector: str = "dbnet-onnx"
    _default_recognizer: str = "crnn-onnx"

    # Кэш инстансов (ключ: имя + все аргументы конструктора)
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

        return _cached(cls._detector_cache, name, cls._detectors[name], kwargs)

    @classmethod
    def get_recognizer(cls, name: str = None, **kwargs) -> Any:
        """Получить экземпляр распознавателя по имени (с кэшированием)"""
        name = name or cls._default_recognizer
        if name not in cls._recognizers:
            raise ValueError(f"Recognizer '{name}' not found. Available: {list(cls._recognizers.keys())}")

        return _cached(cls._recognizer_cache, name, cls._recognizers[name], kwargs)

    @classmethod
    def list_detectors(cls) -> list:
        """Список доступных детекторов"""
        return list(cls._detectors.keys())

    @classmethod
    def list_recognizers(cls) -> list:
        """Список доступных распознавателей"""
        return list(cls._recognizers.keys())
