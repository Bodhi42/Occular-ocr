"""Models module for OCR"""

from .dbnet import DBNet
from .crnn_mobilenet import CRNN, crnn_mobilenet_v3_large

__all__ = ["DBNet", "CRNN", "crnn_mobilenet_v3_large"]
