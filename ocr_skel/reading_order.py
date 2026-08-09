"""Reading-order инференс через AR-layout модель (опциональная, качается с HuggingFace).

Модель (DonutSwin-энкодер + AR-декодер, ONNX) генерит регионы страницы В ПОРЯДКЕ ЧТЕНИЯ.
Мы назначаем каждую распознанную строку своему региону и сортируем строки по
(порядок региона, затем сверху-вниз внутри региона) — это правильно упорядочивает
многоколоночные страницы (газеты/научные тексты), чего простая сортировка по Y не может.

Веса не входят в комплект (большие) — качаются: from ocr_skel import download_reading_order.
Формат ONNX: encoder.onnx (картинка→memory) + decoder.onnx (шаг генерации, фикс L=140).
"""
from pathlib import Path
import numpy as np

_RO_DIR = Path(__file__).parent / "weights" / "reading_order"

# Константы обученной модели (res1024/ckpt_35000, one-class)
_RES = 1024
_NBINS = 1000
_MAXLEN = 140          # фикс. длина декодера (как в ONNX-экспорте)
_MAXB = 100
_BOS, _EOS, _PAD, _REGION = 1, 2, 0, 3


class ReadingOrderModel:
    """Ленивая обёртка над layout-ONNX. Даёт регионы в порядке чтения и переупорядочивает результаты OCR."""

    def __init__(self, num_threads: int = 4, gpu: bool = False):
        import onnxruntime as ort
        enc, dec = _RO_DIR / "encoder.onnx", _RO_DIR / "decoder.onnx"
        if not enc.exists() or not dec.exists():
            raise FileNotFoundError(
                f"Модель порядка чтения не найдена в {_RO_DIR}.\n"
                "Скачайте её:  from ocr_skel import download_reading_order; download_reading_order()"
            )
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(num_threads))
        so.inter_op_num_threads = 1
        from ._runtime import resolve_providers
        providers = resolve_providers(gpu)     # CPU по умолчанию; CUDA при gpu=True
        self.enc = ort.InferenceSession(str(enc), so, providers=providers)
        self.dec = ort.InferenceSession(str(dec), so, providers=providers)

    def _preprocess(self, image: np.ndarray):
        """RGB numpy -> pixel_values [1,3,1024,1024] (aspect-preserving pad, норма [-1,1]). Вернуть (px, S)."""
        from PIL import Image
        im = Image.fromarray(image) if image.ndim == 3 else Image.fromarray(image).convert("RGB")
        W, H = im.size
        S = max(W, H)
        sc = _RES / S
        im = im.resize((max(1, int(W * sc)), max(1, int(H * sc))))
        canvas = Image.new("RGB", (_RES, _RES), (255, 255, 255))
        canvas.paste(im, (0, 0))
        a = (np.asarray(canvas, np.float32) / 255.0 - 0.5) / 0.5
        return a.transpose(2, 0, 1)[None], S

    def regions_in_order(self, image: np.ndarray):
        """Вернуть список регионов [x1,y1,x2,y2] (пиксели ориг.картинки) В ПОРЯДКЕ ЧТЕНИЯ."""
        px, S = self._preprocess(image)
        mem = self.enc.run(["memory"], {"px": px})[0]
        inb = np.zeros((1, _MAXLEN, 4), np.float32)
        inl = np.full((1, _MAXLEN), _PAD, np.int64)
        am = np.zeros((1, _MAXLEN), bool)
        inl[0, 0] = _BOS
        am[0, 0] = True
        cur = 1
        boxes = []
        for st in range(min(_MAXB, _MAXLEN - 1)):
            coord, cls = self.dec.run(["coord", "cls"],
                                      {"memory": mem, "in_boxes": inb, "in_labels": inl, "attn_mask": am})
            last = cur - 1
            c = int(cls[0, last].argmax())
            q = coord[0, last].argmax(-1).astype(np.float32) / (_NBINS - 1)   # x1,y1,x2,y2 в [0,1]
            cx = (q[0] + q[2]) / 2; cy = (q[1] + q[3]) / 2
            w = max(0.0, q[2] - q[0]); h = max(0.0, q[3] - q[1])
            if st > 0 and c in (_EOS, _PAD, _BOS):
                break
            boxes.append([(cx - w / 2) * S, (cy - h / 2) * S, (cx + w / 2) * S, (cy + h / 2) * S])
            if cur >= _MAXLEN:
                break
            inb[0, cur] = [cx, cy, w, h]
            inl[0, cur] = max(c, _REGION)
            am[0, cur] = True
            cur += 1
        return boxes
