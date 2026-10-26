"""Layout-регионы страницы (D-FINE-S, ONNX, CPU) — геометрическая основа для порядка чтения.

    from occular import LayoutDetector
    layout = LayoutDetector().detect("page.png")
    for r in layout.regions:
        print(r.label, r.score, r.bbox)        # bbox = (x1, y1, x2, y2) в пикселях исходной страницы

Модель отдаёт прямоугольники регионов, score и одну из 18 меток (Text, Table, Form, Picture, stamp, ...).
Метки — справочная информация: пайплайн по ним НИЧЕГО не удаляет и не пропускает (регион Picture
не отключает OCR, отдельный TableRecognizer не заменяется). Веса качаются с HuggingFace при первом
использовании (layout_dfine_s144k_fp32.onnx, ~42 МБ) или берутся локально из occular/weights/.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Union
import numpy as np

LAYOUT_WEIGHT = "layout_dfine_s144k_fp32.onnx"
_RES = 960                       # вход модели: растяжение страницы до 960×960 (как при обучении)
DEFAULT_SCORE_THRESHOLD = 0.6    # прод-порог; ниже — больше перекрывающихся дублей

# id → имя метки (id 1..18 — как в обучающей разметке и в выходе модели)
LAYOUT_LABELS = {
    1: "Caption", 2: "Code", 3: "Equation", 4: "Figure", 5: "Footnote", 6: "Form", 7: "ListItem",
    8: "PageFooter", 9: "PageHeader", 10: "Picture", 11: "SectionHeader", 12: "Table",
    13: "TableOfContents", 14: "Text", 15: "TextInlineMath", 16: "stamp", 17: "signature",
    18: "handwritten_text",
}


@dataclass
class LayoutRegion:
    """Один регион: прямоугольник в пикселях входного изображения, уверенность и метка."""
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2
    score: float
    label: str
    label_id: int

    def to_dict(self) -> dict:
        return {"bbox": [float(v) for v in self.bbox], "score": float(self.score),
                "label": self.label, "label_id": int(self.label_id)}


@dataclass
class LayoutResult:
    """Регионы одной страницы + размер страницы, в системе координат которой они даны."""
    width: int
    height: int
    regions: List[LayoutRegion] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"width": int(self.width), "height": int(self.height),
                "regions": [r.to_dict() for r in self.regions]}


def _to_rgb_array(image) -> np.ndarray:
    """Путь / PIL.Image / numpy (H,W,3 RGB или H,W серый) → uint8 RGB numpy."""
    from PIL import Image
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        with Image.open(image) as im:
            return np.array(im.convert("RGB"))
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    a = np.asarray(image)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"Expected an RGB image (H,W,3), got {a.shape}")
    return np.ascontiguousarray(a[:, :, :3].astype(np.uint8, copy=False))


class LayoutDetector:
    """D-FINE-S layout-детектор (ONNX Runtime). Потокобезопасен для detect() из нескольких потоков."""

    def __init__(self, score_threshold: float = DEFAULT_SCORE_THRESHOLD, num_threads: int = 4,
                 gpu: bool = False, model_path: str = None):
        import onnxruntime as ort
        from .model_files import ensure_weight
        from ._runtime import resolve_providers
        self.score_threshold = float(score_threshold)
        if model_path is None:
            try:
                model_path = ensure_weight(LAYOUT_WEIGHT)
            except Exception as e:
                raise RuntimeError(
                    f"Could not get the layout model weights '{LAYOUT_WEIGHT}' ({type(e).__name__}: {e}).\n"
                    "Put the file in occular/weights/ or pass LayoutDetector(model_path=...)."
                ) from e
        path = model_path
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(num_threads))
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), so, providers=resolve_providers(gpu))

    def detect(self, image: Union[str, np.ndarray, "PIL.Image.Image"],
               score_threshold: float = None) -> LayoutResult:
        """Найти регионы. Координаты — в пикселях ПЕРЕДАННОГО изображения (после вашего препроцессинга).
        Регионы отсортированы по убыванию score."""
        from PIL import Image
        a = _to_rgb_array(image)
        H, W = a.shape[:2]
        thr = self.score_threshold if score_threshold is None else float(score_threshold)
        if H == 0 or W == 0:
            return LayoutResult(width=W, height=H)
        x = np.asarray(Image.fromarray(a).resize((_RES, _RES), Image.BILINEAR), np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]
        labels, boxes, scores = self.session.run(
            None, {"images": x, "orig_target_sizes": np.array([[W, H]], np.int64)})
        regions = []
        for lab, box, sc in zip(labels[0], boxes[0], scores[0]):
            if sc < thr or int(lab) not in LAYOUT_LABELS:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            x1, x2 = max(0.0, min(x1, W)), max(0.0, min(x2, W))
            y1, y2 = max(0.0, min(y1, H)), max(0.0, min(y2, H))
            if x2 <= x1 or y2 <= y1:
                continue
            regions.append(LayoutRegion((x1, y1, x2, y2), float(sc), LAYOUT_LABELS[int(lab)], int(lab)))
        regions.sort(key=lambda r: (-r.score, r.bbox))
        return LayoutResult(width=W, height=H, regions=regions)
