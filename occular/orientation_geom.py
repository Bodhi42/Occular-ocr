"""Ориентация/геометрия текстовых регионов (изолированный слой над DBNet).

Итерация DET_REWORK_01: между «DBNet output» и «recognizer crop» появляется расширенное
представление региона, в котором НЕ теряется угол и координаты исходной страницы.

Фаза 1 (этот файл на старте) — только ГЕОМЕТРИЯ, без решения о направлении чтения:
  • TextRegion — расширенный контракт региона (осевой bbox для обратной совместимости + oriented);
  • build_region(polygon) — из сырого полигона DBNet считает minAreaRect / oriented-quad / bbox / ось;
  • estimate_text_axis — угол длинной оси, устойчивый к перестановке точек, low-conf для near-square;
  • oriented_crop — perspective-warp длинной оси в горизонталь + transform_to_page (обратимость).

ЗАПРЕЩЕНО правило height>width => rotate 90. Различение rotated_line vs vertical_stack и 0/180 —
это Фаза 2 (кандидаты + скоринг рекогнайзером), не геометрия. Здесь vertical long-axis остаётся
layout_type="uncertain" до арбитража.

Прод-путь (detect()/_crop_quad) этот файл НЕ меняет — включается флагом на уровне пайплайна.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Tuple
import numpy as np
import cv2

# --- пороги геометрии (не хардкодить под одну DPI: всё относительное/угловое) ---
NEAR_SQUARE_ASPECT = 1.5     # aspect < этого → near-square, ось не определяется уверенно
HORIZ_ANGLE_DEG = 30.0       # |angle| ниже → длинная ось считается «горизонтальной»
CONF_ASPECT_LO = 1.3         # aspect, при котором уверенность оси = 0
CONF_ASPECT_HI = 3.0         # aspect, при котором уверенность оси = 1

LayoutType = Literal["horizontal", "rotated_line", "vertical_stack", "uncertain"]


@dataclass
class AxisEstimate:
    angle_deg: float          # угол длинной оси, нормализован в (-90, 90]; горизонталь≈0, вертикаль≈±90
    aspect_ratio: float       # длинная/короткая сторона minAreaRect (≥1)
    confidence: float         # 0..1, низкая для near-square
    ambiguous: bool           # True → геометрия не должна форсить направление


@dataclass
class TextRegion:
    # --- обратная совместимость: осевой прямоугольник как у detect() ---
    bbox_xyxy: Tuple[float, float, float, float]        # (x0,y0,x1,y1) в координатах страницы
    # --- сырая геометрия DBNet (сохранена ДО схлопывания в bbox) ---
    polygon: np.ndarray                                 # (N,2) unclipped-контур в координатах страницы
    oriented_quad: np.ndarray                           # (4,2) minAreaRect boxPoints, порядок TL,TR,BR,BL
    # --- оценка оси ---
    local_angle_deg: Optional[float]                    # угол длинной оси (из AxisEstimate)
    geometry_confidence: Optional[float]                # уверенность оси/геометрии
    aspect_ratio: Optional[float] = None
    # --- layout/ориентация (заполняется Фазой 2; в Фазе 1 = uncertain/None) ---
    layout_type: LayoutType = "uncertain"
    orientation_deg: Optional[Literal[0, 90, 180, 270]] = None
    orientation_confidence: Optional[float] = None
    # --- преобразования crop<->page (заполняет oriented_crop) ---
    transform_to_crop: Optional[np.ndarray] = None      # 3x3: page → crop
    transform_to_page: Optional[np.ndarray] = None      # 3x3: crop → page (обратимость координат)

    def axis_aligned_quad(self) -> np.ndarray:
        """Осе-выровненный quad (4,2) — идентично старому формату detect()."""
        x0, y0, x1, y1 = self.bbox_xyxy
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def normalize_quad(pts: np.ndarray) -> np.ndarray:
    """Упорядочить 4 точки как TL, TR, BR, BL — устойчиво к любой входной перестановке.
    TL = min(x+y), BR = max(x+y), TR = min(y−x), BL = max(y−x)."""
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = p.sum(axis=1)
    d = p[:, 1] - p[:, 0]
    tl = p[np.argmin(s)]
    br = p[np.argmax(s)]
    tr = p[np.argmin(d)]
    bl = p[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def estimate_text_axis(oriented_quad: np.ndarray) -> AxisEstimate:
    """Угол длинной оси региона из oriented-quad. Устойчив к перестановке точек: угол берётся
    из ДЛИННОГО ребра boxPoints (а не из сырого cv2 angle), диапазон нормализуется в (-90, 90].
    near-square → низкая уверенность и ambiguous=True (геометрия не решает направление)."""
    q = normalize_quad(oriented_quad)                  # §3: устойчивость к любой перестановке точек
    # два ребра из одной вершины: q0->q1 и q1->q2
    e1 = q[1] - q[0]
    e2 = q[2] - q[1]
    l1 = float(np.hypot(*e1))
    l2 = float(np.hypot(*e2))
    long_e, long_len, short_len = (e1, l1, l2) if l1 >= l2 else (e2, l2, l1)
    aspect = long_len / max(short_len, 1e-6)
    ang = float(np.degrees(np.arctan2(long_e[1], long_e[0])))
    # нормализация в (-90, 90] — длинная ось не имеет «направления», только наклон
    while ang > 90:
        ang -= 180
    while ang <= -90:
        ang += 180
    conf = float(np.clip((aspect - CONF_ASPECT_LO) / (CONF_ASPECT_HI - CONF_ASPECT_LO), 0.0, 1.0))
    ambiguous = aspect < NEAR_SQUARE_ASPECT
    if ambiguous:
        conf = min(conf, 0.2)
    return AxisEstimate(angle_deg=round(ang, 2), aspect_ratio=round(aspect, 3),
                        confidence=round(conf, 3), ambiguous=ambiguous)


def build_region(polygon_xy: np.ndarray) -> TextRegion:
    """Из сырого полигона DBNet (в координатах СТРАНИЦЫ) построить TextRegion.
    Геометрия only — layout_type остаётся 'horizontal' лишь для явных широких почти-прямых строк,
    иначе 'uncertain' (вертикальная длинная ось = rotated_line ИЛИ stack → решает Фаза 2)."""
    poly = np.asarray(polygon_xy, dtype=np.float32).reshape(-1, 2)
    x0, y0 = float(poly[:, 0].min()), float(poly[:, 1].min())
    x1, y1 = float(poly[:, 0].max()), float(poly[:, 1].max())
    rect = cv2.minAreaRect(poly)                       # ((cx,cy),(w,h),angle)
    quad = normalize_quad(cv2.boxPoints(rect))
    axis = estimate_text_axis(quad)

    if axis.ambiguous:
        layout: LayoutType = "uncertain"               # near-square (одиночная буква и т.п.)
    elif abs(axis.angle_deg) <= HORIZ_ANGLE_DEG:
        layout = "horizontal"                          # явно широкая почти-прямая строка
    else:
        layout = "uncertain"                           # вертикальная длинная ось → арбитраж в Фазе 2

    return TextRegion(
        bbox_xyxy=(x0, y0, x1, y1),
        polygon=poly,
        oriented_quad=quad,
        local_angle_deg=axis.angle_deg,
        geometry_confidence=axis.confidence,
        aspect_ratio=axis.aspect_ratio,
        layout_type=layout,
    )


def _map_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Применить 3x3 перспективное преобразование к точкам (N,2)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, M)
    return out.reshape(-1, 2)


def oriented_crop(image: np.ndarray, region: TextRegion,
                  pad_ratio: float = 0.0) -> np.ndarray:
    """Perspective-warp oriented-quad в горизонтальный crop: длинная ось → слева-направо.
    Заполняет region.transform_to_crop (page→crop) и transform_to_page (crop→page) для обратимости.
    Возвращает crop (H,W,3). Направление чтения 0/180 здесь НЕ разрешается (это Фаза 2).
    Для маленького наклона это == локальный deskew строки; для 90/270 == разворот в горизонталь."""
    q = normalize_quad(region.oriented_quad)            # TL,TR,BR,BL
    # ДЛИННАЯ ось → горизонталь: если длинное ребро вертикально (повёрнутая строка), крутим порядок
    top_len = np.hypot(*(q[1] - q[0])); left_len = np.hypot(*(q[3] - q[0]))
    if left_len > top_len:                              # длинная ось вертикальна → развернуть в горизонталь
        q = q[[3, 0, 1, 2]]                             # новый top-edge = старый left-edge (длинный)
    wA = np.hypot(*(q[1] - q[0])); wB = np.hypot(*(q[2] - q[3]))
    hA = np.hypot(*(q[3] - q[0])); hB = np.hypot(*(q[2] - q[1]))
    W = int(round(max(wA, wB))); H = int(round(max(hA, hB)))
    if W < 2 or H < 2:
        # вырожденный регион — вернуть осевой кроп как фолбэк, без преобразований
        x0, y0, x1, y1 = [int(round(v)) for v in region.bbox_xyxy]
        return image[max(0, y0):max(1, y1), max(0, x0):max(1, x1)]
    if pad_ratio > 0:
        px = W * pad_ratio; py = H * pad_ratio; W = int(round(W + 2 * px)); H = int(round(H + 2 * py))
        dst = np.array([[px, py], [W - px, py], [W - px, H - py], [px, H - py]], dtype=np.float32)
    else:
        dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)   # page → crop
    region.transform_to_crop = M
    region.transform_to_page = np.linalg.inv(M)                  # crop → page (восстановление координат)
    return cv2.warpPerspective(image, M, (W, H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


@dataclass
class UnstackResult:
    image: np.ndarray                          # горизонтальная псевдострока (upright-глифы слева-направо)
    n_bands: int
    mapping: List[Tuple[int, int, Tuple[int, int, int, int]]]  # (x_start,x_end,band_bbox_in_crop_xyxy)
    ok: bool                                   # >=2 полос → регион правдоподобно stack


def _foreground_mask(gray: np.ndarray) -> np.ndarray:
    """Робастная ink-mask: Otsu + polarity (текст = меньшинство пикселей), лёгкий close.
    Не считаем, что текст всегда чёрный на белом."""
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = gray < t
    fg = dark if dark.mean() <= 0.5 else ~dark      # foreground = меньшинство
    m = fg.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return m


def unstack_vertical_text(image: np.ndarray, canon_h: int = 48, gap_ratio: float = 0.12,
                          merge_gap_ratio: float = 0.6, pad_ratio: float = 0.12,
                          min_area_ratio: float = 0.0008) -> UnstackResult:
    """Вертикальный stack (буквы стоят сверху-вниз, каждая upright) → горизонтальная псевдострока
    БЕЗ поворота глифов. Полосы ищутся проекцией+CC с ОТНОСИТЕЛЬНЫМ мерджем (диакритика ё/й, точки,
    двоеточие присоединяются к своей полосе). Полосы приводятся к canon_h и ставятся слева-направо
    с крошечным (не пробельным) зазором, фон = медиана background. Возвращает mapping x→bbox_in_crop.

    ЭТО image-transform, а не распознавание по GT. Если <2 полос → ok=False (регион не stack)."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape[:2]
    if H < 6 or W < 3:
        return UnstackResult(image, 1, [], ok=False)
    m = _foreground_mask(gray)
    n, _lbl, stats, _c = cv2.connectedComponentsWithStats(m, 8)
    min_area = max(4.0, min_area_ratio * H * W)
    comps = [tuple(int(v) for v in stats[i, :5]) for i in range(1, n) if stats[i, 4] >= min_area]  # x,y,w,h,area
    if len(comps) < 2:
        return UnstackResult(image, 1, [], ok=False)
    med_h = float(np.median([c[3] for c in comps]))
    comps.sort(key=lambda c: c[1])                  # по верхней Y
    # группировка в полосы: следующий компонент присоединяется, если перекрывается или зазор мал
    bands: List[List] = []
    for c in comps:
        x, y, w, h, _a = c
        if bands and y <= bands[-1][1] + merge_gap_ratio * med_h:
            bands[-1][0] = min(bands[-1][0], y)
            bands[-1][1] = max(bands[-1][1], y + h)
            bands[-1][2].append(c)
        else:
            bands.append([y, y + h, [c]])
    if len(bands) < 2:
        return UnstackResult(image, 1, [], ok=False)
    bg = int(np.median(gray[m == 0])) if (m == 0).any() else 255
    tiles = []
    for ymin, ymax, cs in bands:
        xs0 = min(cc[0] for cc in cs); xs1 = max(cc[0] + cc[2] for cc in cs)
        ph = int(round((ymax - ymin) * pad_ratio)); pw = int(round((xs1 - xs0) * pad_ratio))
        y0 = max(0, ymin - ph); y1 = min(H, ymax + ph)
        x0 = max(0, xs0 - pw); x1 = min(W, xs1 + pw)
        if y1 - y0 < 2 or x1 - x0 < 2:
            continue
        tile = image[y0:y1, x0:x1]
        th, tw = tile.shape[:2]
        nw = max(1, int(round(tw * canon_h / th)))
        tiles.append((cv2.resize(tile, (nw, canon_h)), (x0, y0, x1, y1)))
    if len(tiles) < 2:
        return UnstackResult(image, 1, [], ok=False)
    gap = max(1, int(round(canon_h * gap_ratio)))
    total_w = sum(t[0].shape[1] for t in tiles) + gap * (len(tiles) - 1)
    shape = (canon_h, total_w, 3) if image.ndim == 3 else (canon_h, total_w)
    canvas = np.full(shape, bg, np.uint8)
    mapping = []
    cx = 0
    for tile_r, bbox in tiles:
        w = tile_r.shape[1]
        canvas[:, cx:cx + w] = tile_r
        mapping.append((cx, cx + w, bbox))
        cx += w + gap
    return UnstackResult(canvas, len(tiles), mapping, ok=True)


def crop_point_to_page(region: TextRegion, xy: Tuple[float, float]) -> Tuple[float, float]:
    """Восстановить точку из координат crop обратно в координаты страницы (обратимость §1)."""
    if region.transform_to_page is None:
        raise ValueError("region ещё не кропнут (transform_to_page пуст)")
    p = _map_points(region.transform_to_page, np.array([xy], dtype=np.float32))[0]
    return float(p[0]), float(p[1])
