"""Разрешение ориентации/layout региона кандидатами + скоринг рекогнайзером (DET_REWORK_01, Фаза 2).

Геометрия (orientation_geom) снимает наклон и ±90 (длинная ось → горизонталь), но НЕ решает
направление чтения (0 vs 180) и НЕ отличает rotated_line от vertical_stack. Здесь строятся
кандидаты и выбираются РЕКОГНАЙЗЕРОМ (не GT), по его реальной CTC-уверенности:

  LINE_0   = oriented_crop (длинная ось горизонтальна)          — обычная/наклонённая/±90 строка
  LINE_180 = LINE_0, повёрнутый на 180                          — снятие 0/180
  STACK    = unstack_vertical_text(осевой кроп)                 — upright-глифы сверху-вниз

fast-path: уверенная почти-прямая горизонталь → один прогон (локальный deskew), без кандидатов.
ambiguous-path: вертикальная ось / near-square / низкая conf → строим и сравниваем кандидаты.

Рекогнайзер НЕ обучается и его веса не трогаются — он только scorer. Ориентационная CNN в этой
итерации НЕ обучается (см. §17): накапливаем hard-set для возможного будущего классификатора.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import cv2

from .orientation_geom import TextRegion, oriented_crop, unstack_vertical_text


@dataclass
class ResolveConfig:
    fast_conf: float = 0.80        # geometry_confidence выше → доверяем горизонтали (fast-path)
    fast_angle: float = 8.0        # |local_angle| ниже → почти-прямая строка
    fast_accept_conf: float = 0.55  # recognizer conf на LINE_0 выше → fast-path принят без кандидатов
    min_crop_px: int = 4
    try_stack_max_aspect: float = 2.5  # STACK пробуем только если регион не сильно вытянут по ширине
    fast_path: bool = True             # 1-прогон для уверенной горизонтали (прод: страница уже upright
                                       # после page-orientation). Бенч ориентации гоняем с False.


@dataclass
class ResolveResult:
    text: str
    confidence: float
    candidate: str                 # LINE_0 | LINE_180 | STACK | fast/LINE_0
    orientation_deg: Optional[int]  # 0/90/180/270 относительно страницы (None для чистого stack-layout)
    layout_type: str
    scores: Dict[str, float]       # все посчитанные кандидаты (для отчёта/hard-set)


def _score_crop(recognizer, crop: np.ndarray) -> Tuple[str, float]:
    """Прогнать кроп через рекогнайзер как full-frame quad → (text, CTC-conf). Пустой/крошечный → 0."""
    if crop is None or crop.ndim < 2 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return "", 0.0
    h, w = crop.shape[:2]
    quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    try:
        text, conf = recognizer.recognize(crop, [quad])[0]
    except Exception:
        return "", 0.0
    return text, float(conf)


def _cand_score(text: str, conf: float) -> float:
    """Основной скор = CTC-conf рекогнайзера, но пустой/безбуквенный кандидат штрафуется
    (иначе высокая conf на мусоре/бланках выигрывает ложно)."""
    n_alpha = sum(ch.isalpha() for ch in text)
    if n_alpha == 0:
        return conf * 0.1
    return conf


def _orientation_from(region: TextRegion, cand: str) -> Tuple[int, str]:
    """orientation_deg (0/90/180/270) и layout_type по выбранному LINE-кандидату и углу оси."""
    ang = abs(region.local_angle_deg or 0.0)
    axis_vertical = ang >= 45.0
    if axis_vertical:
        # длинная ось вертикальна → строка повёрнута; калибровано на controlled-rotations
        # (warp длинной-оси-в-горизонталь + выбор 0/180): LINE_180→90° CW, LINE_0→270° CW
        return (90 if cand == "LINE_180" else 270), "rotated_line"
    else:
        return (180 if cand == "LINE_180" else 0), "horizontal"


def resolve_region(image: np.ndarray, region: TextRegion, recognizer,
                   cfg: ResolveConfig = ResolveConfig()) -> ResolveResult:
    """Выбрать канонический вход рекогнайзера для региона и вернуть текст + orientation/layout metadata."""
    scores: Dict[str, float] = {}

    # --- fast-path: уверенная почти-прямая горизонталь → один прогон ---
    if (cfg.fast_path
            and region.layout_type == "horizontal"
            and (region.geometry_confidence or 0.0) >= cfg.fast_conf
            and abs(region.local_angle_deg or 0.0) <= cfg.fast_angle):
        crop = oriented_crop(image, region)                 # локальный deskew по oriented-quad
        text, conf = _score_crop(recognizer, crop)
        scores["LINE_0"] = round(_cand_score(text, conf), 4)
        if _cand_score(text, conf) >= cfg.fast_accept_conf:
            region.layout_type = "horizontal"; region.orientation_deg = 0
            region.orientation_confidence = round(conf, 3)
            return ResolveResult(text, conf, "fast/LINE_0", 0, "horizontal", scores)
        # низкая conf → падаем в ambiguous (проверим 180 и, возможно, stack)

    # --- ambiguous-path: строим и сравниваем кандидаты ---
    line0 = oriented_crop(image, region)
    t0, c0 = _score_crop(recognizer, line0)
    line180 = cv2.rotate(line0, cv2.ROTATE_180)
    t180, c180 = _score_crop(recognizer, line180)
    cand_text = {"LINE_0": t0, "LINE_180": t180}
    scores["LINE_0"] = round(_cand_score(t0, c0), 4)
    scores["LINE_180"] = round(_cand_score(t180, c180), 4)
    cand_conf = {"LINE_0": c0, "LINE_180": c180}

    # STACK — только если регион не сильно горизонтально-вытянут (у широкой строки stack бессмыслен)
    aspect = region.aspect_ratio or 1.0
    horizontal_wide = (region.layout_type == "horizontal" and aspect > cfg.try_stack_max_aspect)
    if not horizontal_wide:
        x0, y0, x1, y1 = [int(round(v)) for v in region.bbox_xyxy]
        aa = image[max(0, y0):max(1, y1), max(0, x0):max(1, x1)]   # осевой кроп: глифы в ориентации страницы
        us = unstack_vertical_text(aa)
        if us.ok:
            ts, cs = _score_crop(recognizer, us.image)
            cand_text["STACK"] = ts; cand_conf["STACK"] = cs
            scores["STACK"] = round(_cand_score(ts, cs), 4)

    best = max(scores, key=scores.get)
    text = cand_text[best]; conf = cand_conf[best]
    if best == "STACK":
        region.layout_type = "vertical_stack"; region.orientation_deg = None
        region.orientation_confidence = round(conf, 3)
        return ResolveResult(text, conf, "STACK", None, "vertical_stack", scores)
    odeg, layout = _orientation_from(region, best)
    region.layout_type = layout; region.orientation_deg = odeg
    region.orientation_confidence = round(conf, 3)
    return ResolveResult(text, conf, best, odeg, layout, scores)
