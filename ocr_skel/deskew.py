"""Классический OpenCV deskew (Hough / fht) — выпрямляет наклон скана ПЕРЕД детекцией.

Без нейросети и весов, быстрый (~0.1-0.2с/страница). Включён по умолчанию в препроцессинге;
отключить: OCRPipeline(deskew=False) или ocr(path, deskew=False).

Guard «не навреди»: поворачивает ТОЛЬКО если детекция уверенная (линии согласны) и угол
в разумном диапазоне [0.7°, 20°]. Иначе оставляет как есть — чтобы не портить прямые страницы.
Замер: на реальных сканах остаточный наклон ~5.8° → ~1.4° (median 0.23°, ложно вертит ~2% чистых).
"""
import cv2
import numpy as np


def detect_skew_angle(gray, min_conf: float = 0.6, lo: float = 0.7, hi: float = 20.0):
    """fht (HoughLinesP + взвешенная медиана углов). Вернуть (угол°, применять?)."""
    if max(gray.shape) > 1600:                       # ускорение: детекция на уменьшенной
        s = 1600 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=s, fy=s)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is None:                                # мягче, если ничего не нашли
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=20)
    if lines is None:
        return 0.0, False
    angs, ws = [], []
    for l in lines:
        x1, y1, x2, y2 = [float(v) for v in np.asarray(l).ravel()[:4]]
        L = np.hypot(x2 - x1, y2 - y1)
        if L < 20:
            continue
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        while a > 45:                                # skew, не ориентация -> в ±45°
            a -= 90
        while a < -45:
            a += 90
        angs.append(a); ws.append(L)
    if not angs:
        return 0.0, False
    angs = np.array(angs); ws = np.array(ws); tot = ws.sum()
    order = np.argsort(angs); c = 0.0; med = float(np.median(angs))
    for i in order:                                  # взвешенная медиана (длинные линии важнее)
        c += ws[i]
        if c >= tot / 2:
            med = float(angs[i]); break
    conf = ws[np.abs(angs - med) <= 2.0].sum() / tot  # доля линий, согласных с медианой
    apply = (conf >= min_conf) and (lo <= abs(med) <= hi)
    return round(med, 2), apply


def deskew_image(image: np.ndarray):
    """image: RGB numpy-массив (как отдаёт пайплайн). Вернуть (выпрямленное изображение, применённый угол°)."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    angle, apply = detect_skew_angle(gray)
    if not apply:
        return image, 0.0
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border = (255, 255, 255) if image.ndim == 3 else 255
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=border)
    return rotated, angle
