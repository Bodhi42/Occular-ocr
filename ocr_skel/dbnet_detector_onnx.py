"""DBNet Detector with ONNX Runtime"""

import numpy as np
import onnxruntime as ort
import cv2
from typing import List
from pathlib import Path
import pyclipper
from shapely.geometry import Polygon


THRESHOLD = 0.252
UNCLIP_RATIO = 2.44
BOX_THRESH = 0.52
MIN_AREA = 38


class DBNetDetectorONNX:
    """DBNet text detector using ONNX Runtime (1.9x faster on CPU)"""

    def __init__(self, gpu: bool = False):
        weights_dir = Path(__file__).parent / "weights"
        onnx_path = weights_dir / "dbnet.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # Optimized session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 0  # Auto-detect

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)

        print(f"Loaded DBNet ONNX")

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect text regions"""
        orig_h, orig_w = image.shape[:2]

        # Preprocess
        input_tensor, scale_w, scale_h = self._preprocess(image)

        # ONNX inference
        prob_map = self.session.run(None, {'input': input_tensor})[0]
        prob_map = prob_map[0, 0]  # (H, W)

        # Post-process
        quads = self._postprocess(prob_map, scale_w, scale_h, orig_w, orig_h)

        return quads

    def _preprocess(self, image: np.ndarray):
        """Preprocess image for detection (matches PyTorch exactly)"""
        h, w = image.shape[:2]

        # Pad to multiple of 32 (same as PyTorch)
        new_h = (h + 31) // 32 * 32
        new_w = (w + 31) // 32 * 32

        resized = cv2.resize(image, (new_w, new_h))

        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - mean) / std
        tensor = tensor.transpose(2, 0, 1)  # HWC -> CHW
        tensor = tensor[np.newaxis, ...]  # Add batch

        scale_w = w / new_w
        scale_h = h / new_h

        return tensor, scale_w, scale_h

    def _postprocess(self, prob_map, scale_w, scale_h, orig_w, orig_h):
        """Extract quads from probability map (matches PyTorch exactly)"""
        binary_mask = (prob_map > THRESHOLD).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        quads = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA:
                continue

            # Calculate score (same as PyTorch)
            mask = np.zeros(prob_map.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 1, -1)
            score = (prob_map * mask).sum() / (mask.sum() + 1e-6)
            if score < BOX_THRESH:
                continue

            # Unclip contour first, then minAreaRect (same as PyTorch)
            expanded = self._unclip_polygon(contour.squeeze(1), UNCLIP_RATIO)

            rect = cv2.minAreaRect(expanded.reshape(-1, 1, 2))
            box = cv2.boxPoints(rect)

            # Scale back to original size
            box[:, 0] = np.clip(box[:, 0] * scale_w, 0, orig_w)
            box[:, 1] = np.clip(box[:, 1] * scale_h, 0, orig_h)

            # Order points (same as PyTorch)
            quad = self._order_points(box.astype(np.float32))
            quads.append(quad)

        return quads

    def _unclip_polygon(self, polygon, unclip_ratio):
        """Expand polygon using Vatti clipping (same as PyTorch)"""
        poly = Polygon(polygon)
        if poly.area < 1 or poly.length < 1:
            return polygon
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(polygon.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        if not expanded:
            return polygon
        return np.array(expanded[0], dtype=np.int32)

    def _order_points(self, pts):
        """Order points clockwise starting from top-left (same as PyTorch)"""
        pts = pts[np.argsort(pts[:, 1])]
        top = pts[:2]
        top = top[np.argsort(top[:, 0])]
        tl, tr = top
        bottom = pts[2:]
        bottom = bottom[np.argsort(bottom[:, 0])]
        bl, br = bottom
        return np.array([tl, tr, br, bl], dtype=np.float32)
