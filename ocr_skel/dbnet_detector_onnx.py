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
        """Preprocess image for detection"""
        h, w = image.shape[:2]

        # Resize to multiple of 32, max 1280
        max_size = 1280
        scale = min(max_size / max(h, w), 1.0)
        new_w = int(w * scale / 32) * 32
        new_h = int(h * scale / 32) * 32
        new_w = max(32, new_w)
        new_h = max(32, new_h)

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
        """Extract quads from probability map"""
        binary = (prob_map > THRESHOLD).astype(np.uint8)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        quads = []
        for contour in contours:
            if cv2.contourArea(contour) < MIN_AREA:
                continue

            # Get minimum area rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)

            # Check score
            mask = np.zeros_like(prob_map, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 1)
            score = (prob_map * mask).sum() / mask.sum()

            if score < BOX_THRESH:
                continue

            # Unclip
            poly = Polygon(box)
            if not poly.is_valid or poly.area < 1:
                continue

            distance = poly.area * UNCLIP_RATIO / poly.length
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(box.astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = offset.Execute(distance)

            if not expanded:
                continue

            expanded = np.array(expanded[0])
            if len(expanded) < 4:
                continue

            # Get bounding box
            rect = cv2.minAreaRect(expanded)
            box = cv2.boxPoints(rect)

            # Scale back to original size
            box[:, 0] *= scale_w
            box[:, 1] *= scale_h

            # Clip to image bounds
            box[:, 0] = np.clip(box[:, 0], 0, orig_w)
            box[:, 1] = np.clip(box[:, 1], 0, orig_h)

            # Order points: top-left, top-right, bottom-right, bottom-left
            box = self._order_points(box)

            quads.append(box.astype(np.float32))

        return quads

    def _order_points(self, pts):
        """Order points clockwise starting from top-left"""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect
