"""DBNet Detector wrapper (F1=0.8897)"""

import numpy as np
import torch
import cv2
from typing import List
from pathlib import Path
import pyclipper
from shapely.geometry import Polygon
from .models.dbnet import DBNet


# Hyperparameters (best model)
THRESHOLD = 0.252
UNCLIP_RATIO = 2.44
BOX_THRESH = 0.52
MIN_AREA = 38


class DBNetDetector:
    """DBNet text detector (F1=0.8897)"""

    def __init__(self, gpu: bool = True):
        """
        Args:
            gpu: использовать GPU если доступен
        """
        self.gpu = gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.gpu else 'cpu')

        # Initialize DBNet model
        self.model = DBNet(backbone='resnet18', pretrained=False)

        # Load weights
        weights_path = Path(__file__).parent / "weights" / "dbnet_weights.pth"
        try:
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict)
            print(f"Loaded DBNet weights from {weights_path}")
        except Exception as e:
            print(f"Warning: Could not load DBNet weights: {e}")
            print("Using randomly initialized weights")

        self.model.to(self.device)
        self.model.eval()

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Детектировать текстовые области на изображении

        Args:
            image: изображение в формате RGB (H, W, C)

        Returns:
            Список quad-контуров [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        # Preprocess image
        img_resized, scale_x, scale_y = self._preprocess_image(image)

        # Convert to tensor
        x = torch.from_numpy(img_resized).permute(2, 0, 1).float()  # (H, W, C) -> (C, H, W)
        x = x.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
        x = x.to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(x)

        # Extract probability map (use prob, not binary for better control)
        prob_map = outputs['prob'][0, 0].cpu().numpy()  # (H, W)

        # Get bounding boxes from probability map
        quads = self._get_boxes(prob_map, scale_x, scale_y, image.shape)

        return quads

    def _preprocess_image(self, image: np.ndarray):
        """
        Preprocess image for DBNet

        Args:
            image: RGB image (H, W, C)

        Returns:
            Tuple of (resized_image, scale_x, scale_y)
        """
        h, w = image.shape[:2]

        # Pad to multiple of 32
        new_h = (h + 31) // 32 * 32
        new_w = (w + 31) // 32 * 32

        # Calculate scale factors for reverse mapping
        scale_x = w / new_w
        scale_y = h / new_h

        # Resize image
        img_resized = cv2.resize(image, (new_w, new_h))

        # Normalize to [0, 1]
        img_resized = img_resized.astype(np.float32) / 255.0

        # Mean normalization (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_resized = (img_resized - mean) / std

        return img_resized, scale_x, scale_y

    def _unclip_polygon(self, polygon, unclip_ratio):
        """Expand polygon using Vatti clipping algorithm"""
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

    def _get_boxes(self, prob_map, scale_x, scale_y, orig_shape):
        """
        Get bounding boxes from probability map

        Args:
            prob_map: probability map (H, W)
            scale_x: scale factor for x dimension
            scale_y: scale factor for y dimension
            orig_shape: original image shape

        Returns:
            List of quads
        """
        # Convert to binary mask using threshold
        binary_mask = (prob_map > THRESHOLD).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        quads = []
        h, w = orig_shape[:2]

        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            if area < MIN_AREA:
                continue

            # Calculate score (mean probability in contour region)
            mask = np.zeros(prob_map.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 1, -1)
            score = (prob_map * mask).sum() / (mask.sum() + 1e-6)
            if score < BOX_THRESH:
                continue

            # Expand contour using unclip
            expanded = self._unclip_polygon(contour.squeeze(1), UNCLIP_RATIO)

            # Get minimum area rectangle (rotated bbox)
            rect = cv2.minAreaRect(expanded.reshape(-1, 1, 2))
            box = cv2.boxPoints(rect)  # 4 corner points

            # Scale back to original image size
            box[:, 0] = np.clip(box[:, 0] * scale_x, 0, w)
            box[:, 1] = np.clip(box[:, 1] * scale_y, 0, h)

            # Convert to quad format (4 points)
            quad = box.astype(np.float32)

            # Order points: top-left, top-right, bottom-right, bottom-left
            quad = self._order_points(quad)

            quads.append(quad)

        return quads

    def _order_points(self, pts):
        """
        Order points in clockwise order starting from top-left

        Args:
            pts: 4 points (4, 2)

        Returns:
            ordered points
        """
        # Sort by y-coordinate
        pts = pts[np.argsort(pts[:, 1])]

        # Top two points
        top = pts[:2]
        # Sort by x-coordinate
        top = top[np.argsort(top[:, 0])]
        tl, tr = top

        # Bottom two points
        bottom = pts[2:]
        # Sort by x-coordinate
        bottom = bottom[np.argsort(bottom[:, 0])]
        bl, br = bottom

        return np.array([tl, tr, br, bl], dtype=np.float32)
