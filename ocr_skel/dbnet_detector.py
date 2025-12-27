"""DBNet Detector wrapper"""

import numpy as np
import torch
import cv2
from typing import List
from .models.dbnet_model import DBNet
from .models.model_utils import load_dbnet_weights


class DBNetDetector:
    """DBNet text detector (pretrained)"""

    def __init__(self, gpu: bool = True, backbone: str = 'resnet18'):
        """
        Args:
            gpu: использовать GPU если доступен
            backbone: 'resnet18' or 'resnet50'
        """
        self.gpu = gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.gpu else 'cpu')

        # Initialize DBNet model
        self.model = DBNet(backbone=backbone, pretrained=True, freeze_backbone=False)

        # Load pretrained weights
        try:
            self.model = load_dbnet_weights(self.model, device=self.device)
        except Exception as e:
            print(f"Warning: Could not load DBNet weights: {e}")
            print("Using model with ImageNet pretrained backbone only")

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

        # Extract binary map
        binary_map = outputs['binary'][0, 0].cpu().numpy()  # (H, W)

        # Get bounding boxes from binary map
        quads = self._get_boxes(binary_map, scale_x, scale_y, image.shape)

        return quads

    def _preprocess_image(self, image: np.ndarray, target_size=640):
        """
        Preprocess image for DBNet

        Args:
            image: RGB image (H, W, C)
            target_size: target long side size

        Returns:
            Tuple of (resized_image, scale_x, scale_y)
        """
        h, w = image.shape[:2]

        # Calculate target size maintaining aspect ratio
        ratio = target_size / max(h, w)
        target_h = int(h * ratio)
        target_w = int(w * ratio)

        # Make dimensions divisible by 32
        target_h = target_h - target_h % 32
        target_w = target_w - target_w % 32

        # Calculate scale factors for reverse mapping
        scale_x = w / target_w
        scale_y = h / target_h

        # Resize image
        img_resized = cv2.resize(image, (target_w, target_h))

        # Normalize to [0, 1]
        img_resized = img_resized.astype(np.float32) / 255.0

        # Mean normalization (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_resized = (img_resized - mean) / std

        return img_resized, scale_x, scale_y

    def _get_boxes(self, binary_map, scale_x, scale_y, orig_shape,
                   min_area=100, min_score=0.3):
        """
        Get bounding boxes from binary map

        Args:
            binary_map: binary segmentation map (H, W)
            scale_x: scale factor for x dimension
            scale_y: scale factor for y dimension
            orig_shape: original image shape
            min_area: minimum area threshold
            min_score: minimum score threshold

        Returns:
            List of quads
        """
        # Convert to binary mask
        binary_mask = (binary_map > min_score).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        quads = []
        h, w = orig_shape[:2]

        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Get minimum area rectangle (rotated bbox)
            rect = cv2.minAreaRect(contour)
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
