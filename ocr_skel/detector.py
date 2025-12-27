"""CRAFT Detector wrapper"""

import numpy as np
import torch
import cv2
from typing import List
from .models.craft_model import CRAFT
from .models.model_utils import load_craft_weights


class CRAFTDetector:
    """CRAFT text detector (pretrained)"""

    def __init__(self, gpu: bool = True):
        """
        Args:
            gpu: использовать GPU если доступен
        """
        self.gpu = gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.gpu else 'cpu')

        # Initialize CRAFT model
        self.model = CRAFT(pretrained=True, freeze=False)

        # Load pretrained weights
        try:
            self.model = load_craft_weights(self.model, device=self.device)
        except Exception as e:
            print(f"Warning: Could not load CRAFT weights: {e}")
            print("Using model with random initialization")

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
        img_resized, target_ratio, size_heatmap = self._preprocess_image(image)

        # Convert to tensor
        x = torch.from_numpy(img_resized).permute(2, 0, 1).float()  # (H, W, C) -> (C, H, W)
        x = x.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
        x = x.to(self.device)

        # Forward pass
        with torch.no_grad():
            y = self.model(x)  # (1, 2, H, W)

        # Extract region score and affinity score
        score_text = y[0, 0, :, :].cpu().numpy()  # Region score
        score_link = y[0, 1, :, :].cpu().numpy()  # Affinity score

        # Get bounding boxes from heatmaps
        quads = self._get_boxes(score_text, score_link, target_ratio, image.shape)

        return quads

    def _preprocess_image(self, image: np.ndarray, target_size=1280, mag_ratio=1.5):
        """
        Preprocess image for CRAFT

        Args:
            image: RGB image (H, W, C)
            target_size: target long side size
            mag_ratio: magnification ratio

        Returns:
            Tuple of (resized_image, ratio, heatmap_size)
        """
        h, w = image.shape[:2]

        # Calculate target size
        ratio = target_size / max(h, w)
        target_h = int(h * ratio)
        target_w = int(w * ratio)

        # Make dimensions divisible by 32
        target_h = target_h - target_h % 32
        target_w = target_w - target_w % 32

        # Resize image
        img_resized = cv2.resize(image, (target_w, target_h))

        # Normalize to [0, 1]
        img_resized = img_resized.astype(np.float32) / 255.0

        # Mean normalization (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_resized = (img_resized - mean) / std

        return img_resized, ratio, (target_h, target_w)

    def _get_boxes(self, score_text, score_link, ratio, orig_shape, text_threshold=0.7,
                   link_threshold=0.4, low_text=0.4):
        """
        Get bounding boxes from score maps

        Args:
            score_text: region score map
            score_link: affinity score map
            ratio: resize ratio
            orig_shape: original image shape
            text_threshold: threshold for text region
            link_threshold: threshold for link region
            low_text: lower threshold for text

        Returns:
            List of quads
        """
        # Simple thresholding approach
        # Convert scores to binary masks
        text_mask = score_text > text_threshold
        link_mask = score_link > link_threshold

        # Combine masks
        combined_mask = np.logical_or(text_mask, link_mask).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        quads = []
        h, w = orig_shape[:2]

        for contour in contours:
            # Get bounding rectangle
            x, y, box_w, box_h = cv2.boundingRect(contour)

            # Filter small regions
            if box_w < 10 or box_h < 10:
                continue

            # Scale back to original image size
            scale_x = w / score_text.shape[1]
            scale_y = h / score_text.shape[0]

            x = int(x * scale_x)
            y = int(y * scale_y)
            box_w = int(box_w * scale_x)
            box_h = int(box_h * scale_y)

            # Create quad (4-point polygon)
            quad = np.array([
                [x, y],
                [x + box_w, y],
                [x + box_w, y + box_h],
                [x, y + box_h]
            ], dtype=np.float32)

            quads.append(quad)

        return quads
