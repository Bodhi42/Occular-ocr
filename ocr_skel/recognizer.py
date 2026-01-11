"""CRNN MobileNetV3 Recognizer"""

import numpy as np
import torch
from PIL import Image
from typing import List, Tuple
from pathlib import Path

from .models.crnn_mobilenet import crnn_mobilenet_v3_large


# Model configuration
INPUT_HEIGHT = 32


class CRNNRecognizer:
    """CRNN text recognizer (MobileNetV3-Large)"""

    def __init__(self, languages: List[str] = None, gpu: bool = True):
        """
        Args:
            languages: список языков (не используется, оставлен для совместимости)
            gpu: использовать GPU если доступен
        """
        self.languages = languages or ['ru', 'en']
        self.gpu = gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.gpu else 'cpu')

        # Load weights
        weights_path = Path(__file__).parent / "weights" / "crnn_mobilenet_large.pth"

        if not weights_path.exists():
            raise FileNotFoundError(f"Recognition weights not found: {weights_path}")

        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.vocab = checkpoint['vocab']
        val_cer = checkpoint.get('val_cer', 0)

        # Create model
        self.model = crnn_mobilenet_v3_large(vocab=self.vocab)

        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded CRNN weights (CER={val_cer:.4f})")

        self.model.to(self.device)
        self.model.eval()

    def recognize(self, image: np.ndarray, quads: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Распознать текст в заданных областях

        Args:
            image: изображение в формате RGB (H, W, C)
            quads: список quad-контуров [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns:
            Список пар (текст, confidence)
        """
        results = []

        for quad in quads:
            # Crop and preprocess region
            crop = self._crop_quad(image, quad)

            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                results.append(("", 0.0))
                continue

            # Preprocess for model
            img_tensor = self._preprocess_image(crop)
            img_tensor = img_tensor.to(self.device)

            # Forward pass
            with torch.no_grad():
                output = self.model(img_tensor)

            # Get predictions
            if 'preds' in output and output['preds']:
                text, confidence = output['preds'][0]
            else:
                text, confidence = "", 0.0

            results.append((text, confidence))

        return results

    def _crop_quad(self, image: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """Crop quad region from image"""
        quad = np.array(quad)
        x_coords = quad[:, 0]
        y_coords = quad[:, 1]

        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())

        # Clamp to image bounds
        h, w = image.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        return image[y_min:y_max, x_min:x_max]

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for CRNN (matches training preprocessing exactly)

        Args:
            image: RGB image (H, W, C)

        Returns:
            Tensor (1, 3, 32, W) - variable width
        """
        h, w = image.shape[:2]

        # Scale to target height, keeping aspect ratio
        scale = INPUT_HEIGHT / h
        new_w = int(w * scale)
        new_w = max(new_w, 8)  # Minimum width

        # Use PIL BILINEAR resize (same as training)
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((new_w, INPUT_HEIGHT), Image.BILINEAR)

        # Convert to tensor and normalize (same as training)
        tensor = torch.tensor(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)

        return tensor
