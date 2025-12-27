"""CRNN Recognizer wrapper"""

import numpy as np
import torch
import cv2
from typing import List, Tuple
from .models.crnn_model import create_crnn
from .models.model_utils import load_crnn_weights


# Character set: digits + lowercase letters + blank
# Total 37 characters (0-9, a-z, blank)
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"
BLANK_IDX = len(CHARSET)  # CTC blank token


class CRNNRecognizer:
    """CRNN text recognizer (CNN+BiLSTM+CTC, pretrained)"""

    def __init__(self, languages: List[str] = None, gpu: bool = True):
        """
        Args:
            languages: список языков для распознавания (не используется, оставлен для совместимости)
            gpu: использовать GPU если доступен
        """
        self.languages = languages or ['en']
        self.gpu = gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.gpu else 'cpu')

        # Character set
        self.charset = CHARSET
        self.num_classes = len(self.charset) + 1  # +1 for CTC blank

        # Initialize CRNN model
        self.model = create_crnn(num_classes=self.num_classes, pretrained=False)

        # Load pretrained weights
        try:
            self.model = load_crnn_weights(self.model, device=self.device)
        except Exception as e:
            print(f"Warning: Could not load CRNN weights: {e}")
            print("Using model with random initialization")

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

            # Preprocess for CRNN
            img_tensor = self._preprocess_image(crop)
            img_tensor = img_tensor.to(self.device)

            # Forward pass
            with torch.no_grad():
                output = self.model(img_tensor)  # (1, W/4, num_classes)

            # Decode CTC output
            text, confidence = self._decode_output(output)

            results.append((text, confidence))

        return results

    def _crop_quad(self, image: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """
        Crop quad region from image

        Args:
            image: RGB image (H, W, C)
            quad: quad coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns:
            Cropped region
        """
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

        # Crop region
        crop = image[y_min:y_max, x_min:x_max]

        return crop

    def _preprocess_image(self, image: np.ndarray, target_height=32) -> torch.Tensor:
        """
        Preprocess image for CRNN

        Args:
            image: RGB image (H, W, C)
            target_height: target height (32 for CRNN)

        Returns:
            Tensor (1, 1, 32, W')
        """
        # Convert to grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        h, w = gray.shape

        # Calculate target width maintaining aspect ratio
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)

        # Minimum width
        target_width = max(target_width, target_height)

        # Resize to target height
        resized = cv2.resize(gray, (target_width, target_height))

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Convert to tensor: (H, W) -> (1, 1, H, W)
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

        return tensor

    def _decode_output(self, output: torch.Tensor) -> Tuple[str, float]:
        """
        Decode CTC output to text

        Args:
            output: model output (1, T, num_classes)

        Returns:
            Tuple of (text, confidence)
        """
        # Get predictions: (1, T, num_classes) -> (T, num_classes)
        output = output.squeeze(0)

        # Get most likely character at each timestep
        probs = torch.softmax(output, dim=1)
        max_probs, preds = probs.max(dim=1)

        # CTC greedy decoding
        preds = preds.cpu().numpy()
        max_probs = max_probs.cpu().numpy()

        # Remove consecutive duplicates and blanks
        chars = []
        prev_char = None
        confidences = []

        for pred, prob in zip(preds, max_probs):
            if pred == BLANK_IDX:
                prev_char = None
                continue

            if pred != prev_char:
                if pred < len(self.charset):
                    chars.append(self.charset[pred])
                    confidences.append(prob)
                prev_char = pred

        text = ''.join(chars)
        confidence = float(np.mean(confidences)) if confidences else 0.0

        return text, confidence
