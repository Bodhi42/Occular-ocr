"""CRNN Recognizer with ONNX Runtime"""

import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import List, Tuple
from pathlib import Path
from itertools import groupby


INPUT_HEIGHT = 32


class CRNNRecognizerONNX:
    """CRNN text recognizer using ONNX Runtime (2.4x faster on CPU)"""

    def __init__(self, languages: List[str] = None, gpu: bool = False):
        self.languages = languages or ['ru', 'en']

        weights_dir = Path(__file__).parent / "weights"
        onnx_path = weights_dir / "crnn_encoder.onnx"
        pth_path = weights_dir / "crnn_mobilenet_large.pth"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # Load vocab from PyTorch checkpoint
        import torch
        checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
        self.vocab = checkpoint['vocab']

        # Create ONNX session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 0  # Auto-detect

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)

        print(f"Loaded CRNN ONNX (vocab={len(self.vocab)} chars)")

    def recognize(self, image: np.ndarray, quads: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Recognize text in given regions using batch inference with width bucketing"""
        if not quads:
            return []

        # Preprocess all crops
        preprocessed = []
        valid_indices = []
        results = [("", 0.0)] * len(quads)  # Pre-fill with empty results

        for i, quad in enumerate(quads):
            crop = self._crop_quad(image, quad)
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            tensor = self._preprocess_single(crop)
            preprocessed.append(tensor)
            valid_indices.append(i)

        if not preprocessed:
            return results

        # Group by similar width to minimize padding impact
        # Sort by width and process in buckets
        width_indices = sorted(range(len(preprocessed)), key=lambda i: preprocessed[i].shape[2])

        BUCKET_SIZE = 4  # Process in small batches to minimize padding

        for bucket_start in range(0, len(width_indices), BUCKET_SIZE):
            bucket_indices = width_indices[bucket_start:bucket_start + BUCKET_SIZE]
            bucket_tensors = [preprocessed[i] for i in bucket_indices]

            # Create batch for this bucket
            batch_tensor = self._create_batch(bucket_tensors)

            # ONNX inference
            logits_batch = self.session.run(None, {'input': batch_tensor})[0]

            # Decode each result
            for batch_idx, prep_idx in enumerate(bucket_indices):
                orig_idx = valid_indices[prep_idx]
                logits = logits_batch[batch_idx:batch_idx+1]
                text, confidence = self._ctc_decode(logits)
                results[orig_idx] = (text, confidence)

        return results

    def _preprocess_single(self, image: np.ndarray) -> np.ndarray:
        """Preprocess single crop (without batch dim)"""
        h, w = image.shape[:2]
        scale = INPUT_HEIGHT / h
        new_w = max(8, int(w * scale))

        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((new_w, INPUT_HEIGHT), Image.BILINEAR)

        tensor = np.array(pil_img).transpose(2, 0, 1).astype(np.float32) / 255.0
        return tensor  # (C, H, W)

    def _create_batch(self, tensors: List[np.ndarray]) -> np.ndarray:
        """Create batch by padding to max width"""
        max_w = max(t.shape[2] for t in tensors)

        batch = []
        for t in tensors:
            c, h, w = t.shape
            if w < max_w:
                # Pad with zeros on the right
                padded = np.zeros((c, h, max_w), dtype=np.float32)
                padded[:, :, :w] = t
                batch.append(padded)
            else:
                batch.append(t)

        return np.stack(batch, axis=0)  # (N, C, H, W)

    def _crop_quad(self, image: np.ndarray, quad: np.ndarray) -> np.ndarray:
        quad = np.array(quad)
        x_min, x_max = int(quad[:, 0].min()), int(quad[:, 0].max())
        y_min, y_max = int(quad[:, 1].min()), int(quad[:, 1].max())

        h, w = image.shape[:2]
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)

        return image[y_min:y_max, x_min:x_max]

    def _ctc_decode(self, logits: np.ndarray) -> Tuple[str, float]:
        """CTC best path decoding"""
        blank_idx = len(self.vocab)

        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        # Confidence
        max_probs = probs.max(axis=-1)
        confidence = float(max_probs.min())

        # Decode
        pred_indices = logits.argmax(axis=-1)[0]  # (T,)

        chars = []
        for k, _ in groupby(pred_indices.tolist()):
            if k != blank_idx and 0 <= k < len(self.vocab):
                chars.append(self.vocab[k])

        return ''.join(chars), confidence
