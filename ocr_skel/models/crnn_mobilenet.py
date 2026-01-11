"""CRNN with MobileNetV3-Large backbone for text recognition (standalone, no docTR dependency)"""

from typing import Any, Tuple, List
from itertools import groupby
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenetv3


def create_mobilenet_v3_large_r() -> nn.Module:
    """Create MobileNetV3-Large with rectangular strides for text recognition

    Rectangular strides (2, 1) preserve width while reducing height,
    which is important for text recognition where width carries sequence information.
    """
    model = mobilenetv3.mobilenet_v3_large(weights=None)

    # Rectangular stride layers - same as docTR
    rect_stride_layers = [
        "features.4.block.1.0",
        "features.7.block.1.0",
        "features.13.block.1.0"
    ]

    for layer_name in rect_stride_layers:
        m = model
        for child in layer_name.split("."):
            m = getattr(m, child)
        m.stride = (2, 1)

    return model.features  # Return only feature extractor part


class CRNN(nn.Module):
    """CRNN with MobileNetV3-Large backbone for text recognition

    Architecture:
    - Feature extractor: MobileNetV3-Large with rectangular strides
    - Sequence modeling: 2-layer bidirectional LSTM (rnn_units=128)
    - Output: Linear layer for CTC decoding

    Compatible with docTR trained weights.
    """

    def __init__(
        self,
        vocab: str,
        input_shape: Tuple[int, int, int] = (3, 32, 128),
        rnn_units: int = 128,
    ):
        super().__init__()
        self.vocab = vocab
        self.max_length = 128

        # Feature extractor (MobileNetV3-Large with rect strides)
        self.feat_extractor = create_mobilenet_v3_large_r()

        # Compute LSTM input size from feature extractor output
        with torch.inference_mode():
            out_shape = self.feat_extractor(torch.zeros((1, *input_shape))).shape
        lstm_in = out_shape[1] * out_shape[2]  # C * H

        # Bidirectional LSTM decoder
        self.decoder = nn.LSTM(
            input_size=lstm_in,
            hidden_size=rnn_units,
            batch_first=True,
            num_layers=2,
            bidirectional=True,
        )

        # Output layer: 2*rnn_units (bidirectional) -> vocab_size + 1 (blank)
        self.linear = nn.Linear(
            in_features=2 * rnn_units,
            out_features=len(vocab) + 1
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize decoder and linear layer weights"""
        for n, m in self.named_modules():
            if n.startswith("feat_extractor."):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        target: List[str] = None,
        return_preds: bool = True,
    ) -> dict:
        """
        Forward pass

        Args:
            x: Input tensor (B, C, H, W)
            target: Target strings for training (optional)
            return_preds: Whether to return decoded predictions

        Returns:
            Dictionary with 'preds' and optionally 'loss'
        """
        # Extract features: (B, C, H, W) -> (B, C', H', W')
        features = self.feat_extractor(x)

        # Reshape for LSTM: (B, C', H', W') -> (B, W', C'*H')
        b, c, h, w = features.shape
        features_seq = features.reshape(b, c * h, w)
        features_seq = features_seq.transpose(1, 2)  # (B, W', C'*H')

        # LSTM: (B, W', C'*H') -> (B, W', 2*rnn_units)
        logits, _ = self.decoder(features_seq)

        # Linear: (B, W', 2*rnn_units) -> (B, W', vocab_size+1)
        logits = self.linear(logits)

        out = {}

        if return_preds:
            out["preds"] = self._ctc_decode(logits)

        if target is not None:
            out["loss"] = self._compute_loss(logits, target)

        return out

    def _ctc_decode(self, logits: torch.Tensor) -> List[Tuple[str, float]]:
        """CTC best path decoding

        Args:
            logits: Model output (B, T, vocab_size+1)

        Returns:
            List of (text, confidence) tuples
        """
        blank_idx = len(self.vocab)

        # Get probabilities and predictions
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values.min(dim=1).values  # confidence per sequence
        pred_indices = torch.argmax(logits, dim=-1)  # (B, T)

        results = []
        for seq, conf in zip(pred_indices, max_probs):
            # Collapse repeated characters and remove blanks
            chars = []
            for k, _ in groupby(seq.tolist()):
                if k != blank_idx and 0 <= k < len(self.vocab):
                    chars.append(self.vocab[k])
            text = ''.join(chars)
            results.append((text, float(conf)))

        return results

    def _compute_loss(self, logits: torch.Tensor, target: List[str]) -> torch.Tensor:
        """Compute CTC loss

        Args:
            logits: Model output (B, T, vocab_size+1)
            target: List of target strings

        Returns:
            CTC loss value
        """
        # Encode targets
        gt = []
        seq_len = []
        for text in target:
            encoded = [self.vocab.index(c) for c in text if c in self.vocab]
            gt.extend(encoded)
            seq_len.append(len(encoded))

        batch_len = logits.shape[0]
        input_length = logits.shape[1] * torch.ones(size=(batch_len,), dtype=torch.int32)

        # N x T x C -> T x N x C
        log_probs = F.log_softmax(logits.permute(1, 0, 2), dim=-1)

        ctc_loss = F.ctc_loss(
            log_probs,
            torch.tensor(gt, dtype=torch.int32),
            input_length,
            torch.tensor(seq_len, dtype=torch.int32),
            len(self.vocab),  # blank index
            zero_infinity=True,
        )

        return ctc_loss


def crnn_mobilenet_v3_large(
    vocab: str,
    input_shape: Tuple[int, int, int] = (3, 32, 128),
    **kwargs: Any
) -> CRNN:
    """Create CRNN with MobileNetV3-Large backbone

    Args:
        vocab: Character vocabulary string
        input_shape: Input tensor shape (C, H, W)

    Returns:
        CRNN model
    """
    return CRNN(vocab=vocab, input_shape=input_shape, **kwargs)
