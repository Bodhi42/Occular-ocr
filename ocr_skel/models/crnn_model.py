"""CRNN: Convolutional Recurrent Neural Network for text recognition

Original paper: https://arxiv.org/abs/1507.05717
Architecture: CNN (feature extraction) + BiLSTM (sequence modeling) + CTC (decoding)
"""

import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM layer"""

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, seq_len, output_size)
        """
        recurrent, _ = self.rnn(x)
        output = self.linear(recurrent)
        return output


class CRNN(nn.Module):
    """CRNN text recognition model"""

    def __init__(self, img_height=32, num_channels=1, num_classes=37, hidden_size=256):
        """
        Args:
            img_height: input image height (fixed to 32)
            num_channels: number of input channels (1 for grayscale, 3 for RGB)
            num_classes: number of character classes (including blank for CTC)
            hidden_size: LSTM hidden size
        """
        super(CRNN, self).__init__()

        self.img_height = img_height
        self.num_channels = num_channels
        self.num_classes = num_classes

        # CNN backbone (feature extraction)
        # Input: (B, C, 32, W)
        # Output: (B, 512, 1, W/4)
        self.cnn = nn.Sequential(
            # Conv1: (B, C, 32, W) -> (B, 64, 32, W)
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 64, 16, W/2)

            # Conv2: (B, 64, 16, W/2) -> (B, 128, 16, W/2)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 128, 8, W/4)

            # Conv3: (B, 128, 8, W/4) -> (B, 256, 8, W/4)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Conv4: (B, 256, 8, W/4) -> (B, 256, 8, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # (B, 256, 4, W/4)

            # Conv5: (B, 256, 4, W/4) -> (B, 512, 4, W/4)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Conv6: (B, 512, 4, W/4) -> (B, 512, 4, W/4)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # (B, 512, 2, W/4)

            # Conv7: (B, 512, 2, W/4) -> (B, 512, 1, W/4)
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # RNN (sequence modeling)
        # Input: (B, W/4, 512)
        # Output: (B, W/4, num_classes)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: input image tensor (B, C, H, W) where H should be 32

        Returns:
            output: (B, W/4, num_classes) - log probabilities for CTC
        """
        # CNN feature extraction
        conv = self.cnn(x)  # (B, 512, 1, W/4)

        # Reshape for RNN: (B, 512, 1, W/4) -> (B, W/4, 512)
        b, c, h, w = conv.size()
        assert h == 1, "Height of conv output must be 1"
        conv = conv.squeeze(2)  # (B, 512, W/4)
        conv = conv.permute(0, 2, 1)  # (B, W/4, 512)

        # RNN sequence modeling
        output = self.rnn(conv)  # (B, W/4, num_classes)

        return output


def create_crnn(num_classes=37, pretrained=False):
    """
    Create CRNN model

    Args:
        num_classes: number of character classes (default 37: 0-9, a-z, blank)
        pretrained: load pretrained weights (not implemented yet)

    Returns:
        CRNN model
    """
    model = CRNN(
        img_height=32,
        num_channels=1,  # Grayscale
        num_classes=num_classes,
        hidden_size=256
    )

    return model
