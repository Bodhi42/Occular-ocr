"""
DBNet (Differentiable Binarization Network) for text detection.

Architecture:
- Backbone: ResNet18 (ImageNet pretrained)
- Neck: FPNC (lateral_channels=256, out_channels=64)
- Head: DBHead (in_channels=256 from 64*4 concat)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from typing import List, Dict


class ConvModule(nn.Module):
    """Conv + optional BN + optional ReLU"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        norm: bool = False,
        activation: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels) if norm else None
        self.relu = nn.ReLU(inplace=False) if activation else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FPNC(nn.Module):
    """
    FPN-like feature fusion module.

    - lateral_channels=256 (internal)
    - out_channels=64 (after smooth conv)
    - Output is concatenation of all scales: 64*4=256 channels
    """
    def __init__(
        self,
        in_channels: List[int],
        lateral_channels: int = 256,
        out_channels: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.lateral_channels = lateral_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        self.lateral_convs = nn.ModuleList()
        self.smooth_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                lateral_channels,
                kernel_size=1,
                bias=False,
            )
            smooth_conv = ConvModule(
                lateral_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            self.lateral_convs.append(l_conv)
            self.smooth_convs.append(smooth_conv)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1e-4)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: [C2, C3, C4, C5] feature maps from backbone

        Returns:
            Concatenated features of shape (N, out_channels*4, H, W)
        """
        assert len(inputs) == len(self.in_channels)

        # Build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest'
            )

        # Build outputs with smooth convs
        outs = [
            self.smooth_convs[i](laterals[i])
            for i in range(len(laterals))
        ]

        # Upsample all to first level size
        target_size = outs[0].shape[2:]
        for i in range(1, len(outs)):
            outs[i] = F.interpolate(outs[i], size=target_size, mode='nearest')

        # Concatenate: 64*4 = 256 channels
        out = torch.cat(outs, dim=1)

        return out


class DBHead(nn.Module):
    """
    DBNet detection head.

    Takes concatenated FPN features (256 channels) and outputs:
    - prob_map: probability of text
    - thresh_map: adaptive threshold
    - binary_map: differentiable binarization result
    """
    def __init__(self, in_channels: int = 256, k: int = 50):
        super().__init__()
        self.k = k

        # Binarize branch (probability map)
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, stride=2),
            nn.Sigmoid()
        )

        # Threshold branch
        self.threshold = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, stride=2),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1e-4)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: FPN features (N, 256, H, W)

        Returns:
            dict with 'prob', 'thresh', 'binary' maps
        """
        prob_map = self.binarize(x)
        thresh_map = self.threshold(x)

        # Differentiable binarization
        if self.training:
            binary_map = torch.reciprocal(
                1.0 + torch.exp(-self.k * (prob_map - thresh_map))
            )
        else:
            binary_map = (prob_map > thresh_map).float()

        return {
            'prob': prob_map,
            'thresh': thresh_map,
            'binary': binary_map
        }


class DBNet(nn.Module):
    """
    DBNet text detector.

    Architecture:
        Backbone (ResNet18) -> FPNC -> DBHead
    """

    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.backbone_name = backbone

        if backbone != 'resnet18':
            raise ValueError(f"Only resnet18 is supported. Got: {backbone}")

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)
        in_channels_list = [64, 128, 256, 512]

        # Extract backbone layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if freeze_backbone:
            for param in [self.conv1, self.bn1, self.layer1, self.layer2,
                         self.layer3, self.layer4]:
                for p in param.parameters():
                    p.requires_grad = False

        # FPN neck
        self.neck = FPNC(
            in_channels=in_channels_list,
            lateral_channels=256,
            out_channels=64
        )

        # Detection head
        self.head = DBHead(in_channels=256, k=50)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: input image (N, 3, H, W)

        Returns:
            dict with 'prob', 'thresh', 'binary' maps
        """
        # Backbone
        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.maxpool(c1)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        features = [c2, c3, c4, c5]

        # FPN neck
        fused = self.neck(features)

        # Detection head
        outputs = self.head(fused)

        return outputs
