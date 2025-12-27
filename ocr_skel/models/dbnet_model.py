"""DBNet: Differentiable Binarization Network for text detection

Original paper: https://arxiv.org/abs/1911.08947
Based on ResNet backbone with FPN neck and DB head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights


class ConvBnRelu(nn.Module):
    """Convolution + BatchNorm + ReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FPN(nn.Module):
    """Feature Pyramid Network neck"""
    def __init__(self, in_channels_list, out_channels=256):
        """
        Args:
            in_channels_list: list of input channels from backbone [C2, C3, C4, C5]
            out_channels: output channels for all FPN levels
        """
        super(FPN, self).__init__()

        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )

        # Smooth layers (3x3 conv after upsampling)
        self.smooth_convs = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            self.smooth_convs.append(
                ConvBnRelu(out_channels, out_channels, kernel_size=3, padding=1)
            )

    def forward(self, features):
        """
        Args:
            features: list of feature maps from backbone [C2, C3, C4, C5]

        Returns:
            list of FPN feature maps [P2, P3, P4, P5]
        """
        # Build lateral connections
        laterals = []
        for feat, lateral_conv in zip(features, self.lateral_convs):
            laterals.append(lateral_conv(feat))

        # Build top-down path
        fpn_features = []
        for i in range(len(laterals) - 1, -1, -1):
            if i == len(laterals) - 1:
                # Top level
                fpn_feat = laterals[i]
            else:
                # Upsample and add
                upsampled = F.interpolate(
                    fpn_feat,
                    size=laterals[i].shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                fpn_feat = laterals[i] + upsampled

            # Apply smooth conv
            fpn_feat = self.smooth_convs[i](fpn_feat)
            fpn_features.insert(0, fpn_feat)

        return fpn_features


class DBHead(nn.Module):
    """Differentiable Binarization Head"""
    def __init__(self, in_channels=256, k=50):
        """
        Args:
            in_channels: input channels from FPN
            k: amplification factor for binarization
        """
        super(DBHead, self).__init__()
        self.k = k

        # Probability map branch
        self.prob_conv = nn.Sequential(
            ConvBnRelu(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

        # Threshold map branch
        self.thresh_conv = nn.Sequential(
            ConvBnRelu(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: fused FPN features (B, C, H, W)

        Returns:
            dict with keys:
                - 'binary': approximate binary map (B, 1, H, W)
                - 'prob': probability map (B, 1, H, W)
                - 'thresh': threshold map (B, 1, H, W)
        """
        # Generate probability and threshold maps
        prob_map = self.prob_conv(x)
        thresh_map = self.thresh_conv(x)

        # Differentiable binarization
        # binary_map ≈ 1 / (1 + exp(-k * (prob_map - thresh_map)))
        if self.training:
            binary_map = 1.0 / (1.0 + torch.exp(-self.k * (prob_map - thresh_map)))
        else:
            # During inference, use hard threshold
            binary_map = (prob_map > thresh_map).float()

        return {
            'binary': binary_map,
            'prob': prob_map,
            'thresh': thresh_map
        }


class DBNet(nn.Module):
    """DBNet text detector model"""

    def __init__(self, backbone='resnet18', pretrained=True, freeze_backbone=False):
        """
        Args:
            backbone: 'resnet18' or 'resnet50'
            pretrained: use ImageNet pretrained backbone
            freeze_backbone: freeze backbone weights
        """
        super(DBNet, self).__init__()

        # Build backbone
        if backbone == 'resnet18':
            if pretrained:
                resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                resnet = resnet18(weights=None)
            in_channels_list = [64, 128, 256, 512]  # C2, C3, C4, C5
        elif backbone == 'resnet50':
            if pretrained:
                resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                resnet = resnet50(weights=None)
            in_channels_list = [256, 512, 1024, 2048]  # C2, C3, C4, C5
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract backbone layers (remove avgpool and fc)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # C2
        self.layer2 = resnet.layer2  # C3
        self.layer3 = resnet.layer3  # C4
        self.layer4 = resnet.layer4  # C5

        # Freeze backbone if requested
        if freeze_backbone:
            for param in [self.conv1, self.bn1, self.layer1, self.layer2,
                         self.layer3, self.layer4]:
                for p in param.parameters():
                    p.requires_grad = False

        # Build FPN neck
        self.fpn = FPN(in_channels_list, out_channels=256)

        # Fuse FPN features
        self.fuse_conv = ConvBnRelu(256 * 4, 256, kernel_size=3, padding=1)

        # Build DB head
        self.head = DBHead(in_channels=256, k=50)

    def forward(self, x):
        """
        Args:
            x: input image tensor (B, 3, H, W)

        Returns:
            dict with keys:
                - 'binary': binary text map (B, 1, H, W)
                - 'prob': probability map (B, 1, H, W)
                - 'thresh': threshold map (B, 1, H, W)
        """
        # Backbone feature extraction
        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.maxpool(c1)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        features = [c2, c3, c4, c5]

        # FPN
        fpn_features = self.fpn(features)

        # Upsample all FPN features to same size (C2 size)
        target_size = fpn_features[0].shape[2:]
        upsampled_features = []
        for feat in fpn_features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled_features.append(feat)

        # Concatenate and fuse
        fused = torch.cat(upsampled_features, dim=1)
        fused = self.fuse_conv(fused)

        # DB head
        outputs = self.head(fused)

        return outputs
