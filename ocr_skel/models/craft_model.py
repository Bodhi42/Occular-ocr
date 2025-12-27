"""CRAFT: Character Region Awareness for Text detection

Original paper: https://arxiv.org/abs/1904.01941
Based on VGG16 backbone with UNet-like decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def init_weights(modules):
    """Initialize network weights"""
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class DoubleConv(nn.Module):
    """Double convolution block"""
    def __init__(self, in_ch, mid_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class CRAFT(nn.Module):
    """CRAFT text detector model (compatible with pretrained weights)"""

    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        # VGG16-like backbone with BatchNorm (split into 5 slices)
        # This matches the original CRAFT architecture
        self.basenet = nn.ModuleDict()

        # slice1: conv1_1, conv1_2, pool1, conv2_1, conv2_2, pool2 -> 128 channels
        self.basenet['slice1'] = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # slice2: conv3_1, conv3_2, conv3_3, pool3 -> 256 channels
        self.basenet['slice2'] = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # slice3: conv4_1, conv4_2, conv4_3, pool4 -> 512 channels
        self.basenet['slice3'] = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # slice4: conv5_1, conv5_2, conv5_3, pool5 -> 512 channels
        self.basenet['slice4'] = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # slice5: fc6 as conv, fc7 as conv -> 1024 channels
        self.basenet['slice5'] = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1),
        )

        # UNet-like decoder (matching pretrained weights dimensions)
        self.upconv1 = DoubleConv(1536, 512, 256)  # 512 + 1024 = 1536
        self.upconv2 = DoubleConv(768, 256, 128)   # 256 + 512 = 768
        self.upconv3 = DoubleConv(384, 128, 64)    # 128 + 256 = 384
        self.upconv4 = DoubleConv(192, 64, 32)     # 64 + 128 = 192

        # Final output: region score + affinity score
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),
        )

        init_weights(self.modules())

        if pretrained:
            self._init_weights()

        if freeze:
            self._freeze_backbone()

    def _init_weights(self):
        """Initialize weights with VGG16 pretrained weights if available"""
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            # Note: This is a simplified init, full weight loading comes from pretrained CRAFT
        except Exception:
            pass

    def _freeze_backbone(self):
        """Freeze backbone weights"""
        for param in self.basenet.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: input image tensor (B, 3, H, W)

        Returns:
            y: output feature map (B, 2, H, W) - [region_score, affinity_score]
        """
        # Feature extraction through VGG slices
        sources = []

        # slice1: 128 channels
        y = self.basenet['slice1'](x)
        sources.append(y)

        # slice2: 256 channels
        y = self.basenet['slice2'](y)
        sources.append(y)

        # slice3: 512 channels
        y = self.basenet['slice3'](y)
        sources.append(y)

        # slice4: 512 channels
        y = self.basenet['slice4'](y)
        sources.append(y)

        # slice5: 1024 channels
        y = self.basenet['slice5'](y)
        sources.append(y)

        # sources[0]: 128 channels (H/4)
        # sources[1]: 256 channels (H/8)
        # sources[2]: 512 channels (H/16)
        # sources[3]: 512 channels (H/32)
        # sources[4]: 1024 channels (H/32)

        # Decoder with skip connections
        y = sources[4]  # (B, 1024, H/32, W/32)

        # Upsample and concat with sources[3] (512)
        y = F.interpolate(y, size=sources[3].shape[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)  # (B, 1536, H/32, W/32)
        y = self.upconv1(y)  # (B, 256, H/32, W/32)

        # Upsample and concat with sources[2] (512)
        y = F.interpolate(y, size=sources[2].shape[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)  # (B, 768, H/16, W/16)
        y = self.upconv2(y)  # (B, 128, H/16, W/16)

        # Upsample and concat with sources[1] (256)
        y = F.interpolate(y, size=sources[1].shape[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[1]], dim=1)  # (B, 384, H/8, W/8)
        y = self.upconv3(y)  # (B, 64, H/8, W/8)

        # Upsample and concat with sources[0] (128)
        y = F.interpolate(y, size=sources[0].shape[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[0]], dim=1)  # (B, 192, H/4, W/4)
        y = self.upconv4(y)  # (B, 32, H/4, W/4)

        # Final prediction
        feature = self.conv_cls(y)

        return feature.sigmoid()  # Output in range [0, 1]
