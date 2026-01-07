"""DBNet model compatible with trial41 checkpoint structure"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import cv2
import pyclipper
from shapely.geometry import Polygon
from pathlib import Path


class ConvModule(nn.Module):
    """Conv + BN module (matches checkpoint's lateral_convs structure)"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

    def forward(self, x):
        return self.conv(x)


class SmoothConv(nn.Module):
    """Smooth conv for FPN (matches checkpoint's smooth_convs structure)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=False)

    def forward(self, x):
        return self.conv(x)


class Neck(nn.Module):
    """FPN Neck matching trial41 checkpoint structure"""
    def __init__(self, in_channels_list=[64, 128, 256, 512], inner_channels=256, out_channels=64):
        super().__init__()

        # Lateral convs: reduce channel to inner_channels
        self.lateral_convs = nn.ModuleList([
            ConvModule(in_ch, inner_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # Smooth convs: inner_channels -> out_channels
        self.smooth_convs = nn.ModuleList([
            SmoothConv(inner_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features):
        """
        Args:
            features: [C2, C3, C4, C5] from backbone

        Returns:
            fused feature map
        """
        # Lateral connections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down path with upsampling
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode='bilinear', align_corners=False
            )

        # Smooth and upsample all to C2 size
        target_size = laterals[0].shape[2:]
        smoothed = []
        for i, (lateral, smooth) in enumerate(zip(laterals, self.smooth_convs)):
            feat = smooth(lateral)
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            smoothed.append(feat)

        # Concatenate all levels
        fused = torch.cat(smoothed, dim=1)  # 64*4 = 256 channels
        return fused


class Head(nn.Module):
    """DB Head matching trial41 checkpoint structure"""
    def __init__(self, in_channels=256, k=50):
        super().__init__()
        self.k = k

        # Binarize branch (probability map)
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid()
        )

        # Threshold branch
        self.threshold = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        prob = self.binarize(x)
        thresh = self.threshold(x)

        if self.training:
            binary = 1.0 / (1.0 + torch.exp(-self.k * (prob - thresh)))
        else:
            binary = (prob > thresh).float()

        return {
            'binary': binary,
            'prob': prob,
            'thresh': thresh
        }


class DBNetTrial41(nn.Module):
    """DBNet model matching trial41 checkpoint architecture"""

    def __init__(self, pretrained=True):
        super().__init__()

        # ResNet18 backbone
        if pretrained:
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet18(weights=None)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Neck (FPN)
        self.neck = Neck(
            in_channels_list=[64, 128, 256, 512],
            inner_channels=256,
            out_channels=64
        )

        # Head
        self.head = Head(in_channels=256, k=50)

    def forward(self, x):
        # Backbone
        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.maxpool(c1)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Neck
        fused = self.neck([c2, c3, c4, c5])

        # Head
        outputs = self.head(fused)

        return outputs


def load_trial41_model(weights_path, device='cuda'):
    """Load trial41 checkpoint into compatible model"""
    model = DBNetTrial41(pretrained=False)

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Remove module. prefix if present
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Combine params with top-level metrics
    params = checkpoint.get('params', {})
    params['f1'] = checkpoint.get('f1', 0)
    params['precision'] = checkpoint.get('precision', 0)
    params['recall'] = checkpoint.get('recall', 0)

    return model, params


def unclip_polygon(polygon, unclip_ratio):
    """Expand polygon using pyclipper (reverses shrink from GT generation)"""
    poly = Polygon(polygon)
    if poly.area < 1:
        return polygon

    distance = poly.area * unclip_ratio / poly.length

    offset = pyclipper.PyclipperOffset()
    offset.AddPath(
        polygon.astype(np.int32).tolist(),
        pyclipper.JT_ROUND,
        pyclipper.ET_CLOSEDPOLYGON
    )

    expanded = offset.Execute(distance)
    if len(expanded) == 0:
        return polygon
    return np.array(expanded[0])


class DBNetTrial41Detector:
    """Ready-to-use detector with trial41 weights"""

    def __init__(self, weights_path=None, device='cuda'):
        if weights_path is None:
            weights_path = Path(__file__).parent.parent / 'weights' / 'dbnet_trial41.pth'

        self.device = device
        self.model, self.params = load_trial41_model(weights_path, device)

        # Extract params
        self.threshold = self.params.get('threshold', 0.24)
        self.min_area = self.params.get('min_area', 38)
        self.unclip_ratio = self.params.get('unclip_ratio', 2.0)

    def preprocess(self, image, target_size=640):
        """Preprocess image for inference"""
        h, w = image.shape[:2]

        ratio = target_size / max(h, w)
        new_h = int(h * ratio) - int(h * ratio) % 32
        new_w = int(w * ratio) - int(w * ratio) % 32

        scale_x = w / new_w
        scale_y = h / new_h

        img_resized = cv2.resize(image, (new_w, new_h))
        img_resized = img_resized.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_resized = (img_resized - mean) / std

        return img_resized, scale_x, scale_y

    def detect(self, image):
        """
        Detect text regions in image.

        Args:
            image: RGB numpy array (H, W, 3)

        Returns:
            List of quads (4 points each)
        """
        h, w = image.shape[:2]

        # Preprocess
        img_resized, scale_x, scale_y = self.preprocess(image)

        # To tensor
        x = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0)
        x = x.to(self.device)

        # Forward
        with torch.no_grad():
            outputs = self.model(x)

        prob = outputs['prob'][0, 0].cpu().numpy()

        # Get contours
        binary_mask = (prob > self.threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        quads = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue

            # Get polygon points
            polygon = contour.reshape(-1, 2).astype(np.float64)

            # Unclip - expand the polygon
            try:
                polygon = unclip_polygon(polygon, self.unclip_ratio)
            except:
                pass

            if len(polygon) < 4:
                continue

            # Get minimum area rect
            rect = cv2.minAreaRect(polygon.astype(np.int32))
            box = cv2.boxPoints(rect)

            # Scale back to original size
            box[:, 0] = np.clip(box[:, 0] * scale_x, 0, w)
            box[:, 1] = np.clip(box[:, 1] * scale_y, 0, h)

            quads.append(box.astype(np.float32))

        return quads
