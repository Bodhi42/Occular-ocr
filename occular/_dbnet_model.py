"""DBNet (ResNet50 backbone) — PyTorch-описание детектора для GPU-бэкенда.
Совпадает 1-в-1 с архитектурой, из которой экспортирован detector_dbnet_fp32.onnx
(DBNet(backbone='resnet50', head='db', inner=256), без DCN/ASF).
Backbone создаётся БЕЗ ImageNet-весов (weights=None) — их всё равно перезаписывает наш чекпоинт.
Используется ТОЛЬКО при gpu=True; CPU-путь работает на ONNX без torch/torchvision."""
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision


class ASF(nn.Module):
    """DBNet++ Adaptive Scale Fusion (только если head='dbpp')."""
    def __init__(self, inner, n=4):
        super().__init__(); self.n = n
        self.conv = nn.Conv2d(inner, inner, 3, padding=1)
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(inner, inner, 1), nn.ReLU(True),
                                nn.Conv2d(inner, inner, 1), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(inner, 1, 3, padding=1), nn.ReLU(True), nn.Conv2d(1, n, 1), nn.Sigmoid())
    def forward(self, fuse, outs):
        f = self.conv(fuse); f = f * self.ca(f)
        w = self.sa(f)
        return torch.cat([outs[i] * w[:, i:i + 1] for i in range(self.n)], 1)


class DCNv2(nn.Module):
    """Модулированный deformable conv (только если dcn=True)."""
    def __init__(self, src):
        super().__init__()
        from torchvision.ops import DeformConv2d
        co, ci, kh, kw = src.weight.shape; st = src.stride; pd = src.padding; dl = src.dilation
        self.dcn = DeformConv2d(ci, co, (kh, kw), stride=st, padding=pd, dilation=dl, bias=(src.bias is not None))
        self.off = nn.Conv2d(ci, 2 * kh * kw, (kh, kw), stride=st, padding=pd, dilation=dl)
        self.msk = nn.Conv2d(ci, kh * kw, (kh, kw), stride=st, padding=pd, dilation=dl)
    def forward(self, x):
        return self.dcn(x, self.off(x), torch.sigmoid(self.msk(x)))


def replace_dcn(layer):
    for blk in layer:
        c = getattr(blk, 'conv2', None)
        if isinstance(c, nn.Conv2d) and c.kernel_size == (3, 3): blk.conv2 = DCNv2(c)


class DBNet(nn.Module):
    def __init__(self, inner=256, backbone='resnet50', head='db', dcn=False):
        super().__init__()
        self.mb = (backbone == 'mobilenet_v3_large')
        if self.mb:
            m3 = torchvision.models.mobilenet_v3_large(weights=None)
            self.features = m3.features; self.taps = [4, 7, 13, 17]; ch = [24, 40, 112, 960]
        else:
            if backbone == 'resnet50':
                bb = torchvision.models.resnet50(weights=None); ch = [256, 512, 1024, 2048]
            elif backbone == 'resnet34':
                bb = torchvision.models.resnet34(weights=None); ch = [64, 128, 256, 512]
            else:
                bb = torchvision.models.resnet18(weights=None); ch = [64, 128, 256, 512]
            self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
            self.l1, self.l2, self.l3, self.l4 = bb.layer1, bb.layer2, bb.layer3, bb.layer4
            if dcn: replace_dcn(self.l3); replace_dcn(self.l4)
        self.lat = nn.ModuleList([nn.Conv2d(c, inner, 1) for c in ch])
        self.smooth = nn.ModuleList([nn.Conv2d(inner, inner // 4, 3, padding=1) for _ in ch])
        self.head = nn.Sequential(nn.Conv2d(inner, inner // 4, 3, padding=1), nn.BatchNorm2d(inner // 4), nn.ReLU(True),
                                  nn.ConvTranspose2d(inner // 4, inner // 4, 2, 2), nn.BatchNorm2d(inner // 4), nn.ReLU(True),
                                  nn.ConvTranspose2d(inner // 4, 1, 2, 2))
        self.thr = nn.Sequential(nn.Conv2d(inner, inner // 4, 3, padding=1), nn.BatchNorm2d(inner // 4), nn.ReLU(True),
                                 nn.ConvTranspose2d(inner // 4, inner // 4, 2, 2), nn.BatchNorm2d(inner // 4), nn.ReLU(True),
                                 nn.ConvTranspose2d(inner // 4, 1, 2, 2))
        self.asf = ASF(inner) if head == 'dbpp' else None
        self.k = 50
    def forward(self, x):
        if self.mb:
            t = self.taps; c2 = self.features[:t[0]](x); c3 = self.features[t[0]:t[1]](c2)
            c4 = self.features[t[1]:t[2]](c3); c5 = self.features[t[2]:t[3]](c4)
        else:
            x = self.stem(x); c2 = self.l1(x); c3 = self.l2(c2); c4 = self.l3(c3); c5 = self.l4(c4)
        feats = [c2, c3, c4, c5]; p = [self.lat[i](f) for i, f in enumerate(feats)]
        for i in range(3, 0, -1): p[i - 1] = p[i - 1] + F.interpolate(p[i], size=p[i - 1].shape[-2:], mode='nearest')
        outs = [self.smooth[i](p[i]) for i in range(4)]
        outs = [F.interpolate(o, size=outs[0].shape[-2:], mode='nearest') for o in outs]
        fuse = torch.cat(outs, 1)
        if self.asf is not None: fuse = self.asf(fuse, outs)
        prob = torch.sigmoid(self.head(fuse)); thr = torch.sigmoid(self.thr(fuse))
        binr = torch.reciprocal(1 + torch.exp(-self.k * (prob - thr)))
        return prob.squeeze(1), thr.squeeze(1), binr.squeeze(1)
