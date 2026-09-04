"""Архитектура лёгкого распознавателя svtr_lcnet (PP-LCNet стем + глобальные SVTR-блоки).

Нужна только для GPU-пути: на CPU модель исполняется из ONNX, а на CUDA — нативным torch
(так же, как svtr_t). Совпадает 1-в-1 с той, из которой экспортированы
recognizer_svtr_lcnet_fp32.onnx и recognizer_svtr_lcnet_cyr12_fp32.onnx.

Контракт как у SVTR: вход [B,3,48,W] -> CTC-логиты [B,T,nclass], blank=0.
Стем снижает высоту 48 -> 3 и усредняет её, ширину -> W/4; дальше 6 блоков глобального
внимания над последовательностью длины W/4.
"""
import torch
import torch.nn as nn

from ._svtr import Block


class SE(nn.Module):
    """Squeeze-and-Excitation: канальное перевзвешивание."""

    def __init__(s, c, r=4):
        super().__init__()
        s.f = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // r, 1),
                            nn.ReLU(inplace=True), nn.Conv2d(c // r, c, 1),
                            nn.Hardsigmoid(inplace=True))

    def forward(s, x):
        return x * s.f(x)


class DepthSep(nn.Module):
    """PP-LCNet depthwise-separable: dw(k×k, stride) + pw(1×1), BN+Hardswish, опц. SE."""

    def __init__(s, i, o, k=3, stride=(1, 1), use_se=False):
        super().__init__()
        p = k // 2
        s.dw = nn.Conv2d(i, i, k, stride, p, groups=i, bias=False)
        s.bdw = nn.BatchNorm2d(i)
        s.pw = nn.Conv2d(i, o, 1, 1, 0, bias=False)
        s.bpw = nn.BatchNorm2d(o)
        s.se = SE(o) if use_se else nn.Identity()
        s.act = nn.Hardswish(inplace=True)

    def forward(s, x):
        x = s.act(s.bdw(s.dw(x)))
        x = s.act(s.bpw(s.pw(x)))
        return s.se(x)


class LCNetStem(nn.Module):
    """Стем: высота 48 -> 3 (затем усреднение), ширина -> W/width_ds. Каналы 16→32→64→128→out."""

    def __init__(s, out=256, inc=3, width_ds=8):
        super().__init__()
        last_wstride = 2 if width_ds == 8 else 1
        s.stem = nn.Sequential(nn.Conv2d(inc, 16, 3, (2, 2), 1, bias=False),
                               nn.BatchNorm2d(16), nn.Hardswish(inplace=True))
        s.body = nn.Sequential(
            DepthSep(16, 32, 3, (2, 2)),
            DepthSep(32, 64, 3, (2, 1)),
            DepthSep(64, 64, 3, (1, 1)),
            DepthSep(64, 128, 3, (2, 1), use_se=True),
            DepthSep(128, 128, 3, (1, 1), use_se=True),
            DepthSep(128, out, 3, (1, last_wstride), use_se=True),
        )
        s.out = out

    def forward(s, x):
        x = s.body(s.stem(x))                  # [B, out, 3, W/ds]
        return x.mean(2).transpose(1, 2)       # усреднить высоту -> [B, W/ds, out]


class SVTRLCNet(nn.Module):
    """PP-LCNet стем + N глобальных SVTR-блоков над 1D-последовательностью + CTC."""

    def __init__(s, nclass, dim=256, n_global=2, heads=8, width_ds=8):
        super().__init__()
        s.bb = LCNetStem(dim, width_ds=width_ds)
        s.blocks = nn.ModuleList([Block(dim, heads, 'global', None) for _ in range(n_global)])
        s.norm = nn.LayerNorm(dim)
        s.head = nn.Linear(dim, nclass)

    def forward(s, x):
        x = s.bb(x)
        for b in s.blocks:
            x = b(x)
        return s.head(s.norm(x))


def build_svtr_lcnet(nclass: int) -> SVTRLCNet:
    """Боевая конфигурация svtr_lcnet (в лаборатории — svtr_lcnet6_w4_d320):
    6 глобальных блоков, dim 320, ширина делится на 4."""
    return SVTRLCNet(nclass, dim=320, n_global=6, width_ds=4)
