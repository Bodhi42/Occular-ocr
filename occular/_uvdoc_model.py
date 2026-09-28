"""Вендор-реализация сети UVDoc на чистом torch (без transformers/torchvision).

Портировано из архитектуры UVDoc (Apache-2.0). Имена модулей зеркалируют оригинал 1:1,
поэтому state_dict из официального чекпойнта (.safetensors) грузится strict без переименований.

Сеть: ResNet-backbone (resnet_head + 3 стадии residual-блоков) -> bridge (6 ПАРАЛЛЕЛЬНЫХ ветвей
дилатированных свёрток на одном входе) -> head (конкат 6×128=768 -> 1×1 conv -> предсказание 2D-сетки).
Вход: BGR [0,1], resize 712×488. Выход forward: (B,2,H',W') — сэмпл-сетка в норм.координатах [-1,1].

Зачем вендор: официальный путь тянет transformers>=5.17 (+ torchvision), что конфликтует с
ядром occular (transformers 4.57) и раздувает зависимости. Реальные операции UVDoc — только
torch.nn / F.interpolate / F.grid_sample, поэтому сеть самодостаточна.
"""
from __future__ import annotations
import torch
import torch.nn as nn

_ACT = {"relu": nn.ReLU, "prelu": nn.PReLU, "gelu": nn.GELU, "silu": nn.SiLU, "identity": nn.Identity}


def _act(name):
    if name is None:
        return nn.Identity()
    return _ACT[name]()


class UVDocConvLayer(nn.Module):
    """Conv2d + BatchNorm2d + активация (имена: convolution/normalization/activation)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 padding_mode="zeros", bias=False, dilation=1, activation="relu"):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                     padding=padding, padding_mode=padding_mode, bias=bias, dilation=dilation)
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = _act(activation)

    def forward(self, x):
        return self.activation(self.normalization(self.convolution(x)))


class UVDocResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, downsample=False, activation="relu"):
        super().__init__()
        self.conv_down = (UVDocConvLayer(in_channels, out_channels, kernel_size, stride,
                                         padding=kernel_size // 2, bias=True, activation=None)
                          if downsample else nn.Identity())
        self.conv_start = UVDocConvLayer(in_channels, out_channels, kernel_size, stride,
                                         padding=padding, dilation=dilation, bias=True)
        self.conv_final = UVDocConvLayer(out_channels, out_channels, kernel_size, stride=1,
                                         padding=padding, bias=True, dilation=dilation, activation=None)
        self.act_fn = _act(activation)

    def forward(self, x):
        residual = self.conv_down(x)
        h = self.conv_final(self.conv_start(x))
        return self.act_fn(h + residual)


class UVDocResNetStage(nn.Module):
    def __init__(self, resnet_configs_stage, kernel_size):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_c, out_c, dilation, downsample in resnet_configs_stage:
            self.layers.append(UVDocResidualBlock(
                in_c, out_c, kernel_size=kernel_size, stride=2 if downsample else 1,
                padding=dilation * 2, dilation=dilation, downsample=downsample))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class UVDocResNet(nn.Module):
    def __init__(self, resnet_head, resnet_configs, kernel_size):
        super().__init__()
        self.resnet_head = nn.ModuleList([
            UVDocConvLayer(h[0], h[1], kernel_size=kernel_size, stride=2, padding=kernel_size // 2)
            for h in resnet_head])
        self.resnet_down = nn.ModuleList([
            UVDocResNetStage(stage, kernel_size) for stage in resnet_configs])

    def forward(self, x):
        for head in self.resnet_head:
            x = head(x)
        for stage in self.resnet_down:
            x = stage(x)
        return x


class UVDocBridgeBlock(nn.Module):
    """Ветвь bridge: последовательные дилатированные свёртки (kernel 3, same-padding=dilation)."""

    def __init__(self, stage_cfg):
        super().__init__()
        self.blocks = nn.ModuleList([
            UVDocConvLayer(in_c, in_c, padding=dilation, dilation=dilation)
            for in_c, dilation in stage_cfg])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class UVDocBridge(nn.Module):
    """6 НЕЗАВИСИМЫХ ветвей, каждая обрабатывает один и тот же вход backbone."""

    def __init__(self, stage_configs):
        super().__init__()
        self.bridge = nn.ModuleList([UVDocBridgeBlock(s) for s in stage_configs])

    def forward(self, x):
        return [branch(x) for branch in self.bridge]   # список из len(stage_configs) карт


class UVDocBackbone(nn.Module):
    def __init__(self, bcfg, kernel_size):
        super().__init__()
        self.resnet = UVDocResNet(bcfg["resnet_head"], bcfg["resnet_configs"], kernel_size)
        self.bridge = UVDocBridge(bcfg["stage_configs"])

    def forward(self, x):
        h = self.resnet(x)
        return self.bridge(h)                          # список карт (по числу stage_configs)


class UVDocPointPositions2D(nn.Module):
    def __init__(self, out_pp2d, kernel_size, padding_mode, hidden_act):
        super().__init__()
        self.conv_down = UVDocConvLayer(out_pp2d[0][0], out_pp2d[0][1], kernel_size=kernel_size,
                                        stride=1, padding=kernel_size // 2, padding_mode=padding_mode,
                                        activation=hidden_act)
        self.conv_up = nn.Conv2d(out_pp2d[1][0], out_pp2d[1][1], kernel_size=kernel_size, stride=1,
                                 padding=kernel_size // 2, padding_mode=padding_mode)

    def forward(self, x):
        return self.conv_up(self.conv_down(x))


class UVDocHead(nn.Module):
    def __init__(self, cfg, num_bridge):
        super().__init__()
        bc = cfg["bridge_connector"]
        self.bridge_connector = UVDocConvLayer(bc[0] * num_bridge, bc[1], kernel_size=1,
                                               stride=1, padding=0, dilation=1)
        self.out_point_positions2D = UVDocPointPositions2D(
            cfg["out_point_positions2D"], cfg["kernel_size"], cfg["padding_mode"], cfg["hidden_act"])

    def forward(self, x):
        return self.out_point_positions2D(self.bridge_connector(x))


class UVDocNet(nn.Module):
    """Полная сеть UVDoc. forward(pixel_values (B,3,712,488)) -> mesh (B,2,H',W') в [-1,1]."""

    def __init__(self, cfg: dict):
        super().__init__()
        bcfg = cfg["backbone_config"]
        kernel = bcfg.get("kernel_size", cfg.get("kernel_size", 5))
        self.backbone = UVDocBackbone(bcfg, kernel)
        self.head = UVDocHead(cfg, num_bridge=len(bcfg["stage_configs"]))

    def forward(self, pixel_values):
        feats = self.backbone(pixel_values)            # список из 6 карт (128 каналов)
        fused = torch.cat(feats, dim=1)                # (B, 768, H', W')
        return self.head(fused)                        # (B, 2, H', W')
