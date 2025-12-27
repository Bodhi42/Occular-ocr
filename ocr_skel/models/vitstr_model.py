"""ViTSTR: Vision Transformer for Scene Text Recognition

Based on: https://arxiv.org/abs/2105.08582
Architecture: Vision Transformer (ViT-small) + character classification head
"""

import torch
import torch.nn as nn
import math


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self Attention"""

    def __init__(self, dim, num_heads=6, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x: (B, N, C)
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP with GELU activation"""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block: Attention + MLP with residual connections"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        """
        Args:
            x: (B, N, C)
        Returns:
            (B, N, C)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTSTR(nn.Module):
    """Vision Transformer for Scene Text Recognition (ViT-Small)"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=1,
        num_classes=37,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        max_length=25,
    ):
        """
        Args:
            img_size: input image size (height, assumed square for simplicity)
            patch_size: patch size for embedding
            in_channels: number of input channels (1 for grayscale)
            num_classes: number of character classes (including blank/special tokens)
            embed_dim: embedding dimension
            depth: number of transformer blocks
            num_heads: number of attention heads
            mlp_ratio: ratio of mlp hidden dim to embedding dim
            qkv_bias: enable bias for qkv projection
            max_length: maximum text length
        """
        super().__init__()
        self.num_classes = num_classes
        self.max_length = max_length
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Positional embedding (learnable)
        # +1 for class token, but we'll use patch tokens directly for OCR
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head for CTC decoding
        # Each patch can produce a character prediction
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        # Positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Args:
            x: input image (B, C, H, W)

        Returns:
            output: (B, num_patches, num_classes) - logits for CTC
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Classification head
        logits = self.head(x)  # (B, num_patches, num_classes)

        return logits


def create_vitstr(num_classes=37, pretrained=False, img_size=224):
    """
    Create ViTSTR model (ViT-Small configuration)

    Args:
        num_classes: number of character classes (default 37: 0-9, a-z, blank)
        pretrained: load pretrained weights (not implemented yet)
        img_size: input image size

    Returns:
        ViTSTR model
    """
    model = ViTSTR(
        img_size=img_size,
        patch_size=16,
        in_channels=1,  # Grayscale
        num_classes=num_classes,
        embed_dim=384,  # ViT-Small
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        max_length=25,
    )

    return model
