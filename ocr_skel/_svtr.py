"""SVTR (IJCAI'22) — PyTorch-описание рекогнайзера для GPU-бэкенда.
Совпадает 1-в-1 с архитектурой, из которой экспортирован recognizer_svtr_fp32.onnx.
Используется ТОЛЬКО при gpu=True (torch на CUDA); CPU-путь работает на ONNX без torch."""
import torch, torch.nn as nn, torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(s, i, o, k=3, st=1, p=1, act=nn.GELU):
        super().__init__(); s.c = nn.Conv2d(i, o, k, st, p, bias=False); s.b = nn.BatchNorm2d(o); s.a = act()
    def forward(s, x): return s.a(s.b(s.c(x)))


class PatchEmbed(nn.Module):
    def __init__(s, inc=3, dim=64):
        super().__init__(); s.p1 = ConvBNAct(inc, dim // 2, 3, 2, 1); s.p2 = ConvBNAct(dim // 2, dim, 3, 2, 1)
    def forward(s, x): return s.p2(s.p1(x))


class Mixer(nn.Module):
    def __init__(s, dim, heads, mixer='global', HW=None, local_k=(7, 11)):
        super().__init__(); s.mixer = mixer; s.HW = HW; s.heads = heads; s.scale = (dim // heads) ** -0.5
        s.qkv = nn.Linear(dim, dim * 3); s.proj = nn.Linear(dim, dim)
        if mixer == 'local' and HW is not None:
            H, W = HW; kh, kw = local_k
            mask = torch.zeros(H * W, H * W)
            for i in range(H):
                for j in range(W):
                    a = max(0, i - kh // 2); b = min(H, i + kh // 2 + 1); c = max(0, j - kw // 2); d = min(W, j + kw // 2 + 1)
                    m = torch.zeros(H, W); m[a:b, c:d] = 1.; mask[i * W + j] = m.flatten()
            s.register_buffer('mask', (1 - mask) * (-1e9))
        else: s.mask = None
    def forward(s, x):
        B, N, C = x.shape; qkv = s.qkv(x).reshape(B, N, 3, s.heads, C // s.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=s.mask)
        x = x.transpose(1, 2).reshape(B, N, C); return s.proj(x)


class Block(nn.Module):
    def __init__(s, dim, heads, mixer, HW, mlp=4., drop_path=0.):
        super().__init__(); s.n1 = nn.LayerNorm(dim); s.mix = Mixer(dim, heads, mixer, HW)
        s.n2 = nn.LayerNorm(dim); h = int(dim * mlp); s.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))
        s.dp = drop_path
    def drop(s, x):
        if not s.training or s.dp == 0: return x
        keep = 1 - s.dp; m = torch.rand(x.shape[0], 1, 1, device=x.device) < keep; return x * m / keep
    def forward(s, x): x = x + s.drop(s.mix(s.n1(x))); x = x + s.drop(s.mlp(s.n2(x))); return x


class Merge(nn.Module):
    def __init__(s, dim, out):
        super().__init__(); s.c = nn.Conv2d(dim, out, 3, (2, 1), 1); s.n = nn.LayerNorm(out)
    def forward(s, x, H, W):
        B, N, C = x.shape; x = x.transpose(1, 2).reshape(B, C, H, W); x = s.c(x)
        H2 = x.shape[2]; x = x.flatten(2).transpose(1, 2); return s.n(x), H2, W


class SVTR(nn.Module):
    def __init__(s, nclass, inc=3, dims=(64, 128, 256), depths=(3, 6, 3), heads=(2, 4, 8),
                 mixers=None, img_h=48, img_w=320, out_dim=192, neck='none'):
        super().__init__()
        s.embed = PatchEmbed(inc, dims[0]); H = img_h // 4; W0 = img_w // 4
        s.stages = nn.ModuleList(); s.merges = nn.ModuleList(); s.Hs = []
        curH = H
        for st in range(3):
            blocks = nn.ModuleList([Block(dims[st], heads[st], 'global', (curH, W0)) for i in range(depths[st])])
            s.stages.append(blocks); s.Hs.append(curH)
            if st < 2: s.merges.append(Merge(dims[st], dims[st + 1])); curH = (curH + 1) // 2 if curH > 1 else 1
        s.lastH = curH
        s.comb = nn.Linear(dims[-1], out_dim)
        s.neck = nn.LSTM(out_dim, out_dim // 2, batch_first=True, bidirectional=True) if neck == 'lstm' else None
        s.head = nn.Linear(out_dim, nclass)
    def forward(s, x):
        x = s.embed(x); B, C, H, W = x.shape; x = x.flatten(2).transpose(1, 2)
        for st in range(3):
            for blk in s.stages[st]: x = blk(x)
            if st < 2: x, H, W = s.merges[st](x, H, W)
        x = x.reshape(B, s.lastH, W, -1).mean(1)
        x = s.comb(x)
        if s.neck is not None: x = s.neck(x)[0]
        return s.head(x)                                # [B, W, nclass] логиты CTC


def build_svtr(nclass, arch='svtr_t', img_h=48, img_w=320, neck='none'):
    cfg = {'svtr_t': dict(dims=(64, 128, 256), depths=(3, 6, 3), heads=(2, 4, 8), out_dim=192),
           'svtr_s': dict(dims=(96, 192, 256), depths=(3, 6, 6), heads=(3, 6, 8), out_dim=256),
           'svtr_b': dict(dims=(128, 256, 384), depths=(3, 6, 9), heads=(4, 8, 12), out_dim=384)}[arch]
    return SVTR(nclass, img_h=img_h, img_w=img_w, neck=neck, **cfg)
