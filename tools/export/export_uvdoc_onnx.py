"""Экспорт UVDoc (расправление изгиба страницы) в один ONNX-граф с динамическим размером страницы.

Граф = весь прежний torch-путь occular.unwarp_uvdoc:
  BGR [0,1] (1,3,H,W) -> resize до входа сети (align_corners) -> сеть -> mesh (1,2,h',w')
  -> resize mesh до (H,W) (align_corners) -> grid_sample по оригиналу (align_corners) -> (1,3,H,W) в [0,1]
Нужен torch + safetensors (только здесь, не в пакете).  Запуск:
  python tools/export/export_uvdoc_onnx.py <hf_repo_или_папка_с_весами> <out_dir>
"""
import json, os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from uvdoc_model import UVDocNet            # noqa: E402

_DEFAULT_IN_H, _DEFAULT_IN_W = 712, 488


def load(model_id):
    from safetensors.torch import load_file
    if os.path.isdir(model_id):
        cfg_p, w_p, prep_p = (os.path.join(model_id, n) for n in ("config.json", "model.safetensors", "preprocessor_config.json"))
    else:
        from huggingface_hub import hf_hub_download
        cfg_p, w_p = hf_hub_download(model_id, "config.json"), hf_hub_download(model_id, "model.safetensors")
        try:
            prep_p = hf_hub_download(model_id, "preprocessor_config.json")
        except Exception:
            prep_p = None
    cfg = json.load(open(cfg_p))
    for k, v in (("kernel_size", 5), ("padding_mode", "reflect"), ("hidden_act", "prelu")):
        cfg.setdefault(k, v)
    in_h, in_w = _DEFAULT_IN_H, _DEFAULT_IN_W
    if prep_p and os.path.exists(prep_p):
        size = json.load(open(prep_p)).get("size", {})
        in_h, in_w = int(size.get("height", in_h)), int(size.get("width", in_w))
    net = UVDocNet(cfg).eval()
    net.load_state_dict(load_file(w_p), strict=True)
    return net, in_h, in_w


class Unwarp(nn.Module):
    def __init__(s, net, in_h, in_w):
        super().__init__(); s.net, s.in_h, s.in_w = net, in_h, in_w

    def forward(s, orig):                                       # (1,3,H,W) BGR в [0,1]
        H, W = orig.shape[-2], orig.shape[-1]
        px = F.interpolate(orig, size=(s.in_h, s.in_w), mode="bilinear", align_corners=True)
        mesh = s.net(px)
        up = F.interpolate(mesh, size=(H, W), mode="bilinear", align_corners=True)
        return F.grid_sample(orig, up.permute(0, 2, 3, 1), align_corners=True)


def export(model_id, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    net, in_h, in_w = load(model_id)
    with torch.no_grad():
        torch.onnx.export(Unwarp(net, in_h, in_w), (torch.rand(1, 3, 900, 640),), str(out / "model.onnx"),
                          input_names=["image"], output_names=["unwarped"],
                          dynamic_axes={"image": {2: "H", 3: "W"}, "unwarped": {2: "H", 3: "W"}},
                          opset_version=17, dynamo=False)
    print("готово:", out / "model.onnx", f"(вход сети {in_h}x{in_w})")


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
