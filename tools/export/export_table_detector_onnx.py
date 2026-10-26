"""Экспорт детектора таблиц D-FINE-S в ONNX: модель + постпроцессор одним графом.
Нужны torch и репозиторий D-FINE (в пакет не входит; пакет исполняет граф через onnxruntime).

  DFINE_ROOT=<путь к D-FINE> python tools/export/export_table_detector_onnx.py <cfg.yml> <best_stg2.pth> table_detect_dfine_v4_fp32.onnx

Препроцессинг как при обучении/валидации: растяжение PIL-bilinear до 640×640, RGB 0..1.
Входы : images [N,3,640,640] float32, orig_target_sizes [N,2] int64 (W,H).
Выходы: labels [N,300] int64, boxes [N,300,4] float32 xyxy в пикселях исходника, scores [N,300] float32.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.environ.get("DFINE_ROOT", "vendor/D-FINE"))
from src.core import YAMLConfig  # noqa: E402

CFG, CK, OUT = sys.argv[1:4]
cfg = YAMLConfig(CFG, resume=CK)
cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
cfg.yaml_cfg["eval_spatial_size"] = [640, 640]
c = torch.load(CK, map_location="cpu")
print("load", cfg.model.load_state_dict(c["ema"]["module"] if "ema" in c else c["model"], strict=False))


class W(nn.Module):
    def __init__(s):
        super().__init__(); s.m = cfg.model.deploy(); s.post = cfg.postprocessor.deploy()

    def forward(s, im, sz):
        return s.post(s.m(im), sz)


m = W().eval()
x, sz = torch.rand(1, 3, 640, 640), torch.tensor([[640, 640]])
torch.onnx.export(m, (x, sz), OUT, input_names=["images", "orig_target_sizes"], output_names=["labels", "boxes", "scores"],
                  dynamic_axes={k: {0: "N"} for k in ("images", "orig_target_sizes", "labels", "boxes", "scores")},
                  opset_version=17, do_constant_folding=True, dynamo=False)
import onnx  # noqa: E402
onnx.checker.check_model(onnx.load(OUT))
print("ok", OUT)
