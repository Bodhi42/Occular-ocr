"""Экспорт модели структуры таблиц (split+merge, table_struct_split_merge_v2.pt) в два ONNX-графа.

Модель принимает решения по своим же выходам (число полос, границы, вырезка клеток), поэтому
целиком в ONNX не трассируется: трассировщик зашивает сетку примера как константу. Режем её:

  table_struct_v3_main_fp32.onnx  — картинка 768×768 → логиты счётчиков и сепараторов + карта merge-признаков
  (numpy в occular.tables)        — пиковый декод границ, R/C, усреднение карты по клеткам, выборка на границах
  table_struct_v3_merge_fp32.onnx — Grid-CNN по сетке клеток + MLP рёбер; оси r, c — динамические

Нужен torch (только здесь, не в пакете).  Запуск:
  python tools/export/export_tables_onnx.py <out_dir>
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from tables_model import FastTabSplit          # noqa: E402  (определение сети, перенесено из пакета)


def load_torch_model(pt_path):
    sd = torch.load(pt_path, map_location="cpu", weights_only=False)
    cfg = sd["cfg"]
    m = FastTabSplit(sd["dmodel"], T=cfg["T"], L=cfg["L"], vdown=cfg["vdown"], pool=cfg["pool"],
                     head=cfg["head"], upax=cfg["upax"], merge=cfg.get("merge", False),
                     bedge=bool(cfg.get("bedge", 0)), mgrid=bool(cfg.get("mgrid", 0)))
    if cfg.get("upc"):
        m.upax_col = cfg["upc"]
    m.load_state_dict(sd["model"])
    return m.eval()


class Main(nn.Module):
    """Всё до первого решения: энкодер, контекст, split-головы, проекция карты для merge."""
    def __init__(s, m):
        super().__init__(); s.m = m

    def forward(s, x):
        m = s.m
        Fm = m.enc(x); z = m.trm(Fm)
        rcnt, rsep = m.row(m._seq(Fm, -1), z)
        ccnt, csep = m.col(m._seq(Fm, -2), z)
        return rcnt, rsep, ccnt, csep, m.mrg.proj(Fm)


class Merge(nn.Module):
    """MergeHeadGrid после пулинга клеток: Grid-CNN + MLP по парам соседей (признаки границы и геометрия — входом)."""
    def __init__(s, mrg):
        super().__init__(); s.mrg = mrg

    def forward(s, f, fb_h, lij_h, fb_v, lij_v):
        g = s.mrg.gcnn(f)[0].permute(1, 2, 0)                           # (r, c, d)
        hor = s.mrg.mlp(torch.cat([g[:, :-1], g[:, 1:], fb_h, lij_h], -1)).squeeze(-1)
        ver = s.mrg.mlp(torch.cat([g[:-1], g[1:], fb_v, lij_v], -1)).squeeze(-1)
        return hor, ver


def export(pt_path, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    m = load_torch_model(pt_path)
    assert m.head == "seg" and m.merge and type(m.mrg).__name__ == "MergeHeadGrid" and m.mrg.bedge
    torch.backends.mha.set_fastpath_enabled(False)      # слитый оператор трансформера в ONNX не экспортируется
    d = m.mrg.proj.out_channels
    with torch.no_grad():
        torch.onnx.export(Main(m), (torch.zeros(1, 3, 768, 768),), str(out_dir / "table_struct_v3_main_fp32.onnx"),
                          input_names=["x"], output_names=["rcnt", "rsep", "ccnt", "csep", "pmap"],
                          opset_version=17, dynamo=False)
        r, c = 5, 4
        args = (torch.zeros(1, d, r, c), torch.zeros(r, c - 1, d), torch.zeros(r, c - 1, 5),
                torch.zeros(r - 1, c, d), torch.zeros(r - 1, c, 5))
        torch.onnx.export(Merge(m.mrg), args, str(out_dir / "table_struct_v3_merge_fp32.onnx"),
                          input_names=["f", "fb_h", "lij_h", "fb_v", "lij_v"], output_names=["hor", "ver"],
                          dynamic_axes={"f": {2: "r", 3: "c"}, "fb_h": {0: "r", 1: "c1"}, "lij_h": {0: "r", 1: "c1"},
                                        "fb_v": {0: "r1", 1: "c"}, "lij_v": {0: "r1", 1: "c"},
                                        "hor": {0: "r", 1: "c1"}, "ver": {0: "r1", 1: "c"}},
                          opset_version=17, dynamo=False)
    print("готово:", sorted(p.name for p in out_dir.glob("table_struct_v3_*.onnx")))


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download
    export(hf_hub_download("Shivin11/occular-ocr", "table_struct_split_merge_v2.pt"), sys.argv[1])
