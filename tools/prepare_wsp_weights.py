"""Verify the exact research edit and stage files for a later approved HF upload.

Development-only dependencies: numpy, onnx, onnxruntime. No upload, training,
parameter fitting or checkpoint modification is performed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

ORIGINAL = "3de3ad346181af4e4446e6c365d5d9a8bfb187ad803d6f419dd61ad3509a65f2"
EDITED = "24c5db4e64ad0a9bb8859b501670eff68969264ff0bee8af7130e3eb50f565f6"
CHARSET = "b4943823e5f46eb162fb1bce8ba341da89729cb62ed0617ee1c1f27ea0497303"
FILENAME = "recognizer_svtr_lcnet_wsp_fp32.onnx"
EXPECTED = {"bb.body.3.se.f.1.weight":128, "bb.body.3.se.f.1.bias":1,
            "bb.body.3.se.f.3.weight":256}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):digest.update(chunk)
    return digest.hexdigest()


def verify(original, edited, charset):
    import numpy as np
    import onnx
    from onnx import numpy_helper
    import onnxruntime as ort
    if sha(original) != ORIGINAL or sha(edited) != EDITED:
        raise ValueError("Research checkpoint SHA-256 mismatch; refusing integration.")
    if sha(charset) != CHARSET:raise ValueError("Original RU/EN charset SHA-256 mismatch.")
    base, new = onnx.load(str(original)), onnx.load(str(edited))
    onnx.checker.check_model(base);onnx.checker.check_model(new)
    tensors = {t.name:t for t in base.graph.initializer}
    if set(tensors) != {t.name for t in new.graph.initializer}:
        raise ValueError("Initializer names differ")
    changed = {}
    for t in new.graph.initializer:
        if t.SerializeToString() != tensors[t.name].SerializeToString():
            a,b = numpy_helper.to_array(tensors[t.name]),numpy_helper.to_array(t)
            if a.shape != b.shape or a.dtype != b.dtype:raise ValueError("Tensor layout changed")
            changed[t.name] = int(np.count_nonzero(a != b))
    if changed != EXPECTED:raise ValueError("Unexpected edits: " + repr(changed))
    restored = copy.deepcopy(new)
    for t in restored.graph.initializer:t.CopyFrom(tensors[t.name])
    if restored.SerializeToString() != base.SerializeToString():
        raise ValueError("Graph, metadata, operations or non-weight content changed")
    vocab = Path(charset).read_text(encoding="utf-8").split("\n")
    options = ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    cases = []
    for label,path in [("original",original),("wsp",edited)]:
        session = ort.InferenceSession(str(path),options,providers=["CPUExecutionProvider"])
        for batch,width in [(1,80),(2,128),(1,640)]:
            x = np.ones((batch,3,48,width),np.float32)
            y = session.run(None,{session.get_inputs()[0].name:x})[0]
            assert y.shape[0] == batch and y.shape[-1] == len(vocab)+1 and np.isfinite(y).all()
            cases.append({"model":label,"input_shape":list(x.shape),"output_shape":list(y.shape)})
    assert sha(original)==ORIGINAL and sha(edited)==EDITED
    return {"original_sha256":ORIGINAL,"edited_sha256":EDITED,"charset_sha256":CHARSET,
            "changed_tensors":changed,"changed_parameters":385,"changed_weights":384,"changed_biases":1,
            "non_parameter_graph_and_metadata_identical":True,"onnxruntime":ort.__version__,
            "providers":["CPUExecutionProvider"],"ctc_classes":len(vocab)+1,"runtime_cases":cases}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["original","edited","charset","output"]:p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();report=verify(a.original,a.edited,a.charset)
    a.output.mkdir(parents=True,exist_ok=True)
    target=a.output/FILENAME
    if target.exists() and sha(target)!=EDITED:raise FileExistsError("Different file already staged")
    if not target.exists():shutil.copyfile(a.edited,target)
    root=Path(__file__).resolve().parents[1]
    license_name="recognizer_svtr_lcnet_wsp_LICENSE.md"
    (a.output/license_name).write_text(
        "# SVTR-LCNet WSP derivative checkpoint\n\n"
        "Derived from the public Occular-OCR RU/EN SVTR-LCNet. "
        "The research edit changes 384 weights and one bias; no new operations or training.\n\n"
        "Original SHA-256: `"+ORIGINAL+"`\n\nEdited SHA-256: `"+EDITED+"`\n\n"
        "The original model-weights terms and downstream restrictions remain applicable. "
        "This is not an Apache-licensed weight file.\n\n"+(root/"WEIGHTS_LICENSE.md").read_text(),encoding="utf-8")
    metadata={**report,"filename":FILENAME,"architecture":"svtr_lcnet_wsp","experimental":True,
              "languages":["ru","en"],"cpu_only":True,"cuda_gate":"failed 18 of 48; quality not validated",
              "weights_license":license_name,"default_model_unchanged":True}
    (a.output/"recognizer_svtr_lcnet_wsp.metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    (a.output/"UPLOAD_INSTRUCTIONS.md").write_text(
        "# Pending approval: optional WSP weights\n\n"
        "Upload only the ONNX, metadata JSON and derivative license file to "
        "`Shivin11/occular-ocr`. Do not overwrite any original weight, charset or repository README. "
        "No upload is performed by this preparation command. See docs/releases/0.5.2.md.\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
