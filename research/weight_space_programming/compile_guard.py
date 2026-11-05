"""Rebuild the two frozen LCNet checkpoints from public ONNX and recipe only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

ORIGINAL='3de3ad346181af4e4446e6c365d5d9a8bfb187ad803d6f419dd61ad3509a65f2'
HASHES={'guard':'24c5db4e64ad0a9bb8859b501670eff68969264ff0bee8af7130e3eb50f565f6',
        'ablation':'ce84b3c5fb43b36a34f716852cf8a7bdd320e5f600fb5c0c8ea2d49bb5ab4b37'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','recipe','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if sha(args.source)!=ORIGINAL:raise ValueError('Recipe requires the pinned original ONNX bytes')
    recipe=json.loads(args.recipe.read_text());base=onnx.load(args.source)
    first,bias,last='bb.body.3.se.f.1.weight','bb.body.3.se.f.1.bias','bb.body.3.se.f.3.weight'
    original={t.name:numpy_helper.to_array(t).copy() for t in base.graph.initializer};report={}
    args.output.mkdir(parents=True,exist_ok=True)
    for name in ['guard','ablation']:
        target=args.output/('lcnet_'+name+'.onnx')
        if target.exists():raise FileExistsError(target)
        model=copy.deepcopy(base);values={key:original[key].copy() for key in [first,bias,last]}
        values[first][12,:,0,0]=np.array(recipe['direction'],np.float32)
        values[bias][12]=-np.float32(recipe['threshold']);values[last][:,1,0,0]=0
        # Preserve the compiler's signed FP32 zeros for an exact serialized hash.
        values[last][:,12,0,0]=(np.asarray(recipe['readout'],np.float64)*(1. if name=='guard' else 0.)).astype(np.float32)
        for t in model.graph.initializer:
            if t.name in values:t.CopyFrom(numpy_helper.from_array(values[t.name],name=t.name))
        onnx.checker.check_model(model);temporary=target.with_suffix('.onnx.tmp');onnx.save(model,temporary)
        if sha(temporary)!=HASHES[name]:raise RuntimeError('Output SHA differs from evaluated '+name+'; temporary retained')
        temporary.replace(target)
        count=sum(int(np.count_nonzero(v!=original[k])) for k,v in values.items());assert count==385
        report[name]={'sha256':sha(target),'changed_values':count,'delta_l2':float(np.sqrt(sum(np.square(v.astype(np.float64)-original[k]).sum() for k,v in values.items())))}
    assert sha(args.source)==ORIGINAL
    (args.output/'compilation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
