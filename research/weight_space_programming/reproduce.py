"""Replay the paper's frozen synthetic inputs and decoder, separately from the API.

No private documents, training or weight changes. The fixture contains the
original 2048 rendered records, including two duplicate PNG hashes. Python 3.8+.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import types

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT=Path(__file__).resolve().parent
SHA={'original':'3de3ad346181af4e4446e6c365d5d9a8bfb187ad803d6f419dd61ad3509a65f2',
     'wsp':'24c5db4e64ad0a9bb8859b501670eff68969264ff0bee8af7130e3eb50f565f6'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(row[j]+1,nxt[-1]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]


def extract_fixture(target):
    archive=ROOT/'synthetic_2048.tar.gz'
    expected=json.loads((ROOT/'fixture_sha256.json').read_text())['synthetic_2048.tar.gz']
    if sha(archive)!=expected:raise ValueError('Frozen fixture archive SHA mismatch')
    with tarfile.open(archive) as stream:
        for member in stream:
            path=Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                raise ValueError('Unexpected fixture archive member')
            output=target/path;output.parent.mkdir(parents=True,exist_ok=True)
            with stream.extractfile(member) as source,output.open('wb') as dest:
                import shutil
                shutil.copyfileobj(source,dest)


def frozen_decoder(vocab,lm_dir):
    path=ROOT/'frozen_decoder/occular'
    if sha(path/'decoder_lm.py')!='2b87572a46a9441d2f28295f07042f3b6a312efa8d0ab6dcec15af3357f26fbc':
        raise ValueError('Frozen beam wrapper SHA mismatch')
    if sha(lm_dir/'compact_lm.npz')!='b2f60561802930b7bf5a1926253a4e3279e7507f701d8612bc631b6070e49e9d' or sha(lm_dir/'unigrams.txt')!='290deae673303ca03b067153a4d9d04c52d316bb6487931744c6656e11c08109':
        raise ValueError('Pinned LM SHA mismatch')
    # Separate namespace: never accidentally import the installed release wrapper.
    name='_occular_wsp_frozen';package=types.ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
    module=importlib.import_module(name+'.decoder_lm')
    lm=module.LMDecoder(vocab,alpha=.5,beta=1.,beam_width=100,
                        lm_files=(str(lm_dir/'compact_lm.npz'),str(lm_dir/'unigrams.txt')),lock_codes=True)
    if lm.native is None:raise RuntimeError('Install the pinned occular_decode native wheel for the recorded scientific beam harness')
    return lm


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True);p.add_argument('--wsp',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lm-dir',type=Path);p.add_argument('--greedy',action='store_true');p.add_argument('--allow-runtime-drift',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if ort.__version__!='1.23.2' and not a.allow_runtime_drift:raise RuntimeError('Paper harness uses CPU ONNX Runtime 1.23.2; use a dedicated environment or explicitly record --allow-runtime-drift')
    for name in SHA:
        if sha(getattr(a,name))!=SHA[name]:raise ValueError('Wrong '+name+' checkpoint')
    if not a.greedy and a.lm_dir is None:p.error('--lm-dir is required for the frozen beam comparison (or select --greedy)')
    os.environ.setdefault('RAYON_NUM_THREADS','2');cv2.setNumThreads(1)
    vocab=(ROOT/'recognizer_charset.txt').read_text(encoding='utf-8').split('\n')
    if sha(ROOT/'recognizer_charset.txt')!='b4943823e5f46eb162fb1bce8ba341da89729cb62ed0617ee1c1f27ea0497303':raise ValueError('Wrong charset')
    lm=None if a.greedy else frozen_decoder(vocab,a.lm_dir)
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    nets={name:ort.InferenceSession(str(getattr(a,name)),opts,providers=['CPUExecutionProvider']) for name in SHA}
    totals={name:{view:{'character_errors':0,'characters':0,'word_errors':0,'words':0} for view in ['natural','pad2x','pad4x']} for name in SHA}
    rows=[];mismatches=[];natural_equal=0
    with tempfile.TemporaryDirectory(prefix='occular-wsp-fixture-') as tmp:
        data=Path(tmp);extract_fixture(data)
        manifest=json.loads((data/'manifest.json').read_text());recorded=json.loads((data/'recorded_predictions.json').read_text())
        mode='greedy' if a.greedy else 'beam'
        expected={name:{view:{r['id']:r['prediction'] for r in modes[mode]} for view,modes in views.items()} for name,views in recorded.items()}
        for index,row in enumerate(manifest):
            image_path=data/row['path']
            if sha(image_path)!=row['sha256']:raise ValueError('Frozen image changed')
            with Image.open(image_path) as image:rgb=np.asarray(image.convert('RGB'))
            width=max(8,min(640,round(rgb.shape[1]*48/rgb.shape[0])))
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);x=((cv2.resize(bgr,(width,48)).astype(np.float32)/255-.5)/.5).transpose(2,0,1)[None]
            for view,factor in [('natural',1),('pad2x',2),('pad4x',4)]:
                batch=np.ones((1,3,48,min(640,width*factor)),np.float32);batch[:,:,:,:width]=x
                raw={name:net.run(None,{net.get_inputs()[0].name:batch})[0] for name,net in nets.items()}
                if lm:predictions={name:r[0] for name,r in zip(raw,lm.decode_many(list(raw.values())))}
                else:
                    predictions={}
                    for name,value in raw.items():
                        chars=[];last=-1
                        for k in value[0].argmax(-1):
                            k=int(k)
                            if k and k!=last:chars.append(vocab[k-1])
                            last=k
                        predictions[name]=''.join(chars)
                if view=='natural':natural_equal+=predictions['original']==predictions['wsp']
                result={'id':row['id'],'view':view,'reference':row['text'],'predictions':predictions}
                for name,pred in predictions.items():
                    stat=totals[name][view];stat['character_errors']+=distance(row['text'],pred);stat['characters']+=len(row['text']);stat['word_errors']+=distance(row['text'].split(),pred.split());stat['words']+=len(row['text'].split())
                    key='baseline' if name=='original' else 'wsp'
                    if pred!=expected[key][view][row['id']]:mismatches.append({'id':row['id'],'view':view,'model':name})
                rows.append(result)
            if (index+1)%256==0:print('REPLAY',index+1,'/',len(manifest),flush=True)
    for views in totals.values():
        for stat in views.values():stat['cer']=stat['character_errors']/stat['characters'];stat['wer']=stat['word_errors']/stat['words']
    report={'evaluation_scope':'Frozen scientific harness; never a release API benchmark','runtime':{'onnxruntime':ort.__version__,'numpy':np.__version__,'opencv':cv2.__version__,'provider':'CPUExecutionProvider'},'decoder':mode,'summary':totals,'natural_predictions_equal':natural_equal,'records':2048,'recorded_prediction_mismatches':mismatches,'rows':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['evaluation_scope','runtime','decoder','summary','natural_predictions_equal']},indent=2));print('Recorded prediction mismatches:',len(mismatches),flush=True)


if __name__=='__main__':main()
