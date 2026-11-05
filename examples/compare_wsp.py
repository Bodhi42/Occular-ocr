"""Compare original and optional WSP recognition on identical detector crops.

    python examples/compare_wsp.py document.pdf --html comparison.html --json comparison.json
    python examples/compare_wsp.py document.png --greedy --html comparison.html

One DBNet pass per page, identical RGB pixels/quads, and byte-checked batches.
This is a comparison through the release runtime, not the frozen paper harness.
Without references, changed predictions are never classified as improvements.
"""
import argparse
import base64
from difflib import SequenceMatcher
from html import escape
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def distance(a, b):
    row = list(range(len(b)+1))
    for i,left in enumerate(a,1):
        next_row = [i]
        for j,right in enumerate(b,1):
            next_row.append(min(row[j]+1,next_row[-1]+1,row[j-1]+(left!=right)))
        row = next_row
    return row[-1]


def scores(reference, prediction):
    return {"character_errors":distance(reference,prediction),"characters":len(reference),
            "word_errors":distance(reference.split(),prediction.split()),"words":len(reference.split())}


def compare_page(image, quads, original, wsp):
    """Use the unmodified recognizer's crop/batch/decode paths, checking equality."""
    batches=[];position=[0];a_infer=original._infer;b_infer=wsp._infer
    def record(batch):
        batches.append(batch.copy())
        return a_infer(batch)
    def checked(batch):
        index=position[0]
        if index>=len(batches) or not np.array_equal(batch,batches[index]):
            raise RuntimeError("Original/WSP preprocessing or batches differ")
        position[0]+=1
        return b_infer(batch)
    original._infer=record;wsp._infer=checked
    try:
        left=original.recognize(image,quads)
        right=wsp.recognize(image,quads)
    finally:
        original._infer=a_infer;wsp._infer=b_infer
    if position[0]!=len(batches) or len(left)!=len(right) or len(left)!=len(quads):
        raise RuntimeError("Paired crop coverage differs")
    fingerprints=[{"shape":list(x.shape),"dtype":str(x.dtype),"sha256":hashlib.sha256(x.tobytes()).hexdigest()} for x in batches]
    return left,right,fingerprints


def pages(path,dpi):
    if path.suffix.lower()==".pdf":
        from occular.pipeline import pymupdf
        pdf=pymupdf()
        with pdf.open(str(path)) as doc:
            for index,page in enumerate(doc):
                pix=page.get_pixmap(matrix=pdf.Matrix(dpi/72,dpi/72),colorspace=pdf.csRGB,alpha=False)
                yield index,np.asarray(Image.frombytes("RGB",(pix.width,pix.height),pix.samples))
    else:
        with Image.open(path) as im:yield 0,np.asarray(im.convert("RGB"))


def marked(text,other):
    parts=[]
    for kind,start,end,_,_ in SequenceMatcher(None,text,other,autojunk=False).get_opcodes():
        value=escape(text[start:end])
        parts.append(value if kind=="equal" else "<mark>"+value+"</mark>")
    return "".join(parts)


def html_report(report):
    rows=[]
    for row in report["lines"]:
        gt="<p>Reference: <code>"+escape(row["reference"])+"</code></p>" if "reference" in row else ""
        rows.append('<article class="'+row["status"]+'"><header>'+escape(row["id"])+
                    ' · '+escape(row["status"])+"</header><img alt=\"Detector crop\" src=\"data:image/png;base64,"+
                    row["crop_png_base64"]+'\">'+gt+'<div class="pred"><b>Original</b><code>'+marked(row["original"],row["wsp"])+
                    '</code><b>WSP</b><code>'+marked(row["wsp"],row["original"])+"</code></div></article>")
    counts=report["counts"]
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Occular original / WSP comparison</title><style>body{max-width:1100px;margin:35px auto;padding:0 20px;font:16px/1.6 system-ui;color:#24343a}article{border:1px solid #cfd8d3;padding:18px;margin:18px 0;border-left:5px solid #8b9b91}.changed{border-left-color:#b78a35}.improved{border-left-color:#277e58}.regressed{border-left-color:#ae4d4d}header{font-weight:650;margin-bottom:10px}img{display:block;max-width:100%;max-height:130px;object-fit:contain;object-position:left;margin:14px 0}.pred{display:grid;grid-template-columns:100px 1fr;gap:10px}code{white-space:pre-wrap;overflow-wrap:anywhere}mark{background:#fff0bc}#changed:checked~main .same{display:none}</style>
<h1>Original SVTR-LCNet / experimental WSP</h1><p>'''+str(counts["total"])+" crops; "+str(counts["changed"])+''' changed predictions. One detector pass per page; paired batch equality checked.</p>
<p>'''+escape(report["decoder"])+''' · CPU · release-runtime comparison. A changed answer is not necessarily better. Only supplied references permit improvement/regression labels.</p>
<input id="changed" type="checkbox" checked><label for="changed"> Show differing predictions only</label><main>'''+"".join(rows)+"</main></html>"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path);parser.add_argument("--html",type=Path,default=Path("wsp_comparison.html"));parser.add_argument("--json",type=Path)
    parser.add_argument("--greedy",action="store_true");parser.add_argument("--dpi",type=int,default=200);parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--ground-truth",type=Path,help='Optional JSON object: {"p1-l1": "reference text", ...}; IDs are1-based')
    args=parser.parse_args()
    from occular.dbnet_detector_onnx import DBNetDetectorONNX
    from occular.recognizer_onnx import CRNNRecognizerONNX
    from occular.model_files import WSP_RECOGNIZER_SHA256
    import occular
    # Construct WSP first so an unavailable alternative fails before detector work.
    wsp=CRNNRecognizerONNX(arch="svtr_lcnet_wsp",gpu=False,lm=not args.greedy,num_threads=args.threads)
    original=CRNNRecognizerONNX(arch="svtr_lcnet",gpu=False,lm=not args.greedy,num_threads=args.threads)
    detector=DBNetDetectorONNX(gpu=False,num_threads=args.threads)
    references=json.loads(args.ground_truth.read_text(encoding="utf-8")) if args.ground_truth else {}
    report={"runtime":"Occular "+occular.__version__,"decoder":"greedy CTC" if args.greedy else "release beam+LM (unchanged)",
            "wsp_sha256":WSP_RECOGNIZER_SHA256,"lines":[],"batch_checks":[],"counts":{"total":0,"changed":0,"same":0},"evaluation_scope":"Release runtime; not the frozen scientific harness"}
    for page_index,image in pages(args.input,args.dpi):
        quads=detector.detect(image)
        left,right,batches=compare_page(image,quads,original,wsp)
        report["batch_checks"].append({"page":page_index+1,"batches":batches})
        for index,(quad,a,b) in enumerate(zip(quads,left,right)):
            key="p%d-l%d"%(page_index+1,index+1);status="same" if a[0]==b[0] else "changed"
            row={"id":key,"page":page_index+1,"quad":np.asarray(quad).tolist(),"original":a[0],"wsp":b[0],"original_confidence":float(a[1]),"wsp_confidence":float(b[1]),"status":status}
            crop=original._crop_quad(image,quad);buffer=io.BytesIO()
            Image.fromarray(crop if crop.size else np.full((1,1,3),255,np.uint8)).save(buffer,format="PNG")
            row["crop_png_base64"]=base64.b64encode(buffer.getvalue()).decode("ascii")
            report["counts"]["total"]+=1;report["counts"][status]+=1
            if key in references:
                gt=references[key];row.update(reference=gt,original_metrics=scores(gt,a[0]),wsp_metrics=scores(gt,b[0]))
                delta=row["wsp_metrics"]["character_errors"]-row["original_metrics"]["character_errors"]
                row["status"]="improved" if delta<0 else "regressed" if delta>0 else status
            report["lines"].append(row)
    if references:
        report["metrics"]={}
        for name in ["original","wsp"]:
            records=[r[name+"_metrics"] for r in report["lines"] if "reference" in r]
            ce=sum(r["character_errors"] for r in records);nc=sum(r["characters"] for r in records);we=sum(r["word_errors"] for r in records);nw=sum(r["words"] for r in records)
            report["metrics"][name]={"character_errors":ce,"characters":nc,"word_errors":we,"words":nw,"cer":ce/max(1,nc),"wer":we/max(1,nw),"lines":len(records)}
    args.html.parent.mkdir(parents=True,exist_ok=True);args.html.write_text(html_report(report),encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True);args.json.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"html":str(args.html),"json":str(args.json) if args.json else None,"counts":report["counts"],"metrics":report.get("metrics")},ensure_ascii=False))


if __name__=="__main__":main()
