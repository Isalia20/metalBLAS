"""Replay the M5 Pro f16 vector comparison from the original-shape benchmark."""
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import metalblas
from bench.bench_matmul import SHAPES

torch.manual_seed(7742)
# Replay the benchmark's allocations through the failing f16 case.
for shapes in SHAPES.values():
    for m,n,k in shapes:
        a=torch.randn((m,k),dtype=torch.float16,device='mps')
        b=torch.randn((k,n),dtype=torch.float16,device='mps')
        if (m,n,k)==(4096,1,4096):
            break
    else:
        continue
    break
ref=(a.cpu().double()@b.cpu().double()).half()
static=metalblas.matmul(a,b).cpu()
mps=(a@b).cpu()
report={}
for label,out in [('static',static),('torch_mps',mps)]:
    delta=(out.float()-ref.float()).abs()
    allowed=0.002+0.001*ref.float().abs()
    report[label]={'max_absolute_error_vs_cpu':delta.max().item(),
                   'failures_vs_cpu':int((delta>allowed).sum()),
                   'max_tolerance_ratio_vs_cpu':(delta/allowed).max().item(),
                   'unequal_vs_cpu':int((out!=ref).sum())}
errors=(static.float()-mps.float()).abs()
allowed=0.002+0.001*mps.float().abs()
report['comparison']={'max_error':errors.max().item(),'failures':int((errors>allowed).sum())}
indices=torch.topk(errors.flatten()/allowed.flatten(),5).indices.tolist()
report['worst_relative']=[{'index':i,'cpu':ref.flatten()[i].item(),'static':static.flatten()[i].item(),'torch_mps':mps.flatten()[i].item()} for i in indices]
print(json.dumps(report,indent=2))
