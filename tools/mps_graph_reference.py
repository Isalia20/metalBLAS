"""Direct MPSGraph fused-matmul reference, using CPU copies only at its boundary."""
import ctypes
from functools import cache
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import torch


@cache
def _library():
    source = Path(__file__).with_suffix('.m')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    target = Path(tempfile.gettempdir()) / ('metalblas_graph_reference_' + digest + '.dylib')
    if not target.exists():
        subprocess.run(['clang', '-dynamiclib', '-fobjc-arc', '-framework', 'Foundation',
                        '-framework', 'Metal', '-framework', 'MetalPerformanceShaders',
                        '-framework', 'MetalPerformanceShadersGraph', '-o', str(target), str(source)],
                       check=True, capture_output=True, text=True)
    lib = ctypes.CDLL(str(target))
    lib.mb_graph_addmm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p,
                                  ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.mb_graph_addmm.restype = ctypes.c_int
    return lib


def addmm(bias, a, b, *, alpha=1., beta=1., reduce_batch=False):
    ta = a.stride(-2) == 1 and a.shape[-2] > 1
    tb = b.stride(-2) == 1 and b.shape[-2] > 1
    aa = (a.transpose(-2, -1) if ta else a).cpu().contiguous()
    bb = (b.transpose(-2, -1) if tb else b).cpu().contiguous()
    x = bias.cpu().contiguous()
    shape = (a.shape[-2], b.shape[-1]) if reduce_batch else (*a.shape[:-1], b.shape[-1])
    output = torch.empty(shape, dtype=a.dtype, device='cpu')
    spec = dict(dtype={torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[a.dtype],
                shapes=[list(aa.shape), list(bb.shape), list(x.shape)], ta=ta, tb=tb,
                alpha=alpha, beta=beta, reduce_batch=reduce_batch)
    error = ctypes.create_string_buffer(1024)
    status = _library().mb_graph_addmm(aa.data_ptr(), bb.data_ptr(), x.data_ptr(), json.dumps(spec).encode(),
                                      output.data_ptr(), error, len(error))
    if status:
        raise RuntimeError(error.value.decode())
    return output
