"""Compile source launch closures once using metadata-only Torch tensors."""
from contextvars import ContextVar
from functools import cache
import torch
from ._native import submission as _native

_building = ContextVar('native_program_builder', default=None)
_NAMES = {'a18_gemm', 'a18_global_initialize', 'a18_global_reduce',
          'a16_gemm', 'a16_reduce', 'nna14_gemm', 'nna14_reduce',
          'mps_lora', 'mps_vector', 'mps_batch_reduce', 'mps_identity'}

class _Kernel:
    def __init__(self, function): self.function = function
    def __getattr__(self, name): return getattr(self.function, name)
    def __call__(self, *args, threads, group_size):
        builder = _building.get()
        if builder is None:
            return self.function(*args, threads=threads, group_size=group_size)
        return builder.launch(self.function, args, threads, group_size)

class Library:
    def __init__(self, library): self.library = library
    @cache
    def __getattr__(self, name):
        function = getattr(self.library, name)
        return _Kernel(function) if name in _NAMES else function

class _Builder:
    def __init__(self, inputs):
        self.inputs = inputs
        self.storage_ids = {value.untyped_storage()._cdata: i for i, value in enumerate(inputs)}
        assert len(self.storage_ids) == len(inputs), 'Metadata inputs must have distinct storage'
        self.temporaries, self.launches, self.groups = [], [], []
        self.graph_results = []
    def reference(self, value):
        assert value.is_meta, 'Only metadata operations may run during plan construction'
        storage = value.untyped_storage()._cdata
        if storage not in self.storage_ids:
            self.storage_ids[storage] = len(self.inputs) + len(self.temporaries)
            self.temporaries.append(value)
        return _native().BufferReference(self.storage_ids[storage], value.storage_offset() * value.element_size())
    def launch(self, function, args, threads, group):
        args = tuple(self.reference(a) if isinstance(a, torch.Tensor) and a.is_meta else a for a in args)
        if isinstance(threads, int): threads = (threads,)
        if isinstance(group, int): group = (group,)
        self.launches.append(_native().Launch(function, args, threads, group))
    def graph_result(self, value):
        self.reference(value)
        self.graph_results.append(self.storage_ids[value.untyped_storage()._cdata])
    def finish_group(self, produced):
        self.groups.append((self.launches, produced))
        self.launches = []

def _finish_group(produced): _building.get().finish_group(produced)

def _program(function, inputs, produced=None):
    builder = _Builder(inputs)
    token = _building.set(builder)
    try:
        function(*inputs)
        if produced is not None: builder.finish_group(produced)
        assert not builder.launches, 'Missing logical operation boundary'
    finally:
        _building.reset(token)
    return _native().Program(builder.groups, builder.temporaries, len(inputs), builder.graph_results)

@cache
def _executable(cache_key, generation, cores):
    # GPURegionRuntime::evaluateOps in the captured MPSGraph binaries.
    # The public graphs have <=2 eligible operations, so graph-size adjustments
    # do not apply. See docs/mps_dispatch_sources.json for addresses and hashes.
    effective = cores * (3 if generation >= 22 else 1)
    first_ops, first_values, later_values = 8, 200_000, 300_000
    if effective >= 20:
        factor = min(7 * effective, 512)
        first_ops, first_values, later_values = 11, factor * 3125, factor * 300_000 >> 6
    return _native().Executable((first_ops, first_values, 12, later_values, 45, 30_000_000))


def _key(op, dtype, a_shape, b_shape, bias_shape=(), beta=1., alpha=1.):
    # Torch's graph key includes shapes/type and std::to_string(double) scalars.
    # Native executable specialization adds the concrete bmm input shapes.
    if op in ('addmm', 'addbmm'):
        bias_shape = (a_shape[-2], b_shape[-1])
    elif op != 'baddbmm':
        bias_shape = ()
    scalars = ((_native().scalar_key(beta), _native().scalar_key(alpha))
               if op in ('addmm', 'baddbmm', 'addbmm') else ())
    return op, dtype, tuple(a_shape), tuple(b_shape), tuple(bias_shape), scalars

def _execute(program, inputs, cache_key, dev):
    state = _executable(cache_key, dev.generation, dev.cores)
    program.run(inputs, state)

def _meta(dtype, shape, strides): return torch.empty_strided(shape, strides, dtype=dtype, device='meta')

def _matrix_meta(dtype, M, N, K, lda, ldb, ldc, ta, tb, batch, bsa, bsb, bsc, batched, epilogue):
    a_shape, b_shape, c_shape = (M,K), (K,N), (M,N)
    a_stride, b_stride, c_stride = (1,lda) if ta else (lda,1), (1,ldb) if tb else (ldb,1), (ldc,1)
    if batched:
        a_shape,b_shape,c_shape = (batch,*a_shape),(batch,*b_shape),(batch,*c_shape)
        a_stride,b_stride,c_stride = (bsa,*a_stride),(bsb,*b_stride),(bsc,*c_stride)
    inputs = [_meta(dtype,a_shape,a_stride),_meta(dtype,b_shape,b_stride),_meta(dtype,c_shape,c_stride)]
    if epilogue is not None:
        br,bc,bb,_,_ = epilogue
        inputs.append(_meta(dtype,c_shape,(bb,br,bc) if batched else (br,bc)))
    return inputs

@cache
def launch_plan(dtype,M,N,K,lda,ldb,ldc,ta,tb,batch,bsa,bsb,bsc,batched,dev,macos_major,epilogue=None):
    from .mps_dispatch import _launch_plan
    raw, selected = _launch_plan(dtype,M,N,K,lda,ldb,ldc,ta,tb,batch,bsa,bsb,bsc,batched,dev,macos_major,epilogue)
    inputs = _matrix_meta(dtype,M,N,K,lda,ldb,ldc,ta,tb,batch,bsa,bsb,bsc,batched,epilogue)
    program = _program(raw,inputs,produced=batch*M*N)
    op = ('bmm' if batched else 'mm') if epilogue is None else ('baddbmm' if batched else 'addmm')
    base = _key(op,dtype,inputs[0].shape,inputs[1].shape,(),*(epilogue[-2:] if epilogue is not None else (1.,1.)))
    def run(*args):
        cache_key = base if op != 'baddbmm' else (*base[:4],tuple(args[3].shape),base[-1])
        _execute(program,args,cache_key,dev)
    return run, selected

@cache
def _addbmm_plan(dtype,M,N,K,lda,ldb,ldc,ta,tb,batch,bsa,bsb,br,bc,beta,alpha,dev,macos_major):
    from .mps_dispatch import _launch_plan
    from . import mps_reduction
    inputs = _matrix_meta(dtype,M,N,K,lda,ldb,ldc,ta,tb,batch,bsa,bsb,M*ldc,True,None)
    inputs[2] = _meta(dtype,(M,N),(ldc,1))
    inputs.append(_meta(dtype,(M,N),(br,bc)))
    if batch == 1:
        product,_ = _launch_plan(dtype,M,N,K,lda,ldb,ldc,ta,tb,1,0,0,0,False,dev,macos_major,(br,bc,0,beta,alpha))
        def raw(a,b,out,bias):
            product(a[0],b[0],out,bias)
            _finish_group(M*N)
    else:
        product,_ = _launch_plan(dtype,M,N,K,lda,ldb,N,ta,tb,batch,bsa,bsb,M*N,True,dev,macos_major)
        def raw(a,b,out,bias):
            values = a.new_empty((batch,M,N))
            _building.get().graph_result(values)
            product(a,b,values)
            _finish_group(batch*M*N)
            mps_reduction.reduce_batch(values,out,bias,br,bc,beta,alpha,dev=dev,macos_major=macos_major)
            _finish_group(M*N)
    program = _program(raw,inputs)
    return program, _key('addbmm',dtype,inputs[0].shape,inputs[1].shape,(M,N),beta,alpha)

def addbmm(a, b, out, epilogue):
    from .dispatch import _resolve_bmm_inputs
    from .mps_dispatch import _macos_major
    from .mps_heuristics import device_info
    batch, M, K = a.shape
    N = b.shape[-1]
    bias, br, bc, _, beta, alpha = epilogue
    A, B, _, _, _, lda, ldb, ta, tb, bsa, bsb = _resolve_bmm_inputs(a, b)
    if A.is_contiguous(): ta, lda = False, K
    if B.is_contiguous(): tb, ldb = False, N
    dev = device_info()
    program, cache_key = _addbmm_plan(a.dtype, M, N, K, lda, ldb, out.stride(-2), ta, tb,
                                      batch, bsa, bsb, br, bc, beta, alpha, dev, _macos_major())
    _execute(program, (A, B, out, bias), cache_key, dev)
