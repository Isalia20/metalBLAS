"""Compare the public static API with MPS and an isolated old-autotuner snapshot.

The baseline is loaded under a separate package name with legacy tuning enabled.
Its measurements never select a parameter for the current implementation.
"""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import metalblas
from metalblas import mps_dispatch
from metalblas.dispatch import _resolve_inputs, _resolve_bmm_inputs
from metalblas.mps_heuristics import device_info
from bench.bench_matmul import SHAPES

DTYPES = {'f32': torch.float32, 'f16': torch.float16, 'bf16': torch.bfloat16}


def load_baseline(root, *, autotune=True):
    root = root.resolve()
    if root == ROOT:
        raise ValueError('The baseline must be an isolated snapshot')
    spec = importlib.util.spec_from_file_location('metalblas_baseline', root / 'metalblas/__init__.py',
                                                submodule_search_locations=[str(root / 'metalblas')])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    keys = {'METALBLAS_AUTOTUNE': '1' if autotune else '0', 'METALBLAS_TILE_POLICY': 'legacy'}
    saved = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update(keys)
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if autotune and not sys.modules['metalblas_baseline.float_dispatch']._AUTOTUNE:
        raise ValueError('The baseline snapshot does not enable the old autotuner')
    return module


def source_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / 'metalblas').rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.h', '.metal', '.json')}


def elapsed(fn, iterations):
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def cpu_check_on_mismatch(a, b, got, ref, atol, rtol, candidate_label='static'):
    """Diagnose a disagreeing reference without changing the MPS comparison."""
    expected = (a.cpu().double() @ b.cpu().double()).to(a.dtype).float()
    result = {}
    for label, value in ((candidate_label, got), ('torch_mps', ref)):
        value = value.cpu().float()
        delta = (value - expected).abs()
        allowed = atol + rtol * expected.abs()
        result[label] = {'correct': torch.allclose(value, expected, atol=atol, rtol=rtol),
                         'max_error': delta.max().item(),
                         'failures': int((delta > allowed).sum())}
    return result


def selected_plan(a, b):
    batched = a.ndim == 3
    if batched:
        A, B, m, n, k, lda, ldb, ta, tb, bsa, bsb = _resolve_bmm_inputs(a, b)
        batch, bsc = a.shape[0], m * n
    else:
        A, B, m, n, k, lda, ldb, ta, tb = _resolve_inputs(a, b)
        batch, bsa, bsb, bsc = 1, 0, 0, 0
    if A.is_contiguous(): ta, lda = False, k
    if B.is_contiguous(): tb, ldb = False, n
    _, plan = mps_dispatch._launch_plan(a.dtype, m, n, k, lda, ldb, n, ta, tb,
                                       batch, bsa, bsb, bsc, batched, device_info(),
                                       int(platform.mac_ver()[0].split('.')[0]))
    return dataclasses.asdict(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--cases', type=Path, help='JSON rows: [dtype, layout, M, N, K, optional batch]')
    parser.add_argument('--dtypes', default='f16,bf16')
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--trials', type=int, default=5)
    args = parser.parse_args()
    if min(args.iterations, args.trials) <= 0:
        parser.error('iterations and trials must be positive')
    baseline = load_baseline(args.baseline_root)
    cases = (json.loads(args.cases.read_text()) if args.cases else
             [(dtype, 'nn', *shape) for dtype in args.dtypes.split(',')
              for shapes in SHAPES.values() for shape in shapes])
    result = {'device': dataclasses.asdict(device_info()), 'macos': platform.mac_ver()[0],
              'torch': torch.__version__, 'iterations': args.iterations, 'trials': args.trials,
              'method': 'interleaved synchronized wall-clock medians; public APIs including planning and output allocation',
              'source_sha256': source_hashes(ROOT), 'baseline_root': str(args.baseline_root.resolve()),
              'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'seed': 7742,
              'baseline_source_sha256': source_hashes(args.baseline_root),
              'baseline_environment': {'METALBLAS_AUTOTUNE': '1', 'METALBLAS_TILE_POLICY': 'legacy'},
              'cases': []}
    torch.manual_seed(7742)
    for name, layout, m, n, k, *batch in cases:
        dtype = DTYPES[name]
        prefix = tuple(batch)
        a = torch.randn((*prefix, k, m) if layout[0] == 't' else (*prefix, m, k), dtype=dtype, device='mps')
        b = torch.randn((*prefix, n, k) if layout[1] == 't' else (*prefix, k, n), dtype=dtype, device='mps')
        if layout[0] == 't': a = a.transpose(-1, -2)
        if layout[1] == 't': b = b.transpose(-1, -2)
        functions = {'mps': lambda: torch.matmul(a, b), 'static': lambda: metalblas.matmul(a, b),
                     'autotuned': lambda: baseline.matmul(a, b)}
        # Compile and finish the baseline's candidate timing outside the trials.
        print(f'Preparing {name} {layout} {m}x{n}x{k} batch={batch or [1]}', flush=True)
        for fn in functions.values():
            for _ in range(10): fn()
        ref = functions['mps']()
        got = functions['static']()
        previous = functions['autotuned']()
        error = (got.float() - ref.float()).abs().max().item()
        atol, rtol = {torch.float32: (1e-3, 2e-5), torch.float16: (2e-3, 1e-3),
                      torch.bfloat16: (2e-2, 8e-3)}[dtype]
        correct = torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol)
        cpu_check = None if correct else cpu_check_on_mismatch(a, b, got, ref, atol, rtol)
        baseline_error = (previous.float() - ref.float()).abs().max().item()
        baseline_correct = torch.allclose(previous.float(), ref.float(), atol=atol, rtol=rtol)
        baseline_cpu_check = None if baseline_correct else cpu_check_on_mismatch(
            a, b, previous, ref, atol, rtol, candidate_label='autotuned')
        del ref, got, previous
        times = {key: [] for key in functions}
        labels = list(functions)
        for trial in range(args.trials):
            order = labels[trial % 3:] + labels[:trial % 3]
            for key in order[::1 if trial % 2 == 0 else -1]:
                times[key].append(elapsed(functions[key], args.iterations))
        medians = {key: statistics.median(values) for key, values in times.items()}
        row = {'dtype': name, 'layout': layout, 'shape': [m,n,k], 'batch': batch[0] if batch else None,
               'plan': selected_plan(a,b), 'max_error': error, 'correct': correct,
               'cpu_reference_on_mismatch': cpu_check,
               'baseline_max_error': baseline_error, 'baseline_correct': baseline_correct,
               'baseline_cpu_reference_on_mismatch': baseline_cpu_check,
               'tolerance': {'atol':atol,'rtol':rtol}, 'trials_us': times, 'median_us': medians,
               'speedup_mps': medians['mps']/medians['static'],
               'speedup_autotuned': medians['autotuned']/medians['static']}
        result['cases'].append(row)
        args.json.write_text(json.dumps(result,indent=2)+'\n')
        print(f"  MPS {row['speedup_mps']:.3f}x; autotuner {row['speedup_autotuned']:.3f}x; "
              f"error {error:g}; correct={correct}; baseline_correct={baseline_correct}", flush=True)


if __name__ == '__main__':
    main()
