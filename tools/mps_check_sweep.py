"""Check the recovered MPS selection against a Metal dispatch trace.

    python tools/mps_check_sweep.py sweep.json --macos-major 26

The GEMM check covers the kernel, grid, threadgroup size/memory, split count,
K per split, and reduction dispatch. Other routes are reported separately;
they are not counted as validated GEMMs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from metalblas.mps_a14 import gemm_plan, vector_plan, vector_preferred
from metalblas.mps_heuristics import DeviceInfo

DTYPES = {'bf16': torch.bfloat16, 'f16': torch.float16, 'f32': torch.float32}
GEMM_PREFIXES = ('NDArrayMatrixMultiplyNNA14', 'mmul_kernel_a16_')


def dispatches(row):
    """Remove the byte-less wrapper copies in the earliest tracer output."""
    out = []
    for d in row['dispatches']:
        if out and not out[-1]['bytes'] and all(out[-1][k] == d[k] for k in ('kernel', 'grid', 'tg')):
            out[-1] = d
        else:
            out.append(d)
    return out


def check_sweep(data, macos_major):
    dev = DeviceInfo(**data['device'])
    counts = Counter()
    mismatches = []
    for row in data['rows']:
        ds = dispatches(row)
        gemms = [d for d in ds if d['kernel'].startswith(GEMM_PREFIXES)]
        key = [row[k] for k in ('dtype', 'layout', 'M', 'N', 'K')]
        batch = row.get('batch', 1)
        if 'batch' in row:
            key.append(batch)
        vectors = [d for d in ds if d['kernel'].startswith(('NDArrayVectorMatrixMultiply_', 'LORADOWN_'))]
        # A transpose of a tensor with a unit dimension is still contiguous;
        # PyTorch presents that operand to MPSGraph in contiguous layout.
        ta = row['layout'][0] == 't' and row['M'] > 1 and row['K'] > 1
        tb = row['layout'][1] == 't' and row['N'] > 1 and row['K'] > 1
        if gemms or vectors:
            counts['routing_cases'] += 1
            predicted_vector = vector_preferred(
                row['M'], row['N'], row['K'], DTYPES[row['dtype']], dev=dev,
                trans_a=ta, trans_b=tb, macos_major=macos_major)
            if predicted_vector == bool(vectors):
                counts['routing_matches'] += 1
            else:
                mismatches.append({'case': key, 'expected_vector': predicted_vector,
                                   'observed_vector': bool(vectors)})
        if vectors:
            counts['vector_cases'] += 1
            p = vector_plan(row['M'], row['N'], row['K'], DTYPES[row['dtype']], dev=dev,
                            trans_a=ta, trans_b=tb, macos_major=macos_major, batch=batch)
            expected = [(p.kernel, p.grid, p.threadgroup, p.tgmem)]
            observed = [(d['kernel'], tuple(d['grid']), tuple(d['tg']),
                         tuple(d['tgmem'].get(str(i), 0) for i in (0, 1))) for d in ds]
            if observed == expected:
                counts['vector_matches'] += 1
            else:
                mismatches.append({'case': key, 'expected': expected, 'observed': observed})
            continue
        if not gemms:
            counts['other:' + '+'.join(d['kernel'] for d in ds)] += 1
            continue
        counts['gemm_cases'] += 1
        p = gemm_plan(row['M'], row['N'], row['K'], DTYPES[row['dtype']], dev=dev, macos_major=macos_major, batch=batch)
        expected = [{'kernel': p.kernel, 'grid': list(p.grid), 'tg': [p.threads, 1, 1],
                     'tgmem': list(p.tgmem), 'splits': p.splits, 'k_per_split': p.k_per_split}]
        if p.reduction_grid:
            expected.append({'kernel': 'NDArrayMatrixMultiplyReduction', 'grid': list(p.reduction_grid),
                             'tg': [32, 32, 1], 'tgmem': [0, 0],
                             'splits': p.splits, 'k_per_split': p.k_per_split})
        observed = []
        for d in ds:
            param = d['bytes'].get('26' if d['kernel'].startswith(GEMM_PREFIXES) else '27', '')
            words = struct.unpack('<25I', bytes.fromhex(param)) if len(param) == 200 else None
            observed.append({'kernel': d['kernel'], 'grid': d['grid'], 'tg': d['tg'],
                             'tgmem': [d['tgmem'].get(str(i), 0) for i in (0, 1)],
                             'splits': words[23] if words else None,
                             'k_per_split': words[24] if words else None})
        if observed != expected:
            mismatches.append({'case': key, 'expected': expected, 'observed': observed})
        else:
            counts['gemm_matches'] += 1
    return {'device': data['device'], 'macos_major': macos_major, 'counts': dict(counts), 'mismatches': mismatches}


def load_sweep(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        return json.load(f)


def compact_sweep(data, source):
    """Keep comparison inputs and provenance; omit unrelated buffer contents."""
    out = {k: v for k, v in data.items() if k != 'rows'}
    out['raw_trace_sha256'] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    out['rows'] = []
    for row in data['rows']:
        kept = {k: v for k, v in row.items() if k != 'dispatches'}
        kept['dispatches'] = []
        for d in dispatches(row):
            record = {k: d[k] for k in ('kernel', 'grid', 'tg', 'tgmem')}
            record['bytes'] = {k: v for k, v in d['bytes'].items() if k in ('25', '26', '27')}
            if d['bytes'] and not record['bytes']:
                # Preserve the distinction from a byte-less wrapper record.
                record['bytes']['other'] = ''
            kept['dispatches'].append(record)
        out['rows'].append(kept)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('sweep', type=Path)
    ap.add_argument('--macos-major', type=int, choices=(26, 27), required=True)
    ap.add_argument('--json', type=Path)
    ap.add_argument('--compact', type=Path, help='Write a compact .json.gz fixture after a successful check')
    args = ap.parse_args()
    data = load_sweep(args.sweep)
    report = check_sweep(data, args.macos_major)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + '\n')
    c = report['counts']
    print(f"{report['device']}: {c.get('gemm_matches', 0)}/{c.get('gemm_cases', 0)} GEMM cases match")
    print(f"GEMM/vector routing: {c.get('routing_matches', 0)}/{c.get('routing_cases', 0)} match")
    print(f"Vector dispatches: {c.get('vector_matches', 0)}/{c.get('vector_cases', 0)} match")
    print('Other routes (outside these MPS encoders):')
    for k, n in c.items():
        if k.startswith('other:'):
            print(f'  {n}: {k[6:]}')
    for mismatch in report['mismatches'][:10]:
        print(json.dumps(mismatch))
    failed = bool(report['mismatches']) or not c.get('routing_cases')
    if args.compact and not failed:
        compact = json.dumps(compact_sweep(data, args.sweep), separators=(',', ':')).encode()
        args.compact.write_bytes(gzip.compress(compact, mtime=0))
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
