"""Large-output tiling from MPSNDArray's A14 EncodeArrayMultiply."""
from dataclasses import dataclass

import torch

from .dispatch import _ceildiv


@dataclass(frozen=True)
class ChunkPlan:
    size: int
    rows: tuple
    columns: tuple


def schedule(M, N, batch, dev):
    # The enclosing encoder also excludes vector routes and nontrivial
    # prefix DAGs. Our caller handles ordinary matmul and linked bias only.
    if batch != 1 or dev.generation < 11 or max(M, N) <= 32768:
        return None
    size = 1024 if dev.cores < 9 else 8192
    return ChunkPlan(size, tuple((start, min(size, M-start)) for start in range(0, M, size)),
                     tuple((start, min(size, N-start)) for start in range(0, N, size)))


def row_stride(columns, dtype):
    # MPSCore -[MPSNDArray initWithDevice:descriptor:isTextureBacked:]
    # rounds the default row length to 128 bits (0x18eb840ac..0x18eb840f0).
    elements = 4 if dtype is torch.float32 else 8
    return _ceildiv(columns, elements) * elements


def _temporary(reference, rows, columns):
    return reference.new_empty(rows, row_stride(columns, reference.dtype))[:, :columns]


def launch_plan(dtype, M, N, K, lda, ldb, ldc, ta, tb, batched, dev, macos_major,
                plan, epilogue=None):
    from .mps_dispatch import _launch_plan
    from .mps_identity import launch_plan as copy_plan
    copy_a, copy_b = len(plan.rows) > 1, len(plan.columns) > 1
    plans, a_copies = [], []
    for _, m in plan.rows:
        if copy_a:
            a_copies.append(copy_plan(dtype, m, K, 1 if ta else lda, lda if ta else 1,
                                      row_stride(K, dtype), 1, dev.cores)[0])
        row = []
        for _, n in plan.columns:
            chunk_epilogue = None
            if epilogue is not None:
                br, bc, _, beta, alpha = epilogue
                chunk_epilogue = (row_stride(n if bc else 1, dtype) if br else 0,
                                  1 if bc else 0, 0, beta, alpha)
            run, selected = _launch_plan(dtype, m, n, K,
                row_stride(K, dtype) if copy_a else lda,
                row_stride(n, dtype) if copy_b else ldb, row_stride(n, dtype),
                False if copy_a else ta, False if copy_b else tb,
                1, 0, 0, 0, False, dev, macos_major, chunk_epilogue)
            copy_right = (copy_plan(dtype, K, n, 1 if tb else ldb, ldb if tb else 1,
                                    row_stride(n, dtype), 1, dev.cores)[0] if copy_b else None)
            copy_result = copy_plan(dtype, m, n, row_stride(n, dtype), 1, ldc, 1, dev.cores)[0]
            copy_bias = None
            if epilogue is not None:
                br, bc = epilogue[:2]
                copy_bias = copy_plan(dtype, m if br else 1, n if bc else 1, br, bc,
                                       row_stride(n if bc else 1, dtype), 1, dev.cores)[0]
            row.append((run, selected, copy_right, copy_result, copy_bias))
        plans.append(row)

    def run(a, b, out, bias=None):
        if batched:
            a, b, out = a[0], b[0], out[0]
        for i, (start_m, m) in enumerate(plan.rows):
            left = a
            if copy_a:
                left = _temporary(a, m, K)
                a_copies[i](a[start_m:start_m+m, :], left)
            for j, (start_n, n) in enumerate(plan.columns):
                launch, _, copy_right, copy_result, copy_bias = plans[i][j]
                right = b
                if copy_b:
                    right = _temporary(b, K, n)
                    copy_right(b[:, start_n:start_n+n], right)
                result = _temporary(out, m, n)
                if epilogue is None:
                    launch(left, right, result)
                else:
                    br, bc, _, _, _ = epilogue
                    shape = (m if br else 1, n if bc else 1)
                    view = bias.as_strided(shape, (br, bc),
                        bias.storage_offset() + start_m * br + start_n * bc)
                    chunk_bias = _temporary(bias, *shape)
                    copy_bias(view, chunk_bias)
                    launch(left, right, result, chunk_bias)
                copy_result(result, out[start_m:start_m+m, start_n:start_n+n])

    return run, plan
