"""Batch-reduction schedules against captures, and execution against MPSGraph."""
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import patch

import torch

import metalblas
from metalblas import float_dispatch, mps_dispatch, mps_reduction
from metalblas.mps_heuristics import DeviceInfo, device_info
from tools.mps_graph_reference import addmm as graph_addmm

FIXTURES = Path(__file__).parent / 'fixtures'
DTYPES = (torch.float32, torch.float16, torch.bfloat16)


class TestReductionSchedules(unittest.TestCase):
    def test_source_plans_against_hardware(self):
        total_cases = total_passes = 0
        for directory in ('mps_reduction', 'mps_epilogues'):
            for name in ('M2_26.3', 'M3_Ultra_26.5.2', 'M4_26.6', 'M5_Pro_27.2'):
                capture = json.loads(gzip.decompress((FIXTURES / directory / (name + '.json.gz')).read_bytes()))
                dev = DeviceInfo(**capture['device'])
                major = int(capture['macos'].split('.')[0])
                for entry in capture['cases']:
                    c = entry['case']
                    if c['op'] != 'addbmm':
                        continue
                    with self.subTest(capture=name, case=c):
                        actual = [d for d in entry['dispatches']
                                  if d['kernel'] == 'reduce_multiple_passes_axes_add']
                        plans = mps_reduction.reduction_plans(
                            *c['shape'][:2], c.get('batch', 3), dev=dev, macos_major=major,
                            max_threads=min((d.get('maxThreads', 1024) for d in actual), default=1024))
                        self.assertEqual(len(plans), len(actual))
                        for plan, dispatch in zip(plans, actual):
                            raw = next(v['hex'] for v in dispatch['bytes'].values() if v['len'] == 148)
                            self.assertEqual(plan.parameters, struct.unpack('<37I', bytes.fromhex(raw)))
                            self.assertEqual(plan.grid, tuple(dispatch['grid']))
                            self.assertEqual(plan.threadgroup, tuple(dispatch['tg']))
                            self.assertEqual(plan.threadgroup_memory, dispatch['tgmem']['0'])
                        total_cases += 1
                        total_passes += len(plans)
        self.assertEqual((total_cases, total_passes), (480, 508))


@unittest.skipUnless(torch.backends.mps.is_available(), 'Requires an Apple GPU')
class TestBatchReductionExecution(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1642)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(float_dispatch, '_TILE_POLICY', 'mps'))
        for name in ('_gemm_plan', '_gemm_trans_plan', '_bmm_plan', '_fastest'):
            self.stack.enter_context(patch.object(float_dispatch, name,
                side_effect=AssertionError(f'Default addbmm reached {name}')))

    def test_layouts_and_typed_matmul_intermediate(self):
        shapes = ((1, 1, 7), (1, 17, 33), (17, 1, 33), (7, 65, 129),
                  (65, 8, 2176), (31, 49, 257), (64, 64, 128), (17, 19, 8193))
        for dtype in DTYPES:
            alpha = torch.tensor(.73, dtype=dtype).float()
            beta = torch.tensor(.37, dtype=dtype).float()
            for layout, (ta, tb) in enumerate(((False, False), (False, True),
                                               (True, False), (True, True))):
                for m, n, k in shapes:
                    with self.subTest(dtype=dtype, transpose=(ta, tb), shape=(m, n, k)):
                        def operand(rows, cols, transposed):
                            r, c = (cols, rows) if transposed else (rows, cols)
                            storage = torch.randint(-3, 4, (3, r, c + 3), device='mps').to(dtype)
                            view = storage[..., 1:c + 1]
                            return view.transpose(-1, -2) if transposed else view
                        a, b = operand(m, k, ta), operand(k, n, tb)
                        if layout % 2:
                            a = a[:1].expand(3, -1, -1)
                        bias_shape = ((), (n,), (m, 1), (n, m))[layout]
                        bias = torch.randn(bias_shape, device='mps', dtype=dtype)
                        if layout == 3:
                            bias = bias.t()
                        storage = torch.full((m, n + 3), -123, device='mps', dtype=dtype)
                        out = storage[:, 1:n + 1]
                        self.assertIs(metalblas.addbmm(bias, a, b, alpha=.73, beta=.37, out=out), out)
                        # Integer inputs make each matmul accumulator exact;
                        # the input-dtype store occurs before summing batches.
                        products = (a.cpu().double() @ b.cpu().double()).to(dtype).float()
                        expected = (products.sum(0) * alpha + bias.cpu().float() * beta).to(dtype)
                        torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
                        reference = graph_addmm(bias, a, b, alpha=.73, beta=.37, reduce_batch=True)
                        torch.testing.assert_close(out.cpu(), reference, atol=0, rtol=0)
                        self.assertTrue(bool((storage[:, 0] == -123).all()))
                        self.assertTrue(bool((storage[:, n + 1:] == -123).all()))

    def test_sum_order_and_multiple_passes(self):
        shapes = [(m, n, batch) for m, n in ((1, 1), (1, 3), (3, 1), (7, 5), (8, 8),
                                            (17, 9), (31, 49), (64, 64), (129, 127))
                  for batch in (2, 33, 1025)]
        shapes += [(1, 1, 8193), (3, 1, 65537), (2, 9, 1048577)]
        for dtype in DTYPES:
            for m, n, batch in shapes:
                with self.subTest(dtype=dtype, shape=(batch, m, n)):
                    # K=1 with integer multipliers gives an exact, typed
                    # intermediate. The independent graph then tests only
                    # the reduction order and final linked arithmetic.
                    a = torch.randint(-2, 3, (batch, m, 1), device='mps').to(dtype)
                    b = torch.randn((batch, 1, n), device='mps', dtype=dtype) * .125
                    values = (a * b).contiguous()
                    bias = torch.randn((m, n), device='mps', dtype=dtype)
                    out = torch.empty_like(bias)
                    mps_reduction.reduce_batch(values, out, bias, n, 1, .37, .73,
                                                dev=device_info(), macos_major=mps_dispatch._macos_major())
                    reference = graph_addmm(bias, a, b, alpha=.73, beta=.37, reduce_batch=True)
                    torch.testing.assert_close(out.cpu(), reference, atol=0, rtol=0)
                    for fn, _, _, plan in mps_reduction._launch_plan(
                            dtype, m, n, batch, n, device_info(), mps_dispatch._macos_major(), .37, .73):
                        self.assertEqual(fn.static_thread_group_memory_length, plan.threadgroup_memory)

    def test_unit_batch_removes_intermediate_rounding(self):
        for dtype, delta in ((torch.float16, 2. ** -11), (torch.bfloat16, 2. ** -8)):
            for batch in (1, 3):
                with self.subTest(dtype=dtype, batch=batch):
                    a = torch.tensor([1., delta], device='mps', dtype=dtype).reshape(1, 1, 2).expand(batch, -1, -1)
                    b = torch.ones((batch, 2, 1), device='mps', dtype=dtype)
                    bias = torch.tensor(-batch, device='mps', dtype=dtype)
                    got = metalblas.addbmm(bias, a, b)
                    expected = torch.tensor([[delta if batch == 1 else 0.]], dtype=dtype)
                    torch.testing.assert_close(got.cpu(), expected, atol=0, rtol=0)
                    torch.testing.assert_close(got.cpu(), graph_addmm(bias, a, b, reduce_batch=True), atol=0, rtol=0)

    def test_output_views_empty_inputs_and_zero_beta(self):
        for dtype in DTYPES:
            for batch, k in ((0, 7), (3, 0), (3, 7)):
                for layout in ('padded', 'strided', 'transposed'):
                    with self.subTest(dtype=dtype, batch=batch, k=k, layout=layout):
                        a = torch.ones((batch, 17, k), device='mps', dtype=dtype)
                        b = torch.ones((batch, k, 19), device='mps', dtype=dtype)
                        bias = torch.ones((17, 1), device='mps', dtype=dtype)
                        storage = torch.full((19, 17) if layout == 'transposed' else (17, 38),
                                             -123, device='mps', dtype=dtype)
                        out = (storage.t() if layout == 'transposed' else storage[:, ::2]
                               if layout == 'strided' else storage[:, :19])
                        got = metalblas.addbmm(bias, a, b, alpha=.73, beta=.37, out=out)
                        expected = (torch.tensor(batch * k).float() * torch.tensor(.73, dtype=dtype).float()
                                    + torch.tensor(.37, dtype=dtype).float()).to(dtype)
                        self.assertIs(got, out)
                        torch.testing.assert_close(got.cpu(), torch.full((17, 19), expected.item(), dtype=dtype),
                                                   atol=0, rtol=0)
                        if layout != 'transposed':
                            untouched = storage[:, 1::2] if layout == 'strided' else storage[:, 19:]
                            self.assertTrue(bool((untouched == -123).all()))
                        got = metalblas.addbmm(bias * float('nan'), a, b, beta=0)
                        torch.testing.assert_close(got.cpu(), torch.full((17, 19), batch * k, dtype=dtype),
                                                   atol=0, rtol=0)
            for m, n in ((0, 3), (3, 0)):
                a = torch.empty((3, m, 7), device='mps', dtype=dtype)
                b = torch.empty((3, 7, n), device='mps', dtype=dtype)
                bias = torch.empty((m, n), device='mps', dtype=dtype)
                self.assertEqual(tuple(metalblas.addbmm(bias, a, b).shape), (m, n))


if __name__ == '__main__':
    unittest.main()
