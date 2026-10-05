"""A18 launch geometry checked against captured encoder dispatches."""
import gzip
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import patch

import torch

import metalblas
from metalblas import float_dispatch, mps_a18_global, mps_heuristics


class TestA18LaunchGeometry(unittest.TestCase):
    def test_partial_uber_tiles_against_mps(self):
        path = Path(__file__).parent / 'fixtures' / 'mps_a18' / 'M5_Pro_geometry_27.2.json.gz'
        capture = json.loads(gzip.decompress(path.read_bytes()))
        dev = mps_heuristics.DeviceInfo(**capture['device'])
        self.assertEqual(len(capture['cases']), 16)
        for row in capture['cases']:
            case = row['case']
            with self.subTest(case=case):
                dtype = {'f16': torch.float16, 'bf16': torch.bfloat16}[case['dtype']]
                m, n, k = case['shape']
                ta, tb = (p == 't' for p in case['layout'])
                batch = case['batch']
                key = mps_heuristics.make_key(m, n, k, ta, tb,
                                               mps_heuristics.DTYPE_CODE[dtype], batch)
                self.assertEqual(tuple(key), tuple(row['mps_key']))
                _, params = mps_heuristics.select_params(m, n, k, ta, tb, dtype, dev, batch)
                self.assertEqual(params, tuple(row['mps_params']))
                # No GPU compilation is needed to check host launch geometry.
                with patch.object(float_dispatch.kernels, 'a18_gemm', return_value=(None, None)):
                    _, threads, group, _ = float_dispatch._a18_launch(
                        dtype, m, n, k, params, ta, tb, batch=batch)
                dispatch = next(d for d in row['dispatches'] if d['kernel'] == 'gemm_a18')
                self.assertTrue(all(t % g == 0 for t, g in zip(threads, group)))
                self.assertEqual(tuple(t // g for t, g in zip(threads, group)), tuple(dispatch['grid']))
                self.assertEqual(group, tuple(dispatch['tg']))

    def test_global_schedules_against_native_encoder(self):
        for mode in (0, 1):
            path = Path(__file__).parent / 'fixtures' / 'mps_a18' / f'M5_Pro_explicit_global_mode{mode}_27.2.json.gz'
            capture = json.loads(gzip.decompress(path.read_bytes()))
            self.assertTrue(capture['selection_override'])
            self.assertTrue(capture['logged_params_verified'])
            self.assertEqual(len(capture['cases']), 64)
            for row in capture['cases']:
                case = row['case']
                with self.subTest(mode=mode, case=case):
                    self.assertEqual(row['mps_key'], case['key'])
                    self.assertEqual(row['mps_params'], case['params'])
                    plan = mps_a18_global.schedule(*case['shape'], case['batch'], case['params'])
                    dispatch = next(d for d in row['dispatches'] if d['kernel'] == 'gemm_a18')
                    params = bytes.fromhex(dispatch['bytes']['26']['hex'])
                    self.assertEqual(plan.grid, tuple(dispatch['grid']))
                    self.assertEqual(plan.threadgroup, tuple(dispatch['tg']))
                    self.assertEqual(plan.k_per_split, struct.unpack_from('<I', params, 12)[0])
                    self.assertEqual(plan.k_iterations, struct.unpack_from('<I', params, 320)[0])
                    self.assertEqual(plan.partial_elements * 4, dispatch['bufs']['25']['len'])
                    dtype = {'f16': torch.float16, 'bf16': torch.bfloat16}[case['dtype']]
                    ta, tb = (p == 't' for p in case['layout'])
                    with patch.object(float_dispatch.kernels, 'a18_gemm', return_value=(None, None)):
                        _, threads, group, dims = float_dispatch._a18_launch(
                            dtype, *case['shape'], case['params'], ta, tb,
                            batch=case['batch'] if case['batch'] > 1 else None)
                    self.assertEqual(tuple(t // g for t, g in zip(threads, group)), plan.grid)
                    self.assertEqual(group, plan.threadgroup)
                    self.assertEqual(dims[5] & 0xffffffff, plan.k_iterations)
                    if mode:
                        helper = next(d for d in row['dispatches'] if 'InitializeAtomicCounter' in d['kernel'])
                        self.assertEqual(plan.counter_elements * 4, dispatch['bufs']['24']['len'])
                        self.assertEqual(plan.initializer_grid, tuple(helper['grid']))
                        self.assertEqual(helper['tg'], [32, 1, 1])
                    else:
                        helper = next(d for d in row['dispatches'] if 'gemmA18GlobalReduction' in d['kernel'])
                        self.assertEqual(plan.reduction_grid, tuple(helper['grid']))
                        self.assertEqual(helper['tg'], [256, 1, 1])


@unittest.skipUnless(torch.backends.mps.is_available() and mps_heuristics.device_info().uses_a18_kernel,
                     'Requires an A18-class Apple GPU')
class TestA18GlobalExecution(unittest.TestCase):
    source_device = mps_heuristics.DeviceInfo('Source-model 80-core A18', 80, 23)

    def test_global_modes_layouts_and_partitions(self):
        torch.manual_seed(4076)
        for mode in (0, 1):
            for dtype in (torch.float16, torch.bfloat16):
                for ta, tb in ((False, False), (False, True), (True, False), (True, True)):
                    for batch in (1, 3):
                        for m, n, k in ((64, 64, 2048), (17, 19, 1025), (65, 129, 1024), (128, 128, 12289)):
                            with self.subTest(mode=mode, dtype=dtype, transpose=(ta, tb), batch=batch, shape=(m,n,k)):
                                prefix = () if batch == 1 else (batch,)
                                def operand(rows, cols, transpose):
                                    r, c = (cols, rows) if transpose else (rows, cols)
                                    store = torch.randint(-2, 3, (*prefix, r, c+3), device='mps').to(dtype)
                                    view = store[..., 1:c+1]
                                    return view.transpose(-1,-2) if transpose else view
                                a, b = operand(m, k, ta), operand(k, n, tb)
                                _, params = mps_heuristics.select_params(m, n, k, ta, tb, dtype, self.source_device, batch)
                                self.assertEqual(params[6:8], (1, 1))
                                params = list(params)
                                params[7] = mode
                                storage = torch.full((*prefix,m,n+3), -123, device='mps', dtype=dtype)
                                out = storage[..., 1:n+1]
                                run, _ = mps_a18_global.launch_plan(dtype, m, n, k,
                                    a.stride(-1 if ta else -2), b.stride(-1 if tb else -2), n+3, ta, tb, batch,
                                    a.stride(0) if prefix else 0, b.stride(0) if prefix else 0,
                                    storage.stride(0) if prefix else 0, bool(prefix), tuple(params))
                                run(a, b, out)
                                torch.testing.assert_close(out.cpu(), (a.cpu().double() @ b.cpu().double()).to(dtype), atol=0, rtol=0)
                                self.assertTrue(bool((storage[..., 0] == -123).all()))
                                self.assertTrue(bool((storage[..., n+1:] == -123).all()))

    def test_public_route_fusion_and_output_views(self):
        torch.manual_seed(8649)
        with patch.object(mps_heuristics, 'device_info', return_value=self.source_device), \
             patch.object(float_dispatch, '_TILE_POLICY', 'mps'), \
             patch.object(float_dispatch, '_fastest', side_effect=AssertionError('Runtime timing')):
            for dtype in (torch.float16, torch.bfloat16):
                for batch in (1, 3):
                    for ta, tb in ((False, False), (False, True), (True, False), (True, True)):
                        for layout in ('padded', 'strided', 'transposed'):
                            with self.subTest(dtype=dtype, batch=batch, transpose=(ta,tb), out=layout):
                                m, n, k = 17, 19, 1025
                                prefix = () if batch == 1 else (batch,)
                                a = torch.randint(-2,3,(*prefix,k,m) if ta else (*prefix,m,k),device='mps').to(dtype)
                                b = torch.randint(-2,3,(*prefix,n,k) if tb else (*prefix,k,n),device='mps').to(dtype)
                                if ta: a = a.transpose(-1,-2)
                                if tb: b = b.transpose(-1,-2)
                                bias = torch.randn((*prefix,m,1),device='mps',dtype=dtype)
                                storage = torch.full((*prefix,n,m) if layout=='transposed' else (*prefix,m,n*2), -123, device='mps',dtype=dtype)
                                out = storage.transpose(-1,-2) if layout=='transposed' else storage[...,::2] if layout=='strided' else storage[...,:n]
                                op = metalblas.addmm if batch==1 else metalblas.baddbmm
                                result = op(bias,a,b,alpha=.73,beta=.37,out=out)
                                self.assertIs(result,out)
                                expected = (a.cpu().double() @ b.cpu().double()).float() * torch.tensor(.73,dtype=dtype).float()
                                expected += bias.cpu().float() * torch.tensor(.37,dtype=dtype).float()
                                torch.testing.assert_close(out.cpu(), expected.to(dtype), atol=0,rtol=0)
                                if layout!='transposed':
                                    untouched = storage[...,1::2] if layout=='strided' else storage[...,n:]
                                    self.assertTrue(bool((untouched==-123).all()))


if __name__ == '__main__':
    unittest.main()
