"""Public static dispatch: routing, layouts, output views and no timing calls."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import torch

import metalblas
from metalblas import float_dispatch, mps_dispatch
import test_mps_a16 as a16_tests
from tools.mps_graph_reference import addmm as graph_addmm


def public_matmul(a, b, *, out=None, macos_major=None):
    return metalblas.matmul(a, b, out=out)


@unittest.skipUnless(torch.backends.mps.is_available(), "Requires an Apple GPU")
class TestMPSDefaultDispatch(unittest.TestCase):
    matmul_fn = staticmethod(public_matmul)
    check_product = a16_tests.TestA16Execution.check_product

    def setUp(self):
        torch.manual_seed(8418)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(float_dispatch, '_TILE_POLICY', 'mps'))
        # Public matmul/bmm must reach the source-derived planners even if
        # old planning functions remain available for explicit comparisons.
        for name in ('_gemm_plan', '_gemm_trans_plan', '_bmm_plan', '_fastest'):
            self.stack.enter_context(patch.object(float_dispatch, name,
                side_effect=AssertionError(f'Default dispatch reached {name}')))

    def test_layouts_and_batches(self):
        shapes = ((1, 17, 33), (17, 1, 33), (7, 65, 129), (65, 8, 2176),
                  (33, 49, 257), (64, 64, 64), (17, 19, 4097))
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for ta, tb in ((False, False), (False, True), (True, False), (True, True)):
                for batch in (None, 3):
                    for shape in shapes:
                        with self.subTest(dtype=dtype, layout=(ta, tb), batch=batch, shape=shape):
                            self.check_product(*shape, dtype, ta, tb, batch=batch,
                                               padded=True, broadcast=batch is not None, integral=True)
        for shape in ((5, 3, 129), (12, 7, 8193)):
            self.check_product(*shape, torch.float16, False, True, padded=True, integral=True)

    def test_output_views_and_empty_inputs(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for batch in (None, 2):
                prefix = () if batch is None else (batch,)
                for k in (0, 33):
                    a = torch.randint(-2, 3, (*prefix, 17, k), device='mps').to(dtype)
                    b = torch.randint(-2, 3, (*prefix, k, 19), device='mps').to(dtype)
                    expected = (a.cpu().double() @ b.cpu().double()).to(dtype)
                    for column_major in (False, True):
                        storage = torch.full((*prefix, 19, 17) if column_major else (*prefix, 17, 38),
                                             -123, dtype=dtype, device='mps')
                        out = storage.transpose(-1, -2) if column_major else storage[..., ::2]
                        got = metalblas.matmul(a, b, out=out)
                        self.assertIs(got, out)
                        torch.testing.assert_close(got.cpu(), expected, atol=0, rtol=0)
                        if not column_major:
                            self.assertTrue(bool((storage[..., 1::2] == -123).all()))
            for ashape, bshape in (((0, 7), (7, 3)), ((3, 7), (7, 0)),
                                    ((0, 3, 7), (0, 7, 5))):
                a = torch.empty(ashape, dtype=dtype, device='mps')
                b = torch.empty(bshape, dtype=dtype, device='mps')
                torch.testing.assert_close(metalblas.matmul(a, b).cpu(), a.cpu() @ b.cpu())

    def test_promotion_broadcast_and_pooled_outputs(self):
        shapes = (((33,), (33,)), ((33,), (33, 19)), ((17, 33), (33,)),
                  ((17, 33), (2, 33, 19)), ((2, 17, 33), (33, 19)),
                  ((2, 1, 17, 33), (1, 3, 33, 19)), ((64, 64), (64, 64)))
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for ashape, bshape in shapes:
                a = torch.randint(-2, 3, ashape, device='mps').to(dtype)
                b = torch.randint(-2, 3, bshape, device='mps').to(dtype)
                expected = (a.cpu().double() @ b.cpu().double()).to(dtype)
                first = metalblas.matmul(a, b)
                second = metalblas.matmul(a + 1, b)
                torch.testing.assert_close(first.cpu(), expected, atol=0, rtol=0)
                torch.testing.assert_close(second.cpu(), ((a.cpu().double()+1) @ b.cpu().double()).to(dtype),
                                           atol=0, rtol=0)

    def test_explicit_backend(self):
        a = torch.randint(-2, 3, (33, 17), device='mps').float()
        b = torch.randint(-2, 3, (17, 49), device='mps').float()
        with patch.object(mps_dispatch, 'matmul', side_effect=AssertionError('Ignored explicit backend')):
            got = metalblas.matmul(a, b, backend='simd', tile=(16, 16, 16, 1, 1), swizzle_log=0)
        torch.testing.assert_close(got.cpu(), a.cpu() @ b.cpu(), atol=0, rtol=0)

    def test_rank_one_against_cpu(self):
        # PyTorch 2.12 uses its own matmul_half fallback for this shape;
        # its rounding error can exceed the benchmark tolerance. Validate
        # our recovered vector path against an independent double reference.
        a = torch.randn((4096, 4096), dtype=torch.float16, device='mps')
        b = torch.randn((4096, 1), dtype=torch.float16, device='mps')
        expected = (a.cpu().double() @ b.cpu().double()).half()
        got = metalblas.matmul(a, b).cpu()
        torch.testing.assert_close(got, expected, atol=0.002, rtol=0.001)


@unittest.skipUnless(torch.backends.mps.is_available(), "Requires an Apple GPU")
class TestMPSDefaultEpilogues(a16_tests.TestA16LinkedEpilogue):
    shapes = ((33, 33, 33), (31, 49, 2073), (1, 17, 129), (17, 1, 129),
              (7, 65, 129), (65, 8, 2176), (7, 3, 129), (12, 7, 8193))
    require_split = False
    setUp = TestMPSDefaultDispatch.setUp
    # Some installed PyTorch versions use addmm_half/gemv_* Metal kernels
    # for sliced transposes or rank-one products. They keep float scalars;
    # this reference exercises MPSGraph's recovered scalar-rounding rules.
    reference = staticmethod(graph_addmm)

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(8418)

    def fused(self, a, b, bias, out, beta=1., alpha=1.):
        op = metalblas.addmm if a.ndim == 2 else metalblas.baddbmm
        self.assertIs(op(bias, a, b, beta=beta, alpha=alpha, out=out), out)

    def test_empty_reductions_and_output_views(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for batch in (None, 2):
                prefix = () if batch is None else (batch,)
                op = metalblas.addmm if batch is None else metalblas.baddbmm
                for k in (0, 33):
                    a = torch.ones((*prefix, 17, k), device='mps', dtype=dtype)
                    b = torch.ones((*prefix, k, 19), device='mps', dtype=dtype)
                    bias = torch.ones(19, device='mps', dtype=dtype)
                    expected = (torch.tensor(k, dtype=torch.float32) * torch.tensor(.73, dtype=dtype).float()
                                + torch.tensor(.37, dtype=dtype).float()).to(dtype)
                    for transposed in (False, True):
                        storage = torch.full((*prefix, 19, 17) if transposed else (*prefix, 17, 38),
                                             -123, device='mps', dtype=dtype)
                        out = storage.transpose(-1, -2) if transposed else storage[..., ::2]
                        self.assertIs(op(bias, a, b, alpha=.73, beta=.37, out=out), out)
                        torch.testing.assert_close(out.cpu(), torch.full(out.shape, expected.item(), dtype=dtype),
                                                   atol=0, rtol=0)
                        if not transposed:
                            self.assertTrue(bool((storage[..., 1::2] == -123).all()))


if __name__ == '__main__':
    unittest.main()
