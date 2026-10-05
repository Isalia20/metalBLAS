"""NNA14 GPU correctness; newer GPUs exercise the M2 selection profile.

Running these tests on an M3+ checks the port's arithmetic, not M1/M2 hardware
performance or compiler compatibility. M2 Ultra's linear grid is also simulated.
"""
from functools import partial
import unittest

import torch

from metalblas.mps_nna14 import matmul
from metalblas.mps_nna14 import _launch_plan
from metalblas.mps_heuristics import DeviceInfo, device_info
import test_mps_a16 as a16_tests


@unittest.skipUnless(torch.backends.mps.is_available(), "Requires an Apple GPU")
class TestNNA14Execution(a16_tests.TestA16Execution):
    @classmethod
    def setUpClass(cls):
        actual = device_info()
        cls.dev = actual if actual.generation < 18 else DeviceInfo("Apple M2", 10, 17)
        cls.matmul_fn = staticmethod(partial(matmul, dev=cls.dev))
        torch.manual_seed(8414)

    def test_nna14_tile_shapes(self):
        # 24-wide K tiles and eight-SIMD-group thin matrices are absent on A16.
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for ta, tb in ((False, False), (False, True), (True, False), (True, True)):
                for shape in ((48, 48, 48), (8192, 32, 17), (32, 8192, 17)):
                    with self.subTest(dtype=dtype, layout=(ta, tb), shape=shape):
                        self.check_product(*shape, dtype, ta, tb)

    def test_ultra_linear_grid(self):
        # Source-derived macOS 27 M2 Ultra branch; this does not require an Ultra.
        self.matmul_fn = partial(matmul, dev=DeviceInfo("Apple M2 Ultra", 76, 17))
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for ta, tb in ((False, False), (False, True), (True, False), (True, True)):
                with self.subTest(dtype=dtype, layout=(ta, tb)):
                    self.check_product(47, 256, 2048, dtype, ta, tb, batch=3,
                                       padded=True, broadcast=True, macos_major=27)

    def test_copy_origin_boundary(self):
        # Bounded integers keep these long reductions exactly representable;
        # the test isolates addressing from float32 split-sum rounding.
        for ta, tb in ((False, False), (False, True), (True, False), (True, True)):
            for k in (32766, 32767, 32768):
                with self.subTest(layout=(ta, tb), k=k):
                    self.check_product(17, 33, k, torch.float32, ta, tb,
                                       padded=True, integral=True)
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(dtype=dtype, layout=(ta, tb), m=32767):
                    self.check_product(32767, 17, 7, dtype, ta, tb, padded=True)

    def test_wide_index_path(self):
        # A size-one batch permits unused strides above 2**32 without a huge
        # allocation. This exercises the wide shader, not a >4G-element access.
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for ta, tb in ((False, False), (False, True), (True, False), (True, True)):
                with self.subTest(dtype=dtype, layout=(ta, tb)):
                    a = torch.randint(-2, 3, (13, 17) if ta else (17, 13), device="mps").to(dtype)
                    b = torch.randint(-2, 3, (19, 13) if tb else (13, 19), device="mps").to(dtype)
                    a, b = a.t() if ta else a, b.t() if tb else b
                    a = a.as_strided((1, 17, 13), (2**32 + 64, *a.stride()))
                    b = b.as_strided((1, 13, 19), (2**33 + 32, *b.stride()))
                    out = torch.empty((1, 17, 19), device="mps", dtype=dtype)
                    out = out.as_strided(out.shape, (2**34 + 16, 19, 1))
                    got = self.matmul_fn(a, b, out=out)
                    expected = (a.cpu().double() @ b.cpu().double()).to(dtype)
                    torch.testing.assert_close(got.cpu(), expected, atol=0, rtol=0)


@unittest.skipUnless(torch.backends.mps.is_available(), "Requires an Apple GPU")
class TestNNA14LinkedEpilogue(a16_tests.TestA16LinkedEpilogue):
    @classmethod
    def setUpClass(cls):
        actual = device_info()
        cls.dev = actual if actual.generation < 18 else DeviceInfo("Apple M2", 10, 17)
        torch.manual_seed(8414)

    def fused(self, a, b, bias, out, beta=1., alpha=1.):
        # Reuse the stride/epilogue fixture with the older source family.
        from unittest.mock import patch
        with patch.object(a16_tests, '_launch_plan', _launch_plan), \
             patch.object(a16_tests, 'device_info', lambda: self.dev):
            return super().fused(a, b, bias, out, beta, alpha)


if __name__ == "__main__":
    unittest.main()
