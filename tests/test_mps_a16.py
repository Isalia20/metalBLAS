"""GPU correctness for the recovered A16 schedule, including float split planes."""
import unittest
import platform

import torch

from metalblas.mps_a16 import matmul
from metalblas.mps_heuristics import device_info
from metalblas.mps_a16 import _launch_plan
from metalblas.dispatch import _resolve_inputs, _resolve_bmm_inputs


@unittest.skipUnless(torch.backends.mps.is_available(), "Requires an Apple GPU")
class TestA16Execution(unittest.TestCase):
    matmul_fn = staticmethod(matmul)

    @classmethod
    def setUpClass(cls):
        if device_info().generation < 18:
            raise unittest.SkipTest("A16 requires M3 or newer")
        torch.manual_seed(8416)

    def check_product(self, m, n, k, dtype, ta, tb, batch=None, padded=False,
                      broadcast=False, macos_major=None, integral=False):
        prefix = () if batch is None else (batch,)

        def operand(rows, cols, trans):
            r, c = (cols, rows) if trans else (rows, cols)
            # Odd leading strides and offsets exercise unaligned and edge loads.
            shape = (*prefix, r, c + (3 if padded else 0))
            data = (torch.randint(-2, 3, shape, device="mps").to(dtype) if integral else
                    torch.randn(shape, device="mps", dtype=dtype))
            if padded:
                data = data[..., 1:c + 1]
            return data.transpose(-2, -1) if trans else data

        a, b = operand(m, k, ta), operand(k, n, tb)
        if broadcast:
            a = a[:1].expand(batch, -1, -1)
        storage = torch.full((*prefix, m, n + 3), -123, device="mps", dtype=dtype)
        out = storage[..., 1:n + 1]
        got = self.matmul_fn(a, b, out=out, macos_major=macos_major)
        self.assertIs(got, out)
        expected = (a.cpu().double() @ b.cpu().double()).to(dtype).float()
        atol, rtol = {torch.float32: (1e-4, 2e-5),
                      torch.float16: (2e-3, 1e-3),
                      torch.bfloat16: (2e-2, 8e-3)}[dtype]
        torch.testing.assert_close(got.cpu().float(), expected, atol=atol, rtol=rtol)
        self.assertTrue(bool((storage[..., 0] == -123).all()))
        self.assertTrue(bool((storage[..., n + 1:] == -123).all()))

    def test_layouts_and_edges(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for ta, tb in ((False, False), (False, True), (True, False), (True, True)):
                for shape in ((64, 64, 64), (35, 47, 29), (17, 129, 65), (129, 17, 7)):
                    with self.subTest(dtype=dtype, layout=(ta, tb), shape=shape):
                        self.check_product(*shape, dtype, ta, tb, padded=True)

    def test_batch_and_split_k(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for os_major in (26, 27):
                for ta, tb in ((False, False), (False, True), (True, False), (True, True)):
                    with self.subTest(dtype=dtype, os=os_major, layout=(ta, tb)):
                        self.check_product(31, 49, 2073, dtype, ta, tb, batch=3,
                                           padded=True, broadcast=True, macos_major=os_major)
            self.check_product(64, 64, 64, dtype, False, False, batch=3)

    def test_empty_reduction(self):
        for shape in ((3, 0), (2, 3, 0)):
            a = torch.empty(shape, device="mps")
            b = torch.empty((*shape[:-2], 0, 5), device="mps")
            got = self.matmul_fn(a, b)
            self.assertEqual(tuple(got.shape), (*shape[:-1], 5))
            self.assertEqual(got.count_nonzero().item(), 0)


@unittest.skipUnless(torch.backends.mps.is_available(), "Requires an Apple GPU")
class TestA16StagedAddressWidth(unittest.TestCase):
    matmul_fn = staticmethod(matmul)
    check_product = TestA16Execution.check_product

    @classmethod
    def setUpClass(cls):
        TestA16Execution.setUpClass()

    def test_staged_address_width(self):
        # BK=32 gives lda=1023/1024 on either side of the staged-copy
        # isDim16b bound. The longer reductions cross its dimension bound.
        cases = ((65, 33, 1020, False), (65, 33, 1021, False),
                 (1020, 33, 65, True), (1021, 33, 65, True),
                 (17, 19, 32766, False), (17, 19, 32767, False))
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for m, n, k, ta in cases:
                with self.subTest(dtype=dtype, shape=(m, n, k), trans_a=ta):
                    self.check_product(m, n, k, dtype, ta, False,
                                       padded=True, integral=True)


@unittest.skipUnless(torch.backends.mps.is_available(), "Requires an Apple GPU")
class TestA16UnstagedLoads(unittest.TestCase):
    matmul_fn = staticmethod(matmul)
    check_product = TestA16Execution.check_product

    @classmethod
    def setUpClass(cls):
        TestA16Execution.setUpClass()

    def test_paired_load_edges(self):
        # Cubic reduced-precision products bypass staging. Cross both K-tail
        # boundaries with offset/padded inputs and broadcast batches.
        for dtype in (torch.float16, torch.bfloat16):
            for ta, tb in ((False, False), (False, True), (True, False), (True, True)):
                for size in (31, 33):
                    for batch in (None, 2):
                        with self.subTest(dtype=dtype, layout=(ta, tb), size=size, batch=batch):
                            self.check_product(size, size, size, dtype, ta, tb, batch=batch,
                                               padded=True, broadcast=batch is not None, integral=True)


@unittest.skipUnless(torch.backends.mps.is_available(), "Requires an Apple GPU")
class TestA16LinkedEpilogue(unittest.TestCase):
    shapes = ((33, 33, 33), (31, 49, 2073))
    require_split = True
    @classmethod
    def setUpClass(cls):
        TestA16Execution.setUpClass()

    def fused(self, a, b, bias, out, beta=1., alpha=1.):
        batch = a.shape[0] if a.ndim == 3 else 1
        if a.ndim == 3:
            A, B, m, n, k, lda, ldb, ta, tb, bsa, bsb = _resolve_bmm_inputs(a, b)
            bb, br, bc = bias.expand(batch, m, n).stride()
            bsc = out.stride(0)
        else:
            A, B, m, n, k, lda, ldb, ta, tb = _resolve_inputs(a, b)
            br, bc = bias.expand(m, n).stride()
            bsa = bsb = bsc = bb = 0
        run, plan = _launch_plan(a.dtype, m, n, k, lda, ldb, out.stride(-2), ta, tb,
                                 False, False, batch, bsa, bsb, bsc, device_info(),
                                 int(platform.mac_ver()[0].split('.')[0]),
                                 (br, bc, bb, beta, alpha))
        run(A, B, out, bias)
        return plan

    @staticmethod
    def reference(bias, a, b, *, beta, alpha):
        op = torch.addmm if a.ndim == 2 else torch.baddbmm
        return op(bias, a, b, beta=beta, alpha=alpha).cpu()

    def test_bias_layouts_and_split_reduction(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for i, (ta, tb) in enumerate(((False, False), (False, True), (True, False), (True, True))):
                for m, n, k in self.shapes:
                    for batch in (None, 3):
                        prefix = () if batch is None else (batch,)

                        def operand(rows, cols, trans):
                            r, c = (cols, rows) if trans else (rows, cols)
                            data = torch.randint(-2, 3, (*prefix, r, c + 3), device="mps").to(dtype) / 8
                            data = data[..., 1:c + 1]
                            return data.transpose(-2, -1) if trans else data

                        a, b = operand(m, k, ta), operand(k, n, tb)
                        if batch is not None:
                            a = a[:1].expand(batch, -1, -1)
                        bias = (torch.randint(-8, 9, (*prefix, m, 2 * n + 1), device="mps").to(dtype) / 8)[..., 1::2]
                        if i == 0:
                            bias = bias[..., :1, :]
                        elif i == 1:
                            bias = bias[..., :, :1]
                        elif i == 3:
                            bias = bias[..., :1, :1]
                        if batch is not None and i % 2 == 0:
                            bias = bias[:1]
                        # Products are multiples of 1/64 and accumulate exactly
                        # in float, isolating epilogue rounding from GEMM error.
                        product = a.cpu().float() @ b.cpu().float()
                        for beta, alpha in ((1., 1.), (.37, .73), (0., .73), (.37, 0.)):
                            with self.subTest(dtype=dtype, layout=(ta, tb), shape=(m, n, k), batch=batch,
                                              beta=beta, alpha=alpha):
                                storage = torch.full((*prefix, m, n + 3), -123, device="mps", dtype=dtype)
                                out = storage[..., 1:n + 1]
                                x = torch.full_like(bias, float('nan')) if beta == 0 else bias
                                plan = self.fused(a, b, x, out, beta, alpha)
                                if self.require_split and k == 2073:
                                    self.assertGreater(plan.splits, 1)
                                expected = product * torch.tensor(alpha, dtype=dtype).float()
                                if beta != 0:
                                    expected = expected + bias.cpu().float() * torch.tensor(beta, dtype=dtype).float()
                                torch.testing.assert_close(out.cpu(), expected.to(dtype), atol=0, rtol=0)
                                # PyTorch 2.12's baddbmm graph propagates NaN
                                # bias even when beta=0. Check our ignored-bias
                                # behavior against CPU and use finite bias for
                                # the cross-version MPS arithmetic comparison.
                                reference = self.reference(bias, a, b, beta=beta, alpha=alpha)
                                torch.testing.assert_close(out.cpu(), reference, atol=0, rtol=0)
                                self.assertTrue(bool((storage[..., 0] == -123).all()))
                                self.assertTrue(bool((storage[..., n + 1:] == -123).all()))

    def test_conversion_after_addition(self):
        for dtype, step in ((torch.float16, 2**-10), (torch.bfloat16, 2**-7)):
            a = torch.eye(33, device="mps", dtype=dtype) * (1 + step)
            b = a.clone()
            bias = -torch.eye(33, device="mps", dtype=dtype) * (1 + 2 * step)
            out = torch.empty_like(a)
            self.fused(a, b, bias, out)
            expected = torch.eye(33, device="mps", dtype=dtype) * step**2
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
            torch.testing.assert_close(out, torch.addmm(bias, a, b), atol=0, rtol=0)

    def test_separate_float_multiply_and_add(self):
        a = torch.ones((33, 1), device="mps")
        b = torch.full((1, 33), 65537., device="mps")
        bias = torch.full((33, 33), -65537., device="mps")
        out = torch.empty_like(bias)
        self.fused(a, b, bias, out, alpha=1 + 2**-23)
        torch.testing.assert_close(out, torch.full_like(out, 2**-7), atol=0, rtol=0)
        torch.testing.assert_close(out, torch.addmm(bias, a, b, alpha=1 + 2**-23), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
