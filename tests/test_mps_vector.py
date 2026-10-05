"""GPU correctness of the ordinary vector schedules recovered from MPS AIR."""
import unittest

import torch

from metalblas.mps_vector import matmul
from metalblas.mps_heuristics import device_info
import test_mps_a16 as a16_tests


@unittest.skipUnless(torch.backends.mps.is_available(), "Requires an Apple GPU")
class TestVectorExecution(unittest.TestCase):
    matmul_fn = staticmethod(matmul)
    check_product = a16_tests.TestA16Execution.check_product
    test_empty_reduction = a16_tests.TestA16Execution.test_empty_reduction

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(8417)

    def test_layouts_and_families(self):
        # All four AIR families, both orientations, and each reduction path.
        # W > 128 keeps half NT products outside the separately tested LoRA path.
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for r, k in ((1, 255), (2, 257), (4, 19), (7, 65), (8, 2048),
                         (9, 2176), (12, 129), (15, 8195), (16, 127)):
                for m, n in ((129, r), (r, 129)):
                    for ta, tb in ((False, False), (False, True), (True, False), (True, True)):
                        with self.subTest(dtype=dtype, shape=(m, n, k), layout=(ta, tb)):
                            self.check_product(m, n, k, dtype, ta, tb, padded=True)

    def test_k_loop_boundaries(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for r in (1, 3, 4, 5, 6):
                for k in (1, 2, 3, 4, 127, 128, 129, 256, 2047, 2048,
                          2080, 2176, 2304, 3968, 4096, 4097):
                    with self.subTest(dtype=dtype, r=r, k=k):
                        self.check_product(129, r, k, dtype, False, True, padded=True)

    def test_transposed_wide_edges(self):
        # Exercises x widths 1/2/4/8 and both OS coverage thresholds. Partial
        # output strips still participate in the shared-memory reduction.
        cores = device_info().cores
        for os_major in (26, 27):
            for w in (2, 3, 5, 17, cores * 8 - 1, cores * 16 + 1, cores * 32 + 1):
                for dtype in (torch.float32, torch.float16, torch.bfloat16):
                    with self.subTest(os=os_major, w=w, dtype=dtype):
                        self.check_product(w, 2, 131, dtype, True, False,
                                           padded=True, macos_major=os_major)

    def test_batch_and_broadcast(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for r in (1, 4, 9):
                for m, n in ((31, r), (r, 31)):
                    for ta, tb in ((False, False), (False, True), (True, False), (True, True)):
                        with self.subTest(dtype=dtype, shape=(m, n, 2073), layout=(ta, tb)):
                            self.check_product(m, n, 2073, dtype, ta, tb,
                                               batch=3, padded=True, broadcast=True)

    def test_lora_delegation(self):
        # A strided size-one column exercises LoRA's special vector-load flag.
        for n in (1, 2, 9, 16):
            for k in (128, 129):
                with self.subTest(n=n, k=k):
                    a = torch.randn((32, k + 2), device="mps", dtype=torch.float16)[:, 1:k + 1]
                    b = torch.randn((n, k + 2), device="mps", dtype=torch.float16)[:, 1:k + 1].t()
                    ac, bc = a.cpu().double(), b.cpu().double()
                    end = k - k % 4
                    expected = ac[:, :end] @ bc[:end]
                    for i in range(end, k):
                        expected += (ac[:, i, None] * bc[None, i, :]).half().double()
                    torch.testing.assert_close(matmul(a, b).cpu().float(), expected.half().float(),
                                               atol=2e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
