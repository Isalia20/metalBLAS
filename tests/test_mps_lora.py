"""GPU correctness of the LoRA schedule recovered from MPSNDArray AIR."""
import unittest

import torch

from metalblas.mps_lora import matmul


@unittest.skipUnless(torch.backends.mps.is_available(), "Requires an Apple GPU")
class TestLoRAExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(8415)

    def check_product(self, m, n, k, padding=0, offset=0):
        def operand(rows):
            data = torch.randn((rows, k + padding), device="mps", dtype=torch.float16)
            return data[:, offset:offset + k]

        a, b = operand(m), operand(n).t()
        storage = torch.full((m, n + 3), -123, device="mps", dtype=torch.float16)
        out = storage[:, 1:n + 1]
        self.assertIs(matmul(a, b, out=out), out)
        # AIR accumulates the vector body in float, but multiplies the final
        # K%4 scalar elements in half before converting them to float.
        ac, bc = a.cpu().double(), b.cpu().double()
        end = k - k % 4
        expected = ac[:, :end] @ bc[:end]
        for i in range(end, k):
            expected += (ac[:, i, None] * bc[None, i, :]).half().double()
        torch.testing.assert_close(out.cpu().float(), expected.half().float(),
                                   atol=2e-3, rtol=1e-3)
        self.assertTrue(bool((storage[:, 0] == -123).all()))
        self.assertTrue(bool((storage[:, n + 1:] == -123).all()))

    def test_table_rows_and_simd_doubling(self):
        for n in range(2, 17):
            for k in (17, 2048, 8192, 8193):
                with self.subTest(n=n, k=k):
                    self.check_product(32, n, k)

    def test_k_loops_and_remainders(self):
        # Covers sub-vector K, complete/partial strips, specialized 0/1/2
        # main-loop iterations and the runtime loop, across three table rows.
        for n in (2, 9, 16):
            for k in (1, 2, 3, 4, 127, 128, 129, 255, 256, 257, 1023,
                      1024, 1025, 1536, 4096, 6144, 16384):
                with self.subTest(n=n, k=k):
                    self.check_product(n, n, k)

    def test_strides_offsets_and_output(self):
        for n in (2, 5, 9, 14, 15, 16):
            for k in (19, 2051, 8195):
                # First stride is a multiple of four with an unaligned base;
                # the second takes the scalar addressing path.
                for padding in (1, 2):
                    with self.subTest(n=n, k=k, padding=padding):
                        self.check_product(128, n, k, padding=padding, offset=1)


if __name__ == "__main__":
    unittest.main()
