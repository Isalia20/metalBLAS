"""Native executable reuse and late tensor binding in source launch plans."""
import gc
import gzip
import json
from pathlib import Path
import unittest
import weakref

import torch
import metalblas
from metalblas import mps_submission


@unittest.skipUnless(torch.backends.mps.is_available(), "requires MPS")
class TestSubmissionRuntime(unittest.TestCase):
    def test_native_executable_identities(self):
        paths = sorted((Path(__file__).parent / "fixtures/mps_submission_keys").glob("*.json.gz"))
        self.assertEqual(len(paths), 3)
        cases = 0
        for path in paths:
            fixture = json.loads(gzip.decompress(path.read_bytes()))
            for capture in fixture["captures"]:
                ids = {}
                for case in capture:
                    dtype = getattr(torch, case["dtype"].removeprefix("torch."))
                    key = mps_submission._key(case["operation"], dtype, case["a_shape"], case["b_shape"],
                                              case["bias_shape"], case["beta"], case.get("alpha", 1.0))
                    actual = ids.setdefault(key, len(ids))
                    self.assertEqual(actual, case["executable"], (path.name, case))
                    cases += 1
        self.assertEqual(cases, 453)

    def test_rebinds_aliased_inputs_and_storage_offsets(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            # First use aliases the operands. Subsequent calls with the same
            # shape and strides must bind each new operand independently.
            for offset in (0, 3, 17):
                with self.subTest(dtype=dtype, offset=offset):
                    owner = torch.full((offset + 32 * 32 + 11,), 2.0, dtype=dtype, device="mps")
                    a = owner[offset:offset + 32 * 32].view(32, 32)
                    b = a if offset == 0 else torch.full((32, 32), 3.0, dtype=dtype, device="mps")
                    destination = torch.empty(offset + 32 * 37 + 4, dtype=dtype, device="mps")
                    out = destination.as_strided((32, 32), (37, 1), offset)
                    result = metalblas.matmul(a, b, out=out)
                    self.assertIs(result, out)
                    self.assertTrue((result.cpu() == (128 if offset == 0 else 192)).all().item())

    def test_cached_plan_does_not_retain_inputs(self):
        a = torch.ones(31, 33, device="mps")
        b = torch.ones(33, 49, device="mps")
        out = torch.empty(31, 49, device="mps")
        references = [weakref.ref(value) for value in (a, b, out)]
        metalblas.matmul(a, b, out=out)
        torch.mps.synchronize()
        del a, b, out
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_temporary_lifetime_across_commits_and_errors(self):
        self._check_temporary_lifetime([])

    def test_graph_result_lifetime_across_commits_and_errors(self):
        self._check_temporary_lifetime([4, 5])

    def _check_temporary_lifetime(self, graph_results):
        from metalblas import kernels
        from metalblas._native import submission
        from metalblas.dispatch import _pk

        native = submission()
        fn = kernels.mps_identity('float', 0, False, 1, 32).function
        n = 257
        metadata = [torch.empty(n, device='meta'), torch.empty(n, device='meta')]
        dims = _pk(1, n) + [n, 1, n, 1]

        def copy(src, dst, offset=0):
            return native.Launch(fn, (native.BufferReference(src, offset),
                                      native.BufferReference(dst, 0), dims), (288, 1), (32, 1))

        good = native.Program([([copy(0, 4)], n), ([copy(4, 2)], n),
                               ([copy(1, 5)], n), ([copy(5, 3)], n)], metadata, 4, graph_results)
        # Leave one live temporary, then allocate another before rejecting an
        # invalid offset. Both must be returned even when encoding throws.
        bad = native.Program([([copy(0, 4)], n), ([copy(5, 2, n * 4)], n),
                              ([copy(4, 3)], n)], metadata, 4, graph_results)
        for force_commit in (False, True):
            # Injected diagnostic limits exercise both lifetimes without
            # depending on device-specific production submission thresholds.
            limits = (1,) * 6 if force_commit else (10000, 10000000) * 3
            state = native.Executable(limits)
            for iteration in range(8):
                with self.subTest(force_commit=force_commit, iteration=iteration):
                    a = torch.arange(n, dtype=torch.float32) + iteration
                    b = a + 1000
                    inputs = (a.to('mps'), b.to('mps'),
                              torch.empty(n, device='mps'), torch.empty(n, device='mps'))
                    with self.assertRaisesRegex(RuntimeError, 'Launch offset exceeds storage'):
                        bad.run(inputs, state)
                    torch.mps.synchronize()
                    self.assertEqual(state.count, 0)
                    trace = good.run(inputs, state, True)
                    torch.mps.synchronize()
                    self.assertEqual(state.count, 0)
                    self.assertEqual(trace['commits'], 4 if force_commit else 0)
                    self.assertTrue(torch.equal(inputs[2].cpu(), a))
                    self.assertTrue(torch.equal(inputs[3].cpu(), b))

    def test_graph_result_survives_executable_eviction(self):
        results = []
        # Keep GPU work queued while discarding Python's executable owners.
        # Each call needs distinct values to expose premature storage reuse.
        for iteration in range(12):
            a = torch.full((3, 127, 33), float(iteration + 1), device='mps')
            b = torch.ones(3, 33, 65, device='mps')
            bias = torch.ones(65, device='mps')
            results.append((metalblas.addbmm(bias, a, b), 99 * (iteration + 1) + 1))
            mps_submission._executable.cache_clear()
            del a, b, bias
        gc.collect()
        torch.mps.synchronize()
        for result, expected in results:
            self.assertTrue((result.cpu() == expected).all().item())


if __name__ == "__main__":
    unittest.main()
