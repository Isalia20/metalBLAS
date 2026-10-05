"""Identity-copy schedules from native encoders, plus strided GPU copies."""
import gzip
import itertools
import json
from pathlib import Path
import unittest

import torch

from metalblas import mps_identity
from metalblas.mps_heuristics import device_info


class TestIdentitySchedules(unittest.TestCase):
    def test_native_orders(self):
        count = 0
        for path in (Path(__file__).parent / 'fixtures/mps_identity').glob('*_orders.json.gz'):
            data = json.loads(gzip.decompress(path.read_bytes()))
            self.assertEqual(data['mismatches'], [])
            for row in data['cases']:
                actual = mps_identity.launch_order(row['src'], row['dst'], row['shape'])
                self.assertEqual(actual, (tuple(row['order']), row['contiguous']), row)
                count += 1
        self.assertEqual(count, 10092)

    def test_native_dispatches(self):
        count = 0
        for path in (Path(__file__).parent / 'fixtures/mps_identity').glob('*.json.gz'):
            if '_orders.' in path.name:
                continue
            data = json.loads(gzip.decompress(path.read_bytes()))
            self.assertEqual(data['mismatches'], [])
            for row in data['cases']:
                c = row['case']
                copies = [d for d in row['dispatches'] if 'Identity' in d['kernel']]
                self.assertEqual(len(copies), 1, c)
                d = copies[0]
                plan = mps_identity.schedule(*c['shape'], c['ta'], c['tb'],
                                               data['device']['cores'], d['maxThreads'])
                self.assertEqual((plan.grid, plan.threadgroup, plan.threadgroup_memory),
                                 (tuple(d['grid']), tuple(d['tg']), d['tgmem'].get('0', 0)), c)
                self.assertEqual(d['kernel'], 'ndArrayIdentityTranspose' if plan.transpose
                                 else 'ndArrayIdentity', c)
                count += 1
        self.assertEqual(count, 432)


@unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is required')
class TestIdentityGPU(unittest.TestCase):
    def test_strides_offsets_and_padding(self):
        torch.manual_seed(551)
        shapes = ((1, 1), (1, 33), (33, 1), (2, 3), (17, 33), (33, 17),
                  (127, 129), (128, 128), (129, 511), (511, 512), (513, 8192),
                  (8192, 513), (8192, 33), (8192, 2048), (2048, 8192))
        for dtype, (m, n), ta, tb in itertools.product(
                (torch.float32, torch.float16, torch.bfloat16), shapes, (False, True), (False, True)):
            with self.subTest(dtype=dtype, shape=(m, n), ta=ta, tb=tb):
                def padded(transposed):
                    shape = (n + 2, m + 5) if transposed else (m + 2, n + 5)
                    storage = torch.full(shape, -77., dtype=dtype, device='mps')
                    view = (storage[1:n+1, 2:m+2].t() if transposed
                            else storage[1:m+1, 2:n+2])
                    return storage, view

                _, src = padded(ta)
                storage, dst = padded(tb)
                expected = torch.randn(m, n).to(dtype)
                src.copy_(expected)
                run, _ = mps_identity.launch_plan(dtype, m, n, *src.stride(), *dst.stride(),
                                                   device_info().cores)
                run(src, dst)
                self.assertTrue(torch.equal(dst.cpu(), expected))
                mask = torch.zeros(storage.shape, dtype=torch.bool)
                view = mask[1:n+1, 2:m+2].t() if tb else mask[1:m+1, 2:n+2]
                view.fill_(True)
                self.assertTrue(bool(torch.all(storage.cpu()[~mask] == -77)))


if __name__ == '__main__':
    unittest.main()
