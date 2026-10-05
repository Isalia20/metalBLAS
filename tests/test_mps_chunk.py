"""Large-output host chunks checked against native encoder captures."""
import gzip
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import patch

import torch

import metalblas
from metalblas import float_dispatch, mps_a14, mps_chunk, mps_heuristics


class TestChunkSchedules(unittest.TestCase):
    def test_boundary_captures(self):
        total = 0
        for path in sorted((Path(__file__).parent / 'fixtures/mps_chunk').glob('*.json.gz')):
            data = json.loads(gzip.decompress(path.read_bytes()))
            dev = mps_heuristics.DeviceInfo(**data['device'])
            version = int(data['macos'].split('.')[0])
            for row in data['cases']:
                c = row['case']
                with self.subTest(device=dev.chip, case=c):
                    total += 1
                    dtype = {'f32':torch.float32,'f16':torch.float16,'bf16':torch.bfloat16}[c['dtype']]
                    m, n, k = c['shape']
                    ta, tb = (v == 't' for v in c['layout'])
                    plan = mps_chunk.schedule(m, n, 1, dev)
                    rows, cols = (plan.rows, plan.columns) if plan else (((0,m),),((0,n),))
                    expected = []
                    sizes = []
                    for _, mc in rows:
                        for _, nc in cols:
                            ca = False if len(rows)>1 else ta
                            cb = False if len(cols)>1 else tb
                            vector = mps_a14.vector_preferred(mc,nc,k,dtype,dev=dev,
                                trans_a=ca,trans_b=cb,macos_major=version)
                            if vector:
                                p=mps_a14.vector_plan(mc,nc,k,dtype,dev=dev,
                                    trans_a=ca,trans_b=cb,macos_major=version)
                                expected.append((p.kernel,p.grid,p.threadgroup,p.tgmem))
                            else:
                                p=mps_a14.gemm_plan(mc,nc,k,dtype,dev=dev,macos_major=version)
                                self.assertEqual(p.splits,1)
                                expected.append((p.kernel,p.grid,(p.threads,1,1),p.tgmem))
                                sizes.append((nc,mc,k))
                    dispatches=[d for d in row['dispatches'] if d['kernel'].startswith(
                        ('mmul_kernel','NDArrayMatrixMultiply','NDArrayVectorMatrixMultiply','LORADOWN'))]
                    observed=[(d['kernel'],tuple(d['grid']),tuple(d['tg']),
                               tuple(d['tgmem'].get(str(i),0) for i in (0,1))) for d in dispatches]
                    self.assertEqual(observed,expected)
                    gemms=[d for d in dispatches if d['kernel'].startswith(('mmul_kernel','NDArrayMatrixMultiply'))]
                    self.assertEqual([struct.unpack_from('<3I',bytes.fromhex(d['bytes']['26']['hex'])) for d in gemms],sizes)
                    if plan:
                        elem=4 if dtype is torch.float32 else 2
                        for d,(nc,mc,_) in zip(gemms,sizes):
                            words=struct.unpack('<25I',bytes.fromhex(d['bytes']['26']['hex']))
                            self.assertEqual(words[6],mps_chunk.row_stride(nc,dtype)*elem)
                            if len(rows)>1:self.assertEqual(words[5],mps_chunk.row_stride(k,dtype)*elem)
                            if len(cols)>1:self.assertEqual(words[4],mps_chunk.row_stride(nc,dtype)*elem)
        self.assertEqual(total,320)

    def test_batch_vector_and_a18_exclusions(self):
        total=outside=0
        directory=Path(__file__).parent/'fixtures/mps_chunk/controls'
        for path in sorted(directory.glob('*.json.gz')):
            data=json.loads(gzip.decompress(path.read_bytes()))
            dev=mps_heuristics.DeviceInfo(**data['device'])
            version=int(data['macos'].split('.')[0])
            for row in data['cases']:
                c=row['case']
                with self.subTest(device=dev.chip,case=c):
                    total+=1
                    dtype={'f32':torch.float32,'f16':torch.float16,'bf16':torch.bfloat16}[c['dtype']]
                    m,n,k=c['shape']
                    ta,tb=(v=='t' for v in c['layout'])
                    a18=dev.uses_a18_kernel and dtype is not torch.float32
                    vector=(mps_heuristics.vector_kernel_preferred(m,n,k,dtype,dev,ta,tb) if a18 else
                            mps_a14.vector_preferred(m,n,k,dtype,dev=dev,trans_a=ta,trans_b=tb,macos_major=version))
                    plan=None if a18 or vector else mps_chunk.schedule(m,n,c.get('batch',1),dev)
                    core=[d for d in row['dispatches'] if d['kernel'].startswith(
                        ('mmul_kernel','NDArrayMatrixMultiplyNNA14','NDArrayVectorMatrixMultiply','LORADOWN','gemm_a18'))]
                    if not core:
                        # PyTorch 2.14 handles these cases in its own Metal
                        # kernels. Separate direct-MPSGraph captures cover them.
                        self.assertEqual(data['reference'],'torch')
                        self.assertTrue(all(d['kernel'].startswith(('gemv_','matmul_')) for d in row['dispatches']))
                        outside+=1
                        continue
                    self.assertEqual(len(core),len(plan.rows)*len(plan.columns) if plan else 1)
        self.assertEqual((total,outside),(153,21))


@unittest.skipUnless(torch.backends.mps.is_available(),'Requires MPS')
class TestChunkExecution(unittest.TestCase):
    def test_public_boundary_and_epilogues(self):
        dev=mps_heuristics.device_info()
        types=(torch.float32,) if dev.uses_a18_kernel else (torch.float32,torch.float16,torch.bfloat16)
        torch.manual_seed(308)
        with patch.object(float_dispatch,'_TILE_POLICY','mps'), \
             patch.object(float_dispatch,'_fastest',side_effect=AssertionError('Runtime timing')):
            for dtype in types:
                for ta,tb in ((False,False),(False,True),(True,False),(True,True)):
                    for m,n,k in ((32769,17,33),(17,32769,33)):
                        def operand(rows,cols,transpose):
                            r,c=(cols,rows) if transpose else (rows,cols)
                            base=torch.randint(-2,3,(r,c+3)).to(device='mps',dtype=dtype)
                            view=base[:,1:c+1]
                            return view.t() if transpose else view
                        a,b=operand(m,k,ta),operand(k,n,tb)
                        product=(a.cpu().double()@b.cpu().double()).float()
                        for unit_batch in (False,True):
                            for fused in (False,True):
                                with self.subTest(dtype=dtype,transpose=(ta,tb),shape=(m,n,k),batch=unit_batch,fused=fused):
                                    prefix=(1,) if unit_batch else ()
                                    storage=torch.full((*prefix,m,n+3),-123,device='mps',dtype=dtype)
                                    out=storage[...,1:n+1]
                                    left,right=(a[None],b[None]) if unit_batch else (a,b)
                                    if fused:
                                        bias=torch.randn((m,1)).to(device='mps',dtype=dtype)
                                        fn=metalblas.baddbmm if unit_batch else metalblas.addmm
                                        result=fn(bias,left,right,alpha=.73,beta=.37,out=out)
                                        ref=product*torch.tensor(.73,dtype=dtype).float()+bias.cpu().float()*torch.tensor(.37,dtype=dtype).float()
                                    else:
                                        result=metalblas.matmul(left,right,out=out)
                                        ref=product
                                    self.assertIs(result,out)
                                    torch.testing.assert_close(out.cpu(),ref.to(dtype).reshape(out.shape),atol=0,rtol=0)
                                    self.assertTrue(bool((storage[...,0]==-123).all()))
                                    self.assertTrue(bool((storage[...,n+1:]==-123).all()))


if __name__=='__main__':
    unittest.main()
