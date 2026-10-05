"""Recovered selection checked against hardware traces, without a GPU.

    python -m unittest discover -s tests -p test_mps_selection.py
"""
import copy
from pathlib import Path
import unittest

from tools.mps_check_sweep import check_sweep, load_sweep

FIXTURES = Path(__file__).parent / 'fixtures' / 'mps'


class TestMPSSelection(unittest.TestCase):
    def test_hardware_traces(self):
        # Total cases, GEMMs (including split-K), and vector dispatches.
        cases = {
            'm2_macos26': (26, 2123, 1823, 173),
            'm3ultra_macos26': (26, 2123, 1819, 189),
            'm2_edges_macos26': (26, 303, 303, 0),
            'm3ultra_edges_macos26': (26, 303, 303, 0),
            'm5pro_f32_macos27': (27, 1658, 1396, 262),
            'm2_vectors_macos26': (26, 468, 0, 396),
            'm3ultra_vectors_macos26': (26, 468, 0, 396),
            'm5pro_vectors_macos27': (27, 468, 0, 462),
            'm2_batch_macos26': (26, 252, 144, 108),
            'm3ultra_batch_macos26': (26, 252, 144, 108),
            'm5pro_batch_f32_macos27': (27, 84, 48, 36),
            'm2_lora_macos26': (26, 54, 6, 48),
            'm3ultra_lora_macos26': (26, 54, 6, 48),
            'm5pro_lora_macos27': (27, 54, 0, 54),
            'm2_vector_execution_macos26': (26, 216, 24, 144),
            'm3ultra_vector_execution_macos26': (26, 216, 24, 144),
            'm5pro_vector_execution_macos27': (27, 216, 8, 126),
            'm4_macos26': (26, 2123, 1835, 173),
            'm4_edges_macos26': (26, 303, 303, 0),
            'm4_vectors_macos26': (26, 468, 0, 396),
            'm4_batch_macos26': (26, 252, 144, 108),
            'm4_lora_macos26': (26, 54, 6, 48),
            'm4_vector_execution_macos26': (26, 216, 24, 144),
        }
        for name, (major, total, gemms, vectors) in cases.items():
            with self.subTest(name=name):
                data = load_sweep(FIXTURES / (name + '.json.gz'))
                self.assertEqual(len(data['rows']), total)
                report = check_sweep(data, major)
                self.assertEqual(report['mismatches'], [])
                self.assertEqual(report['counts'].get('gemm_matches', 0), gemms)
                self.assertEqual(report['counts'].get('vector_matches', 0), vectors)
                self.assertEqual(report['counts']['routing_matches'], gemms + vectors)

    def test_trace_checker_detects_bad_dispatch(self):
        data = load_sweep(FIXTURES / 'm3ultra_edges_macos26.json.gz')
        data['rows'] = [next(r for r in data['rows'] if len(r['dispatches']) == 2)]
        for field in ('grid', 'tgmem', 'bytes', 'reduction'):
            with self.subTest(field=field):
                bad = copy.deepcopy(data)
                d = bad['rows'][0]['dispatches'][0]
                if field == 'grid':
                    d['grid'][0] += 1
                elif field == 'tgmem':
                    d['tgmem']['1'] += 16
                elif field == 'bytes':
                    d['bytes'].pop('26')
                else:
                    bad['rows'][0]['dispatches'].pop()
                self.assertTrue(check_sweep(bad, 26)['mismatches'])

    def test_os_revision_is_not_interchangeable(self):
        # On these same shapes the macOS 26 encoder uses floor division and
        # the macOS 27 encoder uses ceil division / caps K at 8192 per split.
        data = load_sweep(FIXTURES / 'm3ultra_edges_macos26.json.gz')
        self.assertTrue(check_sweep(data, 27)['mismatches'])

    def test_linked_epilogues_keep_the_recovered_gemm_schedule(self):
        for name, major, count in (("M2_26.3", 26, 84), ("M3_Ultra_26.5.2", 26, 84),
                                    ("M4_26.6", 26, 84), ("M5_Pro_27.2", 27, 28)):
            with self.subTest(name=name):
                capture = load_sweep(FIXTURES.parent / "mps_epilogues" / (name + ".json.gz"))
                self.assertEqual(len(capture["cases"]), 84)
                rows = []
                for entry in capture["cases"]:
                    case = entry["case"]
                    # Other entries use A18; gather/materialization and the
                    # addbmm batch reduction are outside this GEMM schedule.
                    ds = [{**d, "bytes": {k: v["hex"] for k, v in d["bytes"].items()}}
                          for d in entry["dispatches"] if d["kernel"].startswith(
                              ("NDArrayMatrixMultiplyNNA14", "mmul_kernel_a16_",
                               "NDArrayMatrixMultiplyReduction"))]
                    if not ds:
                        continue
                    m, n, k = case["shape"]
                    row = {"dtype": case["dtype"], "layout": case["layout"],
                           "M": m, "N": n, "K": k, "dispatches": ds}
                    if case["op"] in ("bmm", "baddbmm", "addbmm"):
                        row["batch"] = 3
                    rows.append(row)
                report = check_sweep({"device": capture["device"], "rows": rows}, major)
                self.assertEqual(report["mismatches"], [])
                self.assertEqual(report["counts"]["gemm_matches"], count)
                self.assertEqual(report["counts"]["routing_matches"], count)

    def test_linked_epilogues_keep_the_recovered_vector_schedule(self):
        for name, major in (("M2_vectors_26.3", 26), ("M3_Ultra_vectors_26.5.2", 26),
                           ("M4_vectors_26.6", 26), ("M5_Pro_vectors_27.2", 27)):
            with self.subTest(name=name):
                capture = load_sweep(FIXTURES.parent / "mps_epilogues" / (name + ".json.gz"))
                self.assertEqual(capture["reference"], "mpsgraph")
                self.assertEqual(len(capture["cases"]), 144)
                rows = []
                for entry in capture["cases"]:
                    case = entry["case"]
                    # CPU boundary copies belong to the reference harness,
                    # not the MPSNDArray vector schedule being checked.
                    ds = [{**d, "bytes": {k: v["hex"] for k, v in d["bytes"].items()}}
                          for d in entry["dispatches"] if d["kernel"].startswith(
                              ("NDArrayVectorMatrixMultiply_", "LORADOWN_"))]
                    self.assertEqual(len(ds), 1)
                    m, n, k = case["shape"]
                    row = {"dtype": case["dtype"], "layout": case["layout"],
                           "M": m, "N": n, "K": k, "dispatches": ds}
                    if case["op"] == "baddbmm":
                        row["batch"] = 3
                    rows.append(row)
                report = check_sweep({"device": capture["device"], "rows": rows}, major)
                self.assertEqual(report["mismatches"], [])
                self.assertEqual(report["counts"]["vector_matches"], 144)
                self.assertEqual(report["counts"]["routing_matches"], 144)


if __name__ == '__main__':
    unittest.main()
