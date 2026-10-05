"""Compare the source submission model with captured native state transitions."""
import gzip
import json
from pathlib import Path
import unittest

from tools.mps_submission_model import begin_region, encode_operation, encode_region, thresholds


class TestMPSSubmission(unittest.TestCase):
    def test_native_descriptors_and_commit_boundaries(self):
        root = Path(__file__).parent / "fixtures/mps_submission"
        paths = sorted(root.glob("*.json.gz"))
        self.assertEqual(len(paths), 3)
        transitions = boundaries = 0
        for path in paths:
            fixture = json.loads(gzip.decompress(path.read_bytes()))
            profiles = thresholds(fixture["generation"], fixture["cores"])
            for case in fixture["cases"]:
                with self.subTest(device=path.stem, kind=case["kind"], index=case["index"]):
                    limits = profiles[case["encode_count"] != 0]
                    local = begin_region(case["before"], limits)
                    updated, committed = encode_operation(local, case["produced"], limits)
                    expected = [list(updated)]
                    if fixture["nested_outer_region"]:
                        # Native 27.2 traces show an outer region writing its
                        # saved counters after the inner matmul region returns.
                        expected.append(list(local))
                    self.assertEqual(case["writes"], expected)
                    if case["commits"] is not None:
                        self.assertEqual(case["commits"], int(committed))
                        boundaries += 1
                    transitions += 1
        self.assertEqual(transitions, 692)
        self.assertEqual(boundaries, 392)

    def test_compiled_regions_and_multi_operation_graphs(self):
        root = Path(__file__).parent / "fixtures/mps_submission_regions"
        paths = sorted(root.glob("*.json.gz"))
        self.assertEqual(len(paths), 3)
        public_calls = chains = 0
        for path in paths:
            fixture = json.loads(gzip.decompress(path.read_bytes()))
            for case in fixture["cases"]:
                with self.subTest(device=path.stem, case=case["case"]):
                    writes = []
                    commits = case["scheduler_commits"]
                    for region in case["regions"]:
                        profiles = thresholds(fixture["generation"], fixture["cores"],
                                              *region["graph_statistics"])
                        updated, submitted = encode_region(region["before"], region["produced"],
                                                           profiles, region["encode_count"])
                        writes.append(list(updated))
                        commits += submitted
                    self.assertEqual(writes, case["writes"])
                    self.assertEqual(commits, case["commits"])
                    if case["kind"] == "public":
                        public_calls += 1
                    else:
                        chains += 1
        self.assertEqual(public_calls, 153)
        self.assertEqual(chains, 16)

    def test_native_capture_and_profiler_modes(self):
        root = Path(__file__).parent / "fixtures/mps_submission_modes"
        paths = sorted(root.glob("*.json.gz"))
        self.assertEqual(len(paths), 3)
        cases = disabled = writes = 0
        for path in paths:
            fixture = json.loads(gzip.decompress(path.read_bytes()))
            profiles = thresholds(fixture["generation"], fixture["cores"])
            for case in fixture["cases"]:
                with self.subTest(device=path.stem, mode=case["mode"],
                                  shape=case["shape"], before=case["before"]):
                    updated, submitted = encode_region(
                        case["before"], case["produced"], profiles, case["encode_count"],
                        enable_commit_and_continue=case["enable_commit_and_continue"])
                    expected = [list(updated)]
                    if fixture["nested_outer_region"]:
                        expected.append(list(begin_region(
                            case["before"], profiles[case["encode_count"] != 0])))
                    self.assertEqual(expected, case["writes"])
                    self.assertEqual(submitted, case["threshold_commits"])
                    # Torch commits the whole graph when the stream was created
                    # with commit-and-continue disabled. This is outside the
                    # recovered region's operation threshold logic.
                    self.assertEqual(case["caller_commits"],
                                     int(not case["enable_commit_and_continue"]))
                    cases += 1
                    disabled += not case["enable_commit_and_continue"]
                    writes += len(expected)
        self.assertEqual((cases, disabled, writes), (135, 81, 180))


if __name__ == "__main__":
    unittest.main()
