"""Reference model of MPSGraph's recovered command-buffer submission limits.

This is a recovery/validation tool, not part of metalBLAS's runtime dispatcher.
See docs/mps_matmuls.md and the MPSGraph hashes/addresses in
docs/mps_dispatch_sources.json. No timings are inputs to this model.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Limits:
    operations: int
    values: int
    maximum_operations: int = 45
    maximum_values: int = 30_000_000


def thresholds(generation, cores, graph_operations=0, graph_values=0):
    """Return limits for zero/nonzero MPSGraphExecutable._encodeCount.

    Supply the runtime's graph statistics, not the number of Metal dispatches.
    Ordinary eligible operations contribute output elements; Stitched children
    are excluded. The captured 27.2 walk also counts MemrefBacked wrappers using
    the children's byte-weighted outputs, then visits the children separately.
    The 26.x snapshots exclude the wrappers. See docs/mps_matmuls.md for source
    accounting. The pre-encode flag enables collection.
    """
    effective_cores = cores * (3 if generation >= 22 else 1)
    first_ops, first_values, later_values = 8, 200_000, 300_000
    if effective_cores >= 20:
        factor = min(7 * effective_cores, 512)
        first_ops = 11
        first_values = factor * 3125
        later_values = factor * 300_000 >> 6
    later_ops, maximum_ops = 12, 45
    if graph_operations > 200 and graph_values > 3_600_000:
        first_ops, later_ops, maximum_ops = 13, 15, 75
        first_values *= 12
        later_values *= 12
    elif graph_operations > 130 or graph_values > 3_600_000:
        if graph_operations > 90 and graph_values > 1_800_000:
            later_values *= 4
            later_ops = 15
        elif graph_operations > 200:
            later_values *= 3
            later_ops = 15
        elif graph_operations > 50:
            later_ops = 15
    return (Limits(first_ops, first_values, maximum_ops),
            Limits(later_ops, later_values, maximum_ops))


def begin_region(state, limits):
    """Read the command-buffer counters, seeding an empty region at half limits."""
    operations, values = state
    return ((limits.operations >> 1, limits.values >> 1)
            if operations == 0 else (operations, values))


def encode_operation(state, produced_values, limits, *, enable_commit_and_continue=True):
    """Advance one eligible operation in an ordinary GPU EncodeDescriptor.

    Excluded graph/view/control operations supply zero and do not advance either
    counter. An actual submission clears both counters; a caller starting a new
    region then applies begin_region again. Nested regions maintain local copies
    and write them back to the command buffer when they return. The base
    descriptor's commitAndContinueDynamicallyChanges() returns false in all
    captured binaries; disabled commit-and-continue leaves both counters alone.
    Parallel descriptors with dynamic changes are outside this helper's scope.
    """
    if not enable_commit_and_continue or produced_values == 0:
        return state, False
    operations, values = state[0] + 1, state[1] + produced_values
    commit = (values >= limits.maximum_values
              or operations >= limits.maximum_operations
              or (operations >= limits.operations and values >= limits.values))
    return ((0, 0) if commit else (operations, values)), commit


def encode_region(state, produced_values, profiles, encode_count=0, *,
                  enable_commit_and_continue=True):
    """Return one region's exit counters and number of threshold submissions.

    ``encode_count`` is the executable field sampled at region entry. A decrease
    of the operation counter increments its local copy, selecting the nonzero
    profile for subsequent operations. Counters are seeded only at entry.
    The caller supplies compiled region boundaries and any scheduler submissions
    outside these boundaries; this helper does not predict compiler topology.
    Disabling commit-and-continue still seeds an empty region and writes its
    counters at exit. A caller's final command-buffer commit is separate.
    """
    local = begin_region(state, profiles[encode_count != 0])
    commits = 0
    for produced in produced_values:
        updated, committed = encode_operation(
            local, produced, profiles[encode_count != 0],
            enable_commit_and_continue=enable_commit_and_continue)
        encode_count += updated[0] < local[0]
        local = updated
        commits += committed
    return local, commits


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--cores", type=int, required=True)
    parser.add_argument("--graph-operations", type=int, default=0)
    parser.add_argument("--graph-values", type=int, default=0)
    args = parser.parse_args()
    limits = thresholds(args.generation, args.cores, args.graph_operations, args.graph_values)
    print(json.dumps(dict(zip(("first", "subsequent"), map(asdict, limits))), indent=2))
