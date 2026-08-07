"""Interval algebra behind the overlap harness's trace analysis.

`tools/overlap_harness.py` decides whether a SparseCore gather ran *while* the
TensorCore was computing by intersecting two sets of busy intervals read out of
a Perfetto trace. That arithmetic is repository-invariant, so it is tested here
rather than on the TPU.

The behaviour these lock down is not hypothetical -- both were live bugs in the
first version of the harness:

- summing event durations across every track triple-counts the same work, which
  made a lone matmul look like it had 3x concurrency;
- the TensorCore track carries the SparseCore call's own `call-done` stub, which
  spans the whole SparseCore execution because it *is* the TensorCore waiting.
  Counting it as TensorCore work made a standalone gather look 100% overlapped.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))


@pytest.fixture(scope="module")
def harness():
    """Import without pulling in jax, which the interval math does not need."""
    import importlib.util

    path = Path(__file__).parents[1] / "tools" / "overlap_harness.py"
    spec = importlib.util.spec_from_file_location("overlap_harness", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as error:  # pragma: no cover - only without jax installed
        pytest.skip(f"overlap_harness needs jax to import: {error}")
    return module


def events(*spans):
    return [{"ts": start, "dur": end - start, "name": "op"} for start, end in spans]


def test_union_merges_touching_and_overlapping_spans(harness):
    merged = harness.union_intervals(events((0, 10), (5, 12), (12, 15), (30, 40)))
    assert merged == [(0.0, 15.0), (30.0, 40.0)]
    assert harness.total_length(merged) == 25.0


def test_union_counts_shared_busy_time_once(harness):
    """Three events over the same 10us must be 10us of busy time, not 30us."""
    merged = harness.union_intervals(events((0, 10), (0, 10), (0, 10)))
    assert harness.total_length(merged) == 10.0


def test_union_of_nothing_is_nothing(harness):
    assert harness.union_intervals([]) == []
    assert harness.total_length([]) == 0.0


def test_intersect_finds_only_genuinely_concurrent_time(harness):
    left = [(0.0, 10.0), (20.0, 30.0)]
    right = [(5.0, 25.0)]
    assert harness.intersect(left, right) == [(5.0, 10.0), (20.0, 25.0)]
    assert harness.total_length(harness.intersect(left, right)) == 10.0


def test_intersect_of_adjacent_spans_is_empty(harness):
    """Back-to-back is serial, not concurrent -- a zero-width touch is not overlap."""
    assert harness.intersect([(0.0, 10.0)], [(10.0, 20.0)]) == []


def test_intersect_is_symmetric(harness):
    left = [(0.0, 10.0), (20.0, 30.0)]
    right = [(5.0, 25.0)]
    assert harness.intersect(left, right) == harness.intersect(right, left)


def test_overlap_fraction_is_one_when_fully_nested(harness):
    tensorcore = harness.union_intervals(events((0, 200)))
    sparsecore = harness.union_intervals(events((50, 85)))
    both = harness.intersect(tensorcore, sparsecore)
    fraction = harness.total_length(both) / harness.total_length(sparsecore)
    assert fraction == 1.0


def test_overlap_fraction_is_zero_when_serialized(harness):
    """The standalone-gather case: SparseCore busy, TensorCore idle."""
    tensorcore: list[tuple[float, float]] = []
    sparsecore = harness.union_intervals(events((50, 85)))
    both = harness.intersect(tensorcore, sparsecore)
    assert harness.total_length(both) == 0.0
