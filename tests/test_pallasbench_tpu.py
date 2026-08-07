"""TPU correctness for the PallasBench kernel suite.

This is the corpus's one test file that cuts across families rather than
covering a single one.  PallasBench contributes 43 kernels spread over 19
families, and every one of them has the same shape of check: run
``pallas_<task>`` and compare against upstream's own ``jax_<task>``.  Nineteen
near-identical files would hide that uniformity rather than express it.

Both sides of every comparison are upstream's: the reference from
``pallasbench/baselines/jax_baseline.py`` and the inputs from
``pallasbench/utils.generate_inputs``, both carried in each family's
``baseline.py``.

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest tests/test_pallasbench_tpu.py -q
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp


ROOT = Path(__file__).parents[1]
TASKS = json.loads((ROOT / "tools" / "pallasbench_tasks.json").read_text())

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="PallasBench kernels are Mosaic TPU kernels",
)

#: flash_attention was migrated before this sweep, under its own filename and
#: with its own tests in test_flash_attention_backward_tpu.py; embedding_lookup
#: does not lower on TPU at all (see `excluded` in pallasbench_tasks.json).
SWEPT = sorted(t for t, s in TASKS.items()
               if s["family"] != "flash_attention" and not s.get("excluded"))

#: Upstream's own correctness tolerance, from `pallasbench.utils`.
ATOL = 1e-3

#: Tasks whose kernel contracts over a dimension on the MXU, where TPU's default
#: matmul precision is bf16 -- on *both* sides of the comparison, since the
#: reference is JAX running on the same device.  Measured on multi_head_attention
#: against a float64 host computation: kernel 3.48e-3 off, reference 2.91e-3 off,
#: one bf16 ulp at that magnitude 3.05e-3; the same reference at
#: `Precision.HIGHEST` lands at 2.3e-7.  So the residual is the hardware's, and
#: holding these to 1e-3 would be asking bf16 for f32 answers.
BF16_MATMUL_ULP = 2 ** -8
MATMUL_OPS = ("@", "jnp.dot", "jnp.matmul", "jnp.einsum", "dot_general")


def _contracts_on_the_mxu(task: str) -> bool:
    source = _paths(task)[0].read_text()
    body = source.split('"""', 2)[-1]
    return any(op in body for op in MATMUL_OPS)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _paths(task: str) -> tuple[Path, Path]:
    spec = TASKS[task]
    directory = ROOT / "kernels" / spec["category"] / spec["family"]
    return (directory / f"pallasbench_{task}_optimized.py",
            directory / "baseline.py")


@pytest.fixture(scope="module")
def loaded():
    """Every swept kernel and its family baseline, imported once."""
    out = {}
    for task in SWEPT:
        kernel_path, baseline_path = _paths(task)
        out[task] = (
            load(f"pb_k_{task}", kernel_path),
            load(f"pb_b_{TASKS[task]['family']}", baseline_path),
        )
    return out


@pytest.mark.parametrize("task", SWEPT)
def test_matches_upstream_reference(loaded, task):
    """Each kernel against upstream's own jax_<task>."""
    kernel_module, baseline = loaded[task]
    inputs = baseline.create_inputs(task)
    actual = jax.jit(kernel_module.kernel)(*inputs)
    expected = jax.jit(baseline.REFERENCES[task])(*inputs)
    jax.block_until_ready((actual, expected))

    actual = np.asarray(actual, np.float32)
    expected = np.asarray(expected, np.float32)
    assert actual.shape == expected.shape, task
    # Scale the absolute tolerance by the reference's magnitude: the matmul
    # tasks reach the hundreds, where a fixed 1e-3 would be tighter than f32.
    scale = max(1.0, float(np.max(np.abs(expected))))
    rtol, atol = ATOL, ATOL * scale
    if _contracts_on_the_mxu(task):
        rtol, atol = BF16_MATMUL_ULP, BF16_MATMUL_ULP * scale
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("task", SWEPT)
def test_reaches_pallas(loaded, task):
    """The standing rule: a passing comparison does not prove the kernel ran."""
    sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    kernel_module, baseline = loaded[task]
    launches = count_pallas_launches(
        kernel_module.kernel, tuple(baseline.create_inputs(task)))
    assert launches == 1, f"{task}: {launches} Pallas launches"


def test_every_swept_task_is_a_distinct_file():
    """43 kernels, 43 files, one launch point each."""
    assert len(SWEPT) == 43
    seen = set()
    for task in SWEPT:
        kernel_path, baseline_path = _paths(task)
        assert kernel_path.is_file(), kernel_path
        assert baseline_path.is_file(), baseline_path
        assert kernel_path not in seen
        seen.add(kernel_path)
        source = kernel_path.read_text()
        assert source.count("pl.pallas_call") == 1, task
        assert source.count("\nkernel = ") == 1, task


def test_no_kernel_imports_its_upstream_package():
    """The provenance import is the only repo-local one, and it is stripped."""
    for task in SWEPT:
        kernel_path, _ = _paths(task)
        tree = ast.parse(kernel_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("pallasbench"), task
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("pallasbench"), task


def test_references_are_pallas_free():
    """A reference that lowered to Pallas would not be a valid check."""
    families = sorted({TASKS[t]["family"] for t in SWEPT})
    for family in families:
        category = next(TASKS[t]["category"] for t in SWEPT
                        if TASKS[t]["family"] == family)
        source = (ROOT / "kernels" / category / family / "baseline.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "pallas" not in node.module, family
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "pallas" not in alias.name, family


def test_integer_tasks_get_integer_inputs(loaded):
    """Two tasks index rather than compute, and upstream says so.

    `one_hot` and `nucleotide_onehot` carry `input_dtypes` upstream; feeding
    them float32 would be a different task.  (`embedding_lookup` carries them
    too, but is excluded -- it does not lower on TPU.)
    """
    for task in ("one_hot", "nucleotide_onehot"):
        _, baseline = loaded[task]
        inputs = baseline.create_inputs(task)
        dtypes = [jnp.dtype(a.dtype) for a in inputs]
        assert any(np.issubdtype(d, np.integer) for d in dtypes), (
            f"{task}: expected an integer input, got {dtypes}")


def test_positive_domain_tasks_get_positive_inputs(loaded):
    """`log` and `rsqrt` carry an input_range upstream; without it they are nan."""
    for task in ("log", "rsqrt"):
        _, baseline = loaded[task]
        (x,) = baseline.create_inputs(task)
        assert float(np.min(np.asarray(x))) > 0, task


def test_shape_reductions_are_recorded_on_both_sides():
    """A task validated below its native shape must say so, in both files.

    The risk this guards is not that a smaller shape is wrong -- it is that a
    reader takes `native_shape` for what was actually run.  Seventeen tasks do
    not compile at PallasBench's declared shape on this hardware; each states
    the shape it ran at and why, in the kernel's `SOURCE` and in the baseline's
    `TASK_INPUTS`, and the two must agree.
    """
    def literal(source: str, name: str):
        node = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.Assign) and n.targets[0].id == name)
        return ast.literal_eval(node.value)

    reduced = set()
    for task in SWEPT:
        kernel_path, _ = _paths(task)
        recorded = literal(kernel_path.read_text(), "SOURCE")
        native = [list(s) for s in TASKS[task]["input_shapes"]]
        assert [list(s) for s in recorded["native_shape"]] == native, task

        validation = [list(s) for s in recorded["validation_shape"]]
        if validation == native:
            assert recorded["validation_reason"] is None, task
        else:
            assert recorded["validation_reason"], task
            reduced.add(task)
        # The baseline is what the test actually calls; it must not disagree.
        spec = TASKS[task]
        baseline_src = (ROOT / "kernels" / spec["category"] / spec["family"]
                        / "baseline.py").read_text()
        inputs = literal(baseline_src, "TASK_INPUTS")[task]
        assert [list(s) for s in inputs["input_shapes"]] == validation, task
        assert [list(s) for s in inputs["native_shapes"]] == native, task
        assert inputs["reason"] == recorded["validation_reason"], task
    assert len(reduced) == 17, sorted(reduced)


def test_the_excluded_task_still_does_not_lower():
    """`embedding_lookup` is excluded on evidence, so recheck the evidence.

    Excluding a kernel is a claim about the compiler, and compilers change.
    This reproduces the upstream kernel's one line rather than importing it --
    the file is not in the corpus, precisely because it cannot run -- and fails
    if Mosaic ever grows the support, which is the signal to migrate it.
    """
    def gather_kernel(table_ref, idx_ref, o_ref):
        o_ref[...] = table_ref[idx_ref[...], :]

    from jax.experimental import pallas as pl

    table = jnp.zeros((512, 128), jnp.float32)
    idx = jnp.zeros((8,), jnp.int32)
    with pytest.raises(Exception) as excinfo:
        jax.jit(lambda t, i: pl.pallas_call(
            gather_kernel,
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        )(t, i))(table, idx)
    assert "int indexing" in str(excinfo.value), (
        "Mosaic now accepts a vector gather from a VMEM ref: revisit the "
        "embedding_lookup exclusion in tools/pallasbench_tasks.json")
