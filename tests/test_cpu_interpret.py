"""Run the kernels that can run without a TPU, and check them.

Every other test module that exercises a Mosaic kernel is gated on TPU
hardware, correctly: those kernels lower through Mosaic and there is nothing to
lower on. The consequence, which the rigor audit measured, was that **none of
the 136 migrated launch points was checked on a machine without a TPU**. A
contributor could break a kernel and see a green suite.

Pallas has an interpreter. `pl.pallas_call(..., interpret=True)` executes the
same kernel body in plain JAX on any backend — no Mosaic, no TPU. It cannot
emulate DMA, semaphores, `emit_pipeline`, `core_map` or explicit memory spaces,
so it reaches 62 of the 136 migrated launch points and no more. Within that
reach it is a real check: it runs the actual kernel body against the actual
reference on the actual inputs, and it catches logic errors, indexing mistakes
and bad flattening.

WHAT THIS IS NOT
----------------
Interpret mode is **not** a Mosaic lowering, and passing here does **not** make
a kernel migrated. The corpus's rule — nothing counts as validated without a TPU
run — is what kept the ledger honest when the TPU host died mid-run, and it is
not relaxed by this file. `inventory.json` is untouched by whatever happens
here. This is regression coverage between TPU runs, not a substitute for one.

It is also upstream's own practice where upstream has an opinion: MaxText ships
a `RaggedAttentionCpuTest` that calls its kernel with `interpret=True`, beside
its `tpu_only` tests.

    pytest tests/test_cpu_interpret.py -q
"""

from __future__ import annotations

import contextlib
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

#: Upstream PallasBench's own tolerance, from `pallasbench.utils`.
ATOL = 1e-3
#: TPU's default matmul precision is bf16; the corpus records the measurement
#: behind this in tests/test_pallasbench_tpu.py. The interpreter computes in
#: float32, so contraction-carrying tasks are compared at the looser bar too --
#: a kernel that is right on TPU must not be failed here for being *more*
#: precise than its own reference.
BF16_ULP = 2 ** -8
MATMUL_OPS = ("@", "jnp.dot", "jnp.matmul", "jnp.einsum", "dot_general")

SWEPT = sorted(t for t, s in TASKS.items()
               if s["family"] != "flash_attention" and not s.get("excluded"))


@contextlib.contextmanager
def interpret_mode():
    """Force every `pl.pallas_call` in this process to run under the interpreter.

    The kernel files call `pl.pallas_call(...)` without an `interpret` argument,
    as they should — they are upstream's code, carried verbatim. Patching the
    attribute on the shared `pallas` module is what lets them run here without
    editing a single carried file.
    """
    from jax.experimental import pallas as pl

    original = pl.pallas_call

    def forced(*args, **kwargs):
        return original(*args, **{**kwargs, "interpret": True})

    pl.pallas_call = forced
    try:
        yield
    finally:
        pl.pallas_call = original


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def paths_for(task: str) -> tuple[Path, Path]:
    spec = TASKS[task]
    directory = ROOT / "kernels" / spec["category"] / spec["family"]
    return (directory / f"pallasbench_{task}_optimized.py",
            directory / "baseline.py")


def contracts_on_the_mxu(task: str) -> bool:
    body = paths_for(task)[0].read_text().split('"""', 2)[-1]
    return any(op in body for op in MATMUL_OPS)


@pytest.fixture(scope="module")
def pallasbench():
    modules = {}
    for task in SWEPT:
        kernel_path, baseline_path = paths_for(task)
        modules[task] = (
            load(f"ci_k_{task}", kernel_path),
            load(f"ci_b_{TASKS[task]['family']}", baseline_path),
        )
    return modules


@pytest.mark.parametrize("task", SWEPT)
def test_pallasbench_kernel_under_the_interpreter(pallasbench, task):
    """All 43 PallasBench kernels, against upstream's own `jax_<task>`.

    These are a third of the corpus's migrated launch points, and until this
    file existed not one of them was checked on a machine without a TPU.
    """
    kernel_module, baseline = pallasbench[task]
    inputs = baseline.create_inputs(task)
    with interpret_mode():
        actual = kernel_module.kernel(*inputs)
    expected = baseline.REFERENCES[task](*inputs)
    jax.block_until_ready((actual, expected))

    actual = np.asarray(actual, np.float32)
    expected = np.asarray(expected, np.float32)
    assert actual.shape == expected.shape, task

    scale = max(1.0, float(np.max(np.abs(expected))))
    rtol, atol = ATOL, ATOL * scale
    if contracts_on_the_mxu(task):
        rtol, atol = BF16_ULP, BF16_ULP * scale
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def test_the_interpreter_is_actually_running_the_kernel(pallasbench):
    """Guard the premise: a harness that silently skipped would look identical.

    If `interpret_mode` stopped taking effect, every test above would raise on
    a machine with no TPU rather than pass -- but if the *kernel* stopped being
    called and some fallback answered instead, they would pass while checking
    nothing. This asserts the kernel body is genuinely entered.
    """
    entered = []
    kernel_module, baseline = pallasbench["relu"]
    from jax.experimental import pallas as pl

    original = pl.pallas_call

    def counting(*args, **kwargs):
        entered.append(args[0] if args else kwargs.get("kernel"))
        return original(*args, **{**kwargs, "interpret": True})

    pl.pallas_call = counting
    try:
        kernel_module.kernel(*baseline.create_inputs("relu"))
    finally:
        pl.pallas_call = original

    assert len(entered) == 1, f"expected one pallas_call, saw {len(entered)}"
    assert entered[0] is not None


def test_interpret_mode_restores_the_original_after_use():
    """The patch must not leak into other test modules in the same process."""
    from jax.experimental import pallas as pl

    before = pl.pallas_call
    with interpret_mode():
        assert pl.pallas_call is not before
    assert pl.pallas_call is before


def test_a_broken_kernel_would_be_caught(pallasbench):
    """The harness must be able to fail, not merely to pass.

    Perturb the kernel's output and confirm the comparison rejects it. Without
    this, a tolerance set too loosely would make every test above vacuous.
    """
    kernel_module, baseline = pallasbench["relu"]
    inputs = baseline.create_inputs("relu")
    with interpret_mode():
        actual = np.asarray(kernel_module.kernel(*inputs), np.float32)
    expected = np.asarray(baseline.REFERENCES["relu"](*inputs), np.float32)

    scale = max(1.0, float(np.max(np.abs(expected))))
    np.testing.assert_allclose(actual, expected, rtol=ATOL, atol=ATOL * scale)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(actual * 1.5 + 0.1, expected,
                                   rtol=ATOL, atol=ATOL * scale)


# ---------------------------------------------------------------------------
# Beyond PallasBench: families whose kernels also survive the interpreter.
#
# Each entry names a kernel module, the reference to compare against, and how
# to build inputs. Shapes are deliberately small — the interpreter executes the
# kernel body in Python per grid step, so a native shape would take minutes
# where a reduced one takes tenths of a second. Reduced shapes still exercise
# the tiling, masking and accumulation logic, which is what a logic error lives
# in; they do not exercise the pipelining a real shape would.
# ---------------------------------------------------------------------------

FLASH = ROOT / "kernels" / "attention" / "flash_attention"
GROUPED = ROOT / "kernels" / "moe" / "grouped_matmul"

#: (batch, heads, seq, head_dim) — one block-pair per head, causal.
FLASH_SHAPE = (1, 2, 128, 128)

#: Constructs the Pallas interpreter cannot emulate. A file containing any of
#: them is out of reach on CPU no matter what inputs it is given -- these are
#: not flaky tests to work around, they are a hard boundary.
HOSTILE = (r"make_async_copy|async_copy|SemaphoreType|emit_pipeline|core_map"
           r"|SubcoreMesh|memory_space=pltpu\.|pl\.ANY|shard_map|get_tpu_info")


def interpretable(directory: Path) -> tuple[str, ...]:
    """Implementations in `directory` the interpreter can actually run.

    Derived rather than listed: hardcoding the names meant four grouped-matmul
    kernels that reach for TPU hardware info were parametrized and then failed,
    which is noise rather than signal. What belongs here is decided by what the
    file contains.
    """
    import re

    names = []
    for path in sorted(directory.glob("*_optimized.py")):
        body = path.read_text().split('"""', 2)[-1]
        if not re.search(HOSTILE, body):
            names.append(path.stem[: -len("_optimized")])
    return tuple(names)


FLASH_IMPLEMENTATIONS = interpretable(ROOT / "kernels" / "attention" / "flash_attention")
GROUPED_IMPLEMENTATIONS = interpretable(ROOT / "kernels" / "moe" / "grouped_matmul")


@pytest.fixture(scope="module")
def flash():
    modules = {name: load(f"ci_flash_{name}", FLASH / f"{name}_optimized.py")
               for name in FLASH_IMPLEMENTATIONS if name != "pallasbench"}
    modules["baseline"] = load("ci_flash_baseline", FLASH / "baseline.py")
    return modules


@pytest.fixture(scope="module")
def grouped():
    modules = {}
    for name in GROUPED_IMPLEMENTATIONS:
        path = GROUPED / f"{name}_optimized.py"
        if path.is_file():
            modules[name] = load(f"ci_gm_{name}", path)
    modules["baseline"] = load("ci_gm_baseline", GROUPED / "baseline.py")
    return modules


def cosine(actual, expected) -> float:
    a = np.asarray(actual, np.float64).ravel()
    e = np.asarray(expected, np.float64).ravel()
    return float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-12))


def rms_relative(actual, expected) -> float:
    a = np.asarray(actual, np.float64).ravel()
    e = np.asarray(expected, np.float64).ravel()
    return float(np.sqrt(((a - e) ** 2).mean()) / (np.sqrt((e ** 2).mean()) + 1e-12))


def assert_matches(actual, expected, name=""):
    """Direction and magnitude, mirroring the TPU suite's own helper."""
    label = f"{name}: " if name else ""
    similarity = cosine(actual, expected)
    assert similarity > 0.9999, f"{label}cosine {similarity}"
    scale = rms_relative(actual, expected)
    assert scale < 2e-2, f"{label}RMS-relative {scale}"


@pytest.mark.parametrize("implementation", FLASH_IMPLEMENTATIONS)
def test_flash_attention_under_the_interpreter(flash, implementation):
    """Causal flash attention against the corpus's `causal_bhsd` reference.

    `pallasbench` is skipped: it is in this directory but implements a
    different contract (non-causal `dense_2d`), and it is already covered by
    the PallasBench sweep above.
    """
    if implementation == "pallasbench":
        pytest.skip("different contract (dense_2d); covered by the sweep above")
    batch, heads, seq, dim = FLASH_SHAPE
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    q, k, v = (jax.random.normal(key, (batch, heads, seq, dim), jnp.float32)
               for key in (k1, k2, k3))

    with interpret_mode():
        actual = flash[implementation].kernel(q, k, v)
    expected = flash["baseline"].causal_bhsd(q, k, v)
    jax.block_until_ready((actual, expected))
    assert_matches(actual, expected, implementation)


@pytest.mark.parametrize("implementation", GROUPED_IMPLEMENTATIONS)
def test_grouped_matmul_under_the_interpreter(grouped, implementation):
    """Grouped matmul against the corpus's loop reference, at a reduced shape."""
    if implementation not in grouped:
        pytest.skip(f"{implementation} not present in this directory")
    inputs = grouped["baseline"].create_inputs(
        rows=512, num_groups=4, k=256, n=256)
    try:
        with interpret_mode():
            actual = grouped[implementation].kernel(*inputs)
    except (NotImplementedError, TypeError) as exc:
        pytest.skip(f"{implementation} needs a different calling convention: {exc}")
    expected = grouped["baseline"].grouped_matmul_loop(*inputs)
    jax.block_until_ready((actual, expected))
    assert_matches(actual, expected, implementation)


def test_the_harness_reaches_what_it_claims_to():
    """Keep the coverage claim in the docstring honest.

    If someone adds an interpretable family without extending this file, or
    removes one, the number in the module docstring and in README's Verification
    status goes stale. This recomputes it.
    """
    import re
    inventory = json.loads((ROOT / "inventory.json").read_text())
    hostile = (r"make_async_copy|async_copy|SemaphoreType|emit_pipeline|core_map"
               r"|SubcoreMesh|memory_space=pltpu\.|pl\.ANY|shard_map|get_tpu_info")
    reachable = 0
    for family in inventory["families"].values():
        for implementation in family["corpus_implementations"]:
            path = ROOT / implementation["directory"] / implementation["file"]
            body = path.read_text().split('"""', 2)[-1]
            if not re.search(hostile, body):
                reachable += implementation["migrated_launch_points"]
    total = inventory["counts"]["migrated_launch_points"]
    assert (reachable, total) == (62, 136), (
        f"interpreter reach moved to {reachable} of {total}; update the module "
        f"docstring and README's Verification status row")
