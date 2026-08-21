"""Correctness for MaxText's ragged attention kernel.

This is the corpus's first test file that is **not** TPU-only, and that follows
upstream rather than being a concession to hardware: MaxText's own
`tests/unit/kernels_test.py` has a `RaggedAttentionCpuTest` running the kernel
through `interpret=True` at a reduced shape, beside `@pytest.mark.tpu_only`
tests at the full one. Both halves are reproduced here.

The CPU half is a real check of the flattening and the kernel's logic — it runs
the same Pallas program under the interpreter — but it is **not** a Mosaic
lowering, so it does not make this kernel *migrated* under this corpus's rule.
That still needs the TPU half, which is why `inventory.json` records it
UNVALIDATED with zero migrated launch points until then.

One thing worth knowing before calling any of these: **the kernel and the
reference do not take the same layouts.** `ragged_gqa` wants `k`/`v` as
`[B, S, KV, D]` while `reference_gqa` wants `[B, KV, S, D]` and a squeezed `q`,
and both `mha` and `gqa` return an *unnormalised* output that the caller must
divide by the returned denominator. Upstream's tests do all of that inline; here
it is done in named helpers so the conversion is visible rather than folklore.

Run the CPU half anywhere::

    pytest tests/test_ragged_mqa_attention.py -q -k cpu
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp


ROOT = Path(__file__).parents[1]
FAMILY = ROOT / "kernels" / "attention" / "ragged_mqa_attention"

HAS_TPU = any("TPU" in d.device_kind for d in jax.devices())

#: MaxText's `RaggedAttentionCpuTest` parameterisation.
CPU = dict(batch=2, kv_heads=2, q_heads=4, seq=32, head_dim=32, block_size=16)
#: MaxText's `RaggedAttentionTest` parameterisation.
TPU = dict(batch=4, kv_heads=8, q_heads=32, seq=512, head_dim=128, block_size=256)

#: Upstream's own bars: max |diff| < 1.5e-1, mean |diff| < 1e-2.
MAX_BAR, AVG_BAR = 1.5e-1, 1e-2


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, FAMILY / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    return (load("rmqa_kernel", "maxtext_optimized.py"),
            load("rmqa_baseline", "baseline.py"))


def lengths_for(batch, seq):
    """Ragged prefix lengths, the point of the kernel: every sequence differs."""
    rng = np.random.default_rng(0)
    return jnp.array(rng.integers(1, seq, batch), dtype=jnp.int32)


def compare(got, want, what):
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert got.shape == want.shape, what
    biggest, average = float(np.max(abs(got - want))), float(np.mean(abs(got - want)))
    # Not vacuous: a zeroed kernel differs from the reference by its own scale.
    assert float(np.max(abs(want))) > MAX_BAR, f"{what}: reference too small to check"
    assert biggest < MAX_BAR, f"{what}: max {biggest} >= {MAX_BAR}"
    assert average < AVG_BAR, f"{what}: mean {average} >= {AVG_BAR}"


def run_mqa(kernel, baseline, cfg, interpret):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    b, s, d = cfg["batch"], cfg["seq"], cfg["head_dim"]
    q = jax.random.normal(k1, (b, 1, d), jnp.float32)
    k = jax.random.normal(k2, (b, s, d), jnp.float32)
    v = jax.random.normal(k3, (b, s, d), jnp.float32)
    lengths = lengths_for(b, s)
    got, _, _ = kernel.ragged_mqa(q, k, v, lengths,
                                  block_size=cfg["block_size"], interpret=interpret)
    return got, baseline.reference_mqa(q, k, v, lengths)[0]


def run_mha(kernel, baseline, cfg, interpret):
    """`mha` returns an unnormalised output; the denominator is the second half."""
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    b, s, d, h = cfg["batch"], cfg["seq"], cfg["head_dim"], cfg["q_heads"]
    q = jax.random.normal(k1, (b, 1, h, d), jnp.float32)
    k = jax.random.normal(k2, (b, s, h, d), jnp.float32)
    v = jax.random.normal(k3, (b, s, h, d), jnp.float32)
    lengths = lengths_for(b, s)
    got, _, denominator = kernel.ragged_mha(
        q, k, v, lengths, block_size=cfg["block_size"], interpret=interpret)
    return got / denominator, baseline.reference_mha(q, k, v, lengths)[0]


def run_gqa(kernel, baseline, cfg, interpret):
    """The layouts differ: the kernel takes [B,S,KV,D], the reference [B,KV,S,D]."""
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    b, s, d = cfg["batch"], cfg["seq"], cfg["head_dim"]
    h, kv = cfg["q_heads"], cfg["kv_heads"]
    q = jax.random.normal(k1, (b, 1, h, d), jnp.float32)
    k = jax.random.normal(k2, (b, s, kv, d), jnp.float32)
    v = jax.random.normal(k3, (b, s, kv, d), jnp.float32)
    lengths = lengths_for(b, s)
    got, _, denominator = kernel.ragged_gqa(
        q, k, v, lengths, block_size=cfg["block_size"], interpret=interpret)
    want = baseline.reference_gqa(jnp.squeeze(q), jnp.swapaxes(k, 1, 2),
                                  jnp.swapaxes(v, 1, 2), lengths)[0]
    return got / denominator, want


RUNNERS = {"mqa": run_mqa, "mha": run_mha, "gqa": run_gqa}


@pytest.mark.parametrize("variant", sorted(RUNNERS))
def test_matches_upstream_reference_on_cpu(modules, variant):
    """Upstream's own CPU path: the same Pallas program, under the interpreter."""
    kernel, baseline = modules
    got, want = RUNNERS[variant](kernel, baseline, CPU, interpret=True)
    compare(got, want, f"cpu {variant}")


@pytest.mark.skipif(not HAS_TPU, reason="Mosaic lowering needs a TPU")
@pytest.mark.parametrize("variant", sorted(RUNNERS))
def test_matches_upstream_reference_on_tpu(modules, variant):
    """The half that actually makes this kernel migrated."""
    kernel, baseline = modules
    got, want = RUNNERS[variant](kernel, baseline, TPU, interpret=False)
    compare(got, want, f"tpu {variant}")


@pytest.mark.skipif(not HAS_TPU, reason="Mosaic lowering needs a TPU")
def test_reaches_pallas(modules):
    """One launch point serving three entry points, so count it once."""
    sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    kernel, _ = modules
    b, s, d = TPU["batch"], TPU["seq"], TPU["head_dim"]
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    args = (jax.random.normal(k1, (b, 1, d), jnp.float32),
            jax.random.normal(k2, (b, s, d), jnp.float32),
            jax.random.normal(k3, (b, s, d), jnp.float32),
            lengths_for(b, s))
    assert count_pallas_launches(kernel.kernel, args) == 1


def test_the_kernel_file_holds_exactly_one_launch():
    """Three public entry points, one `pallas_call` -- worth pinning.

    `ragged_mqa`, `ragged_mha` and `ragged_gqa` differ in how they reshape and
    `vmap` around the same kernel body. If a future upstream split them into
    separate launches, the ledger's count for this family would be wrong.
    """
    source = (FAMILY / "maxtext_optimized.py").read_text()
    assert source.count("pl.pallas_call") == 1
    for entry in ("def ragged_mqa(", "def ragged_mha(", "def ragged_gqa("):
        assert entry in source


def test_the_references_left_the_kernel_file():
    """The baseline holds them; the kernel file must not, or they would drift."""
    kernel_source = (FAMILY / "maxtext_optimized.py").read_text()
    baseline_source = (FAMILY / "baseline.py").read_text()
    for name in ("reference_mqa", "reference_mha", "reference_gqa"):
        assert f"def {name}(" in baseline_source, name
        assert f"def {name}(" not in kernel_source, name
    assert "pallas" not in baseline_source.split('"""', 2)[-1]
