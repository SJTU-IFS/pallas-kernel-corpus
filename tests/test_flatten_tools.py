"""Unit tests for the shared flatten helpers.  No TPU required.

These exist because a flatten bug is uniquely expensive: the generated file
parses, imports, passes every static check in the flatten script, and then fails
at kernel run time with an error that points at a line which looks correct.
``strip_module_prefixes`` is the helper that got that wrong for ``batched_rpa``.

Run with::

    uv run --frozen --with pytest python -m pytest tests/test_flatten_tools.py -q
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from flatten_gdn import (  # noqa: E402
    strip_module_prefixes,
    unbound_module_refs,
)
from flatten_fused_mlp import _region  # noqa: E402


def test_strips_a_plain_module_qualifier():
    text = "def f(x):\n    return configs.SIZE + utils.pad(x)\n"
    out, shadowed = strip_module_prefixes(text, {"configs", "utils"})
    assert out == "def f(x):\n    return SIZE + pad(x)\n"
    assert shadowed == []


def test_keeps_a_qualifier_whose_name_is_a_parameter():
    """The batched_rpa bug: a module named `schedule` and a parameter of the
    same name in that very module."""
    text = (
        "def compute(schedule, step):\n"
        "    schedule.s_idx[step] = 1\n"
        "    return configs.SIZE\n"
    )
    out, shadowed = strip_module_prefixes(text, {"schedule", "configs"})
    assert "schedule.s_idx[step] = 1" in out
    assert "return SIZE" in out
    assert [name for _, _, name in shadowed] == ["schedule"]


def test_keeps_a_qualifier_whose_name_is_assigned_locally():
    text = (
        "def f():\n"
        "    utils = make_utils()\n"
        "    return utils.pad(1)\n"
    )
    out, _ = strip_module_prefixes(text, {"utils"})
    assert "return utils.pad(1)" in out


def test_strips_inside_a_function_that_does_not_shadow():
    text = (
        "def outer(schedule):\n"
        "    return schedule.a\n"
        "\n"
        "def other(x):\n"
        "    return schedule.b\n"
    )
    out, shadowed = strip_module_prefixes(text, {"schedule"})
    assert "return schedule.a" in out, "shadowed in outer"
    assert "return b" in out, "not shadowed in other"
    assert len(shadowed) == 1


def test_leaves_comments_and_docstrings_alone():
    """The blind regex also ate prose: `for the RPA kernel.` lost its noun."""
    text = (
        'def f():\n'
        '    """Tuning parameters for the RPA kernel."""\n'
        '    # data into the kernel. Instead, store it.\n'
        '    return kernel.run()\n'
    )
    out, _ = strip_module_prefixes(text, {"kernel"})
    assert '"""Tuning parameters for the RPA kernel."""' in out
    assert "# data into the kernel. Instead, store it." in out
    assert "return run()" in out


def test_nested_function_parameter_does_not_shadow_the_outer_scope():
    text = (
        "def outer():\n"
        "    def inner(configs):\n"
        "        return configs.A\n"
        "    return configs.B\n"
    )
    out, _ = strip_module_prefixes(text, {"configs"})
    assert "return configs.A" in out, "shadowed inside inner"
    assert "return B" in out, "not shadowed in outer"


def test_unbound_module_refs_reports_only_real_leftovers():
    code = (
        "def f(schedule):\n"
        "    return schedule.a\n"
        "\n"
        "def g():\n"
        "    return configs.b\n"
    )
    left = unbound_module_refs(code, {"schedule", "configs"})
    assert [name for _, _, name in left] == ["configs"]


def test_a_qualifier_that_is_not_a_known_module_is_untouched():
    text = "def f(obj):\n    return obj.attr + configs.SIZE\n"
    out, _ = strip_module_prefixes(text, {"configs"})
    assert "obj.attr" in out
    assert "SIZE" in out


@pytest.mark.parametrize("prefix", ["configs", "utils", "schedule", "kernel"])
def test_round_trip_is_syntactically_valid(prefix):
    import ast

    text = (
        f"import {prefix}\n"
        f"def f({prefix}=None):\n"
        f"    if {prefix} is None:\n"
        f"        return {prefix}.default\n"
        f"    return 0\n"
        f"def g():\n"
        f"    return {prefix}.value\n"
    )
    out, _ = strip_module_prefixes(text, {prefix})
    ast.parse(out)


# `flatten_fused_mlp` quotes two regions of sglang-jax's `glm5_moe.py` into the
# generated reference, as evidence for a calling convention written down
# nowhere else.  Several classes in that file define `post_load_weights` and
# `__call__`, so selecting by name alone silently picks whichever comes first
# in the walk -- and the result still parses, still reads like upstream code,
# and is simply the wrong function.  That happened.

TWO_NAMESAKES = '''
class Quantized:
    def post_load_weights(self):
        wq_f32 = self.weight_q.value.astype(jnp.float32)
        self.weight.value = wq_f32 * self.scale

class Fused:
    def post_load_weights(self):
        w_gu = jnp.concatenate([wg_reshaped, wu_reshaped], axis=-1)
        self.w_gu.value = w_gu
'''


def test_region_selects_by_content_not_by_name():
    picked = _region(TWO_NAMESAKES, "post_load_weights", "w_gu = jnp.concatenate")
    assert "w_gu" in picked
    assert "weight_q" not in picked, "picked the wrong namesake"


def test_region_refuses_an_ambiguous_match():
    """Two matches is a mis-selection waiting to happen, so it must not pick."""
    with pytest.raises(ValueError, match="expected 1 match, found 2"):
        _region(TWO_NAMESAKES + TWO_NAMESAKES, "post_load_weights",
                "w_gu = jnp.concatenate")


def test_region_refuses_a_missing_match():
    with pytest.raises(ValueError, match="expected 1 match, found 0"):
        _region(TWO_NAMESAKES, "post_load_weights", "not in this file")
