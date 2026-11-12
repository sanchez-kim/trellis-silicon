"""CPU tests for the chunked naive attention (issue #6) and its patch upgrade path."""

import importlib
import io

import pytest
import torch
import torch.nn.functional as F

from trellis_silicon import attention_ops
from trellis_silicon.attention_ops import naive_attention
from trellis_silicon.patches import attention as attn_patch


def _ref(q, k, v):
    scale = q.shape[-1] ** -0.5
    return torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1) @ v


def _qkv(B=2, H=3, S_q=17, S_kv=17, D=8, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda s: torch.randn(B, H, s, D, generator=g).to(dtype)  # noqa: E731
    return mk(S_q), mk(S_kv), mk(S_kv)


def test_default_threshold_constant():
    assert attention_ops.MPS_BMM_MAX_OUT == 2**32
    # Chunks must stay under the tiled-bmm limit; the default is the chunk budget.
    assert attention_ops.CHUNK_ELEMS <= attention_ops.MPS_BMM_MAX_OUT
    assert naive_attention.__defaults__ == (attention_ops.CHUNK_ELEMS,)


@pytest.mark.parametrize(
    "shape", [(17, 17), (17, 9), (5, 23)], ids=["self", "cross_q>kv", "cross_q<kv"]
)
# max_out = B*H*S_kv*rows: rows=1 edge, 3 and 4 leave a short last chunk
@pytest.mark.parametrize("rows", [1, 3, 4])
def test_chunked_matches_single_shot(shape, rows):
    S_q, S_kv = shape
    q, k, v = _qkv(S_q=S_q, S_kv=S_kv)
    max_out = 2 * 3 * S_kv * rows
    assert 2 * 3 * S_q * S_kv > max_out  # force the chunked path
    out = naive_attention(q, k, v, max_out_elems=max_out)
    assert out.shape == (2, 3, S_q, 8)
    assert torch.allclose(out, _ref(q, k, v), atol=1e-6, rtol=1e-5)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5, rtol=1e-4)


def test_chunked_matches_reference_noncontiguous_call_site_layout():
    g = torch.Generator().manual_seed(1)
    B, S_q, S_kv, H, D = 2, 11, 7, 3, 8
    q = torch.randn(B, S_q, H, D, generator=g).permute(0, 2, 1, 3)
    k = torch.randn(B, S_kv, H, D, generator=g).permute(0, 2, 1, 3)
    v = torch.randn(B, S_kv, H, D, generator=g).permute(0, 2, 1, 3)
    assert not q.is_contiguous()
    out = naive_attention(q, k, v, max_out_elems=B * H * S_kv * 3)
    assert out.shape == (B, H, S_q, D)
    assert torch.allclose(out, _ref(q, k, v), atol=1e-6, rtol=1e-5)


def test_rows_floor_at_one_when_max_out_tiny():
    q, k, v = _qkv(S_q=6, S_kv=7)
    out = naive_attention(q, k, v, max_out_elems=1)
    assert torch.allclose(out, _ref(q, k, v), atol=1e-6, rtol=1e-5)


def test_below_threshold_is_bit_identical():
    q, k, v = _qkv(S_q=19, S_kv=11)
    assert torch.equal(naive_attention(q, k, v), _ref(q, k, v))
    exact = 2 * 3 * 19 * 11
    assert torch.equal(naive_attention(q, k, v, max_out_elems=exact), _ref(q, k, v))


def test_chunked_bf16():
    q, k, v = _qkv(S_q=13, S_kv=10, dtype=torch.bfloat16)
    try:
        ref = _ref(q, k, v)
    except RuntimeError:
        pytest.skip("bf16 matmul unsupported on this CPU")
    out = naive_attention(q, k, v, max_out_elems=2 * 3 * 10 * 4)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), ref.float(), atol=5e-2, rtol=5e-2)


# --- patch upgrade path ---------------------------------------------------

_FRESH = "def f():\n    if x:\n        pass\n" + attn_patch._RAISE_ANCHOR + "\n"


def _state(name):
    if name == "fresh":
        return _FRESH
    block = {"old": attn_patch._OLD_COMBINED_BLOCK, "prev": attn_patch._PREV_SPLIT_BLOCK}[name]
    return "def f():\n    if x:\n        pass\n" + block + attn_patch._RAISE_ANCHOR + "\n"


@pytest.mark.parametrize("state", ["fresh", "old", "prev"])
def test_patch_sparse_attention_upgrades_and_is_idempotent(tmp_path, monkeypatch, state):
    monkeypatch.setattr(attn_patch, "TRELLIS_ROOT", str(tmp_path))
    path = tmp_path / "trellis2/modules/sparse/attention/full_attn.py"
    path.parent.mkdir(parents=True)
    path.write_text(_state(state))

    attn_patch.patch_sparse_attention()
    once = path.read_text()
    assert attn_patch._SPLIT_BLOCK in once
    assert attn_patch._ATTN_MARKER in once
    assert attn_patch._PREV_SPLIT_BLOCK not in once
    assert attn_patch._OLD_COMBINED_BLOCK not in once
    assert once.count(attn_patch._RAISE_ANCHOR) == 1

    attn_patch.patch_sparse_attention()
    assert path.read_text() == once


# --- startup staleness check ----------------------------------------------


def _write_full_attn(root, name):
    path = root / "trellis2/modules/sparse/attention/full_attn.py"
    path.parent.mkdir(parents=True)
    path.write_text(_state(name))
    return path


def test_patch_is_current_true_after_patching(tmp_path, monkeypatch):
    monkeypatch.setattr(attn_patch, "TRELLIS_ROOT", str(tmp_path))
    _write_full_attn(tmp_path, "fresh")
    attn_patch.patch_sparse_attention()
    assert attn_patch.sparse_attention_patch_is_current() is True
    assert attn_patch.sparse_attention_patch_is_current(str(tmp_path)) is True


@pytest.mark.parametrize("state", ["old", "prev"])
def test_patch_is_current_false_for_outdated_blocks(tmp_path, monkeypatch, state):
    monkeypatch.setattr(attn_patch, "TRELLIS_ROOT", str(tmp_path))
    _write_full_attn(tmp_path, state)
    assert attn_patch.sparse_attention_patch_is_current() is False


def test_patch_is_current_none_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(attn_patch, "TRELLIS_ROOT", str(tmp_path))
    assert attn_patch.sparse_attention_patch_is_current() is None


def test_warn_helper_prints_one_line_only_when_outdated(tmp_path):
    _write_full_attn(tmp_path, "old")
    buf = io.StringIO()
    assert attn_patch.warn_if_sparse_attention_patch_outdated(str(tmp_path), file=buf) is True
    assert buf.getvalue() == (
        "[SPARSE] WARNING: TRELLIS.2 sparse attention patch is missing or out of date "
        "(issue #6). Run: trellis-silicon-patch\n"
    )

    missing = io.StringIO()
    assert attn_patch.warn_if_sparse_attention_patch_outdated(str(tmp_path / "nope"), file=missing) is False
    assert missing.getvalue() == ""


def test_warn_helper_silent_when_current(tmp_path, monkeypatch):
    monkeypatch.setattr(attn_patch, "TRELLIS_ROOT", str(tmp_path))
    _write_full_attn(tmp_path, "fresh")
    attn_patch.patch_sparse_attention()
    buf = io.StringIO()
    assert attn_patch.warn_if_sparse_attention_patch_outdated(str(tmp_path), file=buf) is False
    assert buf.getvalue() == ""


def test_attn_marker_module_is_importable():
    stmt = attn_patch._ATTN_MARKER.split()
    assert stmt[0] == "from" and stmt[2] == "import"
    module = importlib.import_module(stmt[1])
    assert hasattr(module, stmt[3])


# --- fresh-insert guards --------------------------------------------------


def _patch_text(tmp_path, monkeypatch, text):
    monkeypatch.setattr(attn_patch, "TRELLIS_ROOT", str(tmp_path))
    path = tmp_path / "trellis2/modules/sparse/attention/full_attn.py"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    attn_patch.patch_sparse_attention()
    return path


def test_patch_refuses_unrecognised_elif_block(tmp_path, monkeypatch, capsys):
    text = (
        "def f():\n    if x:\n        pass\n"
        "    elif config.ATTN in ('sdpa', 'naive'):\n        hand_edited()\n"
        + attn_patch._RAISE_ANCHOR + "\n"
    )
    path = _patch_text(tmp_path, monkeypatch, text)
    assert path.read_text() == text
    assert "Error" in capsys.readouterr().out
    assert attn_patch.sparse_attention_patch_is_current() is False


@pytest.mark.parametrize("anchors", [0, 2])
def test_patch_refuses_when_anchor_count_is_not_one(tmp_path, monkeypatch, capsys, anchors):
    text = "def f():\n    pass\n" + (attn_patch._RAISE_ANCHOR + "\n") * anchors
    path = _patch_text(tmp_path, monkeypatch, text)
    assert path.read_text() == text
    out = capsys.readouterr().out
    assert "Error" in out and "Patched" not in out
