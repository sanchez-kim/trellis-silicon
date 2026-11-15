"""Attention backend patches: add SDPA/naive backends to TRELLIS.2's sparse
attention dispatch so it runs on MPS (no xformers/flash-attn)."""

import os
import sys

from .common import TRELLIS_ROOT, read_file, write_file

# Block states of full_attn.py, oldest to newest. The marker for 'already
# patched' is the naive_attention import, unique to the newest block.
_OLD_COMBINED_BLOCK = """\
    elif config.ATTN in ('sdpa', 'naive'):
        from torch.nn.functional import scaled_dot_product_attention as sdpa_fn
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        H = q.shape[-2]
        max_q = max(q_seqlen)
        max_kv = max(kv_seqlen) if kv_seqlen is not None else max_q
        B = len(q_seqlen)
        C_q = q.shape[-1]
        C_v = v.shape[-1]
        q_padded = torch.zeros(B, max_q, H, C_q, device=device, dtype=q.dtype)
        k_padded = torch.zeros(B, max_kv, H, C_q, device=device, dtype=k.dtype)
        v_padded = torch.zeros(B, max_kv, H, C_v, device=device, dtype=v.dtype)
        q_offset = 0
        kv_offset = 0
        for b in range(B):
            ql = q_seqlen[b]
            kvl = kv_seqlen[b] if kv_seqlen is not None else ql
            q_padded[b, :ql] = q[q_offset:q_offset+ql]
            k_padded[b, :kvl] = k[kv_offset:kv_offset+kvl]
            v_padded[b, :kvl] = v[kv_offset:kv_offset+kvl]
            q_offset += ql
            kv_offset += kvl
        q_padded = q_padded.permute(0, 2, 1, 3)
        k_padded = k_padded.permute(0, 2, 1, 3)
        v_padded = v_padded.permute(0, 2, 1, 3)
        out_padded = sdpa_fn(q_padded, k_padded, v_padded)
        out_padded = out_padded.permute(0, 2, 1, 3)
        out_list = []
        for b in range(B):
            ql = q_seqlen[b]
            out_list.append(out_padded[b, :ql])
        out = torch.cat(out_list, dim=0)
"""

_PREV_SPLIT_BLOCK = """\
    elif config.ATTN in ('sdpa', 'naive'):
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        H = q.shape[-2]
        max_q = max(q_seqlen)
        max_kv = max(kv_seqlen) if kv_seqlen is not None else max_q
        B = len(q_seqlen)
        C_q = q.shape[-1]
        C_v = v.shape[-1]
        q_padded = torch.zeros(B, max_q, H, C_q, device=device, dtype=q.dtype)
        k_padded = torch.zeros(B, max_kv, H, C_q, device=device, dtype=k.dtype)
        v_padded = torch.zeros(B, max_kv, H, C_v, device=device, dtype=v.dtype)
        q_offset = 0
        kv_offset = 0
        for b in range(B):
            ql = q_seqlen[b]
            kvl = kv_seqlen[b] if kv_seqlen is not None else ql
            q_padded[b, :ql] = q[q_offset:q_offset+ql]
            k_padded[b, :kvl] = k[kv_offset:kv_offset+kvl]
            v_padded[b, :kvl] = v[kv_offset:kv_offset+kvl]
            q_offset += ql
            kv_offset += kvl
        q_padded = q_padded.permute(0, 2, 1, 3)
        k_padded = k_padded.permute(0, 2, 1, 3)
        v_padded = v_padded.permute(0, 2, 1, 3)
        if config.ATTN == 'sdpa':
            from torch.nn.functional import scaled_dot_product_attention as sdpa_fn
            out_padded = sdpa_fn(q_padded, k_padded, v_padded)
        else:
            scale = q_padded.shape[-1] ** -0.5
            attn_weight = torch.softmax((q_padded @ k_padded.transpose(-2, -1)) * scale, dim=-1)
            out_padded = attn_weight @ v_padded
        out_padded = out_padded.permute(0, 2, 1, 3)
        out_list = []
        for b in range(B):
            ql = q_seqlen[b]
            out_list.append(out_padded[b, :ql])
        out = torch.cat(out_list, dim=0)
"""

_SPLIT_BLOCK = """\
    elif config.ATTN in ('sdpa', 'naive'):
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        H = q.shape[-2]
        max_q = max(q_seqlen)
        max_kv = max(kv_seqlen) if kv_seqlen is not None else max_q
        B = len(q_seqlen)
        C_q = q.shape[-1]
        C_v = v.shape[-1]
        q_padded = torch.zeros(B, max_q, H, C_q, device=device, dtype=q.dtype)
        k_padded = torch.zeros(B, max_kv, H, C_q, device=device, dtype=k.dtype)
        v_padded = torch.zeros(B, max_kv, H, C_v, device=device, dtype=v.dtype)
        q_offset = 0
        kv_offset = 0
        for b in range(B):
            ql = q_seqlen[b]
            kvl = kv_seqlen[b] if kv_seqlen is not None else ql
            q_padded[b, :ql] = q[q_offset:q_offset+ql]
            k_padded[b, :kvl] = k[kv_offset:kv_offset+kvl]
            v_padded[b, :kvl] = v[kv_offset:kv_offset+kvl]
            q_offset += ql
            kv_offset += kvl
        q_padded = q_padded.permute(0, 2, 1, 3)
        k_padded = k_padded.permute(0, 2, 1, 3)
        v_padded = v_padded.permute(0, 2, 1, 3)
        if config.ATTN == 'sdpa':
            from torch.nn.functional import scaled_dot_product_attention as sdpa_fn
            out_padded = sdpa_fn(q_padded, k_padded, v_padded)
        else:
            from trellis_silicon.attention_ops import naive_attention
            out_padded = naive_attention(q_padded, k_padded, v_padded)
        out_padded = out_padded.permute(0, 2, 1, 3)
        out_list = []
        for b in range(B):
            ql = q_seqlen[b]
            out_list.append(out_padded[b, :ql])
        out = torch.cat(out_list, dim=0)
"""

_ATTN_MARKER = "from trellis_silicon.attention_ops import naive_attention"
_RAISE_ANCHOR = '    else:\n        raise ValueError(f"Unknown attention module: {config.ATTN}")'


def patch_sparse_config():
    """Add 'sdpa' and 'naive' to the allowed attention backends."""
    path = os.path.join(TRELLIS_ROOT, "trellis2/modules/sparse/config.py")
    src = read_file(path)

    if "'sdpa'" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    src = src.replace(
        "env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3']",
        "env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']",
    )
    write_file(path, src)


def patch_sparse_attention():
    """Add SDPA and naive backends to the sparse attention dispatch.

    naive measured ~4x faster than sdpa on MPS at the shape/tex SLat
    attention shapes (B=1/2, S~1591 self-attn, kv=1029 cross-attn, H=12,
    D=128, bf16) -- an even bigger win than the dense structure-phase
    naive discovery (3.3-3.9x). Both backends share the same pad-to-batch
    scheme (no explicit attn_mask, relies on zero-padding); this is safe
    here because every real call site in this pipeline pads with zero
    waste (CFG-batched self/cross-attn always has uniform seqlens per
    batch), so naive is exactly as correct as the existing sdpa branch,
    not less.
    """
    path = os.path.join(TRELLIS_ROOT, "trellis2/modules/sparse/attention/full_attn.py")
    src = read_file(path)

    if _ATTN_MARKER in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    for old in (_PREV_SPLIT_BLOCK, _OLD_COMBINED_BLOCK):
        if old in src:
            # Upgrade from an earlier block: the combined sdpa==naive block, or
            # the split block with the unchunked naive matmul (issue #6).
            write_file(path, src.replace(old, _SPLIT_BLOCK))
            return

    rel = os.path.relpath(path, TRELLIS_ROOT)
    if "elif config.ATTN in ('sdpa', 'naive'):" in src:
        # Unrecognised (e.g. hand-edited) older block. Inserting would add a
        # second elif that never runs and would silence the startup warning.
        print(
            f"  Error: unrecognised sdpa/naive attention block in {rel}. "
            "Restore the file from the TRELLIS.2 git checkout "
            "(git -C TRELLIS.2 checkout -- " + _FULL_ATTN_REL + ") "
            "and re-run trellis-silicon-patch."
        )
        return
    if src.count(_RAISE_ANCHOR) != 1:
        print(
            f"  Error: expected exactly one 'Unknown attention module' anchor in {rel}, "
            f"found {src.count(_RAISE_ANCHOR)}. File not changed."
        )
        return

    # Fresh checkout: neither patch applied yet.
    src = src.replace(_RAISE_ANCHOR, _SPLIT_BLOCK + _RAISE_ANCHOR)
    write_file(path, src)


_FULL_ATTN_REL = "trellis2/modules/sparse/attention/full_attn.py"
_OUTDATED_WARNING = (
    "[SPARSE] WARNING: TRELLIS.2 sparse attention patch is missing or out of date "
    "(issue #6). Run: trellis-silicon-patch"
)


def sparse_attention_patch_is_current(root=None):
    """True if full_attn.py has the newest patched block, False if it does not,
    None if the file is missing or unreadable. Never raises for missing files."""
    if root is None:
        root = TRELLIS_ROOT
    try:
        text = read_file(os.path.join(root, _FULL_ATTN_REL))
    except (OSError, ValueError):
        return None
    return _ATTN_MARKER in text


def warn_if_sparse_attention_patch_outdated(root=None, file=None):
    """Print one warning line when the patched block is outdated. Returns True
    if it warned."""
    if sparse_attention_patch_is_current(root) is not False:
        return False
    print(_OUTDATED_WARNING, file=file if file is not None else sys.stderr, flush=True)
    return True
