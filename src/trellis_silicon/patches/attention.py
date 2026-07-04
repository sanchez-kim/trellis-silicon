"""Attention backend patches: add SDPA/naive backends to TRELLIS.2's sparse
attention dispatch so it runs on MPS (no xformers/flash-attn)."""

import os

from .common import TRELLIS_ROOT, read_file, write_file


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

    if "if config.ATTN == 'sdpa':" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    split_block = """\
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

    old_combined_block = """\
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

    if old_combined_block in src:
        # Upgrade from the earlier combined sdpa==naive block (from before
        # naive was separately benchmarked and found ~4x faster) to real,
        # separate implementations.
        src = src.replace(old_combined_block, split_block)
        write_file(path, src)
        return

    # Fresh checkout: neither patch applied yet.
    src = src.replace(
        '    else:\n        raise ValueError(f"Unknown attention module: {config.ATTN}")',
        split_block
        + '    else:\n        raise ValueError(f"Unknown attention module: {config.ATTN}")',
    )
    write_file(path, src)
