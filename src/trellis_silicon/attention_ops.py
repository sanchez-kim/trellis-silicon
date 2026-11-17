"""Naive attention helper for the MPS backend.

torch's MPS ``bmm`` switches to ``tiled_bmm_out_mps_impl`` when the output has
more than 2**32 elements (torch 2.12.1, LinearAlgebra.mm). With a transposed
(non-contiguous) second operand, as in ``q @ k.transpose(-2, -1)``, that path
SIGSEGVs on M4 Max / macOS 27.0.1 (GitHub issue #6, pipelines 1024 /
1024_cascade) and silently returns wrong scores on M5 / macOS 26.7.1 (a
B*H=24, S=13,400 bf16 probe differs from a CPU fp32 reference by ~90 on values
of ~60; with ``.contiguous()`` on k^T it matches to 0.125).

Real inputs cross that limit: at 1024, four test cutouts gave 9,287-20,043
shape-SLat tokens, three of them above the ~13,400 where B=2, H=12 self-attn
reaches 2**32 score elements. Memory is the tighter bound, though: a 2**32
element bf16 score tensor is 8.6 GB, and the scale and softmax each make
another, which exhausted a 32 GB Mac at 15,626 tokens. So chunks are capped at
CHUNK_ELEMS (2**28 elements, 512 MB bf16), far under the tiled-bmm limit.
"""

import sys

import torch

# torch MPS bmm switches to tiled_bmm_out_mps_impl above this many output elements.
MPS_BMM_MAX_OUT = 2**32
# Score elements per chunk: 512 MB per bf16 score tensor. Must stay <= MPS_BMM_MAX_OUT.
CHUNK_ELEMS = 2**28
# Shapes already reported. One line per new shape, so a cascade run shows both
# shape-SLat stages; the repeated calls within a stage stay silent.
_chunk_notice_shapes = set()


def naive_attention(q, k, v, max_out_elems=CHUNK_ELEMS):
    """q,k,v: [B, H, S, D]. softmax(q @ k^T * D**-0.5) @ v.

    k and v must have the same B and H as q (no broadcasting); v's last dim
    may differ from D. Splits along the query axis so no score matmul output
    exceeds ``max_out_elems`` (issue #6; see module docstring). Softmax is per
    query row, so chunking is exact. At or below the limit this runs the
    single-shot expression unchanged (bit-identical to the unchunked code).
    """
    scale = q.shape[-1] ** -0.5
    B, H, S_q = q.shape[:3]
    S_kv = k.shape[-2]
    if B * H * S_q * S_kv <= max_out_elems:
        return torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1) @ v

    rows = max(1, max_out_elems // (B * H * S_kv))
    shape = (B, H, S_q, S_kv)
    if shape not in _chunk_notice_shapes:
        _chunk_notice_shapes.add(shape)
        print(
            f"[SPARSE] naive attention B={B} H={H} S_q={S_q} S_kv={S_kv}: "
            f"{B * H * S_q * S_kv:,} score elements > {max_out_elems:,}; "
            f"splitting into {-(-S_q // rows)} query blocks (issue #6)",
            file=sys.stderr,
            flush=True,
        )
    # Flatten batch and heads once. With 4D operands, matmul copies k^T and v
    # into a 3D layout on every chunk; the call site passes permuted
    # (non-contiguous) tensors, so each chunk would copy all of k and v.
    # Cost: one copy each of q, k and v lives until return (for the 15,626-
    # token fox at 1024, 96 MB each in bf16, against ~1 GB of per-chunk scores).
    BH, D_v = B * H, v.shape[-1]
    q3 = q.reshape(BH, S_q, q.shape[-1])
    kt3 = k.reshape(BH, S_kv, k.shape[-1]).transpose(1, 2)
    v3 = v.reshape(BH, S_kv, D_v)
    out = q.new_empty(BH, S_q, D_v)
    for i in range(0, S_q, rows):
        scores = q3[:, i : i + rows] @ kt3
        scores.mul_(scale)  # in place: one score-sized tensor fewer per chunk
        out[:, i : i + rows] = torch.softmax(scores, dim=-1) @ v3
        del scores
    return out.view(B, H, S_q, D_v)
