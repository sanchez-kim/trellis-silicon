"""Fair benchmark at OUR realistic shapes: fused mtlgemm sparse attention vs the
SDPA-on-MPS path our pipeline ACTUALLY uses (full_attn.py sdpa branch, no CPU
bounce). Batch 1 => N=1 single sequences, which is the real workload.
Unbuffered, light iter count so it finishes quickly and streams progress.
"""
import os, sys
os.environ.setdefault("FLEX_GEMM_QUIET", "1")
import math, time, torch
import torch.nn.functional as F

assert torch.backends.mps.is_available()
import flex_gemm

DEV = "mps"

def log(*a):
    print(*a); sys.stdout.flush()

def bench(fn, warmup=3, iters=8):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

def sdpa_mps_path(q, k, v, q_seqlen, kv_seqlen):
    """Faithful copy of full_attn.py sdpa branch — runs on MPS, no CPU bounce."""
    device = q.device
    N = len(q_seqlen)
    max_q = max(q_seqlen); max_kv = max(kv_seqlen)
    H = q.shape[-2]; C_q = q.shape[-1]; C_v = v.shape[-1]
    q_dense = q.new_zeros(N, max_q, H, C_q)
    k_dense = k.new_zeros(N, max_kv, H, C_q)
    v_dense = v.new_zeros(N, max_kv, H, C_v)
    attn_mask = torch.zeros(N, max_q, max_kv, dtype=torch.bool, device=device)
    qo = ko = 0
    for i in range(N):
        ql = q_seqlen[i]; kvl = kv_seqlen[i]
        q_dense[i, :ql] = q[qo:qo+ql]; k_dense[i, :kvl] = k[ko:ko+kvl]; v_dense[i, :kvl] = v[ko:ko+kvl]
        attn_mask[i, :ql, :kvl] = True
        qo += ql; ko += kvl
    q_dense = q_dense.permute(0, 2, 1, 3); k_dense = k_dense.permute(0, 2, 1, 3); v_dense = v_dense.permute(0, 2, 1, 3)
    float_mask = torch.zeros(N, 1, max_q, max_kv, dtype=q_dense.dtype, device=device)
    float_mask.masked_fill_(~attn_mask.unsqueeze(1), float('-inf'))
    out = F.scaled_dot_product_attention(q_dense, k_dense, v_dense, attn_mask=float_mask)
    out = out.permute(0, 2, 1, 3)
    return torch.cat([out[i, :q_seqlen[i]] for i in range(N)], dim=0)

def fused(q, k, v, q_seqlen, kv_seqlen):
    device = q.device
    scale = 1.0 / math.sqrt(q.shape[-1])
    csq = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), 0)]).int().to(device)
    cskv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), 0)]).int().to(device)
    return flex_gemm.kernels.cuda.sparse_attention_fwd(
        q.contiguous(), k.contiguous(), v.contiguous(), csq, cskv,
        max(q_seqlen), max(kv_seqlen), scale)

def rand_on(shape, dtype):
    return (torch.randn(*shape, dtype=dtype) * 0.3).to(DEV)

H, C = 8, 64  # TRELLIS.2 decoder-typical (8 heads, head_dim 64)
plan = [
    (torch.float16, [1477, 4096, 8192, 16384]),
    (torch.float32, [1477, 4096, 8192]),
]
for dtype, seqlens in plan:
    log("=" * 78)
    log(f"N=1 single sequence (batch 1), H={H} C={C}, dtype={dtype}")
    log(f"{'seqlen':>7s} {'fused':>10s} {'sdpa-MPS':>10s} {'speedup':>8s} {'max_err':>10s}")
    for seqlen in seqlens:
        q = rand_on((seqlen, H, C), dtype); k = rand_on((seqlen, H, C), dtype); v = rand_on((seqlen, H, C), dtype)
        sl = [seqlen]
        try:
            a = fused(q, k, v, sl, sl).detach().cpu().float()
            b = sdpa_mps_path(q, k, v, sl, sl).detach().cpu().float()
            err = (a - b).abs().max().item()
        except Exception as e:
            log(f"{seqlen:7d} ERROR {type(e).__name__}: {e}")
            continue
        fm = bench(lambda: fused(q, k, v, sl, sl))
        sm = bench(lambda: sdpa_mps_path(q, k, v, sl, sl))
        log(f"{seqlen:7d} {fm:8.3f}ms {sm:8.3f}ms {sm/fm:7.2f}x {err:10.2e}")
log("DONE")
