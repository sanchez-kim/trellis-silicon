"""Numerical sanity: batched CFG vs sequential CFG, one denoising step,
for both the dense structure flow model and the sparse SLat flow model."""
import os
import sys
sys.path.insert(0, os.getcwd())

from trellis_silicon import core as pipeline_core  # sets backends/env before torch
import torch
from PIL import Image

from trellis2.modules.sparse import SparseTensor

DEV = torch.device("mps")


def max_abs_diff(a, b):
    if isinstance(a, SparseTensor):
        a = a.feats
        b = b.feats
    d = (a.float() - b.float()).abs()
    return d.max().item(), a.float().abs().mean().item()


@torch.no_grad()
def run_both(sampler, model, x_t, t, cond, neg_cond, **kw):
    os.environ["CFG_BATCH"] = "0"
    ps, ns = sampler._cfg_dual_forward(model, x_t, t, cond, neg_cond, **kw)
    os.environ["CFG_BATCH"] = "1"
    pb, nb = sampler._cfg_dual_forward(model, x_t, t, cond, neg_cond, **kw)
    return (ps, ns), (pb, nb)


def main():
    print("Loading pipeline (512)...")
    pipe = pipeline_core.load_pipeline("512")
    img = pipe.preprocess_image(Image.open("assets/brighella_input.png"))
    torch.manual_seed(42)

    t = 0.875  # inside the guided interval [0.6, 1.0]

    # ---------- structure (dense) ----------
    cond_d = pipe.get_cond([img], 512)  # {'cond','neg_cond'} on cpu
    fm = pipe.models["sparse_structure_flow_model"].to(DEV)
    reso, ic = fm.resolution, fm.in_channels
    x_t = torch.randn(1, ic, reso, reso, reso, device=DEV)
    cond = cond_d["cond"].to(DEV)
    neg = cond_d["neg_cond"].to(DEV)
    (ps, ns), (pb, nb) = run_both(pipe.sparse_structure_sampler, fm, x_t, t, cond, neg)
    mp, sp = max_abs_diff(ps, pb)
    mn, sn = max_abs_diff(ns, nb)
    print(f"[structure/dense] pred_pos max|diff|={mp:.3e} (mean|val|={sp:.3e}), rel={mp/sp:.3e}")
    print(f"[structure/dense] pred_neg max|diff|={mn:.3e} (mean|val|={sn:.3e}), rel={mn/sn:.3e}")
    fm.cpu()

    # ---------- shape slat (sparse) ----------
    cond_s = pipe.get_cond([img], 512)
    fm2 = pipe.models["shape_slat_flow_model_512"].to(DEV)
    ic2 = fm2.in_channels
    N = 2000
    # random voxel coords in a 64^3 grid, batch col = 0
    vox = torch.randint(0, 64, (N, 3), device=DEV)
    coords = torch.cat([torch.zeros(N, 1, dtype=torch.int32, device=DEV), vox.int()], dim=1)
    feats = torch.randn(N, ic2, device=DEV)
    x_s = SparseTensor(feats=feats, coords=coords)
    cond2 = cond_s["cond"].to(DEV)
    neg2 = cond_s["neg_cond"].to(DEV)
    (ps2, ns2), (pb2, nb2) = run_both(pipe.shape_slat_sampler, fm2, x_s, t, cond2, neg2)
    mp2, sp2 = max_abs_diff(ps2, pb2)
    mn2, sn2 = max_abs_diff(ns2, nb2)
    print(f"[shape/sparse] pred_pos max|diff|={mp2:.3e} (mean|val|={sp2:.3e}), rel={mp2/sp2:.3e}")
    print(f"[shape/sparse] pred_neg max|diff|={mn2:.3e} (mean|val|={sn2:.3e}), rel={mn2/sn2:.3e}")
    fm2.cpu()

    ok = max(mp/sp, mn/sn, mp2/sp2, mn2/sn2) < 5e-2
    print("RESULT:", "PASS (fp16 noise scale)" if ok else "FAIL (batching changed result)")


if __name__ == "__main__":
    main()
