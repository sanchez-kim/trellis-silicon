"""Is skipping initialize_weights safe? Compare a normally-loaded model's
state_dict to a skip-init-loaded model's. If identical, the checkpoint covers
all params/buffers and init is pure waste (overwritten by load_state_dict)."""
import os, sys, time, json
sys.path.insert(0, os.getcwd())
import pipeline_core
import torch
from safetensors.torch import load_file
from trellis2 import models as tmodels

REPO = "/Users/sanchez/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d"
model_map = json.load(open(os.path.join(REPO, "pipeline.json")))['args']['models']

def files(name):
    v = model_map[name]
    if os.path.exists(f"{REPO}/{v}.json"):
        return f"{REPO}/{v}.json", f"{REPO}/{v}.safetensors"
    from huggingface_hub import hf_hub_download
    p = v.split('/'); repo=f"{p[0]}/{p[1]}"; mn='/'.join(p[2:])
    return hf_hub_download(repo, mn+".json"), hf_hub_download(repo, mn+".safetensors")

def build(name, skip_init):
    cfgf, stf = files(name)
    cfg = json.load(open(cfgf))
    cls = tmodels.__getattr__.__self__ if False else None
    klass = getattr(__import__('trellis2.models', fromlist=[cfg['name']]), cfg['name'])
    orig = klass.initialize_weights
    if skip_init:
        klass.initialize_weights = lambda self: None
    t0 = time.time()
    r0 = torch.get_rng_state()
    try:
        model = tmodels.__getattr__(cfg['name'])(**cfg['args'])
    finally:
        klass.initialize_weights = orig
    tcon = time.time() - t0
    rng_moved = not torch.equal(torch.get_rng_state(), r0)
    model.load_state_dict(load_file(stf), strict=False)
    return model, tcon, rng_moved

for name in ['sparse_structure_flow_model', 'shape_slat_flow_model_512', 'tex_slat_decoder']:
    torch.manual_seed(0)
    m_norm, tc_n, rng_n = build(name, skip_init=False)
    torch.manual_seed(0)
    m_skip, tc_s, rng_s = build(name, skip_init=True)
    sd_n, sd_s = m_norm.state_dict(), m_skip.state_dict()
    keys_match = set(sd_n) == set(sd_s)
    maxdiff = 0.0
    mismatched = []
    for k in sd_n:
        d = (sd_n[k].float() - sd_s[k].float()).abs().max().item()
        if d > 0: mismatched.append((k, d))
        maxdiff = max(maxdiff, d)
    print(f"{name}:")
    print(f"  construct: normal={tc_n:.2f}s(rng={rng_n}) skip={tc_s:.2f}s(rng={rng_s})  -> saves {tc_n-tc_s:.1f}s")
    print(f"  state_dict keys match={keys_match}, max|diff|={maxdiff:.3e}, #mismatch={len(mismatched)}")
    if mismatched[:3]:
        for k,d in mismatched[:3]: print(f"    MISMATCH {k}: {d:.3e}")
