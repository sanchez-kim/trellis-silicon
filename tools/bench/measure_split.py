"""Split each model load into construct (RNG) vs weight-I/O (no RNG)."""
import os, sys, time, json
sys.path.insert(0, os.getcwd())
import pipeline_core
import torch
from safetensors.torch import load_file
from trellis2 import models as tmodels

REPO = "/Users/sanchez/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d"
model_map = json.load(open(os.path.join(REPO, "pipeline.json")))['args']['models']

def resolve(name):
    v = model_map[name]
    for base in [f"{REPO}/{v}", None]:
        if base and os.path.exists(base + ".json"):
            return base + ".json", base + ".safetensors"
    # fallback: other repo via hf
    from huggingface_hub import hf_hub_download
    parts = v.split('/'); repo = f"{parts[0]}/{parts[1]}"; mn = '/'.join(parts[2:])
    return hf_hub_download(repo, mn + ".json"), hf_hub_download(repo, mn + ".safetensors")

def measure(name):
    cfgf, stf = resolve(name)
    cfg = json.load(open(cfgf))
    # construct (consumes RNG via initialize_weights)
    r0 = torch.get_rng_state()
    t0 = time.time()
    model = tmodels.__getattr__(cfg['name'])(**cfg['args'])
    t_con = time.time() - t0
    rng_moved = not torch.equal(torch.get_rng_state(), r0)
    # weight I/O (load_file + load_state_dict) — check RNG untouched
    r1 = torch.get_rng_state()
    t0 = time.time()
    sd = load_file(stf)
    t_read = time.time() - t0
    t0 = time.time()
    model.load_state_dict(sd, strict=False)
    t_apply = time.time() - t0
    io_rng_moved = not torch.equal(torch.get_rng_state(), r1)
    return t_con, t_read, t_apply, rng_moved, io_rng_moved

for n in ['sparse_structure_flow_model', 'shape_slat_flow_model_512',
          'tex_slat_flow_model_512', 'shape_slat_decoder', 'tex_slat_decoder',
          'sparse_structure_decoder']:
    tc, tr, ta, rm, im = measure(n)
    print(f"{n:32s} construct={tc:6.2f}s(rng={rm}) read={tr:6.2f}s apply={ta:5.2f}s io_rng={im}")
