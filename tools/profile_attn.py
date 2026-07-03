"""
Non-destructive profiling harness for TRELLIS.2 sampling phases.

Answers: within the three sampling phases (structure / shape-slat / texture-slat),
what fraction of synced wall-clock time is spent in attention (SDPA + pad/unpad)
vs sparse convolution (flex_gemm/conv_none) vs everything else, plus the
padding-waste stat (avg active seq len vs max padded len) for sparse attention.

MPS is async, so every timed region is bracketed with torch.mps.synchronize().
This serializes execution (percentages are self-consistent; absolute numbers run
a bit slower than a normal overlapped run -- see caveat in the printed report).
"""

import sys
import os
import re
import time
import warnings
from collections import Counter

# Importing core performs the backend/env setup (MPS fallback, ATTN/CONV
# backends) and puts TRELLIS.2/ + stubs/ on sys.path. It MUST run before torch.
from trellis_silicon import core  # noqa: F401

import torch
from PIL import Image as PILImage


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# Accumulators, keyed by phase name.
# ---------------------------------------------------------------------------
PHASES = ["structure", "shape", "texture", "decode_shape", "decode_tex"]
CURRENT_PHASE = None

acc = {
    p: {
        "attn_time": 0.0, "attn_calls": 0,
        "conv_time": 0.0, "conv_calls": 0,
        "segred_time": 0.0, "segred_calls": 0,
        "norm_time": 0.0, "norm_calls": 0,          # dense LayerNorm32/GroupNorm32
        "sparsenorm_time": 0.0, "sparsenorm_calls": 0,  # SparseLayerNorm/SparseGroupNorm (Python-loop classes)
        "meshextract_time": 0.0, "meshextract_calls": 0,  # flexible_dual_grid_to_mesh
        "moveto_time": 0.0, "moveto_calls": 0,      # nn.Module.to() device shuffling (low_vram)
        "active_tokens": 0, "padded_tokens": 0, "seqlen_calls": 0,
        "max_seen": 0,
    }
    for p in PHASES
}
phase_total = {p: 0.0 for p in PHASES}

# Global CPU-fallback warning sweep: op name -> {phase -> count}
fallback_warnings = Counter()  # (op_name, phase) -> count
_FALLBACK_RE = re.compile(r"operator '([^']+)' is not currently supported on the MPS backend")


def _record(kind, dt):
    if CURRENT_PHASE is None:
        return
    a = acc[CURRENT_PHASE]
    a[f"{kind}_time"] += dt
    a[f"{kind}_calls"] += 1


def _record_seqlens(seqlens):
    if CURRENT_PHASE is None or not seqlens:
        return
    a = acc[CURRENT_PHASE]
    B = len(seqlens)
    mx = max(seqlens)
    a["active_tokens"] += sum(seqlens)
    a["padded_tokens"] += mx * B
    a["seqlen_calls"] += 1
    a["max_seen"] = max(a["max_seen"], mx)


def install_warning_sweep():
    """
    Capture EVERY 'not currently supported on the MPS backend, will fall
    back to run on the CPU' UserWarning across the whole profiled run,
    grouped by op name and current phase. warnings.simplefilter('always')
    is required -- default filter action shows each (message, module,
    lineno) combo only once, which would undercount repeated CPU
    fallbacks fired from the same internal PyTorch call site.
    """
    warnings.simplefilter("always")
    _orig_showwarning = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):
        msg = str(message)
        m = _FALLBACK_RE.search(msg)
        if m:
            phase = CURRENT_PHASE or "unassigned"
            fallback_warnings[(m.group(1), phase)] += 1
        else:
            _orig_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = showwarning


def print_fallback_sweep():
    print("\n" + "#" * 60)
    print("# CPU-FALLBACK OP SWEEP (torch UserWarning: not supported on MPS)")
    print("#" * 60)
    if not fallback_warnings:
        print("  (none observed)")
        return
    by_op = Counter()
    for (op, phase), n in fallback_warnings.items():
        by_op[op] += n
    for op, total_n in by_op.most_common():
        per_phase = {phase: n for (o, phase), n in fallback_warnings.items() if o == op}
        print(f"  {op}: {total_n} calls total  {dict(per_phase)}")


# ---------------------------------------------------------------------------
# Monkeypatches.
# ---------------------------------------------------------------------------
def install_patches():
    import trellis2.modules.sparse as sp
    VarLenTensor = sp.VarLenTensor

    # --- sparse (padded) attention: modules.py imported the name directly ---
    import trellis2.modules.sparse.attention.modules as sp_attn_mod
    _orig_sparse_attn = sp_attn_mod.sparse_scaled_dot_product_attention

    def timed_sparse_attn(*args, **kwargs):
        # Extract per-batch seqlens from the first VarLenTensor argument.
        seqlens = None
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, VarLenTensor):
                try:
                    seqlens = [a.layout[i].stop - a.layout[i].start
                               for i in range(a.shape[0])]
                except Exception:
                    seqlens = None
                break
        _record_seqlens(seqlens)
        sync()
        t = time.perf_counter()
        out = _orig_sparse_attn(*args, **kwargs)
        sync()
        _record("attn", time.perf_counter() - t)
        return out

    sp_attn_mod.sparse_scaled_dot_product_attention = timed_sparse_attn

    # --- dense attention (used by the sparse-structure flow model) ---
    import trellis2.modules.attention.modules as dense_attn_mod
    _orig_dense_attn = dense_attn_mod.scaled_dot_product_attention

    def timed_dense_attn(*args, **kwargs):
        sync()
        t = time.perf_counter()
        out = _orig_dense_attn(*args, **kwargs)
        sync()
        _record("attn", time.perf_counter() - t)
        return out

    dense_attn_mod.scaled_dot_product_attention = timed_dense_attn

    # --- sparse conv forward (whichever backend is active) ---
    import trellis2.modules.sparse.conv.conv as conv_disp
    from trellis2.modules.sparse import config as sp_config

    backend = sp_config.CONV
    import importlib
    conv_backend_mod = importlib.import_module(
        f"trellis2.modules.sparse.conv.conv_{backend}"
    )
    _orig_conv_fwd = conv_backend_mod.sparse_conv3d_forward

    def timed_conv_fwd(self, x):
        sync()
        t = time.perf_counter()
        out = _orig_conv_fwd(self, x)
        sync()
        _record("conv", time.perf_counter() - t)
        return out

    conv_backend_mod.sparse_conv3d_forward = timed_conv_fwd
    # conv.py caches the backend module in _backends dict; patch that ref too.
    conv_disp._backends[backend] = conv_backend_mod
    print(f"[profile] conv backend = {backend}")

    # --- torch.segment_reduce, called from VarLenTensor.reduce() in
    # trellis2/modules/sparse/basic.py:283. This op is not supported on
    # MPS (UserWarning + CPU fallback), forcing a device sync + CPU<->GPU
    # round trip whenever a VarLenTensor's .mean()/.sum()/.prod()/.std()
    # is used with a non-batch reduction dim (e.g. CFG rescale on
    # shape SLat, guidance_rescale=0.5).
    _orig_segment_reduce = torch.segment_reduce

    def timed_segment_reduce(*args, **kwargs):
        sync()
        t = time.perf_counter()
        out = _orig_segment_reduce(*args, **kwargs)
        sync()
        dt = time.perf_counter() - t
        if CURRENT_PHASE is not None:
            a = acc[CURRENT_PHASE]
            a["segred_time"] += dt
            a["segred_calls"] += 1
        return out

    torch.segment_reduce = timed_segment_reduce
    # basic.py does `import torch` then calls `torch.segment_reduce(...)`
    # as a module attribute lookup at call time, so patching the attribute
    # on the torch module object is sufficient -- no need to patch basic.py.

    # --- dense LayerNorm32/GroupNorm32 (trellis2/modules/norm.py) -- this
    # is what the actual sampling transformer blocks use (blocks.py,
    # modulated.py both do `self.norm1 = LayerNorm32(channels, ...)` and
    # call `self.norm1(x.feats)` directly on the flat [T, C] feats tensor,
    # a single vectorized MPS-native LayerNorm call with no batch loop).
    import trellis2.modules.norm as dense_norm_mod

    _orig_ln32_fwd = dense_norm_mod.LayerNorm32.forward
    _orig_gn32_fwd = dense_norm_mod.GroupNorm32.forward

    def timed_ln32_fwd(self, x):
        sync()
        t = time.perf_counter()
        out = _orig_ln32_fwd(self, x)
        sync()
        dt = time.perf_counter() - t
        if CURRENT_PHASE is not None:
            a = acc[CURRENT_PHASE]
            a["norm_time"] += dt
            a["norm_calls"] += 1
        return out

    def timed_gn32_fwd(self, x):
        sync()
        t = time.perf_counter()
        out = _orig_gn32_fwd(self, x)
        sync()
        dt = time.perf_counter() - t
        if CURRENT_PHASE is not None:
            a = acc[CURRENT_PHASE]
            a["norm_time"] += dt
            a["norm_calls"] += 1
        return out

    dense_norm_mod.LayerNorm32.forward = timed_ln32_fwd
    dense_norm_mod.GroupNorm32.forward = timed_gn32_fwd

    # --- SparseGroupNorm/SparseLayerNorm (trellis2/modules/sparse/norm.py)
    # -- the Python-loop-per-batch-element classes flagged as a suspected
    # bottleneck. Static analysis (grep across TRELLIS.2/) found zero
    # instantiation sites -- only defined/exported, never used by the flow
    # models. Instrumenting anyway to get empirical call-count confirmation
    # rather than relying on grep alone.
    import trellis2.modules.sparse.norm as sparse_norm_mod

    _orig_sgn_fwd = sparse_norm_mod.SparseGroupNorm.forward
    _orig_sln_fwd = sparse_norm_mod.SparseLayerNorm.forward

    def timed_sgn_fwd(self, x):
        sync()
        t = time.perf_counter()
        out = _orig_sgn_fwd(self, x)
        sync()
        dt = time.perf_counter() - t
        if CURRENT_PHASE is not None:
            a = acc[CURRENT_PHASE]
            a["sparsenorm_time"] += dt
            a["sparsenorm_calls"] += 1
        return out

    def timed_sln_fwd(self, x):
        sync()
        t = time.perf_counter()
        out = _orig_sln_fwd(self, x)
        sync()
        dt = time.perf_counter() - t
        if CURRENT_PHASE is not None:
            a = acc[CURRENT_PHASE]
            a["sparsenorm_time"] += dt
            a["sparsenorm_calls"] += 1
        return out

    sparse_norm_mod.SparseGroupNorm.forward = timed_sgn_fwd
    sparse_norm_mod.SparseLayerNorm.forward = timed_sln_fwd

    # --- flexible_dual_grid_to_mesh (backends/mesh_extract.py via
    # stubs/o_voxel_override_convert.py) -- pure-Python/CPU mesh extraction
    # called from FlexiDualGridVaeDecoder.forward inside decode_shape_slat.
    # No MPS work happens inside it, but sync() before/after is kept for
    # consistency and to make sure any pending GPU work feeding its CPU
    # inputs (the .cpu() calls inside the function) is actually done.
    import trellis2.models.sc_vaes.fdg_vae as fdg_vae_mod
    _orig_mesh_extract = fdg_vae_mod.flexible_dual_grid_to_mesh

    def timed_mesh_extract(*args, **kwargs):
        sync()
        t = time.perf_counter()
        out = _orig_mesh_extract(*args, **kwargs)
        sync()
        dt = time.perf_counter() - t
        if CURRENT_PHASE is not None:
            a = acc[CURRENT_PHASE]
            a["meshextract_time"] += dt
            a["meshextract_calls"] += 1
        return out

    fdg_vae_mod.flexible_dual_grid_to_mesh = timed_mesh_extract

    # --- nn.Module.to(device) -- quantifies the "low_vram" cost: pipeline
    # defaults to low_vram=True (pipeline.json doesn't set it, base.py
    # defaults True), which means every sampling/decode call does
    # `model.to(self.device)` before and `model.cpu()` after, shuttling
    # the (large, up to ~1.3B-param) submodel's full weight set between
    # CPU and MPS on every phase transition instead of once at load.
    _orig_module_to = torch.nn.Module.to

    def timed_module_to(self, *args, **kwargs):
        sync()
        t = time.perf_counter()
        out = _orig_module_to(self, *args, **kwargs)
        sync()
        dt = time.perf_counter() - t
        if CURRENT_PHASE is not None:
            a = acc[CURRENT_PHASE]
            a["moveto_time"] += dt
            a["moveto_calls"] += 1
        return out

    torch.nn.Module.to = timed_module_to

    return VarLenTensor


def wrap_phases(pipeline):
    def make_wrapper(name, fn):
        def wrapper(*args, **kwargs):
            global CURRENT_PHASE
            CURRENT_PHASE = name
            sync()
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                sync()
                phase_total[name] += time.perf_counter() - t0
                CURRENT_PHASE = None
                print_phase(name)
        return wrapper

    pipeline.sample_sparse_structure = make_wrapper(
        "structure", pipeline.sample_sparse_structure)
    pipeline.sample_shape_slat = make_wrapper(
        "shape", pipeline.sample_shape_slat)
    pipeline.sample_tex_slat = make_wrapper(
        "texture", pipeline.sample_tex_slat)
    pipeline.decode_shape_slat = make_wrapper(
        "decode_shape", pipeline.decode_shape_slat)
    pipeline.decode_tex_slat = make_wrapper(
        "decode_tex", pipeline.decode_tex_slat)


def print_phase(name):
    a = acc[name]
    tot = phase_total[name]
    attn = a["attn_time"]
    conv = a["conv_time"]
    segred = a["segred_time"]
    norm = a["norm_time"]
    sparsenorm = a["sparsenorm_time"]
    meshextract = a["meshextract_time"]
    moveto = a["moveto_time"]
    other = tot - attn - conv  # segred/norm/meshextract/moveto are subsets of "other"
    print("\n" + "=" * 60)
    print(f"PHASE: {name}   total (synced wall-clock) = {tot:.2f}s")
    if tot > 0:
        print(f"  attention : {attn:7.2f}s  ({100*attn/tot:5.1f}%)  "
              f"[{a['attn_calls']} calls]")
        print(f"  sparse-conv: {conv:6.2f}s  ({100*conv/tot:5.1f}%)  "
              f"[{a['conv_calls']} calls]")
        print(f"  other     : {other:7.2f}s  ({100*other/tot:5.1f}%)")
        if other > 0:
            print(f"    of which segment_reduce: {segred:6.3f}s  "
                  f"({100*segred/other:5.1f}% of 'other', "
                  f"{100*segred/tot:5.1f}% of phase total)  "
                  f"[{a['segred_calls']} calls]")
            print(f"    of which dense LayerNorm32/GroupNorm32: {norm:6.3f}s  "
                  f"({100*norm/other:5.1f}% of 'other', "
                  f"{100*norm/tot:5.1f}% of phase total)  "
                  f"[{a['norm_calls']} calls]")
            print(f"    of which SparseLayerNorm/SparseGroupNorm (Python-loop): "
                  f"{sparsenorm:6.3f}s  [{a['sparsenorm_calls']} calls]")
            print(f"    of which flexible_dual_grid_to_mesh (pure Python/CPU): "
                  f"{meshextract:6.3f}s  ({100*meshextract/other:5.1f}% of 'other', "
                  f"{100*meshextract/tot:5.1f}% of phase total)  "
                  f"[{a['meshextract_calls']} calls]")
            print(f"    of which nn.Module.to() device transfer (low_vram shuffling): "
                  f"{moveto:6.3f}s  ({100*moveto/other:5.1f}% of 'other', "
                  f"{100*moveto/tot:5.1f}% of phase total)  "
                  f"[{a['moveto_calls']} calls]")
    if a["seqlen_calls"] > 0:
        active = a["active_tokens"]
        padded = a["padded_tokens"]
        waste = 100 * (1 - active / padded) if padded else 0.0
        # avg active seq len per batch element and the max padded length.
        # seqlen_calls batches; approximate avg active per element:
        print(f"  padding: active_tokens={active}  padded_tokens={padded}  "
              f"waste={waste:.1f}%  max_seq_seen={a['max_seen']}")
    print("=" * 60)


def main():
    img_path = os.path.join(os.path.dirname(__file__), "assets", "shoe_input.png")

    install_warning_sweep()

    print("[profile] loading pipeline (from_pretrained: disk read + deserialize)...")
    t0 = time.time()
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    t_from_pretrained = time.time() - t0
    print(f"[profile] from_pretrained() in {t_from_pretrained:.1f}s")

    t1 = time.time()
    pipeline.to(torch.device("mps"))
    t_to_mps = time.time() - t1
    print(f"[profile] pipeline.to(mps) in {t_to_mps:.1f}s  "
          f"(low_vram={pipeline.low_vram} -- if True this is nearly a no-op; "
          f"actual model transfers happen per-phase inside sampling/decode)")
    print(f"[profile] total pipeline load: {t_from_pretrained + t_to_mps:.1f}s")

    install_patches()
    wrap_phases(pipeline)

    img = PILImage.open(img_path)
    print("[profile] running (seed=42, pipeline_type=512)...")
    try:
        pipeline.run(img, seed=42, pipeline_type="512")
    except Exception as e:
        print(f"\n[profile] run raised (likely decode/watchdog, AFTER sampling): "
              f"{type(e).__name__}: {e}")

    # Final combined summary.
    print("\n\n" + "#" * 60)
    print("# FINAL SUMMARY (sampling + decode phases)")
    print("#" * 60)
    for p in PHASES:
        print_phase(p)

    print_fallback_sweep()


if __name__ == "__main__":
    main()
