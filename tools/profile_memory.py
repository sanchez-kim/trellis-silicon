"""
Non-destructive phase-level memory profiling for a full TRELLIS.2 generation.

Answers: during a real generation (pipeline load -> structure sampling ->
structure decode -> shape SLat sampling -> shape SLat decode -> tex SLat
sampling -> tex SLat decode -> texture bake), where does unified-memory
usage actually peak, and by how much?

Two measurements are combined:
  - Explicit before/after snapshots at each phase boundary (torch.mps
    .synchronize() + current_allocated_memory()/driver_allocated_memory()),
    printed as the run progresses and collected into a final table.
  - A background poll thread sampling the same two counters every 50ms
    for the whole run, attributed to whichever phase is current. This
    catches intra-phase peaks (e.g. mid-denoising-loop) that a
    before/after-only snapshot would miss. Reading these allocator
    counters does not require a sync -- they reflect allocation
    bookkeeping done on the CPU side when tensors are created/freed, not
    completed GPU work -- so the poller does not serialize the run.

driver_allocated_memory() is the real macOS/Metal-level GPU-side allocation;
current_allocated_memory() is PyTorch's own bookkeeping, included alongside
it since it's what most PyTorch memory guidance quotes. CRITICALLY, neither
captures the whole picture on Apple Silicon: this pipeline's low_vram mode
keeps not-currently-active submodels resident on the CPU side (shuffling
only the active one onto the MPS device), and on unified memory that
CPU-resident footprint counts against the same physical budget a 16GB
machine has to fit everything in, even though it never shows up in the two
MPS-only counters above. So this script also samples process RSS
(psutil, resident set size) at every boundary/poll tick -- RSS is the
number that actually answers "will this fit on a 16GB Mac."

Like tools/profile_attn.py, importing trellis_silicon.core performs the
backend/env setup and MUST happen before torch is imported anywhere else.
Respects ATTN_BACKEND / SPARSE_ATTN_BACKEND / CFG_BATCH env overrides
(core.py uses os.environ.setdefault, so external env wins) -- run this
script under different env vars in fresh processes to compare, e.g.:

    python tools/profile_memory.py
    ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa python tools/profile_memory.py
    CFG_BATCH=0 python tools/profile_memory.py
"""

import datetime
import json
import os
import sys
import tempfile
import threading
import time

from trellis_silicon import core  # noqa: F401 -- sets backends/env before torch

import psutil
import torch
from PIL import Image as PILImage

_PROC = psutil.Process()

assert torch.backends.mps.is_available(), "This profiler requires MPS (Apple Silicon)."

PHASES = [
    "pipeline_load",
    "structure_sampling",
    "structure_decode",
    "shape_slat_sampling",
    "shape_slat_decode",
    "tex_slat_sampling",
    "tex_slat_decode",
    "texture_bake",
]

CURRENT_PHASE = "pipeline_load"
phase_snapshots = {}  # phase -> {"before": (cur, drv), "after": (cur, drv)}

_lock = threading.Lock()
_stop_poll = threading.Event()
per_phase_peak = {p: {"current": 0, "driver": 0, "rss": 0, "sys_available": 0, "sys_used": 0} for p in PHASES}
global_peak = {"current": 0, "driver": 0, "rss": 0, "phase": None, "sys_available": 0, "sys_used": 0}


def snap(phase, when):
    torch.mps.synchronize()
    cur = torch.mps.current_allocated_memory()
    drv = torch.mps.driver_allocated_memory()
    rss = _PROC.memory_info().rss
    vm = psutil.virtual_memory()
    print(f"[mem] {when:>6s}  {phase:<20s} current={cur/1e9:6.2f}GB  driver={drv/1e9:6.2f}GB  rss={rss/1e9:6.2f}GB "
          f"| sys_avail={vm.available/1e9:6.2f}GB  sys_used={vm.used/1e9:6.2f}GB  sys_pct={vm.percent:5.1f}%",
          file=sys.stderr)
    return cur, drv, rss, vm.available, vm.used, vm.percent


def poll_loop():
    while not _stop_poll.is_set():
        cur = torch.mps.current_allocated_memory()
        drv = torch.mps.driver_allocated_memory()
        rss = _PROC.memory_info().rss
        vm = psutil.virtual_memory()
        phase = CURRENT_PHASE
        with _lock:
            if phase in per_phase_peak:
                pp = per_phase_peak[phase]
                if rss > pp["rss"]:
                    pp["driver"] = drv
                    pp["current"] = cur
                    pp["rss"] = rss
                    pp["sys_available"] = vm.available
                    pp["sys_used"] = vm.used
            if rss > global_peak["rss"]:
                global_peak["driver"] = drv
                global_peak["current"] = cur
                global_peak["rss"] = rss
                global_peak["phase"] = phase
                global_peak["sys_available"] = vm.available
                global_peak["sys_used"] = vm.used
        _stop_poll.wait(0.05)


def make_method_wrapper(phase_name, fn):
    def wrapper(*args, **kwargs):
        global CURRENT_PHASE
        CURRENT_PHASE = phase_name
        before = snap(phase_name, "before")
        try:
            return fn(*args, **kwargs)
        finally:
            after = snap(phase_name, "after")
            phase_snapshots[phase_name] = {"before": before, "after": after}
    return wrapper


def make_module_wrapper(module, phase_name):
    """Brackets a single nn.Module's forward as its own phase, nested inside
    whatever phase is already current (saved/restored around the call) --
    used to split sample_sparse_structure's single method call into
    "structure_sampling" (denoising loop) and "structure_decode" (the
    sparse_structure_decoder forward at the end of that same call).
    """
    orig_forward = module.forward

    def wrapped(*args, **kwargs):
        global CURRENT_PHASE
        prev_phase = CURRENT_PHASE
        CURRENT_PHASE = phase_name
        before = snap(phase_name, "before")
        try:
            return orig_forward(*args, **kwargs)
        finally:
            after = snap(phase_name, "after")
            phase_snapshots[phase_name] = {"before": before, "after": after}
            CURRENT_PHASE = prev_phase

    module.forward = wrapped


def make_func_wrapper(obj, attr, phase_name):
    """Brackets a plain function (not a bound nn.Module) as a phase --
    used for the texture-bake backends (o_voxel.postprocess.to_glb, or the
    KDTree fallback's uv_unwrap/bake_texture/export_glb_with_texture), so
    whichever backend actually runs gets attributed to "texture_bake"
    without needing to touch core.py's branching logic.
    """
    orig = getattr(obj, attr)

    def wrapped(*args, **kwargs):
        global CURRENT_PHASE
        prev_phase = CURRENT_PHASE
        CURRENT_PHASE = phase_name
        before = snap(f"{phase_name}:{attr}", "before")
        try:
            return orig(*args, **kwargs)
        finally:
            after = snap(f"{phase_name}:{attr}", "after")
            # Keep the last sub-call's snapshot under the phase name too,
            # so the summary table always has an entry for "texture_bake".
            phase_snapshots[phase_name] = {"before": before, "after": after}
            CURRENT_PHASE = prev_phase

    setattr(obj, attr, wrapped)


def install_texture_bake_wrappers():
    try:
        import o_voxel.postprocess as ovp
        make_func_wrapper(ovp, "to_glb", "texture_bake")
    except (ImportError, AttributeError):
        pass

    from trellis_silicon.backends import texture_baker as tb
    for name in ("uv_unwrap", "bake_texture", "export_glb_with_texture"):
        make_func_wrapper(tb, name, "texture_bake")


def print_summary():
    print("\n" + "#" * 90)
    print("# PHASE MEMORY SUMMARY")
    print("#" * 90)
    header = (f"{'phase':<20s} {'before(rss)':>12s} {'after(rss)':>12s} {'peak(rss)':>12s} "
              f"{'peak(drv)':>12s} {'peak(cur)':>12s} {'sys_avail@pk':>13s}")
    print(header)
    print("-" * len(header))
    for p in PHASES:
        snap_ = phase_snapshots.get(p)
        peak = per_phase_peak[p]
        before_rss = snap_["before"][2] / 1e9 if snap_ else float("nan")
        after_rss = snap_["after"][2] / 1e9 if snap_ else float("nan")
        peak_rss = peak["rss"] / 1e9
        peak_drv = peak["driver"] / 1e9
        peak_cur = peak["current"] / 1e9
        peak_sys_avail = peak["sys_available"] / 1e9
        print(f"{p:<20s} {before_rss:11.2f}G {after_rss:11.2f}G {peak_rss:11.2f}G "
              f"{peak_drv:11.2f}G {peak_cur:11.2f}G {peak_sys_avail:12.2f}G")

    print("\nGlobal peak (process RSS, sampled continuously @ 50ms -- the number that")
    print("determines whether this fits on a given machine's unified memory):")
    print(f"  {global_peak['rss']/1e9:.2f}GB  during phase = {global_peak['phase']}")
    print(f"  (MPS driver_allocated_memory at that instant: {global_peak['driver']/1e9:.2f}GB, "
          f"current_allocated_memory: {global_peak['current']/1e9:.2f}GB)")
    print(f"  (system-wide at that instant: available={global_peak['sys_available']/1e9:.2f}GB, "
          f"used={global_peak['sys_used']/1e9:.2f}GB)")


def log_run(pipeline_type, total_seconds, glb_path):
    """Append this run's full summary to a persistent JSONL log (outputs/ is
    gitignored -- safe scratch space) so results survive across however many
    comparison runs get done, instead of scrolling away in stdout."""
    log_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "mem_experiment_log.jsonl",
    )
    vm = psutil.virtual_memory()
    phases_out = {}
    for p in PHASES:
        snap_ = phase_snapshots.get(p)
        peak = per_phase_peak[p]
        phases_out[p] = {
            "before_rss_gb": round(snap_["before"][2] / 1e9, 4) if snap_ else None,
            "after_rss_gb": round(snap_["after"][2] / 1e9, 4) if snap_ else None,
            "peak_rss_gb": round(peak["rss"] / 1e9, 4),
            "peak_driver_gb": round(peak["driver"] / 1e9, 4),
            "peak_current_gb": round(peak["current"] / 1e9, 4),
            "peak_sys_available_gb": round(peak["sys_available"] / 1e9, 4),
            "peak_sys_used_gb": round(peak["sys_used"] / 1e9, 4),
        }
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "script": "tools/profile_memory.py",
        "env": {
            "ATTN_BACKEND": os.environ.get("ATTN_BACKEND"),
            "SPARSE_ATTN_BACKEND": os.environ.get("SPARSE_ATTN_BACKEND"),
            "CFG_BATCH": os.environ.get("CFG_BATCH", "1"),
            "SKIP_INIT_ON_LOAD": os.environ.get("SKIP_INIT_ON_LOAD", "1"),
        },
        "pipeline_type": pipeline_type,
        "total_seconds": round(total_seconds, 1),
        "glb_path": glb_path,
        "post_run_sys_available_gb": round(vm.available / 1e9, 4),
        "post_run_sys_used_gb": round(vm.used / 1e9, 4),
        "phases": phases_out,
        "global_peak_rss_gb": round(global_peak["rss"] / 1e9, 4),
        "global_peak_phase": global_peak["phase"],
        "global_peak_sys_available_gb": round(global_peak["sys_available"] / 1e9, 4),
        "global_peak_sys_used_gb": round(global_peak["sys_used"] / 1e9, 4),
    }
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\n[profile] appended run summary to {log_path}", file=sys.stderr)


def main():
    img_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets", "brighella_input.png",
    )
    output_base = os.path.join(tempfile.gettempdir(), "trellis_silicon_profile_memory")

    poller = threading.Thread(target=poll_loop, daemon=True)
    poller.start()

    print(f"[profile] ATTN_BACKEND={os.environ.get('ATTN_BACKEND')} "
          f"SPARSE_ATTN_BACKEND={os.environ.get('SPARSE_ATTN_BACKEND')} "
          f"CFG_BATCH={os.environ.get('CFG_BATCH', '1')}", file=sys.stderr)

    vm0 = psutil.virtual_memory()
    print(f"[profile] pre-run system memory: available={vm0.available/1e9:.2f}GB "
          f"used={vm0.used/1e9:.2f}GB percent={vm0.percent:.1f}%", file=sys.stderr)
    log_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "mem_experiment_log.jsonl",
    )
    if os.path.exists(log_path):
        with open(log_path) as f:
            prior_lines = [json.loads(line) for line in f if line.strip()]
        if prior_lines:
            prior_avail = prior_lines[-1].get("post_run_sys_available_gb")
            if prior_avail is not None and abs(vm0.available / 1e9 - prior_avail) > 3.5:
                print(f"[profile] WARNING: system available memory ({vm0.available/1e9:.2f}GB) differs "
                      f"from the previous logged run's post-run reading ({prior_avail:.2f}GB) by more "
                      f"than 3.5GB -- background load has changed meaningfully; treat this run as "
                      f"NOT directly comparable to that prior run.", file=sys.stderr)

    global CURRENT_PHASE
    CURRENT_PHASE = "pipeline_load"
    before = snap("pipeline_load", "before")
    pipeline = core.load_pipeline("512", resident=False)
    after = snap("pipeline_load", "after")
    phase_snapshots["pipeline_load"] = {"before": before, "after": after}

    make_module_wrapper(pipeline.models["sparse_structure_decoder"], "structure_decode")
    pipeline.sample_sparse_structure = make_method_wrapper(
        "structure_sampling", pipeline.sample_sparse_structure)
    pipeline.sample_shape_slat = make_method_wrapper(
        "shape_slat_sampling", pipeline.sample_shape_slat)
    pipeline.decode_shape_slat = make_method_wrapper(
        "shape_slat_decode", pipeline.decode_shape_slat)
    pipeline.sample_tex_slat = make_method_wrapper(
        "tex_slat_sampling", pipeline.sample_tex_slat)
    pipeline.decode_tex_slat = make_method_wrapper(
        "tex_slat_decode", pipeline.decode_tex_slat)
    install_texture_bake_wrappers()

    img = PILImage.open(img_path)
    print("[profile] running generate_glb (seed=42, pipeline_type=512, texture_size=1024)...",
          file=sys.stderr)
    t0 = time.time()
    result = core.generate_glb(
        pipeline, img,
        seed=42, pipeline_type="512", texture_size=1024,
        output_base=output_base,
    )
    elapsed = time.time() - t0
    print(f"[profile] done in {elapsed:.1f}s -> {result['glb_path']}", file=sys.stderr)

    _stop_poll.set()
    poller.join(timeout=1.0)

    print_summary()
    log_run(pipeline_type="512", total_seconds=elapsed, glb_path=result["glb_path"])


if __name__ == "__main__":
    main()
