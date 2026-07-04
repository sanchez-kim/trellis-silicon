"""
Shared core for TRELLIS.2 image-to-3D generation on Apple Silicon.

Both cli.py (CLI) and webui.py (Gradio UI) import this module so the two
front-ends run through exactly one code path. IMPORTANT: importing this module
performs the backend/environment setup below, and that setup MUST run before
torch or trellis are imported anywhere — the same constraint the CLI had
when this code lived at its top. Keep the os.environ/sys.path block first.
"""

import sys
import os

from . import _paths

# Set up backends before any TRELLIS imports. Use setdefault so the caller
# can override from the environment. Default conv backend is flex_gemm since
# Pedro Naugusto's mtlgemm fix (zero-copy on MPS, fp16/bf16 native); fall
# back to conv_none if flex_gemm isn't importable for some reason.
# MPS fallback MUST be set before torch is imported anywhere (including
# transitively via flex_gemm). Without this, segment_reduce and a few other
# ops crash instead of falling back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Dense attention (structure phase only — sparse_structure_flow's self+cross
# attn) defaults to the naive unfused path. At the structure shapes on MPS
# (B=2 CFG, H=12, D=128, bf16; self S=4096, cross q=4096/kv=1029) a naive
# softmax(q@kT*scale)@v measures ~3.3x (self) / ~3.9x (cross) faster than
# torch's fused SDPA, which is slow on MPS at these sizes. Numerics stay at
# bf16 noise (~4e-3 vs fp32). The naive path materializes an ~805MB bf16 S×S
# score tensor per self-attn call — fine on unified memory here. The 'naive'
# and 'sdpa' backends both ship in the pinned TRELLIS.2 dense dispatch
# (trellis2/modules/attention/full_attn.py), so no source patch is needed.
# Escape hatch: ATTN_BACKEND=sdpa restores the fused path.
os.environ.setdefault("ATTN_BACKEND", "naive")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
try:
    import flex_gemm  # noqa: F401

    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
except (ImportError, RuntimeError):
    # ImportError: package not installed (SKIP_METAL=1 or install failed).
    # RuntimeError: metallib load failure — flex_gemm ships an MSL 4.0
    # metallib which only loads on macOS 26+. On older macOS the import
    # itself raises "Failed to load metallib ... language version 4.0
    # which is not supported on this OS". Fall back to conv_none either way.
    os.environ.setdefault("SPARSE_CONV_BACKEND", "none")

# Add paths. stubs/ is appended (not prepended) so a pip-installed o_voxel
# wins over our package stub — the flat override module o_voxel_override_convert
# is still discoverable either way because it doesn't collide with any package.
# TRELLIS.2/ and stubs/ live at the project root; _paths resolves them from the
# installed package location (walk up to the dir containing TRELLIS.2/, or the
# TRELLIS2_ROOT override).
sys.path.insert(0, _paths.trellis2_root())
sys.path.append(_paths.stubs_dir())

import time
import torch
from PIL import Image as PILImage


# Two known watchdog-corruption signatures raised out of pipeline.run:
#   IndexError: max(): Expected reduction dim 0 to have non-zero size
#     — empty SparseTensor in decode_latent's spatial_shape calc
#   AssertionError: BVH needs at least 8 triangles, got 0
#     — empty mesh propagating into o_voxel.postprocess.to_glb
WATCHDOG_SIGNATURES = (
    "non-zero size",
    "BVH needs at least 8 triangles",
)


class WatchdogEmptyMeshError(Exception):
    """The decoder produced an empty mesh — almost always the macOS GPU
    watchdog killing a long-running Metal kernel in the SLat decoder.

    Raised instead of exiting so each front-end decides how to surface it:
    the CLI prints str(self) and exits 2; the UI shows str(self) in its
    status box. str(self) is the full help message from watchdog_help_message().
    """


def watchdog_help_message():
    return (
        "\nERROR: The decoder produced an empty mesh.\n"
        "On Apple Silicon this is almost always the macOS GPU watchdog\n"
        "killing a long-running Metal kernel in the SLat decoder. The Metal\n"
        "error prints to stderr above (look for\n"
        "'kIOGPUCommandBufferCallbackErrorImpactingInteractivity') but does\n"
        "not raise a Python exception, so execution continues with empty\n"
        "tensors and crashes downstream.\n"
        "\n"
        "Workarounds, cheapest first:\n"
        "  1. Run headless — close the lid / unplug external displays and\n"
        "     re-run over SSH. The watchdog tightens with WindowServer load.\n"
        "  2. MTL_CAPTURE_ENABLED=1 trellis-silicon ...   (extends the\n"
        "     watchdog timeout as a side effect of Metal-debugger mode)\n"
        "  3. SPARSE_CONV_BACKEND=none trellis-silicon ... (slower path,\n"
        "     may not help if a single dispatch is the offender)\n"
        "\n"
        "Tracking issue: https://github.com/sanchez-kim/trellis-silicon/issues\n"
    )


def load_pipeline(pipeline_type, resident=False):
    """Load the TRELLIS.2 pipeline for pipeline_type and place it on MPS.

    Only the checkpoints this pipeline_type actually needs are loaded (for
    "512" this skips the two ~2.4GB *_1024 flow models). Returns the loaded,
    device-placed pipeline. Callers own their own timing/status prints.
    """
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    # Task A: only load the checkpoints this pipeline_type actually needs.
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        "microsoft/TRELLIS.2-4B",
        pipeline_type=pipeline_type,
    )
    # Keep upstream's low_vram shuffling as the default — measured faster here
    # than keeping weights resident (the CPU->MPS copies happen either way;
    # resident mode only saves the cheap return trips while adding ~9-11GB of
    # unified-memory pressure across the sampling loop). resident=True opts into
    # the all-on-MPS path for high-memory machines.
    pipeline.low_vram = not resident

    # Move to MPS. Under low_vram this is a no-op; with resident it makes
    # every submodel resident now.
    pipeline.to(torch.device("mps"))
    return pipeline


def generate_glb(
    pipeline, image, *, seed, pipeline_type, texture_size, no_texture=False, output_base, steps=None
):
    """Run the pipeline and bake a textured GLB. Shared by CLI and UI.

    Returns a dict with the output paths and stats:
        glb_path, basecolor_path (or None), verts, faces (numpy arrays for
        the CLI's optional OBJ export), vertices, triangles, gen_time,
        bake_time (or None), backend ("metal" | "kdtree" | "vertex_colors").

    Raises WatchdogEmptyMeshError when the decoder yields an empty mesh.
    """
    t0 = time.time()

    sampler_overrides = {"steps": steps} if steps else {}

    try:
        outputs = pipeline.run(
            image,
            seed=seed,
            pipeline_type=pipeline_type,
            sparse_structure_sampler_params=sampler_overrides,
            shape_slat_sampler_params=sampler_overrides,
            tex_slat_sampler_params=sampler_overrides,
        )
    except (IndexError, AssertionError) as e:
        msg = str(e)
        if any(sig in msg for sig in WATCHDOG_SIGNATURES):
            raise WatchdogEmptyMeshError(watchdog_help_message()) from e
        raise

    t_gen = time.time() - t0

    mesh_out = outputs[0] if isinstance(outputs, list) else outputs

    verts = mesh_out.vertices.cpu().numpy()
    faces = mesh_out.faces.cpu().numpy()
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        raise WatchdogEmptyMeshError(watchdog_help_message())
    print(f"\nMesh: {verts.shape[0]:,} vertices, {faces.shape[0]:,} triangles")
    print(f"Generation time: {t_gen:.1f}s")

    result = {
        "glb_path": None,
        "basecolor_path": None,
        "verts": verts,
        "faces": faces,
        "vertices": int(verts.shape[0]),
        "triangles": int(faces.shape[0]),
        "gen_time": t_gen,
        "bake_time": None,
        "backend": None,
    }

    # Check for voxel texture data
    has_voxels = hasattr(mesh_out, "attrs") and mesh_out.attrs is not None
    tex_size = texture_size

    if has_voxels and not no_texture:
        # Try Metal-accelerated bake via o_voxel + mtldiffrast if available.
        # Catch AttributeError too: our stubs/o_voxel/ stub has no .postprocess
        # submodule, so a shadowing stub package trips getattr, not import.
        try:
            import o_voxel.postprocess

            backend = getattr(o_voxel.postprocess, "_BACKEND", None)
            has_dr = getattr(o_voxel.postprocess, "_HAS_DR", False)
            use_metal = backend == "metal" and has_dr
            if use_metal and not getattr(o_voxel.postprocess, "_HAS_FLEX_GEMM", False):
                # o_voxel's _grid_sample_3d fallback returns [B*C, M] but the
                # bake consumes it as [M, C]. Patch it to transpose before the
                # reshape. We avoid installing flex_gemm itself because its
                # import slows the diffusion hot path ~10x on MPS.
                import torch.nn.functional as _F_gs

                def _gs3d_fix(feats, coords, shape, grid, mode="trilinear"):
                    B, C = shape[0], shape[1]
                    D, H, W = shape[2], shape[3], shape[4]
                    device = feats.device
                    dense_vol = torch.zeros(B, C, D, H, W, dtype=feats.dtype, device=device)
                    batch_idx = coords[:, 0].long()
                    cx = coords[:, 1].long()
                    cy = coords[:, 2].long()
                    cz = coords[:, 3].long()
                    dense_vol[batch_idx, :, cx, cy, cz] = feats
                    grid_norm = torch.stack(
                        [
                            grid[..., 2] / (W - 1) * 2 - 1,
                            grid[..., 1] / (H - 1) * 2 - 1,
                            grid[..., 0] / (D - 1) * 2 - 1,
                        ],
                        dim=-1,
                    ).reshape(B, 1, 1, -1, 3)
                    sampled = _F_gs.grid_sample(
                        dense_vol,
                        grid_norm,
                        mode="bilinear",
                        align_corners=True,
                        padding_mode="border",
                    )
                    M = grid.shape[1]
                    return sampled.reshape(B, C, M).permute(0, 2, 1).reshape(B * M, C)

                o_voxel.postprocess._grid_sample_3d = _gs3d_fix
        except (ImportError, AttributeError):
            use_metal = False

        glb_path = f"{output_base}.glb"
        t_bake = time.time()

        # Decimation cap before Metal BVH / xatlas. Default 200K keeps the Metal
        # builder stable; raise via BAKE_MAX_FACES to let more geometry survive.
        bake_max_faces = int(os.environ.get("BAKE_MAX_FACES", "200000"))

        if use_metal:
            try:
                print(f"\nBaking PBR textures via Metal ({tex_size}x{tex_size})...")
                import o_voxel

                # Pre-simplify mesh to avoid mtlbvh crash on large meshes.
                # Target ~200K faces — keeps detail, avoids Metal BVH issues.
                import fast_simplification

                verts_np = mesh_out.vertices.cpu().numpy()
                faces_np = mesh_out.faces.cpu().numpy()
                target_faces = min(bake_max_faces, len(faces_np))
                if len(faces_np) > target_faces:
                    ratio = 1.0 - (target_faces / len(faces_np))
                    print(f"  Simplifying mesh: {len(faces_np):,} -> ~{target_faces:,} faces")
                    simp_verts, simp_faces = fast_simplification.simplify(verts_np, faces_np, ratio)
                    simp_verts_t = torch.from_numpy(simp_verts).float().to(mesh_out.vertices.device)
                    simp_faces_t = torch.from_numpy(simp_faces.astype("int32")).to(
                        mesh_out.faces.device
                    )
                else:
                    simp_verts_t = mesh_out.vertices
                    simp_faces_t = mesh_out.faces

                # Move all mesh tensors to CPU — o_voxel.to_glb mixes device-neutral
                # AABB tensor with mesh tensors; keep everything on CPU to avoid mismatch.
                glb = o_voxel.postprocess.to_glb(
                    vertices=simp_verts_t.cpu(),
                    faces=simp_faces_t.cpu(),
                    attr_volume=mesh_out.attrs.cpu(),
                    coords=mesh_out.coords.cpu(),
                    attr_layout=mesh_out.layout,
                    voxel_size=mesh_out.voxel_size,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=target_faces,
                    texture_size=tex_size,
                    verbose=True,
                )
                glb.export(glb_path)
                print(f"  Saved: {glb_path}")
                result["backend"] = "metal"
            except RuntimeError as e:
                print(f"\n  Metal bake failed: {e}")
                print("  Falling back to KDTree texture baker...")
                use_metal = False

        if not use_metal:
            print(f"\nBaking PBR textures via KDTree ({tex_size}x{tex_size})...")
            from .backends.texture_baker import uv_unwrap, bake_texture, export_glb_with_texture

            voxel_coords = mesh_out.coords.cpu().float()
            voxel_attrs = mesh_out.attrs.cpu().float()
            origin = mesh_out.origin.cpu().float()
            vs = mesh_out.voxel_size

            # Simplify before UV unwrap — xatlas is very slow on 800K+ vertex meshes
            bake_verts, bake_faces = verts, faces
            target_faces = min(bake_max_faces, len(faces))
            if len(faces) > target_faces:
                try:
                    import fast_simplification

                    ratio = 1.0 - (target_faces / len(faces))
                    print(f"  Simplifying mesh: {len(faces):,} -> ~{target_faces:,} faces")
                    bake_verts, bake_faces = fast_simplification.simplify(verts, faces, ratio)
                except ImportError:
                    print(
                        "  Warning: fast_simplification not installed, UV unwrapping full mesh (slow)"
                    )

            print("  UV unwrapping with xatlas...")
            new_verts, new_faces, uvs, vmapping = uv_unwrap(bake_verts, bake_faces)
            print(f"  UV unwrap: {len(verts):,} -> {len(new_verts):,} vertices")

            base_color_img, mr_img, mask = bake_texture(
                new_verts,
                new_faces,
                uvs,
                voxel_coords.numpy(),
                voxel_attrs.numpy(),
                origin.numpy(),
                vs,
                texture_size=tex_size,
            )

            basecolor_path = f"{output_base}_basecolor.png"
            PILImage.fromarray(base_color_img).save(basecolor_path)
            export_glb_with_texture(new_verts, new_faces, uvs, base_color_img, mr_img, glb_path)
            print(f"  Saved: {glb_path}")
            result["basecolor_path"] = basecolor_path
            result["backend"] = "kdtree"

        t_bake_total = time.time() - t_bake
        print(f"  Bake time: {t_bake_total:.0f}s")
        result["glb_path"] = glb_path
        result["bake_time"] = t_bake_total
    else:
        # Fallback: vertex colors only
        print("\nExporting with vertex colors (use texture baking for PBR textures)...")
        import trimesh

        glb_path = f"{output_base}.glb"
        tm = trimesh.Trimesh(vertices=verts, faces=faces)
        tm.export(glb_path)
        print(f"Saved: {glb_path}")
        result["glb_path"] = glb_path
        result["backend"] = "vertex_colors"

    return result
