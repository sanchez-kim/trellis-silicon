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
from .envflags import env_enabled

# Set up backends before any TRELLIS imports. Use setdefault so the caller
# can override from the environment. Default conv backend is flex_gemm since
# Pedro Augusto's mtlgemm fix (zero-copy on MPS, fp16/bf16 native); fall
# back to conv_none if flex_gemm isn't importable (see convbackend.py).
# MPS fallback MUST be set before torch is imported anywhere (torch is
# imported below, right after these defaults). Without this, segment_reduce and a few other
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
# Sparse attention (shape/tex SLat flow models' padded self+cross attn) also
# defaults to naive. At the shape/tex SLat shapes (self-attn B=2/1, S=1591;
# cross-attn kv=1029; H=12, D=128, bf16) naive measures ~4.0-4.1x faster than
# SDPA — an even bigger win than the dense case. Padding is zero-waste for
# every real call site here (CFG-batched self/cross-attn always duplicates
# the same sparse coords across the batch, so seqlens are always uniform),
# so the naive path's lack of an explicit attn_mask is exactly as correct as
# the existing sdpa path, not less. patches/attention.py adds a real separate
# 'naive' branch to the sparse dispatch (previously it aliased 'sdpa').
# Escape hatch: SPARSE_ATTN_BACKEND=sdpa restores the fused path.
os.environ.setdefault("SPARSE_ATTN_BACKEND", "naive")

# torch is imported here, after the env defaults above and before flex_gemm.
# flex_gemm's _C extension links @rpath/libc10.dylib, and mtlgemm's setup.py
# bakes the build venv's absolute torch/lib path in as its only torch rpath.
# Imported first, a build whose venv has moved (e.g. trellis-mac ->
# trellis-silicon) fails to load, and the old code then fell back to the slow
# conv_none backend without a word. With torch loaded, libc10 is already in the
# process and the extension resolves it whatever its rpath says.
import torch

from .convbackend import select_sparse_conv_backend

# Escape hatch: SPARSE_CONV_BACKEND=none (or any explicit value) skips the
# probe. If flex_gemm cannot be imported, one line on stderr says so and why.
select_sparse_conv_backend()

# Add paths. stubs/ is appended (not prepended) so a pip-installed o_voxel
# wins over our package stub — the flat override module o_voxel_override_convert
# is still discoverable either way because it doesn't collide with any package.
# TRELLIS.2/ and stubs/ live at the project root; _paths resolves them from the
# installed package location (walk up to the dir containing TRELLIS.2/, or the
# TRELLIS2_ROOT override).
sys.path.insert(0, _paths.trellis2_root())
sys.path.append(_paths.stubs_dir())

# Warn when `git pull` brought a newer patch but trellis-silicon-patch was not
# re-run (issue #6). The patches package imports only stdlib and _paths, so
# this import is cheap and does not need torch.
from .patches.attention import warn_if_sparse_attention_patch_outdated

warn_if_sparse_attention_patch_outdated(_paths.trellis2_root())

import time
import numpy as np
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


def color_reference(pipeline, image):
    """RGBA foreground of the input image for the optional colour match: the
    image's own alpha when it has one, else the pipeline's background remover
    (the same model and call preprocess_image uses). Runs after generation,
    so it cannot change the geometry."""
    rgba = image.convert("RGBA")
    if image.mode == "RGBA" and rgba.getextrema()[3][0] < 255:
        return rgba
    scale = min(1, 1024 / max(image.size))  # preprocess_image's input cap
    if scale < 1:
        size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(size, PILImage.Resampling.LANCZOS)
    if pipeline.low_vram:
        pipeline.rembg_model.to(pipeline.device)
    try:
        return pipeline.rembg_model(image.convert("RGB")).convert("RGBA")
    finally:
        if pipeline.low_vram:
            pipeline.rembg_model.cpu()


def generate_glb(
    pipeline,
    image,
    *,
    seed,
    pipeline_type,
    texture_size,
    no_texture=False,
    output_base,
    steps=None,
    color_match=None,
):
    """Run the pipeline and bake a textured GLB. Shared by CLI and UI.

    color_match: move the baked base colour toward the input foreground
    (postprocess.color_match_params). None reads TEXTURE_COLOR_MATCH (default
    off). Only the default remesh bake supports it.

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
                # Without flex_gemm's grid_sample the fork falls back to a
                # dense F.grid_sample whose output layout the bake can't use
                # and which darkens the texture. Swap in a sampler with the
                # upstream flex_gemm semantics (postprocess.sparse_grid_sample_3d).
                from . import postprocess as _pp

                o_voxel.postprocess._grid_sample_3d = _pp.sparse_grid_sample_3d
        except (ImportError, AttributeError):
            use_metal = False

        glb_path = f"{output_base}.glb"
        t_bake = time.time()

        # Face budget of the exported GLB (to_glb's decimation_target). Raise via
        # BAKE_MAX_FACES to let more geometry survive.
        bake_max_faces = int(os.environ.get("BAKE_MAX_FACES", "200000"))
        # Default: the upstream TRELLIS.2 GLB postprocess (postprocess.to_glb —
        # full-resolution decoder mesh in, narrow-band dual-contouring remesh,
        # texels baked from the full-resolution surface, OPAQUE material).
        # GLB_REMESH=0 restores the legacy path: fast_simplification pre-pass to
        # the face budget, then the o_voxel fork's non-remesh cleanup branch
        # (open boundaries, non-manifold edges and floaters survive into the GLB).
        remesh = env_enabled("GLB_REMESH")
        if color_match is None:
            color_match = env_enabled("TEXTURE_COLOR_MATCH", default=False)
        if color_match and not (use_metal and remesh):
            print("\n  Colour match needs the default remesh bake; skipped.")
            color_match = False

        # Computed outside the bake try and before the MPS cache is emptied:
        # a failure here skips the match, never the Metal bake or the run.
        ref = None
        if color_match:
            try:
                ref = np.asarray(color_reference(pipeline, image))
            except Exception as e:
                print(f"\n  Warning: colour match skipped: {e}")

        if use_metal:
            try:
                print(f"\nBaking PBR textures via Metal ({tex_size}x{tex_size})...")
                import o_voxel

                # Keep all tensors on CPU — the Metal kernels read unified memory
                # directly, and to_glb mixes the mesh with CPU-side AABB tensors.
                if remesh:
                    from .postprocess import to_glb

                    # Hand the decoder's cached MPS blocks back before the
                    # remesh allocates (upstream empties the CUDA cache here).
                    torch.mps.empty_cache()
                    glb = to_glb(
                        vertices=mesh_out.vertices.cpu(),
                        faces=mesh_out.faces.cpu(),
                        attr_volume=mesh_out.attrs.cpu(),
                        coords=mesh_out.coords.cpu(),
                        attr_layout=mesh_out.layout,
                        voxel_size=mesh_out.voxel_size,
                        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        decimation_target=bake_max_faces,
                        texture_size=tex_size,
                        remesh_band=1,
                        remesh_project=0,
                        color_reference=ref,
                        verbose=True,
                    )
                else:
                    # Legacy: pre-simplify so the fork's non-remesh branch (and
                    # its BVH) only sees ~target_faces faces. The texture is then
                    # baked from this simplified surface, not the decoder surface.
                    import fast_simplification

                    verts_np = mesh_out.vertices.cpu().numpy()
                    faces_np = mesh_out.faces.cpu().numpy()
                    target_faces = min(bake_max_faces, len(faces_np))
                    bake_verts_t = mesh_out.vertices
                    bake_faces_t = mesh_out.faces
                    if len(faces_np) > target_faces:
                        ratio = 1.0 - (target_faces / len(faces_np))
                        print(f"  Simplifying mesh: {len(faces_np):,} -> ~{target_faces:,} faces")
                        simp_verts, simp_faces = fast_simplification.simplify(
                            verts_np, faces_np, ratio
                        )
                        bake_verts_t = torch.from_numpy(simp_verts).float()
                        bake_faces_t = torch.from_numpy(simp_faces.astype("int32"))

                    glb = o_voxel.postprocess.to_glb(
                        vertices=bake_verts_t.cpu(),
                        faces=bake_faces_t.cpu(),
                        attr_volume=mesh_out.attrs.cpu(),
                        coords=mesh_out.coords.cpu(),
                        attr_layout=mesh_out.layout,
                        voxel_size=mesh_out.voxel_size,
                        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        decimation_target=target_faces,
                        texture_size=tex_size,
                        verbose=True,
                    )
                    # Upstream to_glb always writes OPAQUE. The fork switches to
                    # BLEND when any texel has alpha < 250; viewers then disable
                    # depth writes and back faces show through the model.
                    glb.visual.material.alphaMode = "OPAQUE"
                glb.export(glb_path)
                print(f"  Saved: {glb_path}")
                result["backend"] = "metal"
            except (RuntimeError, AssertionError) as e:
                print(f"\n  Metal bake failed: {e}")
                print("  Falling back to KDTree texture baker...")
                use_metal = False

        if not use_metal:
            print(f"\nBaking PBR textures via KDTree ({tex_size}x{tex_size})...")
            if color_match:
                print("  Colour match skipped: the KDTree baker does not support it.")
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
