"""Mesh-decode compat patches: guard the CUDA-only cumesh/flex_gemm imports, run
decode-time hole filling on the Metal cumesh port, and force the pure-Python
dual-grid mesh extraction (the Metal o_voxel.convert segfaults on decoder output)."""

import os

from .common import TRELLIS_ROOT, read_file, write_file


def patch_mesh_base():
    """Guard cumesh/flex_gemm imports and skip the in-place remove_faces /
    simplify ops (the 512/1024 image-to-3D pipelines never call them; the GLB
    face budget is applied later by postprocess.to_glb). fill_holes, which
    decode_latent does call, is handled by patch_mesh_fill_holes.
    """
    path = os.path.join(TRELLIS_ROOT, "trellis2/representations/mesh/base.py")
    src = read_file(path)

    if "except (ImportError, RuntimeError)" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    # Guard imports — cumesh/flex_gemm may or may not be present
    src = src.replace(
        "import cumesh\nfrom flex_gemm.ops.grid_sample import grid_sample_3d",
        "try:\n"
        "    import cumesh\n"
        "except (ImportError, RuntimeError):\n"
        "    cumesh = None\n"
        "try:\n"
        "    from flex_gemm.ops.grid_sample import grid_sample_3d\n"
        "except (ImportError, RuntimeError):\n"
        "    def grid_sample_3d(*args, **kwargs):\n"
        '        raise RuntimeError("flex_gemm requires CUDA")',
    )

    # Unconditionally return from remove_faces
    src = src.replace(
        "    def remove_faces(self, face_mask: torch.Tensor):\n"
        "        vertices = self.vertices.cuda()\n"
        "        faces = self.faces.cuda()",
        "    def remove_faces(self, face_mask: torch.Tensor):\n"
        "        return\n"
        "        vertices = self.vertices.to(self.device)\n"
        "        faces = self.faces.to(self.device)",
    )

    # Unconditionally return from simplify
    src = src.replace(
        "    def simplify(self, target=1000000, verbose: bool=False, options: dict={}):\n"
        "        vertices = self.vertices.cuda()\n"
        "        faces = self.faces.cuda()",
        "    def simplify(self, target=1000000, verbose: bool=False, options: dict={}):\n"
        "        return\n"
        "        vertices = self.vertices.to(self.device)\n"
        "        faces = self.faces.to(self.device)",
    )

    write_file(path, src)


# Head of the fill_holes body before patching: upstream, and the version an
# earlier patcher left behind (an unconditional return).
_FILL_HOLES_UPSTREAM = (
    "    def fill_holes(self, max_hole_perimeter=3e-2):\n"
    "        vertices = self.vertices.cuda()\n"
    "        faces = self.faces.cuda()\n"
)
_FILL_HOLES_OLD_SKIP = (
    "    def fill_holes(self, max_hole_perimeter=3e-2):\n"
    "        return  # Skip — Metal cumesh segfaults on large decode meshes\n"
    "        vertices = self.vertices.to(self.device)\n"
    "        faces = self.faces.to(self.device)\n"
)
_FILL_HOLES_PREV_PATCHED = (
    "    def fill_holes(self, max_hole_perimeter=3e-2):\n"
    "        # trellis-silicon: run on CPU tensors (the Metal cumesh port reads\n"
    "        # unified memory directly). DECODE_FILL_HOLES=0 skips this step.\n"
    "        import os\n"
    '        if cumesh is None or os.environ.get("DECODE_FILL_HOLES", "1") == "0":\n'
    "            return\n"
    "        vertices = self.vertices.cpu()\n"
    "        faces = self.faces.cpu()\n"
)
# The wrapper checks the switches and holds the autorelease pool; the original
# body below it becomes _fill_holes_metal, so the pool covers every Metal call.
_FILL_HOLES_PATCHED = (
    "    def fill_holes(self, max_hole_perimeter=3e-2):\n"
    "        # trellis-silicon: run on CPU tensors (the Metal cumesh port reads\n"
    "        # unified memory directly). DECODE_FILL_HOLES=0 skips this step.\n"
    "        from trellis_silicon.envflags import env_enabled\n"
    "        from trellis_silicon.postprocess import _autorelease_pool\n"
    "        if cumesh is None or self.faces.shape[0] == 0:\n"
    "            return\n"
    '        if not env_enabled("DECODE_FILL_HOLES"):\n'
    "            return\n"
    "        with _autorelease_pool():\n"
    "            self._fill_holes_metal(max_hole_perimeter)\n"
    "\n"
    "    def _fill_holes_metal(self, max_hole_perimeter=3e-2):\n"
    "        vertices = self.vertices.cpu()\n"
    "        faces = self.faces.cpu()\n"
)


def patch_mesh_fill_holes():
    """Run decode-time hole filling (upstream decode_latent calls
    Mesh.fill_holes on every decoded mesh) through the Metal cumesh port.

    Earlier versions of this patcher made fill_holes return immediately,
    because the Metal port was reported to segfault on decoder-sized meshes.
    That does not reproduce with CPU tensors: on 1.2M-5.3M-face decoder
    meshes it runs in 0.2-0.8s and is bit-identical across runs. Applies to
    a fresh upstream file and migrates a file patched by the old skip.
    """
    path = os.path.join(TRELLIS_ROOT, "trellis2/representations/mesh/base.py")
    src = read_file(path)

    if "_fill_holes_metal" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)} (fill_holes)")
        return
    if _FILL_HOLES_PREV_PATCHED in src:
        src = src.replace(_FILL_HOLES_PREV_PATCHED, _FILL_HOLES_PATCHED)
    elif _FILL_HOLES_OLD_SKIP in src:
        src = src.replace(_FILL_HOLES_OLD_SKIP, _FILL_HOLES_PATCHED)
    elif _FILL_HOLES_UPSTREAM in src:
        src = src.replace(_FILL_HOLES_UPSTREAM, _FILL_HOLES_PATCHED)
    else:
        raise RuntimeError(f"fill_holes patch target not found in {path}")
    write_file(path, src)


def patch_fdg_vae():
    """Force our pure-Python flexible_dual_grid_to_mesh over any installed
    o_voxel. The Metal-port o_voxel.convert segfaults on decoder output even
    when it imports cleanly, so we always prefer our stub implementation.
    """
    path = os.path.join(TRELLIS_ROOT, "trellis2/models/sc_vaes/fdg_vae.py")
    src = read_file(path)

    if "o_voxel_override_convert" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    src = src.replace(
        "from o_voxel.convert import flexible_dual_grid_to_mesh\n",
        "# Force pure-Python mesh extraction — real o_voxel.convert (CUDA or Metal port)\n"
        "# segfaults on decoder output. Import our stub version explicitly.\n"
        "# stubs/ is appended (not prepended) so a pip-installed o_voxel still wins\n"
        "# for other submodules like o_voxel.postprocess.\n"
        "import sys, os\n"
        "_stubs = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'stubs')\n"
        "if _stubs not in sys.path:\n"
        "    sys.path.append(_stubs)\n"
        "try:\n"
        "    from o_voxel_override_convert import flexible_dual_grid_to_mesh\n"
        "except ImportError:\n"
        "    try:\n"
        "        from o_voxel.convert import flexible_dual_grid_to_mesh\n"
        "    except (ImportError, RuntimeError):\n"
        "        def flexible_dual_grid_to_mesh(*args, **kwargs):\n"
        '            raise RuntimeError("flexible_dual_grid_to_mesh unavailable")\n',
    )
    write_file(path, src)
