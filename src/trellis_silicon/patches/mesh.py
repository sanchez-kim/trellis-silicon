"""Mesh-decode compat patches: guard the CUDA-only cumesh/flex_gemm imports and
force the pure-Python dual-grid mesh extraction (the Metal ports segfault on the
large decode meshes)."""

import os

from .common import TRELLIS_ROOT, read_file, write_file


def patch_mesh_base():
    """Guard cumesh/flex_gemm imports and unconditionally skip in-place mesh
    ops. TRELLIS.2 calls fill_holes/remove_faces/simplify during decode on the
    full 400K-vertex mesh; the Metal port of cumesh segfaults on inputs that
    large, so we skip these decode-time ops entirely. Post-decode mesh
    simplification happens later via fast_simplification before texture bake.
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

    # Unconditionally return from fill_holes (Metal cumesh segfaults on large meshes)
    src = src.replace(
        "    def fill_holes(self, max_hole_perimeter=3e-2):\n"
        "        vertices = self.vertices.cuda()\n"
        "        faces = self.faces.cuda()",
        "    def fill_holes(self, max_hole_perimeter=3e-2):\n"
        "        return  # Skip — Metal cumesh segfaults on large decode meshes\n"
        "        vertices = self.vertices.to(self.device)\n"
        "        faces = self.faces.to(self.device)",
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
