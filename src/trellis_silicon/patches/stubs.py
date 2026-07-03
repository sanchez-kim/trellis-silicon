"""Stub/backend installers: copy the pure-PyTorch sparse conv backend and the
pure-Python mesh extraction into place. These write into TRELLIS.2 and the
runtime stubs/ dir; they do not rewrite TRELLIS.2 source in place like the
patch_* functions."""

import os
import shutil

from .common import BACKENDS_DIR, STUBS_DIR, TRELLIS_ROOT


def install_conv_backend():
    """Copy the pure-PyTorch sparse convolution backend into place."""
    src = os.path.join(BACKENDS_DIR, "conv_none.py")
    dst = os.path.join(TRELLIS_ROOT, "trellis2/modules/sparse/conv/conv_none.py")

    if os.path.exists(dst):
        print("  Already installed: trellis2/modules/sparse/conv/conv_none.py")
        return

    shutil.copy2(src, dst)
    print("  Installed: trellis2/modules/sparse/conv/conv_none.py")


def install_mesh_extract():
    """Copy the pure-Python mesh extraction into the o_voxel stub and also as
    a flat override module. The flat module takes precedence over any
    Metal/CUDA o_voxel package that might be installed alongside us.
    """
    stubs_dir = STUBS_DIR
    ovoxel_dir = os.path.join(stubs_dir, "o_voxel")
    os.makedirs(ovoxel_dir, exist_ok=True)

    src = os.path.join(BACKENDS_DIR, "mesh_extract.py")

    # Flat override module — loaded before real o_voxel by fdg_vae patch
    flat_dst = os.path.join(stubs_dir, "o_voxel_override_convert.py")
    shutil.copy2(src, flat_dst)
    print("  Installed: stubs/o_voxel_override_convert.py")

    # Also the stub package for environments without any o_voxel install
    dst = os.path.join(ovoxel_dir, "convert.py")
    shutil.copy2(src, dst)
    print("  Installed: stubs/o_voxel/convert.py")

    # __init__.py
    with open(os.path.join(ovoxel_dir, "__init__.py"), "w") as f:
        f.write("pass\n")

    # io.py stub
    with open(os.path.join(ovoxel_dir, "io.py"), "w") as f:
        f.write(
            'def read(*args, **kwargs):\n    raise RuntimeError("o_voxel.io requires CUDA")\n\n'
        )
        f.write(
            'def write(*args, **kwargs):\n    raise RuntimeError("o_voxel.io requires CUDA")\n\n'
        )
        f.write(
            'def read_vxz(*args, **kwargs):\n    raise RuntimeError("o_voxel.io requires CUDA")\n'
        )

    # rasterize.py stub
    with open(os.path.join(ovoxel_dir, "rasterize.py"), "w") as f:
        f.write("class VoxelRenderer:\n    def __init__(self, *args, **kwargs):\n")
        f.write('        raise RuntimeError("o_voxel.rasterize requires CUDA")\n')
