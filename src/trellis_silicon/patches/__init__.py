"""Apply all MPS compatibility patches to a fresh TRELLIS.2 clone.

Rewrites source files in the vendored ``TRELLIS.2/`` checkout in place to
replace CUDA-only code paths with device-agnostic (MPS) alternatives, and
installs the pure-PyTorch/KDTree fallback backends and CUDA-library stubs.

Run once after cloning TRELLIS.2::

    python -m trellis_silicon.patches      # or: trellis-silicon-patch

Every patch is guarded by a marker string, so re-running is idempotent. Set
``TRELLIS2_ROOT`` to target a non-standard TRELLIS.2 location.
"""

import os

from .attention import patch_sparse_attention, patch_sparse_config
from .cfg_batch import patch_cfg_batching
from .common import TRELLIS_ROOT
from .device import (
    patch_birefnet,
    patch_image_feature_extractor,
    patch_pipeline,
    patch_pipeline_base,
)
from .loading import (
    patch_base_models_to_load,
    patch_pipeline_conditional_load,
    patch_skip_init_on_load,
)
from .mesh import patch_fdg_vae, patch_mesh_base
from .stubs import install_conv_backend, install_mesh_extract

__all__ = ["main"]


def main():
    print("Applying MPS compatibility patches to TRELLIS.2...")
    print(f"  TRELLIS root: {TRELLIS_ROOT}")
    print()

    if not os.path.isdir(TRELLIS_ROOT):
        print(f"Error: TRELLIS.2 not found at {TRELLIS_ROOT}")
        print("Run setup.sh first to clone the repository.")
        return False

    patch_sparse_config()
    patch_sparse_attention()
    patch_image_feature_extractor()
    patch_birefnet()
    patch_mesh_base()
    patch_fdg_vae()
    patch_pipeline()
    patch_pipeline_base()
    patch_base_models_to_load()
    patch_pipeline_conditional_load()
    patch_cfg_batching()
    patch_skip_init_on_load()
    install_conv_backend()
    install_mesh_extract()

    print()
    print("All patches applied.")
    return True
