"""Background-remover patch: load the remover trellis_silicon.rembg_model picks
(MIT BiRefNet at a pinned revision, fp32) instead of the model_name that
TRELLIS.2-4B's pipeline.json passes (briaai/RMBG-2.0, CC BY-NC 4.0, gated)."""

import os

from .common import TRELLIS_ROOT, read_file, write_file

# The load in BiRefNet.__init__. patch_birefnet (device.py) leaves it alone,
# so the same text is the target in a fresh upstream file and in one that an
# earlier patcher run already rewrote.
_INIT_LOAD_UPSTREAM = (
    "        self.model = AutoModelForImageSegmentation.from_pretrained(\n"
    "            model_name, trust_remote_code=True\n"
    "        )\n"
)
_INIT_LOAD_PATCHED = (
    "        # trellis-silicon: the remover comes from trellis_silicon.rembg_model\n"
    "        # (MIT BiRefNet at a pinned revision, fp32; REMBG_MODEL overrides),\n"
    "        # not from model_name. pipeline.json names briaai/RMBG-2.0 (CC BY-NC).\n"
    "        from trellis_silicon.rembg_model import load_rembg_model\n"
    "        self.model = load_rembg_model(AutoModelForImageSegmentation, model_name)\n"
)


def patch_rembg_model():
    """Route BiRefNet.__init__'s model load through rembg_model.load_rembg_model.

    The override sits in the remover class rather than in the pipeline args:
    every path that builds the remover goes through this one constructor, and
    the pipeline's own from_pretrained builds it before any caller could swap
    it, so the gated RMBG-2.0 download never starts.
    """
    path = os.path.join(TRELLIS_ROOT, "trellis2/pipelines/rembg/BiRefNet.py")
    src = read_file(path)

    if "trellis_silicon.rembg_model" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)} (rembg model)")
        return
    if _INIT_LOAD_UPSTREAM not in src:
        raise RuntimeError(f"rembg model patch target not found in {path}")
    src = src.replace(_INIT_LOAD_UPSTREAM, _INIT_LOAD_PATCHED, 1)
    write_file(path, src)
