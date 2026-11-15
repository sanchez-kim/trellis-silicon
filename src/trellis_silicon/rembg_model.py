"""Background-remover choice for TRELLIS.2's rembg step (stdlib only).

The TRELLIS.2-4B ``pipeline.json`` builds its remover from
``briaai/RMBG-2.0`` (CC BY-NC 4.0, gated). This port loads the original
BiRefNet (``ZhengPeng7/BiRefNet``, MIT, not gated) instead, at a pinned
revision: the remote code runs with ``trust_remote_code=True``, so an
unpinned ``main`` would run whatever the repo holds next.

Its weights are stored fp16. The pipeline feeds the remover an fp32 image
tensor, which fails against fp16 weights ("Input type (float) and bias type
(c10::Half) should be the same"), so the model is cast to fp32 on load.
RMBG-2.0 already loads as fp32; the cast is a no-op for it.

``REMBG_MODEL=<hf repo id>`` picks another remover (for example
``REMBG_MODEL=briaai/RMBG-2.0`` with a BRIA license). Such a repo loads at
its ``main`` revision. patches/rembg.py makes TRELLIS.2's ``BiRefNet`` class
call :func:`load_rembg_model`.
"""

import os
import sys

DEFAULT_MODEL = "ZhengPeng7/BiRefNet"
# Full commit sha on huggingface.co/ZhengPeng7/BiRefNet (2026-02-04).
DEFAULT_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"


def rembg_source(environ=None):
    """Return ``(repo_id, revision)`` for the remover.

    ``REMBG_MODEL`` unset, empty, or equal to the default -> the pinned
    BiRefNet. Any other value -> that repo, ``revision=None`` (its main).
    """
    environ = os.environ if environ is None else environ
    requested = environ.get("REMBG_MODEL", "").strip()
    if not requested or requested == DEFAULT_MODEL:
        return DEFAULT_MODEL, DEFAULT_REVISION
    return requested, None


def load_rembg_model(auto_cls, requested=None, environ=None, stream=None):
    """Load the remover with ``auto_cls.from_pretrained`` and cast it to fp32.

    ``requested`` is the model name the TRELLIS.2 pipeline asked for; it is
    only reported. Prints one ``[REMBG]`` line naming the model that loads.
    """
    stream = sys.stdout if stream is None else stream
    name, revision = rembg_source(environ)
    note = f" @ {revision[:12]}" if revision else " @ main"
    if requested and requested != name:
        note += f"; pipeline.json asked for {requested}"
    print(f"[REMBG] Background remover: {name}{note} (fp32)", file=stream, flush=True)
    model = auto_cls.from_pretrained(
        name, trust_remote_code=True, revision=revision, code_revision=revision
    )
    return model.float()
