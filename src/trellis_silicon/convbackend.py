"""Pick the sparse-convolution backend (stdlib only, safe to import early).

TRELLIS.2 reads ``SPARSE_CONV_BACKEND`` once, when ``trellis2.modules.sparse``
is first imported, so the choice has to be made before that. The Metal
``flex_gemm`` backend (mtlgemm) is the default; the pure-PyTorch ``none``
backend is the fallback.

The caller MUST import torch before calling :func:`select_sparse_conv_backend`.
flex_gemm's ``_C`` extension links ``@rpath/libc10.dylib`` and friends, and its
only torch rpath is the absolute path of the venv it was built in. Importing
torch first loads those dylibs into the process, so the extension resolves them
whatever that baked-in path says. Imported before torch, a build whose venv has
since moved fails with ``Library not loaded: @rpath/libc10.dylib``.
"""

import importlib
import os
import sys

DEFAULT_BACKEND = "flex_gemm"
FALLBACK_BACKEND = "none"
VALID_BACKENDS = frozenset({"none", "spconv", "torchsparse", "flex_gemm"})


def _first_line(exc):
    text = str(exc).strip().splitlines()
    msg = text[0].strip() if text else ""
    if len(msg) > 200:
        msg = msg[:197] + "..."
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def fallback_notice(exc):
    """The one-line message printed when flex_gemm cannot be imported."""
    return (
        f"[trellis-silicon] flex_gemm unavailable ({_first_line(exc)}); "
        f"sparse convolution uses the slow pure-PyTorch fallback "
        f"(SPARSE_CONV_BACKEND={FALLBACK_BACKEND}). Re-run setup.sh to rebuild mtlgemm."
    )


def select_sparse_conv_backend(environ=None, import_module=None, stream=None):
    """Set ``SPARSE_CONV_BACKEND`` in ``environ`` and return the backend name.

    - Non-empty valid value (``none``, ``spconv``, ``torchsparse``): left alone,
      flex_gemm is not imported. This is the escape hatch
      (``SPARSE_CONV_BACKEND=none``). Matching is case-sensitive, as in trellis2.
    - Explicit ``flex_gemm``: probed. If the probe fails, raise ImportError with
      the fallback notice (fail fast instead of a raw dlopen error later).
    - Unset, empty, or invalid value: an invalid value prints one warning line
      that names the valid values. Then flex_gemm is probed. Success ->
      ``flex_gemm``. Failure -> ``none`` and one line on ``stream`` (stderr)
      that names the error.

    ImportError covers a missing package and a dylib that fails to load.
    RuntimeError covers a metallib the OS cannot load (flex_gemm ships an MSL
    4.0 metallib, which needs macOS 26+). OSError covers other loader errors.
    """
    environ = os.environ if environ is None else environ
    import_module = importlib.import_module if import_module is None else import_module
    stream = sys.stderr if stream is None else stream

    requested = environ.get("SPARSE_CONV_BACKEND", "").strip()
    if requested in VALID_BACKENDS and requested != DEFAULT_BACKEND:
        environ["SPARSE_CONV_BACKEND"] = requested
        return requested
    if requested and requested not in VALID_BACKENDS:
        print(
            f"[trellis-silicon] invalid SPARSE_CONV_BACKEND={requested!r}; "
            f"valid values: {', '.join(sorted(VALID_BACKENDS))}. Probing as if unset.",
            file=stream,
            flush=True,
        )
        requested = ""
    try:
        import_module("flex_gemm")
    except (ImportError, RuntimeError, OSError) as exc:
        if requested == DEFAULT_BACKEND:
            raise ImportError(fallback_notice(exc)) from exc
        environ["SPARSE_CONV_BACKEND"] = FALLBACK_BACKEND
        print(fallback_notice(exc), file=stream, flush=True)
        return FALLBACK_BACKEND
    environ["SPARSE_CONV_BACKEND"] = DEFAULT_BACKEND
    return DEFAULT_BACKEND
