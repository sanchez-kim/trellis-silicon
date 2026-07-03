"""Repository path resolution for the src-layout package.

The vendored microsoft/TRELLIS.2 checkout and the generated ``stubs/`` directory
live at the *project root*, not inside the installed package. With an editable
install (``uv pip install -e .``) this module's ``__file__`` still resolves into
the source tree, so we locate the project root by walking up until we find the
directory that contains ``TRELLIS.2/``. ``TRELLIS2_ROOT`` overrides the TRELLIS.2
location directly (escape hatch for non-standard layouts / verification).

Pure ``os`` only — this is imported by ``core`` before torch is set up, so it
must stay import-light.
"""

import os


def trellis2_root() -> str:
    """Absolute path to the vendored TRELLIS.2 checkout."""
    env = os.environ.get("TRELLIS2_ROOT")
    if env:
        return os.path.abspath(env)
    return os.path.join(project_root(), "TRELLIS.2")


def project_root() -> str:
    """Directory that contains ``TRELLIS.2/`` (and the runtime ``stubs/``)."""
    env = os.environ.get("TRELLIS2_ROOT")
    if env:
        return os.path.dirname(os.path.abspath(env))
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(d, "TRELLIS.2")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            # No TRELLIS.2 found walking up; fall back to CWD so a clear
            # "TRELLIS.2 not found" error surfaces at the actual use site.
            return os.getcwd()
        d = parent


def stubs_dir() -> str:
    """Runtime stub-package directory, as a sibling of TRELLIS.2.

    Must stay consistent with the hard-coded ``../../../../stubs`` path baked
    into the fdg_vae patch, which resolves stubs relative to the TRELLIS.2 tree.
    """
    return os.path.join(os.path.dirname(trellis2_root()), "stubs")
