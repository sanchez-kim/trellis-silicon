"""Shared helpers and paths for the TRELLIS.2 source patchers.

Each patcher module rewrites specific files in the vendored TRELLIS.2 checkout
in place, guarded by a marker string so re-running is idempotent. The patched
output is byte-identical to what the original monolithic ``mps_compat.py``
produced — these modules are a reorganization, not a rewrite.
"""

import os

from .. import _paths

# Resolved once at import; set TRELLIS2_ROOT before running the patcher to
# target a non-standard TRELLIS.2 location (e.g. verification clones).
TRELLIS_ROOT = _paths.trellis2_root()
STUBS_DIR = _paths.stubs_dir()
# The pure-PyTorch/KDTree backends ship inside the package (this file is at
# trellis_silicon/patches/common.py, so two dirs up + /backends).
BACKENDS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backends")


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)
    print(f"  Patched: {os.path.relpath(path, TRELLIS_ROOT)}")
