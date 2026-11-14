#!/usr/bin/env bash
#
# Set up TRELLIS.2 for Apple Silicon.
# Creates a venv, installs dependencies, clones the repo, and applies patches.
#

set -euo pipefail
cd "$(dirname "$0")"

echo "=== TRELLIS.2 for Apple Silicon — Setup ==="
echo

# ---------------------------------------------------------------------------
# Pre-clone Git dependencies so that all network I/O happens up front.
# If a clone fails you can retry just this section without re-running the
# whole script — the "if [ ! -d …]" guards make it idempotent.
# ---------------------------------------------------------------------------
DEPS_DIR="deps"
mkdir -p "$DEPS_DIR"

clone_dep() {
    local url="$1" dir="$2" ref="${3:-}"
    if [ ! -d "$DEPS_DIR/$dir" ]; then
        echo "Cloning $dir ..."
        git clone --depth 1 ${ref:+--branch "$ref"} "$url" "$DEPS_DIR/$dir"
    else
        echo "  $dir already cloned — skipping"
    fi
}

# utils3d needs a specific commit, so clone without --depth and checkout
if [ ! -d "$DEPS_DIR/utils3d" ]; then
    echo "Cloning utils3d ..."
    git clone https://github.com/EasternJournalist/utils3d.git "$DEPS_DIR/utils3d"
    git -C "$DEPS_DIR/utils3d" checkout 9a4eb15e4021b67b12c460c7057d642626897ec8
else
    echo "  utils3d already cloned — skipping"
fi

# Clone a dependency at an exact commit. An existing clone at the pinned commit
# is left alone; one at a different commit stops the script (rebuild: rm -rf deps).
clone_dep_at() {
    local url="$1" dir="$2" commit="$3"
    if [ ! -d "$DEPS_DIR/$dir" ]; then
        echo "Cloning $dir (pinned @ ${commit:0:12}) ..."
        git init -q "$DEPS_DIR/$dir"
        git -C "$DEPS_DIR/$dir" remote add origin "$url"
        git -C "$DEPS_DIR/$dir" fetch -q --depth 1 origin "$commit"
        git -C "$DEPS_DIR/$dir" checkout -q FETCH_HEAD
    elif [ "$(git -C "$DEPS_DIR/$dir" rev-parse HEAD 2>/dev/null)" != "$commit" ]; then
        echo "ERROR: $DEPS_DIR/$dir is not at the pinned commit ${commit:0:12}." >&2
        echo "       Run: rm -rf deps && bash setup.sh" >&2
        exit 1
    else
        echo "  $dir already cloned at the pinned commit — skipping"
    fi
}

# mtlbvh/mtlmesh are pinned: GLB export needs fixes that older commits lack.
#   mtlbvh  fe6660b includes bc0a534 — the BVH traversal stack (24 slots)
#           can silently drop subtrees once a mesh exceeds ~131K triangles
#           (derived safe bound; first measured failure at 1.18M), so
#           closest-triangle queries can miss the nearest triangle (holey remesh,
#           texels baked from the wrong surface).
#   mtlmesh 5e67493 includes 16a7d0a — bounded hash probe loops and a fix for
#           spurious weak-CAS insert failures.
MTLBVH_COMMIT="${MTLBVH_COMMIT:-fe6660bdb7de5a12ea00bd96df2e21bdcbdd40b1}"
MTLMESH_COMMIT="${MTLMESH_COMMIT:-5e6749356ba39b15489c99487f55840d49923da7}"
clone_dep_at https://github.com/pedronaugusto/mtlbvh.git    mtlbvh  "$MTLBVH_COMMIT"
clone_dep https://github.com/pedronaugusto/mtldiffrast.git   mtldiffrast
clone_dep_at https://github.com/pedronaugusto/mtlmesh.git   mtlmesh "$MTLMESH_COMMIT"
clone_dep https://github.com/pedronaugusto/mtlgemm.git       mtlgemm
clone_dep https://github.com/pedronaugusto/trellis2-apple.git trellis2-apple

# TRELLIS.2 lives at the project root (the patcher and package expect it there).
# Pinned to the commit the patches and the verification gate were validated
# against — upstream moves and can break the patch markers. Override with
# TRELLIS2_COMMIT=<sha> (or TRELLIS2_COMMIT=HEAD for an unpinned clone) at
# your own risk.
TRELLIS2_COMMIT="${TRELLIS2_COMMIT:-75fbf0183001ed9876c8dbb35de6b68552ee08bd}"
if [ ! -d "TRELLIS.2" ]; then
    if [ "$TRELLIS2_COMMIT" = "HEAD" ]; then
        echo "Cloning TRELLIS.2 (unpinned HEAD) ..."
        git clone --depth 1 https://github.com/microsoft/TRELLIS.2.git TRELLIS.2
    else
        echo "Cloning TRELLIS.2 (pinned @ ${TRELLIS2_COMMIT:0:12}) ..."
        git init -q TRELLIS.2
        git -C TRELLIS.2 remote add origin https://github.com/microsoft/TRELLIS.2.git
        git -C TRELLIS.2 fetch -q --depth 1 origin "$TRELLIS2_COMMIT"
        git -C TRELLIS.2 checkout -q FETCH_HEAD
    fi
else
    echo "  TRELLIS.2 already cloned — skipping"
fi

echo

# Check Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Warning: This project requires Apple Silicon (M1 or later)."
fi

# Create venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    if command -v uv &>/dev/null; then
        uv venv .venv --python python3.11
    else
        python3 -m venv .venv
    fi
fi

source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
if command -v uv &>/dev/null; then
    PIP="uv pip install"
else
    PIP="python3 -m pip install"
fi
# Install the locked dependency set first (the exact versions the verification
# gate baseline was established with — see requirements.lock), then the package
# itself without re-resolving deps. This registers the trellis-silicon /
# trellis-silicon-web / trellis-silicon-patch console scripts.
$PIP -r requirements.lock
$PIP -e . --no-deps
$PIP "$DEPS_DIR/utils3d"

METAL_FAILS=""

# Optional Metal acceleration for texture baking.
# Requires Xcode Metal Toolchain:
#     xcodebuild -downloadComponent MetalToolchain
# Without these, we fall back to a pure-Python KDTree-based texture baker.
#
# --no-build-isolation is critical: these packages need torch at build time,
# and uv's default isolated build env has no torch installed.
if [ "${SKIP_METAL:-0}" != "1" ]; then
    # PyTorch's MPS headers require macOS 12.0+. Some Python builds (e.g. uv's
    # prebuilt binaries) set -mmacosx-version-min=11.0 which makes the compiler
    # reject the MPS headers with -Werror. Override to 12.0 for the Metal builds.
    export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-12.0}
    echo
    echo "Installing Metal backends for texture baking (set SKIP_METAL=1 to skip)..."
    # Build deps required by the Metal packages' setup.py
    $PIP setuptools wheel pybind11

    # Always compile the Metal packages against THIS venv's torch.
    # setuptools' build_ext skips the compile when deps/<pkg>/build/ already
    # holds an extension newer than its sources, and packs that old binary
    # into the new wheel (verified here); uv may also reuse a cached wheel for an unchanged
    # source dir. mtlgemm's setup.py bakes the build venv's absolute
    # torch/lib path into flex_gemm's _C as its only torch rpath, so a binary
    # left over from a venv at another path (the trellis-mac -> trellis-silicon
    # rename) cannot load libc10 on its own — and a re-run of this script
    # silently reinstalled exactly that binary. Removing build/ and forcing a
    # reinstall (which refreshes uv's cache entry) makes every run compile.
    install_metal_dep() {
        local dir="$1" dist="$2"
        rm -rf "$DEPS_DIR/$dir/build"
        if command -v uv &>/dev/null; then
            uv pip install --no-build-isolation --reinstall-package "$dist" "$DEPS_DIR/$dir"
        else
            # pip always rebuilds and reinstalls a local path source.
            python3 -m pip install --no-build-isolation "$DEPS_DIR/$dir"
        fi
    }

    install_metal_dep mtlbvh      mtlbvh      || echo "  mtlbvh install failed — continuing without Metal BVH"
    install_metal_dep mtldiffrast mtldiffrast || echo "  mtldiffrast install failed — continuing without Metal rasterizer"
    install_metal_dep mtlmesh     cumesh      || echo "  mtlmesh install failed — continuing without Metal mesh ops"
    # mtlgemm provides flex_gemm: the Metal sparse convolution for the SLat
    # decoders (SPARSE_CONV_BACKEND=flex_gemm, much faster than the pure-PyTorch
    # conv_none fallback) and flex_gemm.ops.grid_sample for the baker; without it
    # the baker uses this port's CPU sparse_grid_sample_3d (same result, slower).
    install_metal_dep mtlgemm     flex_gemm   || echo "  mtlgemm install failed — sparse conv falls back to pure PyTorch and the baker to the CPU sparse sampler (same result, slower)"
    # Pedro Augusto's o_voxel CPU fork — exposes o_voxel.postprocess.to_glb
    # which wraps the Metal stack. Install last so its deps already present.
    install_metal_dep trellis2-apple/o-voxel o_voxel \
        || echo "  o_voxel (Apple fork) install failed — falling back to KDTree baker"

    # Load check: import each package after torch, as trellis_silicon.core does.
    SMOKE_OUT=$(python3 - <<'PY' || true
import importlib
import torch  # noqa: F401  (loads libc10 & co. before the extensions)
for name in ("mtlbvh", "mtldiffrast", "cumesh", "flex_gemm", "o_voxel"):
    try:
        importlib.import_module(name)
        print(f"  {name}: OK")
    except Exception as exc:  # report and keep going; setup.sh treats these as optional
        first = (str(exc).splitlines() or [""])[0][:160]
        print(f"  {name}: NOT available ({type(exc).__name__}: {first})")
PY
)
    echo "$SMOKE_OUT"
    METAL_FAILS=$(printf '%s\n' "$SMOKE_OUT" | grep 'NOT available' | sed 's/^ *//' || true)
fi

# Apply source patches (this also installs stubs and backends)
echo "Applying MPS compatibility patches..."
python3 -m trellis_silicon.patches

# Check HuggingFace auth
echo
if python3 -c "from huggingface_hub import get_token; assert get_token()" 2>/dev/null; then
    echo "HuggingFace auth: OK"
else
    echo "WARNING: Not logged into HuggingFace."
    echo "Some model weights require authentication. Run:"
    echo "  hf auth login"
    echo ""
    echo "You also need to request access to this gated model:"
    echo "  https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m"
fi

if [ -n "$METAL_FAILS" ]; then
    echo
    echo "WARNING: Metal packages not available:"
    printf '%s\n' "$METAL_FAILS" | sed 's/^/  /'
    if [ "${STRICT_METAL:-0}" = "1" ]; then
        echo "STRICT_METAL=1: failing."
        exit 1
    fi
fi

echo
echo "=== Setup complete ==="
echo "Activate the environment:  source .venv/bin/activate"
echo "Generate a 3D model:       trellis-silicon path/to/image.png"
echo "Launch the web UI:         trellis-silicon-web"
