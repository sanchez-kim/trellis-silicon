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

clone_dep https://github.com/pedronaugusto/mtlbvh.git       mtlbvh
clone_dep https://github.com/pedronaugusto/mtldiffrast.git   mtldiffrast
clone_dep https://github.com/pedronaugusto/mtlmesh.git       mtlmesh
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
    PIP="pip install"
fi
# Install the locked dependency set first (the exact versions the verification
# gate baseline was established with — see requirements.lock), then the package
# itself without re-resolving deps. This registers the trellis-silicon /
# trellis-silicon-web / trellis-silicon-patch console scripts.
$PIP -r requirements.lock
$PIP -e . --no-deps
$PIP "$DEPS_DIR/utils3d"

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
    PIP_NB="$PIP --no-build-isolation"
    # Build deps required by the Metal packages' setup.py
    $PIP setuptools wheel pybind11
    $PIP_NB "$DEPS_DIR/mtlbvh"      || echo "  mtlbvh install failed — continuing without Metal BVH"
    $PIP_NB "$DEPS_DIR/mtldiffrast" || echo "  mtldiffrast install failed — continuing without Metal rasterizer"
    $PIP_NB "$DEPS_DIR/mtlmesh"     || echo "  mtlmesh install failed — continuing without Metal mesh ops"
    # mtlgemm provides flex_gemm.ops.grid_sample. The Metal baker in
    # o_voxel.postprocess prefers this over a torch.nn.functional.grid_sample
    # fallback, and the flex_gemm sparse sampling produces noticeably cleaner
    # texture baking (no concentric ring artifacts on curved surfaces).
    $PIP_NB "$DEPS_DIR/mtlgemm"     || echo "  mtlgemm install failed — baker will use a lower-quality torch.nn.functional.grid_sample fallback"
    # Pedro Naugusto's o_voxel CPU fork — exposes o_voxel.postprocess.to_glb
    # which wraps the Metal stack. Install last so its deps already present.
    $PIP_NB "$DEPS_DIR/trellis2-apple/o-voxel" \
        || echo "  o_voxel (Apple fork) install failed — falling back to KDTree baker"
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
    echo "You also need to request access to these gated models:"
    echo "  https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m"
    echo "  https://huggingface.co/briaai/RMBG-2.0"
fi

echo
echo "=== Setup complete ==="
echo "Activate the environment:  source .venv/bin/activate"
echo "Generate a 3D model:       trellis-silicon path/to/image.png"
echo "Launch the web UI:         trellis-silicon-web"
