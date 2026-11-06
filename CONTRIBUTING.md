# Contributing to TRELLIS-Silicon

## Dev setup

```bash
bash setup.sh
source .venv/bin/activate
uv pip install -e ".[dev]"   # ruff + pytest
```

## The one architectural rule

**`TRELLIS.2/` is an untracked upstream checkout.** `setup.sh` clones it at a
pinned commit; it is not vendored in git and you must never edit it directly.

Every change to TRELLIS.2's behavior has to be expressed as an idempotent
patch function in `src/trellis_silicon/patches/`, guarded by a marker-string
check so re-running the patcher is a no-op on an already-patched tree. See
`src/trellis_silicon/patches/device.py` or `src/trellis_silicon/patches/loading.py`
for the pattern: check for a string the patch would introduce (e.g.
`if "def device(self)" in src: return`), then apply the source rewrite.

After adding or changing a patch function, register it in
`src/trellis_silicon/patches/__init__.py:main()` and run:

```bash
trellis-silicon-patch
```

## Verification gate

Any change that can affect generation must reproduce the deterministic
baseline:

- **seed 42 + `assets/brighella_input.png` + `512` pipeline → exactly
  513,120 vertices / 1,071,138 triangles** (the `Mesh:` line, after the
  decode-time hole fill and before the GLB export). The GLB face count is not
  part of the gate: the Metal QEM decimation varies by a few hundred faces
  between runs.

The gate assumes the Metal toolchain is installed, sparse convolution runs on
`flex_gemm` (the run log prints `[SPARSE] Conv backend: flex_gemm`; if
`core.py` prints a "flex_gemm unavailable" line, re-run `setup.sh`), and
decode-time hole fill is on (the default). With `SPARSE_CONV_BACKEND=none`
the same run reproduces the previous baseline, 513,106 vertices / 1,071,104
triangles; with `SPARSE_CONV_BACKEND=none DECODE_FILL_HOLES=0 GLB_REMESH=0`
it reproduces 512,254 vertices / 1,065,930 triangles.

Changes that only reorder floating-point ops (batching, kernel swaps) instead
need to stay within ±1% of that baseline, with the GLB loading successfully
in `trimesh` and a `PBRMaterial` present.

Benchmark timing only on a cool machine — Apple Silicon throttles under
sustained load, which can make the same run several times slower with no
code change. Run benchmarks in a fresh process, with `torch.mps.synchronize()`
around timed regions.

## Tests & lint

```bash
pytest tests/
ruff check src/ && ruff format --check src/
```

CI runs both of these plus a patcher idempotency check on Apple Silicon
runners (the patcher is run twice against a fresh, pinned TRELLIS.2 clone and
must produce the same result both times).

## Dependencies

Dependencies are locked in `requirements.lock`. Bumping a dependency requires
re-running the verification gate above — dependency drift has been measured
to shift the deterministic output mesh.

## Pull requests

Keep PRs small and focused. Explain what was measured, not just what was
changed — for anything touching generation or performance, include the
before/after numbers. Negative results are welcome too: open an issue or
discussion for dead ends you've ruled out, since documented dead ends save
everyone else time.
