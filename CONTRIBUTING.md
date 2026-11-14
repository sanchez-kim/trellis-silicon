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
  514,312 vertices / 1,092,064 triangles** (the `Mesh:` line, after the
  decode-time hole fill and before the GLB export). The GLB face count is not
  part of the gate: the Metal QEM decimation varies by a few hundred faces
  between runs.

The gate input is RGB, so it goes through the background remover; the run log
prints the model on a `[REMBG] Background remover:` line. The baseline above
uses the default BiRefNet (2026-11-14). With `REMBG_MODEL=briaai/RMBG-2.0` the
same run reproduces the earlier baseline, 513,120 vertices / 1,071,138
triangles.

The gate assumes the Metal toolchain is installed, sparse convolution runs on
`flex_gemm` (the run log prints `[SPARSE] Conv backend: flex_gemm`; if
`core.py` prints a "flex_gemm unavailable" line, re-run `setup.sh`), and
decode-time hole fill is on (the default). With RMBG-2.0 as the remover,
`SPARSE_CONV_BACKEND=none` gave 513,106 vertices / 1,071,104 triangles, and
`SPARSE_CONV_BACKEND=none DECODE_FILL_HOLES=0 GLB_REMESH=0` gave 512,254
vertices / 1,065,930 triangles; these two have not been re-measured with
BiRefNet.

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

Dependencies are locked in `requirements.lock`. Bumping a dependency or
rebuilding the Metal packages requires re-running the verification gate above,
and checking the `[SPARSE] Conv backend:` line in the log.
A gate shift seen after a venv rebuild in July 2026 was first blamed on
dependency drift. It was traced to `flex_gemm` silently failing to load, which
switches sparse convolution to the `none` fallback and moves the mesh. All
nine July 2026 run logs kept after the rebuild show `Conv backend: none`.
Re-running with the July settings reproduces both July gate counts when only
`SPARSE_CONV_BACKEND` changes.

## Pull requests

Keep PRs small and focused. Explain what was measured, not just what was
changed — for anything touching generation or performance, include the
before/after numbers. Negative results are welcome too: open an issue or
discussion for dead ends you've ruled out, since documented dead ends save
everyone else time.
