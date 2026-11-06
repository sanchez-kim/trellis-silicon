"""CPU-only tests for the texture colour changes.

- postprocess.sparse_grid_sample_3d: matches a per-query reference of upstream
  flex_gemm trilinear sampling (voxel centre i + 0.5, occupied neighbours only,
  renormalised weights), and fixes the darkening of the dense zero-filled
  fallback it replaces when flex_gemm is unavailable.
- postprocess colour match (TEXTURE_COLOR_MATCH): identity, recovery of an
  exposure drop, brighten-only gain, clamps, surface-area weights.
"""

import itertools
import math

import pytest
import torch

from trellis_silicon import postprocess


def _reference_sample(feats, coords, dims, q):
    """Per-query port of flex_gemm/kernels/cuda/grid_sample/grid_sample.cu
    (trilinear, batch 0)."""
    table = {tuple(c[1:].tolist()): i for i, c in enumerate(coords)}
    out = torch.zeros(q.shape[0], feats.shape[1])
    for j, p in enumerate(q.tolist()):
        base = [math.floor(v - 0.5) for v in p]
        acc, w_sum = torch.zeros(feats.shape[1]), 0.0
        for off in itertools.product((0, 1), repeat=3):
            n = tuple(b + o for b, o in zip(base, off))
            if n in table and all(0 <= n[a] < dims[a] for a in range(3)):
                w = math.prod(1 - abs(p[a] - n[a] - 0.5) for a in range(3))
                acc += w * feats[table[n]]
                w_sum += w
        out[j] = acc / max(w_sum, 1e-12)
    return out


def _dense_zero_filled_sample(feats, coords, shape, grid):
    """The fork's no-flex_gemm fallback (o_voxel.postprocess._grid_sample_3d)
    as this port's core.py used to fix it up: F.grid_sample over a zero-filled
    dense volume, align_corners=True."""
    B, C, D, H, W = shape
    vol = torch.zeros(B, C, D, H, W)
    vol[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]] = feats
    g = torch.stack(
        [
            grid[..., 2] / (W - 1) * 2 - 1,
            grid[..., 1] / (H - 1) * 2 - 1,
            grid[..., 0] / (D - 1) * 2 - 1,
        ],
        dim=-1,
    ).reshape(B, 1, 1, -1, 3)
    out = torch.nn.functional.grid_sample(
        vol, g, mode="bilinear", align_corners=True, padding_mode="border"
    )
    return out.reshape(B, C, -1).permute(0, 2, 1).reshape(-1, C)


def _random_field(gen, n_vox=300, dims=(12, 10, 14), channels=6):
    cells = [torch.randint(0, d, (n_vox,), generator=gen) for d in dims]
    cells = torch.stack(cells, dim=1).unique(dim=0)
    coords = torch.cat([torch.zeros(len(cells), 1, dtype=torch.long), cells], dim=1)
    feats = torch.rand(len(cells), channels, generator=gen)
    return feats, coords


def _shape(dims, channels):
    return torch.Size([1, channels, *dims])


def test_sparse_sample_matches_flex_gemm_reference():
    gen = torch.Generator().manual_seed(0)
    dims = (12, 10, 14)
    feats, coords = _random_field(gen, dims=dims)
    q = torch.rand(500, 3, generator=gen) * torch.tensor(dims, dtype=torch.float32)
    got = postprocess.sparse_grid_sample_3d(feats, coords, _shape(dims, 6), q[None])
    torch.testing.assert_close(got, _reference_sample(feats, coords, dims, q))


def test_sparse_sample_chunking_is_exact(monkeypatch):
    gen = torch.Generator().manual_seed(1)
    dims = (12, 10, 14)
    feats, coords = _random_field(gen, dims=dims)
    q = torch.rand(333, 3, generator=gen) * torch.tensor(dims, dtype=torch.float32)
    whole = postprocess.sparse_grid_sample_3d(feats, coords, _shape(dims, 6), q[None])
    monkeypatch.setattr(postprocess, "_SAMPLE_CHUNK", 50)
    chunked = postprocess.sparse_grid_sample_3d(feats, coords, _shape(dims, 6), q[None])
    assert torch.equal(whole, chunked)


def test_sparse_sample_voxel_centre_and_empty_neighbourhood():
    coords = torch.tensor([[0, 2, 3, 4], [0, 3, 3, 4]])
    feats = torch.tensor([[0.2], [0.8]])
    shape = _shape((8, 8, 8), 1)
    q = torch.tensor([[2.5, 3.5, 4.5], [3.0, 3.5, 4.5], [6.5, 6.5, 6.5]])
    got = postprocess.sparse_grid_sample_3d(feats, coords, shape, q[None])
    # centre of voxel (2,3,4) -> its own value; halfway -> mean; no neighbour -> 0
    torch.testing.assert_close(got[:, 0], torch.tensor([0.2, 0.5, 0.0]))


def _plane_shell(value=1.0, n=16):
    """One-voxel-thick plane z = 5 of constant colour, like a decoder shell."""
    xy = torch.cartesian_prod(torch.arange(n), torch.arange(n))
    coords = torch.cat(
        [torch.zeros(len(xy), 1, dtype=torch.long), xy, torch.full((len(xy), 1), 5)], dim=1
    )
    feats = torch.full((len(xy), 3), value)
    return feats, coords, _shape((n, n, n), 3)


def test_constant_shell_keeps_its_colour_but_dense_fallback_darkens():
    feats, coords, shape = _plane_shell(1.0)
    gen = torch.Generator().manual_seed(2)
    q = torch.rand(400, 3, generator=gen)
    q[:, :2] = q[:, :2] * 12 + 2  # away from the grid border
    q[:, 2] = 5.0 + q[:, 2]  # surface points anywhere inside the shell voxel [5, 6)
    new = postprocess.sparse_grid_sample_3d(feats, coords, shape, q[None])
    torch.testing.assert_close(new, torch.ones_like(new))
    old = _dense_zero_filled_sample(feats, coords, shape, q[None])
    assert old.shape == new.shape
    assert old.mean() < 0.75  # empty neighbours count as black
    assert (old <= 1 + 1e-6).all()


def test_sparse_sample_batch_layout():
    feats, coords, _ = _plane_shell(0.5, n=8)
    coords2 = coords.clone()
    coords2[:, 0] = 1
    feats_b = torch.cat([feats, torch.full_like(feats, 0.25)])
    coords_b = torch.cat([coords, coords2])
    q = torch.tensor([[[3.5, 3.5, 5.5]], [[3.5, 3.5, 5.5]]])
    got = postprocess.sparse_grid_sample_3d(feats_b, coords_b, torch.Size([2, 3, 8, 8, 8]), q)
    torch.testing.assert_close(got, torch.tensor([[0.5] * 3, [0.25] * 3]))


def test_sparse_sample_rejects_other_modes():
    feats, coords, shape = _plane_shell()
    with pytest.raises(NotImplementedError):
        postprocess.sparse_grid_sample_3d(feats, coords, shape, torch.zeros(1, 1, 3), "nearest")


# --- optional input-guided colour match -------------------------------------


def _rgba(rgb, alpha=255):
    import numpy as np

    img = np.zeros((4, 4, 4), np.uint8)
    img[..., :3] = rgb
    img[..., 3] = alpha
    return img


def test_reference_pixels_drops_soft_edges():
    import numpy as np

    img = _rgba((200, 100, 50))
    img[0, :, 3] = 100  # cutout edge, mixed with background
    px = postprocess.reference_pixels(img)
    assert px.shape == (12, 3)
    np.testing.assert_allclose(px[0], np.array([200, 100, 50]) / 255)
    with pytest.raises(ValueError):
        postprocess.reference_pixels(img[..., :3])


def test_color_match_identity_when_texture_matches_input():
    import numpy as np

    gen = np.random.default_rng(0)
    px = gen.uniform(0.1, 0.9, (500, 3))
    gain, chroma = postprocess.color_match_params(px, px)
    assert gain == pytest.approx(1.0, abs=1e-6)
    assert chroma == pytest.approx(1.0, abs=1e-4)
    rgb = (px * 255).astype(np.uint8)
    assert np.abs(postprocess.apply_color_match(rgb, 1.0, 1.0).astype(int) - rgb).max() <= 1


def test_color_match_recovers_a_dark_texture_and_keeps_neutrals_neutral():
    import numpy as np

    gen = np.random.default_rng(1)
    ref = gen.uniform(0.2, 0.8, (800, 3))
    lin = postprocess._srgb_to_linear(ref) * 0.7  # exposure drop within the clamp
    dark = postprocess._linear_to_srgb(lin)
    gain, chroma = postprocess.color_match_params(ref, dark)
    assert gain == pytest.approx(1 / 0.7, rel=1e-3)
    assert chroma == pytest.approx(1.0, abs=0.02)
    grey = np.full((1, 3), 90, np.uint8)
    out = postprocess.apply_color_match(grey, gain, 1.25)[0].astype(int)
    assert out.max() - out.min() <= 1 and out[0] > 90


def test_color_match_is_clamped():
    import numpy as np

    lo, hi = postprocess.COLOR_MATCH_CHROMA_RANGE
    # bright, saturated photo vs a dark, nearly grey texture: both bounds hit
    ref = np.tile([1.0, 0.3, 0.1], (200, 1))
    tex = np.tile([0.05, 0.05, 0.06], (200, 1))
    gain, chroma = postprocess.color_match_params(ref, tex)
    assert gain == postprocess.COLOR_MATCH_MAX_GAIN
    assert chroma == hi
    # grey photo vs a saturated texture: chroma hits the lower bound
    gain, chroma = postprocess.color_match_params(
        np.tile([0.5, 0.5, 0.5], (200, 1)), np.tile([0.8, 0.3, 0.1], (200, 1))
    )
    assert chroma == lo


def test_color_match_empty_reference_is_identity():
    import numpy as np

    tex = np.full((50, 3), 0.4)
    with pytest.warns(UserWarning, match="colour match skipped"):
        assert postprocess.color_match_params(np.zeros((99, 3)), tex) == (1.0, 1.0)
    with pytest.warns(UserWarning, match="colour match skipped"):
        assert postprocess.color_match_params(np.zeros((0, 3)), tex) == (1.0, 1.0)
    with pytest.warns(UserWarning, match="colour match skipped"):
        assert postprocess.color_match_params(np.full((500, 3), 0.5), np.zeros((0, 3))) == (
            1.0,
            1.0,
        )


def test_apply_color_match_keeps_hue_when_gain_overflows():
    import numpy as np

    rgb = np.array([[240, 120, 20]], np.uint8)
    out = postprocess.apply_color_match(rgb, 2.0, 1.0)[0].astype(float)
    assert out.max() <= 255
    # linear-light ratios G/R and B/R survive (a per-channel clip would flatten them)
    lin_in = postprocess._srgb_to_linear(rgb[0] / 255.0)
    lin_out = postprocess._srgb_to_linear(out / 255.0)
    assert lin_out[1] / lin_out[0] == pytest.approx(lin_in[1] / lin_in[0], rel=0.03)
    assert lin_out[2] / lin_out[0] == pytest.approx(lin_in[2] / lin_in[0], rel=0.1)


def test_texel_surface_weights_sliver_does_not_dominate():
    # face 0: 4 texels, area 2.0. face 1: a UV sliver, 1 texel, area 2.0.
    # face 2: unrasterised. Old area3 / area_uv gave the sliver a huge weight.
    area3 = torch.tensor([2.0, 2.0, 5.0])
    texel_face = torch.tensor([0, 0, 0, 0, 1])
    w = postprocess.texel_surface_weights(area3, texel_face)
    torch.testing.assert_close(w, torch.tensor([0.5, 0.5, 0.5, 0.5, 2.0]))
    # each face's texels sum to its exact 3D area
    assert w[:4].sum() == pytest.approx(2.0) and w[4] == pytest.approx(2.0)
    # bounded by the face's own area, whatever its UV area
    assert (w <= area3[texel_face]).all()


def test_sparse_sample_empty_inputs_and_device_roundtrip():
    shape = _shape((8, 8, 8), 3)
    q = torch.rand(1, 5, 3) * 8
    out = postprocess.sparse_grid_sample_3d(
        torch.zeros(0, 3), torch.zeros(0, 4, dtype=torch.long), shape, q
    )
    assert out.shape == (5, 3) and out.dtype == torch.float32 and not out.any()
    feats, coords, shape = _plane_shell(0.5, n=8)
    out = postprocess.sparse_grid_sample_3d(feats, coords, shape, torch.zeros(1, 0, 3))
    assert out.shape == (0, 3)


def test_sparse_sample_matches_real_flex_gemm():
    torch.manual_seed(0)  # import torch before flex_gemm
    pytest.importorskip("flex_gemm.ops.grid_sample")
    from flex_gemm.ops.grid_sample import grid_sample_3d

    gen = torch.Generator().manual_seed(5)
    dims = (12, 10, 14)
    feats, coords = _random_field(gen, dims=dims)
    q = torch.rand(1, 400, 3, generator=gen) * torch.tensor(dims, dtype=torch.float32)
    shape = _shape(dims, 6)
    want = grid_sample_3d(feats, coords.int(), shape, q, mode="trilinear")
    got = postprocess.sparse_grid_sample_3d(feats, coords, shape, q)
    torch.testing.assert_close(got, want.reshape(got.shape), atol=1e-5, rtol=1e-4)


def test_color_match_never_darkens():
    import numpy as np

    gen = np.random.default_rng(3)
    tex = gen.uniform(0.4, 0.9, (300, 3))
    ref = tex * 0.5  # photo darker than the albedo (e.g. shadowed interior)
    gain, _ = postprocess.color_match_params(ref, tex)
    assert gain == 1.0


def test_color_match_texel_weights_follow_surface_area():
    import numpy as np

    ref = np.full((200, 3), 0.5)
    tex = np.array([[0.2] * 3, [0.5] * 3, [0.5] * 3])
    # the dark texel stands for most of the surface -> gain > 1
    gain_w, _ = postprocess.color_match_params(ref, tex, np.array([10.0, 1.0, 1.0]))
    gain_u, _ = postprocess.color_match_params(ref, tex)
    assert gain_u == pytest.approx(1.0)
    assert gain_w > 1.5
