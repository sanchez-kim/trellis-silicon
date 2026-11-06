"""CPU-only tests for the postprocess-parity changes.

- patch_mesh_fill_holes: applies to a fresh upstream Mesh.fill_holes, migrates a
  file patched by the old unconditional skip, and is idempotent.
- postprocess._ChunkedBVH: the min-merge over sub-BVHs returns the global
  nearest triangle with global face ids. The Metal BVH is replaced by a
  brute-force fake (cumesh/mtlbvh are not installed in CI).
"""

import sys
import types

import pytest
import torch

from trellis_silicon import postprocess
from trellis_silicon.patches import mesh as mesh_patch

_BASE_TAIL = (
    "        \n"
    "        mesh = cumesh.CuMesh()\n"
    "        mesh.init(vertices, faces)\n"
    "        mesh.fill_holes(max_hole_perimeter=max_hole_perimeter)\n"
)


def _write_base(root, fill_holes_head):
    path = root / "trellis2/representations/mesh/base.py"
    path.parent.mkdir(parents=True)
    path.write_text("import torch\n\nclass Mesh:\n" + fill_holes_head + _BASE_TAIL)
    return path


@pytest.mark.parametrize(
    "head", [mesh_patch._FILL_HOLES_UPSTREAM, mesh_patch._FILL_HOLES_OLD_SKIP], ids=["fresh", "old"]
)
def test_fill_holes_patch_applies_and_is_idempotent(tmp_path, monkeypatch, head):
    monkeypatch.setattr(mesh_patch, "TRELLIS_ROOT", str(tmp_path))
    path = _write_base(tmp_path, head)

    mesh_patch.patch_mesh_fill_holes()
    once = path.read_text()
    assert mesh_patch._FILL_HOLES_PATCHED in once
    assert "return  # Skip" not in once
    assert ".cuda()" not in once
    assert once.endswith(_BASE_TAIL)

    mesh_patch.patch_mesh_fill_holes()
    assert path.read_text() == once


def test_fill_holes_patch_fails_loudly_on_unknown_source(tmp_path, monkeypatch):
    monkeypatch.setattr(mesh_patch, "TRELLIS_ROOT", str(tmp_path))
    _write_base(tmp_path, "    def fill_holes(self):\n        pass\n")
    with pytest.raises(RuntimeError, match="patch target not found"):
        mesh_patch.patch_mesh_fill_holes()


def _closest_on_triangles(p, tri):
    """Exact point-triangle closest point (Ericson), p: [N,3], tri: [M,3,3] -> d [N,M]."""
    a, b, c = (tri[None, :, i, :] for i in range(3))
    p = p[:, None, :]
    ab, ac = b - a, c - a

    def clamp_seg(s0, s1):
        e = s1 - s0
        t = (((p - s0) * e).sum(-1) / (e * e).sum(-1)).clamp(0, 1)
        return s0 + e * t[..., None]

    n = torch.cross(ab.expand_as(p - a), ac.expand_as(p - a), dim=-1)
    n = n / n.norm(dim=-1, keepdim=True)
    proj = p - ((p - a) * n).sum(-1, keepdim=True) * n
    bary = []
    for s0, s1 in ((a, b), (b, c), (c, a)):
        bary.append((torch.cross(s1 - s0, proj - s0, dim=-1) * n).sum(-1) >= 0)
    inside = bary[0] & bary[1] & bary[2]
    d_face = (p - proj).norm(dim=-1)
    d_edge = torch.stack(
        [(p - clamp_seg(s0, s1)).norm(dim=-1) for s0, s1 in ((a, b), (b, c), (c, a))]
    ).amin(0)
    return torch.where(inside, d_face, d_edge)


class _BruteBVH:
    """Brute-force stand-in for cumesh.cuBVH (same unsigned_distance contract)."""

    def __init__(self, vertices, faces):
        self.tri = vertices[faces.long()]

    def unsigned_distance(self, positions, return_uvw=False):
        d = _closest_on_triangles(positions, self.tri)
        dist, fid = d.min(dim=1)
        uvw = torch.full((positions.shape[0], 3), float(len(self.tri))) if return_uvw else None
        return dist, fid, uvw


def test_chunked_bvh_returns_global_nearest(monkeypatch):
    monkeypatch.setitem(sys.modules, "cumesh", types.SimpleNamespace(cuBVH=_BruteBVH))
    monkeypatch.setattr(postprocess, "_SAFE_BVH_TRIANGLES", 16)  # force 4 sub-BVHs

    gen = torch.Generator().manual_seed(0)
    verts = torch.rand(180, 3, generator=gen)  # 60 non-degenerate random triangles
    faces = torch.arange(180).reshape(60, 3).int()
    pts = torch.rand(200, 3, generator=gen)

    chunked = postprocess._ChunkedBVH(verts, faces)
    assert len(chunked.parts) == 4
    d, f, uvw = chunked.unsigned_distance(pts, return_uvw=True)

    ref_d, ref_f, _ = _BruteBVH(verts, faces).unsigned_distance(pts)
    torch.testing.assert_close(d, ref_d)
    assert torch.equal(f, ref_f)
    # uvw comes from the sub-BVH that won: its sentinel is that part's size
    bounds = postprocess._bvh_part_bounds(60)
    sizes = torch.tensor([e - s for s, e in bounds], dtype=torch.float32)
    owner = torch.bucketize(f, torch.tensor([e - 1 for _, e in bounds]))
    torch.testing.assert_close(uvw[:, 0], sizes[owner])


# --- review-fix tests -------------------------------------------------------


@pytest.mark.parametrize(
    "n", [3_538_943, 4_718_589, 5_767_167, 0, 1, 1 << 17, (1 << 17) + 1, 6_000_000]
)
def test_bvh_part_bounds_cap_and_cover(n):
    bounds = postprocess._bvh_part_bounds(n)
    assert all(0 < e - s <= postprocess._SAFE_BVH_TRIANGLES for s, e in bounds)
    covered = [(s, e) for s, e in bounds]
    assert sum(e - s for s, e in covered) == n
    assert all(a[1] == b[0] for a, b in zip(covered, covered[1:]))
    if n:
        assert (covered[0][0], covered[-1][1]) == (0, n)
    else:
        assert covered == []


def test_bvh_part_bounds_random():
    import random

    rng = random.Random(0)
    for _ in range(50):
        n = rng.randint(1, 6_000_000)
        bounds = postprocess._bvh_part_bounds(n)
        assert all(e - s <= postprocess._SAFE_BVH_TRIANGLES for s, e in bounds)
        assert bounds[0][0] == 0 and bounds[-1][1] == n
        assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))


class _FakeBVH:
    builds = []

    def __init__(self, vertices, faces):
        self.n = faces.shape[0]
        _FakeBVH.builds.append(self.n)


@pytest.fixture
def fake_cumesh(monkeypatch):
    _FakeBVH.builds = []
    mod = types.ModuleType("cumesh")
    mod.cuBVH = _FakeBVH
    monkeypatch.setitem(sys.modules, "cumesh", mod)
    monkeypatch.setattr(postprocess, "_bvh_exact", None)
    return mod


def _faces(n):
    return torch.zeros(n, 3, dtype=torch.int32)


def test_make_bvh_small_mesh_builds_directly_without_selftest(fake_cumesh, monkeypatch):
    def boom():
        raise AssertionError("self-test must not run for small meshes")

    monkeypatch.setattr(postprocess, "_bvh_traversal_is_exact", boom)
    bvh = postprocess.make_bvh(None, _faces(postprocess._SAFE_BVH_TRIANGLES))
    assert isinstance(bvh, _FakeBVH)


def test_make_bvh_large_mesh_exact_build_direct(fake_cumesh, monkeypatch):
    monkeypatch.setattr(postprocess, "_bvh_traversal_is_exact", lambda: True)
    bvh = postprocess.make_bvh(None, _faces(postprocess._SAFE_BVH_TRIANGLES + 1))
    assert isinstance(bvh, _FakeBVH)


def test_make_bvh_large_mesh_falls_back_when_selftest_fails(fake_cumesh, monkeypatch):
    monkeypatch.setattr(postprocess, "_bvh_traversal_is_exact", lambda: False)
    n = postprocess._SAFE_BVH_TRIANGLES * 2 + 5
    bvh = postprocess.make_bvh(None, _faces(n))
    assert isinstance(bvh, postprocess._ChunkedBVH)
    assert len(bvh.parts) == 3
    assert [p[0] for p in bvh.parts] == [
        0,
        postprocess._SAFE_BVH_TRIANGLES,
        2 * postprocess._SAFE_BVH_TRIANGLES,
    ]
    assert sorted(_FakeBVH.builds) == [
        5,
        postprocess._SAFE_BVH_TRIANGLES,
        postprocess._SAFE_BVH_TRIANGLES,
    ]


def test_selftest_result_cached_per_process(fake_cumesh, monkeypatch):
    calls = []

    class _Dist(_FakeBVH):
        def unsigned_distance(self, q):
            calls.append(1)
            return (q[:, 2].clone(),)

    fake_cumesh.cuBVH = _Dist
    assert postprocess._bvh_traversal_is_exact() is True
    assert postprocess._bvh_traversal_is_exact() is True
    assert len(calls) == 1
    assert postprocess._bvh_exact is True


def test_autorelease_pool_noop_off_macos(monkeypatch):
    monkeypatch.setattr(postprocess, "_objc", None)
    with postprocess._autorelease_pool():
        ran = True
    assert ran


def test_autorelease_pool_balances_push_pop(monkeypatch):
    events = []
    fake = types.SimpleNamespace(
        objc_autoreleasePoolPush=lambda: events.append("push") or 7,
        objc_autoreleasePoolPop=lambda ctx: events.append(("pop", ctx)),
    )
    monkeypatch.setattr(postprocess, "_objc", fake)
    with pytest.raises(ValueError):
        with postprocess._autorelease_pool():
            raise ValueError
    assert events == ["push", ("pop", 7)]


@pytest.mark.parametrize(
    "val,expected",
    [
        (None, True),
        ("", True),
        ("1", True),
        ("yes", True),
        ("0", False),
        (" False ", False),
        ("NO", False),
        ("Off", False),
    ],
)
def test_env_enabled(monkeypatch, val, expected):
    from trellis_silicon.envflags import env_enabled

    if val is None:
        monkeypatch.delenv("X_FLAG", raising=False)
    else:
        monkeypatch.setenv("X_FLAG", val)
    assert env_enabled("X_FLAG") is expected


def test_fill_holes_patch_migrates_previous_patched_text(tmp_path, monkeypatch):
    monkeypatch.setattr(mesh_patch, "TRELLIS_ROOT", str(tmp_path))
    path = _write_base(tmp_path, mesh_patch._FILL_HOLES_PREV_PATCHED)
    mesh_patch.patch_mesh_fill_holes()
    once = path.read_text()
    assert mesh_patch._FILL_HOLES_PATCHED in once
    assert "_autorelease_pool" in once and "self.faces.shape[0] == 0" in once
    mesh_patch.patch_mesh_fill_holes()
    assert path.read_text() == once
