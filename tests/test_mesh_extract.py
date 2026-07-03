"""Equivalence tests for the vectorized dual-grid mesh extraction.

`trellis_silicon.backends.mesh_extract.flexible_dual_grid_to_mesh` replaced a
pure-Python dict-loop implementation with a vectorized int64-hash version. The
project has a strict bit-exact verification gate, so this suite pins the new
implementation against the original dict-loop version (embedded below as
`_reference_flexible_dual_grid_to_mesh`) on a spread of synthetic sparse
dual-voxel grids, asserting identical vertices/triangles (values, dtype, shape).

CPU-only, no GPU/MPS. Imports the module directly to avoid trellis_silicon.core's
heavy env setup.
"""

import numpy as np
import pytest
import torch

from trellis_silicon.backends import mesh_extract

# --- Reference oracle: the original dict-loop implementation, verbatim -------
# (recovered from git HEAD~1:src/trellis_silicon/backends/mesh_extract.py). Kept
# self-contained so the test does not depend on any file that git may rewrite.

_EDGE_NEIGHBOR_VOXEL_OFFSET = torch.tensor(
    [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ],
    dtype=torch.int,
).unsqueeze(0)
_QUAD_SPLIT_1 = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long)
_QUAD_SPLIT_2 = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long)


def _reference_flexible_dual_grid_to_mesh(
    coords,
    dual_vertices,
    intersected_flag,
    split_weight,
    aabb,
    voxel_size=None,
    grid_size=None,
    train=False,
):
    device = coords.device
    edge_offset = _EDGE_NEIGHBOR_VOXEL_OFFSET.to(device)
    quad_split_1 = _QUAD_SPLIT_1.to(device)
    quad_split_2 = _QUAD_SPLIT_2.to(device)

    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=device)

    if voxel_size is not None:
        if isinstance(voxel_size, (int, float)):
            voxel_size = [voxel_size] * 3
        if isinstance(voxel_size, (list, tuple, np.ndarray)):
            voxel_size = torch.tensor(np.array(voxel_size), dtype=torch.float32, device=device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        if isinstance(grid_size, int):
            grid_size = [grid_size] * 3
        if isinstance(grid_size, (list, tuple, np.ndarray)):
            grid_size = torch.tensor(np.array(grid_size), dtype=torch.int32, device=device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size.float()

    N = dual_vertices.shape[0]

    # Build coordinate lookup on CPU
    coords_cpu = coords.cpu()
    coord_to_idx = {}
    for i in range(N):
        key = (coords_cpu[i, 0].item(), coords_cpu[i, 1].item(), coords_cpu[i, 2].item())
        coord_to_idx[key] = i

    # Find connected voxels for each intersected edge
    edge_neighbor_voxel = coords.reshape(N, 1, 1, 3) + edge_offset
    connected_voxel = edge_neighbor_voxel[intersected_flag]
    M = connected_voxel.shape[0]

    if M == 0:
        return torch.zeros(0, 3, device=device), torch.zeros(0, 3, dtype=torch.long, device=device)

    # Look up neighbor indices via dict
    connected_cpu = connected_voxel.cpu().reshape(-1, 3)
    indices = []
    for j in range(connected_cpu.shape[0]):
        key = (connected_cpu[j, 0].item(), connected_cpu[j, 1].item(), connected_cpu[j, 2].item())
        indices.append(coord_to_idx.get(key, 0xFFFFFFFF))

    connected_voxel_indices = torch.tensor(indices, dtype=torch.int64, device=device).reshape(M, 4)
    connected_voxel_valid = (connected_voxel_indices != 0xFFFFFFFF).all(dim=1)
    quad_indices = connected_voxel_indices[connected_voxel_valid].long()
    L = quad_indices.shape[0]

    if L == 0:
        return torch.zeros(0, 3, device=device), torch.zeros(0, 3, dtype=torch.long, device=device)

    mesh_vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0].reshape(1, 3)

    if train:
        raise RuntimeError("Training mode not supported in pure-Python mesh extraction")

    if split_weight is None:
        # NOTE: torch.cross without an explicit dim, matching the original and the
        # current implementation verbatim (both rely on the same default-axis
        # behavior). Do NOT add dim= here — it would diverge from the impl under test.
        a1 = quad_indices[:, quad_split_1]
        n0 = torch.cross(
            mesh_vertices[a1[:, 1]] - mesh_vertices[a1[:, 0]],
            mesh_vertices[a1[:, 2]] - mesh_vertices[a1[:, 0]],
        )
        n1 = torch.cross(
            mesh_vertices[a1[:, 2]] - mesh_vertices[a1[:, 1]],
            mesh_vertices[a1[:, 3]] - mesh_vertices[a1[:, 1]],
        )
        align0 = (n0 * n1).sum(dim=1, keepdim=True).abs()

        a2 = quad_indices[:, quad_split_2]
        n0 = torch.cross(
            mesh_vertices[a2[:, 1]] - mesh_vertices[a2[:, 0]],
            mesh_vertices[a2[:, 2]] - mesh_vertices[a2[:, 0]],
        )
        n1 = torch.cross(
            mesh_vertices[a2[:, 2]] - mesh_vertices[a2[:, 1]],
            mesh_vertices[a2[:, 3]] - mesh_vertices[a2[:, 1]],
        )
        align1 = (n0 * n1).sum(dim=1, keepdim=True).abs()

        mesh_triangles = torch.where(align0 > align1, a1, a2).reshape(-1, 3)
    else:
        sw = split_weight[quad_indices]
        sw_02 = (sw[:, 0] * sw[:, 2]).squeeze()
        sw_13 = (sw[:, 1] * sw[:, 3]).squeeze()
        cond = (sw_02 > sw_13).unsqueeze(1).expand(-1, 6)
        mesh_triangles = torch.where(
            cond,
            quad_indices[:, quad_split_1],
            quad_indices[:, quad_split_2],
        ).reshape(-1, 3)

    return mesh_vertices, mesh_triangles


# --- Synthetic case generators ----------------------------------------------

_AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]


def _make_case(n_grid, n_vox, seed, use_split_weight, dense_cluster=False, boundary=False):
    """Build a synthetic sparse dual-voxel grid input dict."""
    g = torch.Generator().manual_seed(seed)
    if n_vox == 0:
        coords = torch.zeros(0, 3, dtype=torch.int32)
    elif dense_cluster:
        # A fully dense small cube => maximal edge connectivity.
        side = int(round(n_vox ** (1 / 3)))
        rng = torch.arange(side)
        coords = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3).int()
    elif boundary:
        # Coords pushed against the grid extreme so neighbor offsets go out of bounds.
        base = torch.randint(0, n_grid, (n_vox, 3), generator=g)
        base[:, 0] = n_grid - 1
        coords = torch.unique(base, dim=0).int()
    else:
        raw = torch.randint(0, n_grid, (n_vox, 3), generator=g)
        coords = torch.unique(raw, dim=0).int()

    n = coords.shape[0]
    dual_vertices = torch.rand(n, 3, generator=g)
    intersected_flag = torch.rand(n, 3, generator=g) > 0.4
    split_weight = torch.rand(n, 1, generator=g) if use_split_weight else None
    return dict(
        coords=coords,
        dual_vertices=dual_vertices,
        intersected_flag=intersected_flag,
        split_weight=split_weight,
        aabb=_AABB,
        grid_size=n_grid,
    )


def _make_dup_case(seed):
    """Coords with deliberate duplicates, to exercise last-index-wins lookup."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randint(0, 8, (400, 3), generator=g).int()
    coords = torch.cat([base, base[:120]], dim=0)
    n = coords.shape[0]
    dual_vertices = torch.rand(n, 3, generator=g)
    intersected_flag = torch.rand(n, 3, generator=g) > 0.4
    return dict(
        coords=coords,
        dual_vertices=dual_vertices,
        intersected_flag=intersected_flag,
        split_weight=None,
        aabb=_AABB,
        grid_size=8,
    )


def _build_cases():
    cases = []
    # empty / near-empty
    cases.append(("empty-grid", _make_case(16, 0, 1, False)))
    cases.append(("single-voxel", _make_case(16, 1, 2, False)))
    cases.append(("two-voxels", _make_case(16, 2, 3, False)))
    # small sparse, both split modes
    for seed in range(4, 12):
        cases.append((f"sparse-g32-s{seed}-splitNone", _make_case(32, 500, seed, False)))
        cases.append((f"sparse-g32-s{seed}-splitW", _make_case(32, 500, seed, True)))
    # dense clusters (max connectivity)
    cases.append(("dense-8cube-splitNone", _make_case(8, 512, 20, False, dense_cluster=True)))
    cases.append(("dense-8cube-splitW", _make_case(8, 512, 21, True, dense_cluster=True)))
    cases.append(("dense-12cube", _make_case(12, 1728, 22, False, dense_cluster=True)))
    # boundary coords (neighbor offsets go out of grid -> missing lookups)
    cases.append(("boundary-splitNone", _make_case(24, 800, 30, False, boundary=True)))
    cases.append(("boundary-splitW", _make_case(24, 800, 31, True, boundary=True)))
    # duplicate coords (last-index-wins path)
    cases.append(("duplicate-coords", _make_dup_case(40)))
    cases.append(("duplicate-coords-2", _make_dup_case(41)))
    # larger, both modes (grid small enough that random voxels are densely adjacent)
    for seed in range(50, 56):
        cases.append((f"med-g20-s{seed}-splitNone", _make_case(20, 8000, seed, False)))
        cases.append((f"med-g20-s{seed}-splitW", _make_case(20, 8000, seed, True)))
    return cases


_CASES = _build_cases()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("kwargs", [c[1] for c in _CASES], ids=[c[0] for c in _CASES])
def test_vectorized_matches_reference(kwargs):
    """New vectorized impl must be bit-exact with the original dict-loop oracle."""
    ref_v, ref_t = _reference_flexible_dual_grid_to_mesh(**kwargs)
    new_v, new_t = mesh_extract.flexible_dual_grid_to_mesh(**kwargs)

    assert new_v.dtype == ref_v.dtype
    assert new_t.dtype == ref_t.dtype
    assert new_v.shape == ref_v.shape
    assert new_t.shape == ref_t.shape
    assert torch.equal(new_v, ref_v)
    assert torch.equal(new_t, ref_t)


def test_case_count():
    """Guard against silently dropping parametrized cases."""
    assert len(_CASES) == 38
