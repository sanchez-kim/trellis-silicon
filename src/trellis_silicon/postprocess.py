"""GLB export with the upstream TRELLIS.2 postprocess, on the Apple Metal stack.

`to_glb` is a port of the production path of upstream
``TRELLIS.2/o-voxel/o_voxel/postprocess.py:to_glb`` (pinned commit 75fbf01,
``remesh=True``, as called by upstream ``app.py``): hole fill, BVH on the
hole-filled full-resolution mesh, narrow-band dual-contouring remesh, QEM
decimation to the face budget, UV unwrap, texel bake snapped to the
full-resolution surface, Telea inpaint, OPAQUE single-sided material.

It runs on the same Metal backends the Apple o_voxel fork wraps (``cumesh``
from mtlmesh, ``mtlbvh``, ``mtldiffrast``) but does not call the fork's
``to_glb``, because the fork diverges from upstream on this path: it runs an
extra cleanup pass after the remesh (its ``fill_holes`` re-opens boundaries
after decimation), always writes ``doubleSided=True``, and switches the
material to ``BLEND`` when any texel has alpha < 250.

`make_bvh` guards against an mtlbvh traversal bug in builds older than
pedronaugusto/mtlbvh bc0a534: a 24-slot traversal stack silently drops
subtrees on meshes with more than ~131K triangles, so ``unsigned_distance``
returns a farther triangle than the nearest one. That corrupts the remesh
distance field (holes, fragments) and the texel snap (texels sample the wrong
surface). A cheap self-test detects such a build; the BVH is then split into
sub-BVHs small enough that the stack cannot overflow.

Every Metal-heavy stage runs inside its own Objective-C autorelease pool
(`_autorelease_pool`). The Metal extensions create autoreleased command
buffers, which keep every buffer bound to them alive, and the Python main
thread never drains a pool: without this, mtlmesh's simplify loop alone grew
the process footprint by ~11GB on a 4.4M-face remesh (peak 20.5GB -> 8.4GB for
the whole export, measured on this project's Mac).
"""

import contextlib
import ctypes

import numpy as np
import torch

try:
    _objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    _objc.objc_autoreleasePoolPush.restype = ctypes.c_void_p
    _objc.objc_autoreleasePoolPop.argtypes = [ctypes.c_void_p]
except OSError:  # not macOS
    _objc = None


@contextlib.contextmanager
def _autorelease_pool():
    """Drain the Objective-C objects the Metal extensions autorelease inside."""
    if _objc is None:
        yield
        return
    ctx = _objc.objc_autoreleasePoolPush()
    try:
        yield
    finally:
        _objc.objc_autoreleasePoolPop(ctx)


def _simplify(mesh, target, verbose=False):
    """mesh.simplify with one autorelease pool per simplify_step call. The
    mtlmesh decimation loop (cumesh/metal_backend.py) calls simplify_step
    repeatedly; each call allocates fresh Metal buffers."""
    step = mesh.simplify_step

    def pooled_step(*args, **kwargs):
        with _autorelease_pool():
            return step(*args, **kwargs)

    mesh.simplify_step = pooled_step
    try:
        mesh.simplify(target, verbose=verbose)
    finally:
        del mesh.simplify_step


# Per sub-BVH triangle cap for the stack-overflow workaround. mtlbvh builds a
# 4-ary median-split tree with 8 triangles per leaf, so N <= 2**17 triangles
# gives <= 7 internal levels and a worst-case traversal stack of 1 + 3*7 = 22
# entries, within the 24 slots of the affected builds.
_SAFE_BVH_TRIANGLES = 1 << 17

_bvh_exact = None  # cached self-test result for this process


def _bvh_traversal_is_exact():
    """Return True if the installed mtlbvh finds the true nearest triangle on a
    mesh deep enough to overflow the 24-slot stack of the affected builds.

    A flat 768x768 grid (1,179,648 triangles, 9 internal levels, up to 28 stack
    entries) queried at known heights above the plane: the exact distance is
    the height. Measured on this project's Mac: affected build 3621/4096 wrong
    (max error 1.2e-3), fixed build 0/4096. Takes ~0.3s, run once per process.
    """
    global _bvh_exact
    if _bvh_exact is None:
        import cumesh

        n = 768
        g = torch.linspace(0, 1, n + 1)
        yy, xx = torch.meshgrid(g, g, indexing="ij")
        verts = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros((n + 1) ** 2)], 1)
        i = torch.arange(n)
        r, c = torch.meshgrid(i, i, indexing="ij")
        v0 = (r * (n + 1) + c).reshape(-1)
        faces = torch.cat(
            [torch.stack([v0, v0 + 1, v0 + n + 2], 1), torch.stack([v0, v0 + n + 2, v0 + n + 1], 1)]
        )
        gen = torch.Generator().manual_seed(0)
        q = torch.rand(4096, 3, generator=gen)
        q[:, :2] = q[:, :2] * 0.8 + 0.1
        q[:, 2] = q[:, 2] * 0.01 + 1e-3
        d = cumesh.cuBVH(verts.float(), faces.int()).unsigned_distance(q)[0]
        _bvh_exact = bool(((d - q[:, 2]).abs() <= 1e-5).all())
    return _bvh_exact


def _bvh_part_bounds(n):
    """(start, end) ranges covering [0, n) exactly, each <= _SAFE_BVH_TRIANGLES.
    Integer arithmetic only: float32 rounding can yield parts above the cap."""
    return [(s, min(s + _SAFE_BVH_TRIANGLES, n)) for s in range(0, n, _SAFE_BVH_TRIANGLES)]


class _ChunkedBVH:
    """Exact closest-triangle queries over sub-BVHs of <= _SAFE_BVH_TRIANGLES
    triangles each. Same ``unsigned_distance`` contract as ``cumesh.cuBVH``
    (distances, global face ids, barycentric uvw), so it can be passed as
    ``bvh=`` to ``cumesh.remeshing.remesh_narrow_band_dc``.

    Assumes the sub-BVHs never return the not-found sentinel (face id -1),
    which holds for queries inside the unit cube; a sentinel would be shifted
    by the part offset and read as a valid id.
    """

    def __init__(self, vertices, faces):
        import cumesh

        self.parts = []
        for s, e in _bvh_part_bounds(faces.shape[0]):
            self.parts.append((s, cumesh.cuBVH(vertices, faces[s:e].contiguous())))

    def unsigned_distance(self, positions, return_uvw=False):
        best_d = best_f = best_uvw = None
        for offset, bvh in self.parts:
            d, f, uvw = bvh.unsigned_distance(positions, return_uvw=return_uvw)
            f = f + offset
            if best_d is None:
                best_d, best_f, best_uvw = d, f, uvw
                continue
            closer = d < best_d
            best_d = torch.where(closer, d, best_d)
            best_f = torch.where(closer, f, best_f)
            if return_uvw:
                best_uvw = torch.where(closer[:, None], uvw, best_uvw)
        return best_d, best_f, best_uvw


def make_bvh(vertices, faces, verbose=False):
    """BVH for exact closest-triangle queries on `faces`."""
    import cumesh

    if faces.shape[0] <= _SAFE_BVH_TRIANGLES or _bvh_traversal_is_exact():
        return cumesh.cuBVH(vertices, faces)
    if verbose:
        print(
            "\n  Installed mtlbvh drops BVH subtrees on large meshes (fixed in "
            "pedronaugusto/mtlbvh bc0a534); using "
            f"{len(_bvh_part_bounds(faces.shape[0]))} exact sub-BVHs instead. "
            "Re-run setup.sh to rebuild the pinned mtlbvh.",
            end="",
            flush=True,
        )
    return _ChunkedBVH(vertices, faces)


def to_glb(
    vertices,
    faces,
    attr_volume,
    coords,
    attr_layout,
    aabb,
    voxel_size,
    decimation_target=1000000,
    texture_size=2048,
    remesh_band=1,
    remesh_project=0,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose=False,
):
    """Upstream to_glb (remesh=True branch) on the Metal backends. All tensors
    are expected on CPU: the Metal kernels read unified memory directly.

    ``remesh_project=0`` follows upstream app.py; the upstream function
    signature default is 0.9.
    """
    import cumesh
    import cv2
    import o_voxel.postprocess as fork
    import trimesh
    import trimesh.visual
    from PIL import Image

    dr = fork.dr  # mtldiffrast.torch, resolved by the fork's backend probe
    device = torch.device("cpu")

    # --- Input normalization (upstream postprocess.py:60-93) ---
    aabb = torch.tensor(np.array(aabb), dtype=torch.float32, device=device)
    if isinstance(voxel_size, float):
        voxel_size = [voxel_size, voxel_size, voxel_size]
    voxel_size = torch.tensor(np.array(voxel_size), dtype=torch.float32, device=device)
    grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()

    if verbose:
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    vertices = vertices.to(device).float().contiguous()
    faces = faces.to(device).int().contiguous()
    coords = coords.to(device)
    attr_volume = attr_volume.to(device)

    mesh = cumesh.CuMesh()
    # --- Initial mesh cleaning: the "original" surface for every later BVH
    # lookup is the hole-filled mesh, not the raw input ---
    with _autorelease_pool():
        mesh.init(vertices, faces)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        vertices, faces = mesh.read()
    if verbose:
        print(f"After filling holes: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

    if verbose:
        print("Building BVH for current mesh...", end="", flush=True)
    with _autorelease_pool():
        bvh = make_bvh(vertices, faces, verbose=verbose)
    if verbose:
        print("Done")

    # --- Remeshing pipeline (upstream postprocess.py:165-187) ---
    center = aabb.mean(dim=0)
    scale = (aabb[1] - aabb[0]).max().item()
    resolution = grid_size.max().item()
    with _autorelease_pool():
        mesh.init(
            *cumesh.remeshing.remesh_narrow_band_dc(
                vertices,
                faces,
                center=center,
                scale=(resolution + 3 * remesh_band) / resolution * scale,
                resolution=resolution,
                band=remesh_band,
                project_back=remesh_project,
                verbose=verbose,
                bvh=bvh,
            )
        )
    if verbose:
        print(f"After remeshing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    _simplify(mesh, int(decimation_target), verbose=verbose)
    if verbose:
        print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

    # --- UV parameterization ---
    if verbose:
        print("Parameterizing new mesh...")
    with _autorelease_pool():
        out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
                "refine_iterations": mesh_cluster_refine_iterations,
                "global_iterations": mesh_cluster_global_iterations,
                "smooth_strength": mesh_cluster_smooth_strength,
            },
            return_vmaps=True,
            verbose=verbose,
        )
        out_vertices = out_vertices.to(device)
        out_faces = out_faces.to(device)
        out_uvs = out_uvs.to(device)
        out_vmaps = out_vmaps.to(device)
        mesh.compute_vertex_normals()
        out_normals = mesh.read_vertex_normals()[out_vmaps]

    # --- Texture baking ---
    if verbose:
        print("Sampling attributes...", end="", flush=True)
    with _autorelease_pool():
        ctx = dr.MtlRasterizeContext()
        uvs_rast = torch.cat(
            [out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])],
            dim=-1,
        ).unsqueeze(0)
        rast = torch.zeros((1, texture_size, texture_size, 4), device=device, dtype=torch.float32)
        # Rasterize in chunks to save memory; face ID goes into the alpha channel
        for i in range(0, out_faces.shape[0], 100000):
            rast_chunk, _ = dr.rasterize(
                ctx, uvs_rast, out_faces[i : i + 100000], resolution=[texture_size, texture_size]
            )
            mask_chunk = rast_chunk[..., 3:4] > 0
            rast_chunk[..., 3:4] += i
            rast = torch.where(mask_chunk, rast_chunk, rast)
        mask = rast[0, ..., 3] > 0

        # 3D position of every texel, snapped to the hole-filled full-resolution
        # surface, so the bake samples the decoder surface, not the decimated one
        pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
        valid_pos = pos[mask]
        _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
        orig_tri_verts = vertices[faces[face_id.long()]]
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

        attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device=device)
        attrs[mask] = fork._grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
            grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
            mode="trilinear",
        )
    if verbose:
        print("Done")

    # --- Texture post-processing and material ---
    if verbose:
        print("Finalizing mesh...", end="", flush=True)
    mask = mask.cpu().numpy()

    def channel(name):
        x = attrs[..., attr_layout[name]].cpu().numpy() * 255
        return np.clip(x, 0, 255).astype(np.uint8)

    base_color = channel("base_color")
    metallic = channel("metallic")
    roughness = channel("roughness")
    alpha = channel("alpha")

    # Inpainting: fill gaps (dilation) to prevent black seams at UV boundaries
    mask_inv = (~mask).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]

    # Upstream: alpha stays in the texture but the material ignores it
    # (OPAQUE), and the remesh output is single-sided.
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(
            np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)
        ),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode="OPAQUE",
        doubleSided=False,
    )

    # --- Y-up to Z-up for GLB: (x, y, z) -> (x, z, -y); flip V ---
    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    normals_np = out_normals.cpu().numpy()
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2].copy(), -normals_np[:, 1].copy()
    uvs_np[:, 1] = 1 - uvs_np[:, 1]

    textured_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material),
    )
    if verbose:
        print("Done")
    return textured_mesh
