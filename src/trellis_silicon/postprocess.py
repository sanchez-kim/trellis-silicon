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
import warnings

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


# Texels sampled per sparse_grid_sample_3d chunk (bounds the [chunk, 8] lookups).
_SAMPLE_CHUNK = 1 << 20


def sparse_grid_sample_3d(feats, coords, shape, grid, mode="trilinear"):
    """Trilinear sample of a sparse voxel field with upstream flex_gemm semantics.

    Same contract as ``o_voxel.postprocess._grid_sample_3d`` (feats [N, C],
    coords [N, 4] as (batch, x, y, z), shape [B, C, X, Y, Z], grid [B, L, 3] in
    voxel units) and returns [B*L, C] float32. Matches
    ``flex_gemm/kernels/cuda/grid_sample/grid_sample.cu`` (what upstream
    TRELLIS.2 bakes with on CUDA): voxel i covers [i, i+1) with its centre at
    i + 0.5, only occupied voxels contribute, and their trilinear weights are
    renormalised to sum to 1. A query with no occupied neighbour returns 0.

    Used only when flex_gemm's grid_sample cannot be imported: the o_voxel
    fork then falls back to a dense ``F.grid_sample`` over a zero-filled volume,
    which counts empty voxels as black (the decoder's PBR field is a thin shell,
    so most texels have empty neighbours), puts voxel i's centre at i instead of
    i + 0.5, and needs a 6 x 512^3 float32 volume (3.2GB). On the fox input
    that fallback baked the base colour ~24% darker (mean sampled alpha 0.77).

    Inputs may live on any device: they move to the CPU first and the result
    returns on ``grid``'s device. Empty ``feats`` or an empty query set give
    zeros of the right shape.
    """
    if mode != "trilinear":
        raise NotImplementedError(f"sparse_grid_sample_3d: mode={mode!r}")
    out_device = grid.device
    feats, coords, grid = feats.cpu(), coords.cpu(), grid.cpu()
    B, dims = shape[0], torch.tensor(list(shape[2:5]), dtype=torch.long)
    X, Y, Z = (int(d) for d in dims)
    feats = feats.float()
    coords = coords.long()
    if feats.shape[0] == 0 or grid.numel() == 0:
        return torch.zeros(B * grid.shape[1], feats.shape[1], dtype=torch.float32).to(out_device)
    keys = ((coords[:, 0] * X + coords[:, 1]) * Y + coords[:, 2]) * Z + coords[:, 3]
    keys, order = torch.sort(keys)
    L = grid.shape[1]
    q_all = grid.reshape(-1, 3).float()
    b_all = torch.arange(B, dtype=torch.long).repeat_interleave(L)
    offsets = torch.tensor(
        [[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], dtype=torch.long
    )
    out = torch.empty(q_all.shape[0], feats.shape[1], dtype=torch.float32)
    for s in range(0, q_all.shape[0], _SAMPLE_CHUNK):
        q = q_all[s : s + _SAMPLE_CHUNK]
        b = b_all[s : s + _SAMPLE_CHUNK]
        base = torch.floor(q - 0.5).long()
        acc = torch.zeros(q.shape[0], feats.shape[1], dtype=torch.float32)
        w_sum = torch.zeros(q.shape[0], dtype=torch.float32)
        for off in offsets:
            n = base + off
            inside = ((n >= 0) & (n < dims)).all(dim=1)
            k = ((b * X + n[:, 0]) * Y + n[:, 1]) * Z + n[:, 2]
            pos = torch.searchsorted(keys, k).clamp_max(keys.shape[0] - 1)
            found = inside & (keys[pos] == k)
            w = (1 - (q - n - 0.5).abs()).prod(dim=1) * found
            acc += w[:, None] * feats[order[pos]]
            w_sum += w
        out[s : s + _SAMPLE_CHUNK] = acc / w_sum.clamp_min(1e-12)[:, None]
    return out.to(out_device)


# --- Optional input-guided colour match (TEXTURE_COLOR_MATCH / --color-match) ---

# Bounds on the correction. The input photo has its lighting baked in (shadows,
# specular highlights), so a full match can over-correct; these cap how far the
# albedo may move. The gain only brightens: TRELLIS.2's measured bias is a
# darker albedo than the photo, and a photo darker than the albedo (shadowed
# interiors) is not evidence that the albedo is too bright. Measured on four
# inputs (fox, tin robot, backpack, egg chair; 512, seed 42): mean |dV|+|dS| of
# the unlit input view vs the input 0.19 -> 0.07, the chair (already close) is
# left as is (gain 1.0, chroma x0.98). A two-sided gain darkened the chair.
COLOR_MATCH_MAX_GAIN = 2.0  # linear-light luminance gain, range [1, max]
COLOR_MATCH_CHROMA_RANGE = (0.75, 1.25)  # Lab chroma scale
# Input pixels with alpha below this are cutout edges mixed with background.
_REF_MIN_ALPHA = 230
# Fewer foreground pixels than this is no usable reference.
_REF_MIN_PIXELS = 100


def _srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def _luminance(lin):
    return lin @ np.array([0.2126, 0.7152, 0.0722])


def _lab(srgb):
    import cv2

    return cv2.cvtColor(srgb.astype(np.float32).reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)


def _weighted_median(x, w):
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cw, 0.5 * cw[-1])])


def reference_pixels(rgba):
    """Foreground pixels [N, 3] (sRGB, 0-1) of an RGBA input image."""
    rgba = np.asarray(rgba)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("colour reference must be an RGBA image")
    px = rgba[..., :3][rgba[..., 3] >= _REF_MIN_ALPHA]
    return px.astype(np.float64) / 255.0


def color_match_params(ref_px, tex_px, tex_w=None):
    """Gain and chroma scale that move the albedo texels toward the input
    foreground, from medians (robust to highlights, shadows and small parts).

    ref_px: input foreground pixels [N, 3] sRGB 0-1. tex_px: used texels
    [M, 3] sRGB 0-1, tex_w: per-texel surface area (texel density varies per
    chart). Returns (gain, chroma): gain multiplies linear RGB, chroma scales
    Lab a/b after the gain. Both are clamped (COLOR_MATCH_*).
    """
    if len(ref_px) < _REF_MIN_PIXELS or len(tex_px) == 0:
        warnings.warn(
            f"colour match skipped: {len(ref_px)} reference foreground pixels "
            f"(need {_REF_MIN_PIXELS}), {len(tex_px)} texels",
            stacklevel=2,
        )
        return 1.0, 1.0
    tex_w = np.ones(len(tex_px)) if tex_w is None else np.asarray(tex_w, np.float64)
    ref_w = np.ones(len(ref_px))
    y_ref = _weighted_median(_luminance(_srgb_to_linear(ref_px)), ref_w)
    y_tex = _weighted_median(_luminance(_srgb_to_linear(tex_px)), tex_w)
    gain = float(np.clip(y_ref / max(y_tex, 1e-6), 1.0, COLOR_MATCH_MAX_GAIN))
    gained = _linear_to_srgb(_srgb_to_linear(tex_px) * gain)
    c_ref = _weighted_median(np.hypot(*_lab(ref_px)[:, 1:].T), ref_w)
    c_tex = _weighted_median(np.hypot(*_lab(gained)[:, 1:].T), tex_w)
    chroma = float(np.clip(c_ref / max(c_tex, 1e-6), *COLOR_MATCH_CHROMA_RANGE))
    return gain, chroma


def texel_surface_weights(area3, texel_face):
    """Per-texel surface area: each face's exact 3D area split evenly over the
    texels it actually covers (a UV sliver covering one texel no longer
    inflates its weight the way area3 / area_uv does).

    area3: [F] 3D face areas. texel_face: [M] face index of every used texel.
    Returns [M] weights.
    """
    cnt = torch.bincount(texel_face, minlength=area3.shape[0])
    return (area3 / cnt.clamp_min(1))[texel_face]


def apply_color_match(rgb, gain, chroma):
    """Apply (gain, chroma) to uint8 sRGB pixels of any shape [..., 3].

    A pixel whose gained linear max channel exceeds 1 is scaled by one factor
    on all three channels, so clipping does not shift its hue.
    """
    import cv2

    shape = rgb.shape
    lin = _srgb_to_linear(rgb.reshape(-1, 3).astype(np.float64) / 255.0) * gain
    lin = lin / np.maximum(lin.max(axis=1, keepdims=True), 1.0)
    lab = _lab(_linear_to_srgb(lin))
    lab[:, 1:] *= chroma
    out = cv2.cvtColor(lab.reshape(-1, 1, 3), cv2.COLOR_Lab2RGB).reshape(-1, 3)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8).reshape(shape)


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
    color_reference=None,
    verbose=False,
):
    """Upstream to_glb (remesh=True branch) on the Metal backends. All tensors
    are expected on CPU: the Metal kernels read unified memory directly.

    ``remesh_project=0`` follows upstream app.py; the upstream function
    signature default is 0.9.

    ``color_reference`` (optional, not upstream): the input image as RGBA. When
    given, the baked base colour is moved toward the input foreground's
    median luminance and chroma (`color_match_params`), measured over the
    texels that map to the surface, weighted by surface area.
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
        texel_weight = None
        if color_reference is not None:
            # Surface area each used texel stands for: chart texel density varies
            try:
                texel_face = rast[0, ..., 3][mask].long() - 1
                tri3 = out_vertices[out_faces.long()]
                area3 = torch.linalg.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0]).norm(
                    dim=1
                )
                texel_weight = texel_surface_weights(area3, texel_face).double().cpu().numpy()
            except Exception as e:
                print(f"\n  Warning: colour match skipped: {e}")

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
    # Texels whose sampled attrs are all zero had no occupied neighbour voxel;
    # they stay in the texture but do not count in the colour statistics.
    sampled = (
        (attrs.abs().sum(dim=-1) > 0).cpu().numpy()[mask] if texel_weight is not None else None
    )

    def channel(name):
        x = attrs[..., attr_layout[name]].cpu().numpy() * 255
        return np.clip(x, 0, 255).astype(np.uint8)

    base_color = channel("base_color")
    metallic = channel("metallic")
    roughness = channel("roughness")
    alpha = channel("alpha")

    if texel_weight is not None:
        try:
            tex = base_color[mask]
            gain, chroma = color_match_params(
                reference_pixels(color_reference), tex[sampled] / 255.0, texel_weight[sampled]
            )
            base_color[mask] = apply_color_match(tex, gain, chroma)
            if verbose:
                print(
                    f" colour match: gain {gain:.3f}, chroma x{chroma:.3f} ...", end="", flush=True
                )
        except Exception as e:  # never let the optional match change or fail the bake
            print(f"\n  Warning: colour match skipped: {e}")

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
