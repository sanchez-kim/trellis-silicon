"""Headless GLB preview renderer (moderngl standalone + trimesh + numpy + PIL).

Real triangle rasterizer with a GPU depth buffer. Replaces the old vertex-splat
approximation, which left speckle and fake holes on adaptively-decimated meshes.

Loads a GLB via trimesh (vertices / faces / vertex normals / UVs / base color
texture), renders it under an orthographic turntable camera with simple lambert +
ambient lighting, and writes a still PNG or a turntable GIF. Anti-aliasing is done
by rendering at a supersample factor and downsampling with PIL LANCZOS.

Importable: `load_mesh`, `Renderer`. CLI: see `main`.
"""

import argparse

import moderngl
import numpy as np
import trimesh
from PIL import Image

LIGHT_DIR = np.array([0.4, 0.7, 0.6], np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
FILL_DIR = np.array([-0.6, 0.1, 0.4], np.float32)  # dim fill from the opposite side
FILL_DIR /= np.linalg.norm(FILL_DIR)
BG = (18, 18, 18)
CLAY_ALBEDO = (0.62, 0.62, 0.62)
FILL = 0.9  # fraction of the frame the mesh's longest axis fills
SUPERSAMPLE = 2  # render at RES * SUPERSAMPLE, then LANCZOS-downsample

VERTEX_SHADER = """
#version 330
uniform mat4 u_mvp;
uniform mat3 u_nrm;
in vec3 in_pos;
in vec3 in_nrm;
in vec2 in_uv;
out vec3 v_nrm;
out vec2 v_uv;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_nrm = u_nrm * in_nrm;
    v_uv = in_uv;
}
"""

FRAGMENT_SHADER = """
#version 330
uniform sampler2D u_tex;
uniform bool u_textured;
uniform vec3 u_clay;
uniform vec3 u_light;
uniform vec3 u_fill;
in vec3 v_nrm;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec3 n = normalize(v_nrm);
    // Two-sided: these meshes have inconsistent winding, so orient the normal
    // toward the camera (view dir +z in screen space) before shading.
    if (n.z < 0.0) n = -n;
    // Key light + dim opposite fill so the shadow side still reads at any turn
    // angle, over an ambient floor. Tuned to lift shadows without washing out.
    float key = max(dot(n, u_light), 0.0);
    float fill = max(dot(n, u_fill), 0.0);
    float shade = 0.23 + 0.70 * key + 0.16 * fill;
    // glTF texcoord origin is top-left; GL sampler origin is bottom-left -> flip V.
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec3 albedo = u_textured ? texture(u_tex, uv).rgb : u_clay;
    f_color = vec4(albedo * shade, 1.0);
}
"""


def load_mesh(path):
    """Return (vertices, faces, normals, uv, texture_image_or_None)."""
    scene = trimesh.load(path, process=False)
    geom = list(scene.geometry.values())[0] if isinstance(scene, trimesh.Scene) else scene
    vertices = np.asarray(geom.vertices, np.float32)
    faces = np.asarray(geom.faces, np.int32)
    normals = np.asarray(geom.vertex_normals, np.float32)

    visual = geom.visual
    uv = getattr(visual, "uv", None)
    uv = np.asarray(uv, np.float32) if uv is not None else None
    material = getattr(visual, "material", None)
    tex = getattr(material, "baseColorTexture", None) if material is not None else None
    if tex is not None:
        tex = tex.convert("RGBA")
    return vertices, faces, normals, uv, tex


def _roty(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


class Renderer:
    """Reusable offscreen renderer for one loaded mesh.

    Holds the GL context, uploaded buffers and texture so a turntable can render
    many angles without re-uploading geometry.
    """

    def __init__(self, vertices, faces, normals, uv, texture, res=512, clay=False):
        self.res = res
        self.clay = clay or texture is None or uv is None
        self.ss = SUPERSAMPLE
        w = res * self.ss

        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)  # inconsistent winding; keep both sides
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

        # Auto-fit: center on bbox, normalize the longest axis to the frame.
        bmin, bmax = vertices.min(0), vertices.max(0)
        self.center = ((bmin + bmax) * 0.5).astype(np.float32)
        longest = float((bmax - bmin).max())
        self.scale = (2.0 * FILL) / longest  # half-longest -> FILL in NDC

        uv_arr = uv if uv is not None else np.zeros((len(vertices), 2), np.float32)
        interleaved = np.hstack([vertices, normals, uv_arr]).astype("f4")
        self.vbo = self.ctx.buffer(interleaved.tobytes())
        self.ibo = self.ctx.buffer(faces.astype("i4").tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "3f 3f 2f", "in_pos", "in_nrm", "in_uv")],
            self.ibo,
        )

        if not self.clay:
            # No mipmaps: this is a fragmented UV atlas (many tiny charts), and
            # mip minification bleeds neighboring charts together into speckle.
            # Plain bilinear on the base level + supersampling keeps it clean.
            # (V is flipped in the fragment shader, not here.)
            self.texture = self.ctx.texture(texture.size, 4, texture.tobytes())
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        else:
            # Clay: bind a 1x1 dummy so sampler unit 0 is never left unbound
            # (avoids a harmless-but-noisy "texture unloadable" GL warning).
            self.texture = self.ctx.texture((1, 1), 4, b"\xff\xff\xff\xff")

        self.color = self.ctx.texture((w, w), 4)
        self.depth = self.ctx.depth_renderbuffer((w, w))
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color], depth_attachment=self.depth)

        self.prog["u_light"].value = tuple(LIGHT_DIR.tolist())
        self.prog["u_fill"].value = tuple(FILL_DIR.tolist())
        self.prog["u_clay"].value = CLAY_ALBEDO
        self.prog["u_textured"].value = not self.clay

    def _mvp(self, deg):
        ry = _roty(deg)
        m = np.eye(4, dtype=np.float32)
        m[:3, :3] = self.scale * ry
        m[:3, 3] = -self.scale * (ry @ self.center)
        return m, ry

    def render(self, deg):
        """Render one frame at rotation `deg` and return a PIL RGB image."""
        m, ry = self._mvp(deg)
        self.fbo.use()
        self.ctx.clear(BG[0] / 255, BG[1] / 255, BG[2] / 255, 1.0, depth=1.0)
        # moderngl matrix uniforms expect column-major; transpose the row-major numpy op.
        self.prog["u_mvp"].write(np.ascontiguousarray(m.T).tobytes())
        self.prog["u_nrm"].write(np.ascontiguousarray(ry.T).tobytes())
        if self.texture is not None:
            self.texture.use(0)
            self.prog["u_tex"].value = 0
        self.vao.render(moderngl.TRIANGLES)

        w = self.res * self.ss
        data = self.fbo.read(components=3, alignment=1)
        img = Image.frombytes("RGB", (w, w), data).transpose(Image.FLIP_TOP_BOTTOM)
        if self.ss != 1:
            img = img.resize((self.res, self.res), Image.LANCZOS)
        return img

    def release(self):
        for obj in (
            self.vao,
            self.vbo,
            self.ibo,
            self.texture,
            self.color,
            self.depth,
            self.fbo,
            self.ctx,
        ):
            if obj is not None:
                obj.release()


def render_still(path, deg=315, res=512, clay=False):
    """Convenience: load `path` and render a single still at `deg`."""
    r = Renderer(*load_mesh(path), res=res, clay=clay)
    try:
        return r.render(deg)
    finally:
        r.release()


def render_turntable(path, start=315, res=512, frames=36, clay=False):
    """Convenience: load `path` and render `frames` evenly around a full turn."""
    r = Renderer(*load_mesh(path), res=res, clay=clay)
    try:
        return [r.render(start + i * 360.0 / frames) for i in range(frames)]
    finally:
        r.release()


def main():
    ap = argparse.ArgumentParser(description="Headless GLB preview renderer.")
    ap.add_argument("mesh", help="path to a .glb file")
    ap.add_argument("--mode", choices=["still", "turntable"], default="still")
    ap.add_argument("--deg", type=float, default=315.0, help="still angle / turntable start angle")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--frames", type=int, default=36, help="turntable frame count")
    ap.add_argument("--duration", type=int, default=110, help="turntable GIF frame duration (ms)")
    ap.add_argument("--clay", action="store_true", help="flat gray albedo instead of texture")
    ap.add_argument("--out", required=True, help="output path (.png for still, .gif for turntable)")
    args = ap.parse_args()

    if args.mode == "still":
        render_still(args.mesh, deg=args.deg, res=args.res, clay=args.clay).save(args.out)
    else:
        imgs = render_turntable(
            args.mesh, start=args.deg, res=args.res, frames=args.frames, clay=args.clay
        )
        imgs[0].save(
            args.out,
            save_all=True,
            append_images=imgs[1:],
            duration=args.duration,
            loop=0,
            optimize=True,
            disposal=2,
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
