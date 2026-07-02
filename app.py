"""
Gradio demo UI for TRELLIS.2 image-to-3D on Apple Silicon.

Run:  python app.py   → opens on http://127.0.0.1:7860 (share=False).

Shares one code path with the CLI via pipeline_core. The pipeline is loaded
lazily on the first Generate click (so the UI opens instantly) and cached in a
module global keyed by pipeline_type; switching pipeline_type replaces the
cached pipeline rather than holding two ~9GB model sets resident at once.
"""

import os
import time
import uuid
import gc

# pipeline_core MUST be imported before torch is imported anywhere — importing
# it performs the backend/env setup (PYTORCH_ENABLE_MPS_FALLBACK, ATTN/CONV
# backends, sys.path). Keep it above gradio too, since gradio pulls in numpy/
# torch transitively in some setups. Do not reorder these.
import pipeline_core
from pipeline_core import load_pipeline, generate_glb, WatchdogEmptyMeshError

import gradio as gr


OUTPUT_DIR = "gradio_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cached pipeline, keyed by pipeline_type. Only one is ever held resident;
# loading a different type frees the previous one first (see _ensure_pipeline).
_PIPELINE = None
_PIPELINE_TYPE = None


def _free_pipeline():
    """Drop the cached pipeline and reclaim its unified memory."""
    global _PIPELINE, _PIPELINE_TYPE
    _PIPELINE = None
    _PIPELINE_TYPE = None
    gc.collect()
    # torch is already imported via pipeline_core; empty the MPS allocator cache
    # so the freed ~9GB isn't held by the caching allocator before the reload.
    try:
        pipeline_core.torch.mps.empty_cache()
    except Exception:
        pass


def _format_status(result, load_time):
    lines = [f"Done — {result['vertices']:,} vertices, {result['triangles']:,} triangles."]
    if load_time:
        lines.append(f"Pipeline load: {load_time:.0f}s")
    lines.append(f"Generation: {result['gen_time']:.1f}s")
    if result.get("bake_time") is not None:
        lines.append(f"Texture bake ({result['backend']}): {result['bake_time']:.0f}s")
    lines.append(f"Output: {result['glb_path']}")
    return "\n".join(lines)


def generate(image, seed, pipeline_type, texture_size):
    """Gradio click handler. Generator: yields (Model3D, status) so the status
    box updates while the (long) load + generation run."""
    global _PIPELINE, _PIPELINE_TYPE

    if image is None:
        # Friendly validation error rather than a worker crash.
        raise gr.Error("Please upload an image first.")

    seed = int(seed)
    texture_size = int(texture_size)

    # Lazy load / swap the cached pipeline.
    if _PIPELINE is None or _PIPELINE_TYPE != pipeline_type:
        if _PIPELINE is not None:
            _free_pipeline()
        yield gr.update(), f"Loading {pipeline_type} pipeline (~1 min on first run)…"
        t0 = time.time()
        _PIPELINE = load_pipeline(pipeline_type)
        _PIPELINE_TYPE = pipeline_type
        load_time = time.time() - t0
    else:
        load_time = 0.0

    yield gr.update(), "Generating (~4-5 min)… progress prints to the terminal."

    # Unique output base so near-simultaneous sessions don't clobber each other.
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_base = os.path.join(OUTPUT_DIR, f"trellis_{stamp}_{uuid.uuid4().hex[:8]}")

    try:
        result = generate_glb(
            _PIPELINE, image,
            seed=seed,
            pipeline_type=pipeline_type,
            texture_size=texture_size,
            no_texture=False,
            output_base=output_base,
        )
    except WatchdogEmptyMeshError as e:
        # Surface the existing help text in the status box instead of crashing
        # the queue worker.
        yield gr.update(value=None), str(e)
        return

    yield result["glb_path"], _format_status(result, load_time)


def build_demo():
    with gr.Blocks(title="TRELLIS.2 — Image to 3D (Apple Silicon)") as demo:
        gr.Markdown(
            "# TRELLIS.2 — Image to 3D\n"
            "Upload an image and generate a textured 3D mesh. First generation "
            "loads the pipeline (~1 min); each generation then takes ~4-5 min on "
            "Apple Silicon. One generation runs at a time."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Input image")
                seed_in = gr.Number(value=42, precision=0, label="Seed")
                pipeline_in = gr.Radio(
                    choices=["512", "1024", "1024_cascade"], value="512",
                    label="Pipeline type",
                )
                texture_in = gr.Radio(
                    choices=["512", "1024", "2048"], value="1024",
                    label="Texture size",
                )
                generate_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                model_out = gr.Model3D(label="Output GLB", clear_color=[0.1, 0.1, 0.1, 1.0])
                status_out = gr.Textbox(
                    label="Status", value="Ready. Upload an image and click Generate.",
                    lines=8, interactive=False,
                )

        generate_btn.click(
            fn=generate,
            inputs=[image_in, seed_in, pipeline_in, texture_in],
            outputs=[model_out, status_out],
        )

    # Exactly one generation at a time — the pipeline holds a single resident
    # model set and is not safe to run concurrently on MPS.
    demo.queue(default_concurrency_limit=1)
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
