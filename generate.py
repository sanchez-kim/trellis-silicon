"""
Generate a 3D mesh from a single image using TRELLIS.2 on Apple Silicon.

Thin CLI wrapper over pipeline_core, which owns the shared load + generate +
bake logic (also used by the Gradio UI in app.py). Importing pipeline_core
performs the backend/env setup that MUST run before torch is imported — so it
is imported at module top here, before argparse/torch, exactly as the inline
setup block used to sit at the top of this file.
"""

import sys
import os

import pipeline_core
from pipeline_core import load_pipeline, generate_glb, WatchdogEmptyMeshError

import argparse
import time


def main():
    parser = argparse.ArgumentParser(description="Generate 3D mesh from an image using TRELLIS.2")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", default="output_3d", help="Output filename without extension (default: output_3d)")
    parser.add_argument(
        "--pipeline-type", default="512",
        choices=["512", "1024", "1024_cascade"],
        help="Pipeline resolution (default: 512)",
    )
    parser.add_argument(
        "--texture-size", type=int, default=1024,
        choices=[512, 1024, 2048],
        help="Texture resolution for PBR baking (default: 1024)",
    )
    parser.add_argument(
        "--no-texture", action="store_true",
        help="Skip texture baking, export geometry only",
    )
    parser.add_argument(
        "--obj", action="store_true",
        help="Also export untextured OBJ geometry (default: GLB only)",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override sampler steps for all three flow phases (default: pipeline JSON, usually 12)",
    )
    parser.add_argument(
        "--resident", action="store_true",
        help="Keep all needed models resident on MPS for the whole run instead of "
             "shuffling each submodel CPU<->MPS around its phase. Measured slower "
             "than the default on a 32GB machine (resident weights add unified-memory "
             "pressure that outweighs the saved transfers); may pay off on 64GB+.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found")
        sys.exit(1)

    print("=" * 60)
    print("TRELLIS.2 on Apple Silicon")
    print("=" * 60)

    # Load pipeline
    print("\nLoading pipeline...")
    t0 = time.time()
    pipeline = load_pipeline(args.pipeline_type, resident=args.resident)
    print(f"Loaded in {time.time() - t0:.0f}s")
    print(f"Device: MPS (low_vram={pipeline.low_vram})")

    # Load image
    img = pipeline_core.PILImage.open(args.image)
    print(f"Input: {args.image} ({img.size[0]}x{img.size[1]})")

    # Generate
    print(f"\nGenerating 3D model (pipeline={args.pipeline_type}, seed={args.seed})...")
    try:
        result = generate_glb(
            pipeline, img,
            seed=args.seed,
            pipeline_type=args.pipeline_type,
            texture_size=args.texture_size,
            no_texture=args.no_texture,
            output_base=args.output,
            steps=args.steps,
        )
    except WatchdogEmptyMeshError as e:
        # Empty mesh from the GPU watchdog: print the help text and exit 2,
        # exactly as before the refactor (str(e) is the full help message).
        print(str(e))
        sys.exit(2)

    # Also save OBJ (CLI-only; the UI never needs untextured OBJ)
    if args.obj:
        verts = result["verts"]
        faces = result["faces"]
        obj_path = f"{args.output}.obj"
        with open(obj_path, "w") as f:
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        print(f"Saved: {obj_path}")

    print(f"\nTotal time: {result['gen_time']:.1f}s generation + baking")


if __name__ == "__main__":
    main()
