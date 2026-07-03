"""TRELLIS.2 image-to-3D generation, native on Apple Silicon (PyTorch MPS).

Importing :mod:`trellis_silicon.core` performs backend/environment setup that
MUST run before torch or the vendored ``trellis2`` package are imported. The
public entry points (:func:`core.load_pipeline`, :func:`core.generate_glb`) and
the CLI/webui front-ends all go through that module.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
