"""Pure-PyTorch / KDTree fallback backends used when the Metal stack is absent.

- ``conv_none``     — pure-PyTorch sparse convolution, installed into TRELLIS.2.
- ``mesh_extract``  — pure-Python dual-grid mesh extraction (o_voxel stand-in).
- ``texture_baker`` — KDTree PBR texture baker (used when o_voxel/Metal is off).
- ``stubs``         — installers for CUDA-only library stubs.
"""
