"""Device patches: replace hardcoded CUDA (`.cuda()`, `to("cuda")`) with
device-agnostic code so the models run on MPS."""

import os

from .common import TRELLIS_ROOT, read_file, write_file


def patch_image_feature_extractor():
    """Add device property and replace .cuda() calls with device-aware code."""
    path = os.path.join(TRELLIS_ROOT, "trellis2/modules/image_feature_extractor.py")
    src = read_file(path)

    if "def device(self)" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    # DinoV2FeatureExtractor: add device property, fix cuda()
    src = src.replace(
        "    def to(self, device):\n"
        "        self.model.to(device)\n"
        "\n"
        "    def cuda(self):\n"
        "        self.model.cuda()\n"
        "\n"
        "    def cpu(self):\n"
        "        self.model.cpu()\n"
        "    \n"
        "    @torch.no_grad()\n"
        "    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:\n"
        '        """\n'
        "        Extract features from the image.",
        "    @property\n"
        "    def device(self):\n"
        "        return next(self.model.parameters()).device\n"
        "\n"
        "    def to(self, device):\n"
        "        self.model.to(device)\n"
        "\n"
        "    def cuda(self):\n"
        "        self.model.to(self.device)\n"
        "\n"
        "    def cpu(self):\n"
        "        self.model.cpu()\n"
        "\n"
        "    @torch.no_grad()\n"
        "    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:\n"
        '        """\n'
        "        Extract features from the image.",
        1,  # only first occurrence
    )

    # Fix hardcoded .cuda() in both extractors
    src = src.replace(
        "            image = torch.stack(image).cuda()",
        "            image = torch.stack(image).to(self.device)",
    )
    src = src.replace(
        "        image = self.transform(image).cuda()",
        "        image = self.transform(image).to(self.device)",
    )

    # DinoV3FeatureExtractor: add device property, fix cuda()
    # The second class has the same to/cuda/cpu pattern
    src = src.replace(
        "    def to(self, device):\n"
        "        self.model.to(device)\n"
        "\n"
        "    def cuda(self):\n"
        "        self.model.cuda()\n"
        "\n"
        "    def cpu(self):\n"
        "        self.model.cpu()",
        "    @property\n"
        "    def device(self):\n"
        "        return next(self.model.parameters()).device\n"
        "\n"
        "    def to(self, device):\n"
        "        self.model.to(device)\n"
        "\n"
        "    def cuda(self):\n"
        "        self.model.to(self.device)\n"
        "\n"
        "    def cpu(self):\n"
        "        self.model.cpu()",
    )

    # Fix DINOv3 model.layer -> model.model.layer (HuggingFace structure)
    src = src.replace(
        "        for i, layer_module in enumerate(self.model.layer):",
        "        layers = self.model.model.layer if hasattr(self.model, 'model') and hasattr(self.model.model, 'layer') else self.model.layer\n"
        "        for i, layer_module in enumerate(layers):",
    )

    write_file(path, src)


def patch_birefnet():
    """Add device property and fix hardcoded .cuda()/.to('cuda') calls."""
    path = os.path.join(TRELLIS_ROOT, "trellis2/pipelines/rembg/BiRefNet.py")
    src = read_file(path)

    if "def device(self)" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    # Replace to/cuda/cpu block
    src = src.replace(
        "    def to(self, device: str):\n"
        "        self.model.to(device)\n"
        "\n"
        "    def cuda(self):\n"
        "        self.model.cuda()\n"
        "\n"
        "    def cpu(self):\n"
        "        self.model.cpu()",
        "    @property\n"
        "    def device(self):\n"
        "        return next(self.model.parameters()).device\n"
        "\n"
        "    def to(self, device):\n"
        "        self.model.to(device)\n"
        "        return self\n"
        "\n"
        "    def cuda(self):\n"
        "        self.model.to(self.device)\n"
        "\n"
        "    def cpu(self):\n"
        "        self.model.cpu()",
    )

    # Fix hardcoded .to("cuda") in __call__
    src = src.replace(
        '.unsqueeze(0).to("cuda")',
        ".unsqueeze(0).to(self.device)",
    )

    write_file(path, src)


def patch_pipeline():
    """Guard torch.cuda.empty_cache() call."""
    path = os.path.join(TRELLIS_ROOT, "trellis2/pipelines/trellis2_image_to_3d.py")
    src = read_file(path)

    if "if torch.cuda.is_available():" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    src = src.replace(
        "        torch.cuda.empty_cache()\n",
        "        if torch.cuda.is_available():\n            torch.cuda.empty_cache()\n",
    )
    write_file(path, src)


def patch_pipeline_base():
    """Fix hardcoded cuda device in Pipeline.cuda()."""
    path = os.path.join(TRELLIS_ROOT, "trellis2/pipelines/base.py")
    src = read_file(path)

    if "torch.backends.mps.is_available()" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    src = src.replace(
        '        self.to(torch.device("cuda"))',
        '        self.to(torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda"))',
    )
    write_file(path, src)
