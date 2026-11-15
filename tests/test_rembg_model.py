"""CPU-only tests for the background-remover switch. No network, no weights.

- rembg_model: the pinned MIT BiRefNet is the default; REMBG_MODEL picks
  another repo; the load passes the pinned revision for weights and remote
  code, and casts to fp32.
- patch_rembg_model: applies to a fresh upstream BiRefNet.py and to one that
  patch_birefnet already rewrote (existing installs), in either order, and is
  idempotent. The patched class loads through rembg_model, not model_name.
"""

import importlib.util
import io
import sys
import types

import pytest

from trellis_silicon import rembg_model
from trellis_silicon.patches import device as device_patch
from trellis_silicon.patches import rembg as rembg_patch

# trellis2/pipelines/rembg/BiRefNet.py at the pinned TRELLIS.2 commit (75fbf01).
_UPSTREAM = (
    "from typing import *\n"
    "from transformers import AutoModelForImageSegmentation\n"
    "import torch\n"
    "from torchvision import transforms\n"
    "from PIL import Image\n"
    "\n"
    "\n"
    "class BiRefNet:\n"
    '    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):\n'
    "        self.model = AutoModelForImageSegmentation.from_pretrained(\n"
    "            model_name, trust_remote_code=True\n"
    "        )\n"
    "        self.model.eval()\n"
    "        self.transform_image = transforms.Compose(\n"
    "            [\n"
    "                transforms.Resize((1024, 1024)),\n"
    "                transforms.ToTensor(),\n"
    "                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),\n"
    "            ]\n"
    "        )\n"
    "    \n"
    "    def to(self, device: str):\n"
    "        self.model.to(device)\n"
    "\n"
    "    def cuda(self):\n"
    "        self.model.cuda()\n"
    "\n"
    "    def cpu(self):\n"
    "        self.model.cpu()\n"
    "        \n"
    "    def __call__(self, image: Image.Image) -> Image.Image:\n"
    "        image_size = image.size\n"
    '        input_images = self.transform_image(image).unsqueeze(0).to("cuda")\n'
    "        # Prediction\n"
    "        with torch.no_grad():\n"
    "            preds = self.model(input_images)[-1].sigmoid().cpu()\n"
    "        pred = preds[0].squeeze()\n"
    "        pred_pil = transforms.ToPILImage()(pred)\n"
    "        mask = pred_pil.resize(image_size)\n"
    "        image.putalpha(mask)\n"
    "        return image\n"
)

_PIN = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"


class _FakeModel:
    def __init__(self):
        self.calls = []

    def float(self):
        self.calls.append("float")
        return self

    def eval(self):
        self.calls.append("eval")
        return self


class _FakeAuto:
    """Stands in for AutoModelForImageSegmentation; records from_pretrained."""

    loads = []

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        model = _FakeModel()
        cls.loads.append((name, kwargs, model))
        return model


@pytest.fixture(autouse=True)
def _reset_fake():
    _FakeAuto.loads = []


# --- rembg_model -----------------------------------------------------------


def test_default_revision_is_a_full_sha():
    assert rembg_model.DEFAULT_REVISION == _PIN
    assert len(_PIN) == 40 and all(c in "0123456789abcdef" for c in _PIN)


@pytest.mark.parametrize("env", [{}, {"REMBG_MODEL": ""}, {"REMBG_MODEL": "  "}])
def test_default_is_pinned_birefnet(env):
    assert rembg_model.rembg_source(env) == ("ZhengPeng7/BiRefNet", _PIN)


def test_naming_the_default_keeps_the_pin():
    env = {"REMBG_MODEL": " ZhengPeng7/BiRefNet "}
    assert rembg_model.rembg_source(env) == ("ZhengPeng7/BiRefNet", _PIN)


def test_opt_in_loads_the_named_repo_at_main():
    env = {"REMBG_MODEL": "briaai/RMBG-2.0"}
    assert rembg_model.rembg_source(env) == ("briaai/RMBG-2.0", None)


def test_load_default_pins_weights_and_code_and_casts_fp32():
    out = io.StringIO()
    model = rembg_model.load_rembg_model(_FakeAuto, "briaai/RMBG-2.0", {}, out)

    [(name, kwargs, loaded)] = _FakeAuto.loads
    assert name == "ZhengPeng7/BiRefNet"
    assert kwargs == {"trust_remote_code": True, "revision": _PIN, "code_revision": _PIN}
    assert model is loaded and loaded.calls == ["float"]
    assert out.getvalue() == (
        "[REMBG] Background remover: ZhengPeng7/BiRefNet @ e2bf8e4460fc; "
        "pipeline.json asked for briaai/RMBG-2.0 (fp32)\n"
    )


def test_load_opt_in_uses_env_model_unpinned():
    out = io.StringIO()
    env = {"REMBG_MODEL": "briaai/RMBG-2.0"}
    rembg_model.load_rembg_model(_FakeAuto, "briaai/RMBG-2.0", env, out)

    [(name, kwargs, loaded)] = _FakeAuto.loads
    assert name == "briaai/RMBG-2.0"
    assert kwargs == {"trust_remote_code": True, "revision": None, "code_revision": None}
    assert loaded.calls == ["float"]
    assert out.getvalue() == "[REMBG] Background remover: briaai/RMBG-2.0 @ main (fp32)\n"


# --- patch_rembg_model -----------------------------------------------------


def _write_birefnet(root, text):
    path = root / "trellis2/pipelines/rembg/BiRefNet.py"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    return path


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(rembg_patch, "TRELLIS_ROOT", str(tmp_path))
    monkeypatch.setattr(device_patch, "TRELLIS_ROOT", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize(
    "order",
    [
        ("rembg",),
        ("birefnet", "rembg"),  # an existing install: old patch already applied
        ("rembg", "birefnet"),
    ],
    ids=["fresh", "after-device-patch", "before-device-patch"],
)
def test_rembg_patch_applies_and_is_idempotent(root, order):
    path = _write_birefnet(root, _UPSTREAM)
    patchers = {"birefnet": device_patch.patch_birefnet, "rembg": rembg_patch.patch_rembg_model}
    for name in order:
        patchers[name]()
    once = path.read_text()

    assert rembg_patch._INIT_LOAD_PATCHED in once
    assert rembg_patch._INIT_LOAD_UPSTREAM not in once
    assert once.count("load_rembg_model(") == 1
    if "birefnet" in order:
        assert "def device(self)" in once

    for name in order:
        patchers[name]()
    assert path.read_text() == once


def test_rembg_patch_fails_loudly_on_unknown_source(root):
    _write_birefnet(root, "class BiRefNet:\n    pass\n")
    with pytest.raises(RuntimeError, match="patch target not found"):
        rembg_patch.patch_rembg_model()


def _import_patched(path, monkeypatch):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageSegmentation = _FakeAuto
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    spec = importlib.util.spec_from_file_location("_patched_birefnet", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "env, expected",
    [
        (None, ("ZhengPeng7/BiRefNet", _PIN)),
        ("briaai/RMBG-2.0", ("briaai/RMBG-2.0", None)),
    ],
    ids=["default", "opt-in"],
)
def test_patched_class_loads_through_rembg_model(root, monkeypatch, capsys, env, expected):
    if env is None:
        monkeypatch.delenv("REMBG_MODEL", raising=False)
    else:
        monkeypatch.setenv("REMBG_MODEL", env)
    path = _write_birefnet(root, _UPSTREAM)
    device_patch.patch_birefnet()
    rembg_patch.patch_rembg_model()
    mod = _import_patched(path, monkeypatch)

    remover = mod.BiRefNet(model_name="briaai/RMBG-2.0")  # what pipeline.json passes

    [(name, kwargs, loaded)] = _FakeAuto.loads
    assert (name, kwargs["revision"]) == expected
    assert loaded.calls == ["float", "eval"]
    assert remover.model is loaded
    assert f"[REMBG] Background remover: {expected[0]}" in capsys.readouterr().out
