"""CPU-only tests for the sparse-conv backend selection.

- convbackend.select_sparse_conv_backend: an explicit SPARSE_CONV_BACKEND is
  left alone; otherwise flex_gemm is probed, and a failed probe falls back to
  ``none`` with exactly one stderr line that names the error.
- core.py import order: torch is imported before the flex_gemm probe runs
  (flex_gemm's _C only finds libc10 through an already-loaded torch when its
  baked-in rpath is stale), and the env defaults stay above torch.
"""

import ast
import io
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from trellis_silicon import convbackend

CORE = Path(__file__).resolve().parents[1] / "src" / "trellis_silicon" / "core.py"

# What dlopen reports for a flex_gemm _C built in a venv that has since moved.
_DLOPEN_ERR = ImportError(
    "dlopen(/x/.venv/lib/python3.11/site-packages/flex_gemm/kernels/metal/"
    "_C.cpython-311-darwin.so, 0x0002): Library not loaded: @rpath/libc10.dylib\n"
    "  Referenced from: <UUID> /x/_C.cpython-311-darwin.so\n"
    "  Reason: tried: '/old/trellis-mac/.venv/.../libc10.dylib' (no such file)"
)


def _importer(exc=None):
    calls = []

    def fake_import(name):
        calls.append(name)
        if exc is not None:
            raise exc

    return fake_import, calls


@pytest.mark.parametrize("value", ["none", "spconv", "torchsparse"])
def test_explicit_backend_is_respected_without_probing(value):
    env = {"SPARSE_CONV_BACKEND": value}
    fake_import, calls = _importer(ImportError("must not be called"))
    out = io.StringIO()

    assert convbackend.select_sparse_conv_backend(env, fake_import, out) == value
    assert env == {"SPARSE_CONV_BACKEND": value}
    assert calls == []
    assert out.getvalue() == ""


def test_flex_gemm_selected_when_importable():
    env = {}
    fake_import, calls = _importer()
    out = io.StringIO()

    assert convbackend.select_sparse_conv_backend(env, fake_import, out) == "flex_gemm"
    assert env == {"SPARSE_CONV_BACKEND": "flex_gemm"}
    assert calls == ["flex_gemm"]
    assert out.getvalue() == ""


@pytest.mark.parametrize(
    "exc, reason",
    [
        (_DLOPEN_ERR, "Library not loaded: @rpath/libc10.dylib"),
        (ModuleNotFoundError("No module named 'flex_gemm'"), "No module named 'flex_gemm'"),
        (
            RuntimeError("Failed to load metallib: language version 4.0 is not supported"),
            "Failed to load metallib",
        ),
        (OSError("dlopen failed"), "dlopen failed"),
    ],
    ids=["stale-rpath", "not-installed", "metallib", "oserror"],
)
def test_fallback_prints_one_line_with_reason(exc, reason):
    env = {}
    fake_import, _ = _importer(exc)
    out = io.StringIO()

    assert convbackend.select_sparse_conv_backend(env, fake_import, out) == "none"
    assert env == {"SPARSE_CONV_BACKEND": "none"}
    text = out.getvalue()
    assert text.count("\n") == 1 and text.endswith("\n")
    assert type(exc).__name__ in text
    assert reason in text
    assert "SPARSE_CONV_BACKEND=none" in text
    assert "slow" in text


def test_notice_truncates_long_reasons():
    line = convbackend.fallback_notice(ImportError("x" * 1000))
    assert "\n" not in line
    assert len(line) < 400


def test_unexpected_errors_are_not_swallowed():
    fake_import, _ = _importer(ValueError("bug inside flex_gemm"))
    with pytest.raises(ValueError):
        convbackend.select_sparse_conv_backend({}, fake_import, io.StringIO())


def test_explicit_flex_gemm_is_probed_and_selected():
    env = {"SPARSE_CONV_BACKEND": "flex_gemm"}
    fake_import, calls = _importer()
    out = io.StringIO()

    assert convbackend.select_sparse_conv_backend(env, fake_import, out) == "flex_gemm"
    assert env == {"SPARSE_CONV_BACKEND": "flex_gemm"}
    assert calls == ["flex_gemm"]
    assert out.getvalue() == ""


def test_explicit_flex_gemm_failure_raises_with_notice():
    env = {"SPARSE_CONV_BACKEND": "flex_gemm"}
    fake_import, _ = _importer(_DLOPEN_ERR)
    out = io.StringIO()

    with pytest.raises(ImportError) as info:
        convbackend.select_sparse_conv_backend(env, fake_import, out)
    assert str(info.value) == convbackend.fallback_notice(_DLOPEN_ERR)
    assert "\n" not in str(info.value)
    assert out.getvalue() == ""


@pytest.mark.parametrize("value", ["None", "NONE", "conv_none", "bogus", "Flex_Gemm"])
def test_invalid_value_warns_then_probes_as_unset(value):
    env = {"SPARSE_CONV_BACKEND": value}
    fake_import, calls = _importer()
    out = io.StringIO()

    assert convbackend.select_sparse_conv_backend(env, fake_import, out) == "flex_gemm"
    assert env == {"SPARSE_CONV_BACKEND": "flex_gemm"}
    assert calls == ["flex_gemm"]
    text = out.getvalue()
    assert text.count("\n") == 1
    assert repr(value) in text
    for valid in ("none", "spconv", "torchsparse", "flex_gemm"):
        assert valid in text


def test_invalid_value_with_failed_probe_falls_back():
    env = {"SPARSE_CONV_BACKEND": "conv_none"}
    fake_import, _ = _importer(ImportError("No module named 'flex_gemm'"))
    out = io.StringIO()

    assert convbackend.select_sparse_conv_backend(env, fake_import, out) == "none"
    assert env == {"SPARSE_CONV_BACKEND": "none"}
    lines = out.getvalue().splitlines()
    assert len(lines) == 2
    assert "invalid" in lines[0]
    assert "flex_gemm unavailable" in lines[1]


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_value_counts_as_unset(value):
    env = {"SPARSE_CONV_BACKEND": value}
    fake_import, calls = _importer()
    out = io.StringIO()

    assert convbackend.select_sparse_conv_backend(env, fake_import, out) == "flex_gemm"
    assert env == {"SPARSE_CONV_BACKEND": "flex_gemm"}
    assert calls == ["flex_gemm"]
    assert out.getvalue() == ""


def _is_flex_module(name):
    return (name or "").split(".")[0] == "flex_gemm"


def _call_name(call):
    return getattr(call.func, "id", None) or getattr(call.func, "attr", None)


def _module_level_lines(tree):
    """Line numbers of torch imports (module level) and flex_gemm imports,
    env defaults and probe calls (anywhere in the module)."""
    torch_imports, flex_imports, setdefaults, probes = [], [], [], []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(a.name == "torch" for a in node.names):
            torch_imports.append(node.lineno)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_is_flex_module(a.name) for a in node.names):
                flex_imports.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if _is_flex_module(node.module):
                flex_imports.append(node.lineno)
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name == "import_module" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and _is_flex_module(arg.value):
                    flex_imports.append(node.lineno)
        if isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, ast.Call):
                name = _call_name(value)
                if name == "select_sparse_conv_backend":
                    probes.append(node.lineno)
                elif name == "setdefault":
                    setdefaults.append(node.lineno)
    return torch_imports, flex_imports, setdefaults, probes


def test_core_imports_torch_before_probing_flex_gemm():
    tree = ast.parse(CORE.read_text())
    torch_imports, flex_imports, setdefaults, probes = _module_level_lines(tree)

    assert len(torch_imports) == 1, "core.py should import torch exactly once at module level"
    assert flex_imports == [], "core.py must not import flex_gemm directly; use convbackend"
    assert len(probes) == 1
    (torch_line,) = torch_imports
    assert torch_line < probes[0]
    # PYTORCH_ENABLE_MPS_FALLBACK and the attention defaults must be set before torch.
    assert setdefaults and max(setdefaults) < torch_line


def test_flex_gemm_probe_runs_after_torch_import(tmp_path):
    """Hermetic: a fake flex_gemm asserts torch is already in sys.modules."""
    (tmp_path / "torch.py").write_text("")
    (tmp_path / "flex_gemm").mkdir()
    (tmp_path / "flex_gemm" / "__init__.py").write_text(
        'import sys\nassert "torch" in sys.modules, "flex_gemm imported before torch"\n'
    )
    code = textwrap.dedent(
        """
        import os
        import torch  # the fake one
        from trellis_silicon.convbackend import select_sparse_conv_backend
        print(select_sparse_conv_backend())
        """
    )
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(tmp_path) + ":" + src,
        "PYTHONNOUSERSITE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "flex_gemm"
    assert result.stderr == ""
