"""MPS shader regressions, including dispatch and fallback behavior."""

import pytest
import torch

import bitsandbytes
from tests.helpers import describe_dtype, get_available_devices


@pytest.fixture
def mps_blockwise():
    if "mps" not in get_available_devices() or not torch.backends.mps.is_available():
        pytest.skip("MPS device required")
    from bitsandbytes.backends.mps import ops, shaders

    return ops, shaders, ops._quantize_blockwise_compute.__wrapped__


@pytest.fixture
def mps_blockwise_shader(mps_blockwise, monkeypatch):
    ops, shaders, reference = mps_blockwise
    if not hasattr(torch.mps, "compile_shader"):
        pytest.skip("torch.mps.compile_shader required")
    assert shaders.get_library() is not None, "Metal shader compilation failed"

    def forbid_fallback(*args, **kwargs):
        pytest.fail("Expected actual Metal execution, not the PyTorch fallback")

    monkeypatch.setattr(ops, "_quantize_blockwise_compute", forbid_fallback)
    return reference


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize("blocksize", [16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided_offset"])
def test_quantize_blockwise_mps_parity(mps_blockwise_shader, dtype, blocksize, layout):
    a = torch.randn(3, blocksize + 17, dtype=dtype, device="mps")
    if layout == "transpose":
        a = a.T
    elif layout == "strided_offset":
        a = a[:, 1::2]
    code = bitsandbytes.functional.create_dynamic_map().to("mps")

    expected, expected_absmax = mps_blockwise_shader(a.reshape(-1).float(), code, blocksize)
    actual, absmax = torch.ops.bitsandbytes.quantize_blockwise(a, code, blocksize)

    torch.testing.assert_close(actual.cpu(), expected.reshape(a.shape).cpu(), rtol=0, atol=0)
    # MPS equality can hide differences between zero and subnormal values.
    torch.testing.assert_close(absmax.cpu().view(torch.int32), expected_absmax.cpu().view(torch.int32))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize("blocksize", [16, 64, 256, 4096])
@pytest.mark.parametrize("position", ["first", "last_full", "partial"])
def test_quantize_blockwise_mps_nan(mps_blockwise_shader, dtype, blocksize, position):
    a = torch.ones(2 * blocksize + blocksize // 2, dtype=dtype, device="mps")
    index = {"first": 0, "last_full": 2 * blocksize - 1, "partial": a.numel() - 1}[position]
    a[index] = float("nan")
    code = bitsandbytes.functional.create_dynamic_map().to("mps")

    expected, expected_absmax = mps_blockwise_shader(a.float(), code, blocksize)
    actual, absmax = torch.ops.bitsandbytes.quantize_blockwise(a, code, blocksize)

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(absmax.cpu(), expected_absmax.cpu(), rtol=0, atol=0, equal_nan=True)
    block = index // blocksize
    assert torch.isnan(absmax.cpu()[block])
    assert (actual.cpu()[block * blocksize : (block + 1) * blocksize] == 0).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize("blocksize", [16, 64, 256, 4096])
@pytest.mark.parametrize("value", [0.0, 1e-39, -1e-39])
@pytest.mark.parametrize("full_blocks", [0, 2])
def test_quantize_blockwise_mps_partial_absmax(mps_blockwise_shader, dtype, blocksize, value, full_blocks):
    a_cpu = torch.full((full_blocks * blocksize + blocksize // 2,), value, dtype=dtype)
    a = a_cpu.to("mps")
    code = bitsandbytes.functional.create_dynamic_map().to("mps")

    expected, _ = mps_blockwise_shader(a.float(), code, blocksize)
    actual, absmax = torch.ops.bitsandbytes.quantize_blockwise(a, code, blocksize)
    magnitude = a_cpu.float().abs().max().item()
    expected_absmax = torch.tensor([magnitude] * full_blocks + [max(magnitude, 1e-38)], dtype=torch.float32)

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    # Full blocks keep their true statistic; only the partial block is clamped.
    torch.testing.assert_close(absmax.cpu().view(torch.int32), expected_absmax.view(torch.int32))


@pytest.mark.parametrize("reason", ["missing_api", "compile_error", "launch_error"])
def test_quantize_blockwise_mps_fallback(mps_blockwise, monkeypatch, reason):
    ops, shaders, reference = mps_blockwise
    a = torch.randn(513, device="mps")
    code = bitsandbytes.functional.create_dynamic_map().to("mps")
    expected = reference(a, code, 256)
    attempts = []
    fallback_calls = []

    def fail(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError("simulated Metal failure")

    def fallback(*args):
        fallback_calls.append(True)
        return reference(*args)

    monkeypatch.setattr(ops, "_quantize_blockwise_compute", fallback)
    monkeypatch.setattr(shaders, "_library", None)
    monkeypatch.setattr(shaders, "_compile_failed", False)
    if reason == "missing_api":
        monkeypatch.delattr(torch.mps, "compile_shader", raising=False)
    elif reason == "compile_error":
        monkeypatch.setattr(torch.mps, "compile_shader", fail, raising=False)
    else:

        class BrokenLibrary:
            quantize_blockwise = staticmethod(fail)

        monkeypatch.setattr(shaders, "_library", BrokenLibrary())

    for _ in range(2):
        actual = torch.ops.bitsandbytes.quantize_blockwise(a, code, 256)
        for result, wanted in zip(actual, expected):
            torch.testing.assert_close(result.cpu(), wanted.cpu(), rtol=0, atol=0)
    assert len(fallback_calls) == 2
    if reason == "compile_error":
        assert len(attempts) == 1  # Cache failed compilation across calls.


def test_quantize_blockwise_mps_short_codebook(mps_blockwise, monkeypatch):
    ops, shaders, reference = mps_blockwise
    a = torch.randn(513, device="mps")
    code = torch.linspace(-1, 1, 16, device="mps")
    monkeypatch.setattr(ops, "_quantize_blockwise_compute", reference)

    def forbid_shader():
        pytest.fail("A short codebook must not reach the 256-entry shader")

    monkeypatch.setattr(shaders, "get_library", forbid_shader)
    actual = torch.ops.bitsandbytes.quantize_blockwise(a, code, 256)
    for result, wanted in zip(actual, reference(a, code, 256)):
        torch.testing.assert_close(result.cpu(), wanted.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("n, blocksize", [(2**32, 256), (2**32 - 256, 4096), (1, 2**32)])
def test_quantize_blockwise_mps_index_limit(monkeypatch, n, blocksize):
    from bitsandbytes.backends.mps import ops, shaders

    def forbid_shader():
        pytest.fail("An unrepresentable index must not reach the uint shader")

    monkeypatch.setattr(shaders, "get_library", forbid_shader)
    # Meta tensors exercise the guard without allocating multi-gigabyte inputs.
    a = torch.empty(n, device="meta")
    code = torch.empty(256, device="meta")
    assert ops._quantize_blockwise_metal(a, code, blocksize) is None
