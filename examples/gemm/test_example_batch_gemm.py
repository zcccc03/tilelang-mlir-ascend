# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""AscendC Batch GEMM lowering and hardware correctness coverage."""

import pytest
import torch
import tilelang
import tilelang.testing
from example_batch_gemm import batch_gemm as batch_gemm_bf16
from example_batch_gemm_persistent import batch_gemm as batch_gemm_fp16
from example_gemm import ascend_matmul


pytestmark = pytest.mark.skipif(
    not tilelang.testing.ascend_backend_compiled(),
    reason="Ascend code generation is not built",
)


@pytest.mark.parametrize("persistent", [False, True], ids=["bf16", "fp16-swizzle"])
@pytest.mark.parametrize("case", ["default", "single-k-tile", "tail-wave"])
def test_batch_gemm_lowering(persistent, case):
    # Include a partial core wave and a swizzle window with a wider final group.
    if case == "default":
        program = batch_gemm_fp16(28, 2048, 2048, 2048) if persistent else batch_gemm_bf16(28, 1024, 1024, 1024, 256, 256, 128)
    else:
        batch, m, n, k = 2, 224, 96, 32 if case == "single-k-tile" else 96
        args = (batch, m, n, k, 32, 32, 32, 11)
        program = batch_gemm_fp16(*args, window=3) if persistent else batch_gemm_bf16(*args)
    artifact = tilelang.lower(program, target="ascend")
    assert artifact.kernel_source
    assert "__cube__" in artifact.kernel_source
    assert "asc_mmad" in artifact.kernel_source or "ascend_gemm_l1" in artifact.kernel_source


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("persistent", [False, True], ids=["bf16", "fp16-swizzle"])
@pytest.mark.parametrize(
    "batch,m,n,k,cores,window",
    [(1, 32, 64, 32, 28, 4), (3, 96, 64, 96, 7, 2), (2, 224, 96, 96, 11, 3)],
    ids=["fewer-tiles-than-cores", "multiple-batches", "tail-wave-and-window"],
)
def test_batch_gemm_correctness(persistent, batch, m, n, k, cores, window):
    args = (batch, m, n, k, 32, 32, 32, cores)
    program = batch_gemm_fp16(*args, window=window) if persistent else batch_gemm_bf16(*args)
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)
    generator = torch.Generator().manual_seed(0)
    dtype = torch.float16 if persistent else torch.bfloat16
    a = (torch.randint(-4, 5, (batch, m, k), generator=generator).float() / 8).to(dtype)
    b_shape = (batch, n, k) if persistent else (batch, k, n)
    b = (torch.randint(-4, 5, b_shape, generator=generator).float() / 8).to(dtype)
    expected = a.float() @ (b.float().transpose(-1, -2) if persistent else b.float())
    actual = kernel(a.npu(), b.npu()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()


@pytest.mark.parametrize("persistent", [False, True], ids=["bf16", "fp16-swizzle"])
@pytest.mark.parametrize("tile", [(256, 256, 64), (128, 128, 128)])
def test_batch_gemm_requested_tiles(persistent, tile):
    factory = batch_gemm_fp16 if persistent else batch_gemm_bf16
    artifact = tilelang.lower(factory(28, 1024, 1024, 1024, *tile), target="ascend")
    assert "__cube__" in artifact.kernel_source
    assert "asc_mmad" in artifact.kernel_source or "ascend_gemm_l1" in artifact.kernel_source


def test_ascend_gemm_fp32_output_lowering():
    # Regression: the generic GPU example used an FP32 fragment -> FP16 DMA copy.
    program = ascend_matmul(1024, 1024, 1024)
    assert str(program.buffer_map[program.params[-1]].dtype) == "float32"
    artifact = tilelang.lower(program, target="ascend")
    assert "__cube__" in artifact.kernel_source
    assert "asc_mmad" in artifact.kernel_source or "ascend_gemm_l1" in artifact.kernel_source


@tilelang.testing.requires_ascend
def test_ascend_gemm_correctness():
    from example_gemm import main_ascend

    main_ascend()


def test_gemm_main_selects_ascend(monkeypatch):
    import example_gemm

    monkeypatch.setenv("TILELANG_DEFAULT_TARGET", "ascend")
    calls = []
    monkeypatch.setattr(example_gemm, "main_ascend", lambda compile_only=False: calls.append(compile_only))
    example_gemm.main(compile_only=True)
    assert calls == [True]
