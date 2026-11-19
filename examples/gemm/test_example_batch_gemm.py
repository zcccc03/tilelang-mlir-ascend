# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""AscendC Batch GEMM lowering and hardware correctness coverage."""

import pytest
import torch
import tilelang
import tilelang.testing
from example_batch_gemm import batch_gemm as batch_gemm_bf16
from example_batch_gemm_persistent import batch_gemm as batch_gemm_fp16


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
