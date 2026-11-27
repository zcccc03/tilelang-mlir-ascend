# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""AscendC Batch GEMM lowering and hardware correctness coverage."""

import pytest
import torch
import tilelang
import tilelang.testing
from example_batch_gemm import batch_gemm as batch_gemm_bf16
from example_batch_gemm_persistent import batch_gemm as batch_gemm_owned
from example_gemm import ascend_matmul


pytestmark = pytest.mark.skipif(
    not tilelang.testing.ascend_backend_compiled(),
    reason="Ascend code generation is not built",
)


@pytest.mark.parametrize("batch_owned", [False, True], ids=["distributed", "batch-owned"])
def test_batch_gemm_lowering(batch_owned):
    factory = batch_gemm_owned if batch_owned else batch_gemm_bf16
    artifact = tilelang.lower(factory(28, 2048, 2048, 2048), target="ascend")
    assert "__cube__" in artifact.kernel_source
    assert "asc_mmad" in artifact.kernel_source


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("batch_owned", [False, True], ids=["distributed", "batch-owned"])
def test_batch_gemm_correctness(batch_owned):
    factory = batch_gemm_owned if batch_owned else batch_gemm_bf16
    kernel = tilelang.compile(factory(28, 2048, 2048, 2048), target="ascend", out_idx=-1)
    generator = torch.Generator().manual_seed(0)
    dtype = torch.bfloat16
    a = torch.randint(-4, 5, (28, 2048, 2048), generator=generator).to(dtype) / 8
    b = torch.randint(-4, 5, (28, 2048, 2048), generator=generator).to(dtype) / 8
    actual = kernel(a.npu(), b.npu()).cpu()
    for i in range(28):
        expected = a[i].float() @ b[i].float().T
        torch.testing.assert_close(actual[i], expected, rtol=0, atol=0)


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
