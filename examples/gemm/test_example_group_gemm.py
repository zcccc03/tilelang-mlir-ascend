"""Target case plus same-allocation group-boundary updates on an Ascend NPU."""

import pytest
import torch

from example_group_gemm import group_gemm, reference
import tilelang


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires an Ascend NPU")
def test_bf16_two_expert_group_gemm():
    generator = torch.Generator().manual_seed(20261126)
    x = (torch.randint(-8, 9, (512, 256), generator=generator).float() / 8).bfloat16()
    w = (torch.randint(-8, 9, (2, 256, 4), generator=generator).float() / 8).bfloat16()
    kernel = tilelang.compile(group_gemm(), target="ascend", out_idx=-1)
    xn, wn = x.npu(), w.npu()
    metadata = torch.tensor([256, 512], dtype=torch.int64, device="npu")
    pool = torch.full((528, 4), float("nan"), dtype=torch.bfloat16, device="npu")
    for split in (256, 0, 511, 1, 512, 257, 256):
        metadata.copy_(torch.tensor([split, 512], dtype=torch.int64))
        pool.fill_(float("nan"))
        kernel(xn, wn, metadata, pool[8:520])
        torch.npu.synchronize()
        torch.testing.assert_close(pool[8:520].cpu(), reference(x, w, [split, 512]), rtol=0, atol=0)
        assert torch.isnan(pool[:8]).all() and torch.isnan(pool[520:]).all()
