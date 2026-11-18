# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Batched GEMM with BF16 inputs and an FP32 result."""

import os

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(out_idx=[-1], target="npuir")
def batch_gemm(batch, m, n, k, block_m, block_n, block_k, cores=28):
    m_tiles = m // block_m
    n_tiles = n // block_n
    tiles_per_batch = m_tiles * n_tiles

    @T.prim_func
    def batch_gemm_bf16(
        A: T.Tensor((batch, m, k), "bfloat16"),
        B: T.Tensor((batch, k, n), "bfloat16"),
        C: T.Tensor((batch, m, n), "float32"),
    ):
        with T.Kernel(cores, is_npu=True) as (cid, _):
            lhs = T.alloc_shared((block_m, block_k), "bfloat16")
            rhs = T.alloc_shared((block_k, block_n), "bfloat16")
            acc = T.alloc_fragment((block_m, block_n), "float32")

            for wave in T.serial(T.ceildiv(batch * tiles_per_batch - cid, cores)):
                tile_id = wave * cores + cid
                batch_id = tile_id // tiles_per_batch
                tile = tile_id % tiles_per_batch
                row = tile // n_tiles * block_m
                col = tile % n_tiles * block_n

                for kt in T.Pipelined(k // block_k, num_stages=2):
                    T.copy(A[batch_id, row, kt * block_k], lhs)
                    T.copy(B[batch_id, kt * block_k, col], rhs)
                    T.gemm(lhs, rhs, acc, initC=(kt == 0))

                T.copy(acc, C[batch_id, row, col])

    return batch_gemm_bf16


def main():
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    batch, m, n, k = 28, 1024, 1024, 1024
    kernel = batch_gemm(batch, m, n, k, 256, 256, 128)

    rng = torch.Generator().manual_seed(0)
    a = (
        torch.randint(-4, 5, (batch, m, k), dtype=torch.int8, generator=rng).bfloat16()
        / 8
    )
    b = (
        torch.randint(-4, 5, (batch, k, n), dtype=torch.int8, generator=rng).bfloat16()
        / 8
    )
    c = kernel(a.npu(), b.npu()).cpu()

    torch.testing.assert_close(c, a.float() @ b.float(), rtol=0, atol=0)
    print("Batch GEMM passed for B=28, M=N=K=1024, BF16 to FP32.")


if __name__ == "__main__":
    main()
