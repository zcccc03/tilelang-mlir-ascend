# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""AscendC batched GEMM with BF16 inputs and an FP32 result."""

import argparse

import tilelang
import tilelang.ascend.language as T
import torch


def batch_gemm(batch, m, n, k, block_m, block_n, block_k, cores=28):
    if min(batch, m, n, k, block_m, block_n, block_k, cores) <= 0:
        raise ValueError("Shapes, tile dimensions and core count must be positive")
    if block_m % 16 or block_n % 16 or block_k % 32:
        raise ValueError("M/N tiles must be multiples of 16 and K tiles of 32")
    if m % block_m or n % block_n or k % block_k:
        raise ValueError("M, N and K must be divisible by their tile sizes")
    m_tiles = m // block_m
    n_tiles = n // block_n
    tiles_per_batch = m_tiles * n_tiles
    tiles = batch * tiles_per_batch
    cores = min(cores, tiles)
    stages = min(2, k // block_k)
    # Reserve half of each 64 KiB L0 operand buffer for pipeline overlap.
    l0_k = min(block_k, 16384 // max(block_m, block_n)) // 32 * 32
    while l0_k > 0 and block_k % l0_k:
        l0_k -= 32
    if l0_k == 0:
        raise ValueError("M/N tiles are too large for double-buffered L0 operands")

    @T.prim_func
    def batch_gemm_bf16(
        A: T.Tensor((batch, m, k), "bfloat16"),
        B: T.Tensor((batch, k, n), "bfloat16"),
        C: T.Tensor((batch, m, n), "float32"),
    ):
        with T.Kernel(cores) as cid:
            lhs = T.alloc_l1((block_m, block_k), "bfloat16")
            rhs = T.alloc_l1((block_k, block_n), "bfloat16")
            lhs_l0 = T.alloc_l0a((block_m, l0_k), "bfloat16")
            rhs_l0 = T.alloc_l0b((block_n, l0_k), "bfloat16")
            acc = T.alloc_l0c((block_m, block_n), "float32")

            for tile_id in T.Persistent([tiles], cores, cid):
                batch_id = tile_id // tiles_per_batch
                tile = tile_id % tiles_per_batch
                row = tile // n_tiles * block_m
                col = tile % n_tiles * block_n

                for kt in T.Pipelined(k // block_k, num_stages=stages):
                    T.copy(A[batch_id, row, kt * block_k], lhs)
                    T.copy(B[batch_id, kt * block_k, col], rhs)
                    # L1 GEMM requires NT operands. Transpose B during the
                    # L1-to-L0B copy to preserve the public [batch, K, N] input.
                    for sub in T.serial(block_k // l0_k):
                        T.copy(lhs[:, sub * l0_k : (sub + 1) * l0_k], lhs_l0)
                        T.copy(rhs[sub * l0_k : (sub + 1) * l0_k, :], rhs_l0, transpose=True)
                        T.gemm(lhs_l0, rhs_l0, acc, transpose_B=True, clear_accum=(kt == 0) & (sub == 0))

                T.copy(acc, C[batch_id, row, col])

    return batch_gemm_bf16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=28)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--block-m", type=int, default=256)
    parser.add_argument("--block-n", type=int, default=256)
    parser.add_argument("--block-k", type=int, default=128)
    parser.add_argument("--cores", type=int, default=28)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    batch, m, n, k = args.batch, args.m, args.n, args.k
    program = batch_gemm(batch, m, n, k, args.block_m, args.block_n, args.block_k, args.cores)
    kernel = tilelang.compile(program, out_idx=-1, target="ascend", execution_backend="tvm_ffi")
    if args.compile_only:
        print("COMPILE_PASS")
        return

    rng = torch.Generator().manual_seed(0)
    a = torch.randint(-4, 5, (batch, m, k), dtype=torch.int8, generator=rng).bfloat16() / 8
    b = torch.randint(-4, 5, (batch, k, n), dtype=torch.int8, generator=rng).bfloat16() / 8
    c = kernel(a.npu(), b.npu()).cpu()

    torch.testing.assert_close(c, a.float() @ b.float(), rtol=0, atol=0)
    print(f"Batch GEMM passed for B={batch}, M={m}, N={n}, K={k}, BF16 to FP32.")


if __name__ == "__main__":
    main()
