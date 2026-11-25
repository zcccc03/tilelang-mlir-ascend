# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""AscendC batched GEMM with BF16 inputs and an FP32 result."""

import argparse

import tilelang
import tilelang.ascend.language as T
import torch


def batch_gemm(batch, m, n, k, block_m, block_n, block_k, cores=28, l1_k=None, stages=2, window=4, prefetch=True, split_m=1):
    if min(batch, m, n, k, block_m, block_n, block_k, cores, stages, window) <= 0:
        raise ValueError("Shapes, tile dimensions and core count must be positive")
    if block_m % 16 or block_n % 16 or block_k % 32:
        raise ValueError("M/N tiles must be multiples of 16 and K tiles of 32")
    if m % block_m or n % block_n or k % block_k:
        raise ValueError("M, N and K must be divisible by their tile sizes")
    if split_m not in (1, 2) or (block_m // split_m) % 16:
        raise ValueError("Split M must leave a multiple-of-16 Cube tile")
    tile_m = block_m // split_m
    m_tiles = m // block_m
    n_tiles = n // block_n
    tiles_per_batch = m_tiles * n_tiles
    tiles = batch * tiles_per_batch
    cores = min(cores, tiles)
    # Reserve half of each 64 KiB L0 operand buffer for pipeline overlap.
    l0_k = min(block_k, 16384 // max(block_m, block_n)) // 32 * 32
    while l0_k > 0 and block_k % l0_k:
        l0_k -= 32
    if l0_k == 0:
        raise ValueError("M/N tiles are too large for double-buffered L0 operands")

    if l1_k is None:
        l1_k = min(k, max(256, block_k), 131072 // (block_m + block_n))
        l1_k = l1_k // l0_k * l0_k
        while l1_k and k % l1_k:
            l1_k -= l0_k
    if not l1_k or k % l1_k or l1_k % l0_k:
        raise ValueError("L1 K must divide K and be a multiple of L0 K")
    stages = min(stages, k // l1_k)
    prefetch = prefetch and k // l1_k > 1
    window = min(window, m_tiles)
    last_window = (m_tiles - 1) // window

    @T.prim_func
    def batch_gemm_bf16(
        A: T.Tensor((batch, m, k), "bfloat16"),
        B: T.Tensor((batch, k, n), "bfloat16"),
        C: T.Tensor((batch, m, n), "float32"),
    ):
        with T.Kernel(cores) as cid:
            lhs = T.alloc_l1((block_m, l1_k), "bfloat16")
            rhs = T.alloc_l1((block_n, l1_k), "bfloat16")
            lhs_l0 = T.alloc_l0a((tile_m, l0_k), "bfloat16")
            rhs_l0 = T.alloc_l0b((block_n, l0_k), "bfloat16")
            acc = T.alloc_l0c((tile_m, block_n), "float32")
            T.annotate_buffer_versions({lhs: 2, rhs: 2, lhs_l0: 2, rhs_l0: 2})

            if split_m == 2:
                lhs1_l0 = T.alloc_l0a((tile_m, l0_k), "bfloat16")
                acc1 = T.alloc_l0c((tile_m, block_n), "float32")
                T.annotate_buffer_versions({lhs1_l0: 2})

            for tile_id in T.Persistent([tiles], cores, cid, num_stages=2):
                batch_id = tile_id // tiles_per_batch
                tile = tile_id % tiles_per_batch
                group = T.min(tile // (n_tiles * window), last_window)
                width = T.min(window, m_tiles - group * window)
                in_window = tile - group * window * n_tiles
                row = (group * window + in_window % width) * block_m
                col_tile = in_window // width
                col = T.Select(group % 2 == 0, col_tile, n_tiles - 1 - col_tile) * block_n

                for kt in T.Pipelined(k // l1_k, num_stages=stages, annotations={"enable_offset": prefetch}):
                    with T.Stage(0):
                        T.copy(A[batch_id, row, kt * l1_k], lhs)
                        T.copy(B[batch_id, kt * l1_k : (kt + 1) * l1_k, col : col + block_n], rhs, transpose=True)
                    with T.Stage(1 if prefetch else 0):
                        # B is already K-major in L1 (DN2NZ); reuse it across M halves.
                        for sub in T.Pipelined(l1_k // l0_k, num_stages=2):
                            T.copy(rhs[:, sub * l0_k : (sub + 1) * l0_k], rhs_l0)
                            if split_m == 2:
                                T.copy(lhs[tile_m:, sub * l0_k : (sub + 1) * l0_k], lhs1_l0)
                                T.gemm(lhs1_l0, rhs_l0, acc1, transpose_B=True, clear_accum=(kt == 0) & (sub == 0))
                            T.copy(lhs[:tile_m, sub * l0_k : (sub + 1) * l0_k], lhs_l0)
                            T.gemm(lhs_l0, rhs_l0, acc, transpose_B=True, clear_accum=(kt == 0) & (sub == 0))

                T.copy(acc, C[batch_id, row, col])
                if split_m == 2:
                    T.copy(acc1, C[batch_id, row + tile_m, col])

    return batch_gemm_bf16.with_attr("l1_tile_k", l1_k).with_attr("l0_tile_k", l0_k).with_attr("prefetch_k", int(prefetch))


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
    parser.add_argument("--l1-k", type=int, default=None, help="GM-to-L1 K tile; default chooses up to 256")
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--swizzle-window", type=int, default=4)
    parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-m", type=int, choices=[1, 2], default=1, help="use two half-height L0C slots")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    batch, m, n, k = args.batch, args.m, args.n, args.k
    program = batch_gemm(
        batch,
        m,
        n,
        k,
        args.block_m,
        args.block_n,
        args.block_k,
        args.cores,
        args.l1_k,
        args.stages,
        args.swizzle_window,
        args.prefetch,
        args.split_m,
    )
    print(
        f"CONFIG BF16/BKN -> FP32 B={batch} M={m} N={n} K={k}; cores={args.cores}; "
        f"L1={args.block_m}x{args.block_n}x{program.attrs['l1_tile_k']}; "
        f"L0={args.block_m // args.split_m}x{args.block_n}x{program.attrs['l0_tile_k']}; "
        f"split_m={args.split_m}; prefetch={program.attrs['prefetch_k']}",
        flush=True,
    )
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
