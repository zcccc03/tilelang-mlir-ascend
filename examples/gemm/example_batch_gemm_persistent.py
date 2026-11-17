# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Persistent BNK Batch GEMM for NPUIR with FP16 inputs and FP32 output."""

import argparse
import math
import os
from tempfile import TemporaryDirectory

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(out_idx=[-1], target="npuir")
def batch_gemm(
    batch, m, n, k, block_m=256, block_n=256, block_k=256, cores=28, stages=2, window=4
):
    if min(batch, m, n, k, block_m, block_n, block_k, cores, stages, window) <= 0:
        raise ValueError(
            "Shapes, tile dimensions, core count, stages and window must be positive"
        )
    if block_m % 16 or block_n % 16 or block_k % 32:
        raise ValueError("M/N tiles must be multiples of 16 and K tiles of 32")
    if m % block_m or n % block_n or k % block_k:
        raise ValueError("M, N and K must be divisible by their tile sizes")

    m_tiles, n_tiles, k_tiles = m // block_m, n // block_n, k // block_k
    tiles_per_batch = m_tiles * n_tiles
    tiles = batch * tiles_per_batch
    cores = min(cores, tiles)
    stages = min(stages, k_tiles)
    window = min(window, m_tiles)
    last_window = m_tiles // window - 1
    tail_width = m_tiles - last_window * window

    @T.prim_func
    def batch_gemm_fp16(
        A: T.Tensor((batch, m, k), "float16"),
        W: T.Tensor((batch, n, k), "float16"),
        C: T.Tensor((batch, m, n), "float32"),
    ):
        with T.Kernel(cores, is_npu=True) as (cid, _):
            lhs = T.alloc_shared((block_m, block_k), "float16")
            rhs = T.alloc_shared((block_n, block_k), "float16")
            acc = T.alloc_fragment((block_m, block_n), "float32")

            # The dynamic trip count assigns each output tile exactly once.
            for wave in T.serial(T.ceildiv(tiles - cid, cores)):
                tile_id = wave * cores + cid
                batch_id = tile_id // tiles_per_batch
                tile = tile_id % tiles_per_batch
                group = T.min(tile // (n_tiles * window), last_window)
                width = T.Select(
                    group == last_window, T.int32(tail_width), T.int32(window)
                )
                in_window = tile - group * window * n_tiles
                row = (group * window + in_window % width) * block_m
                col_tile = in_window // width
                col = (
                    T.Select(group % 2 == 0, col_tile, n_tiles - 1 - col_tile) * block_n
                )

                for kt in T.Pipelined(k_tiles, num_stages=stages):
                    T.copy(A[batch_id, row, kt * block_k], lhs)
                    T.copy(W[batch_id, col, kt * block_k], rhs)
                    T.gemm(lhs, rhs, acc, initC=(kt == 0), b_transpose=True)

                T.copy(acc, C[batch_id, row, col])

    return batch_gemm_fp16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=28)
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--block-m", type=int, default=256)
    parser.add_argument("--block-n", type=int, default=256)
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument("--cores", type=int, default=28)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--swizzle-window", type=int, default=4)
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--compile-only", action="store_true")
    run_mode.add_argument(
        "--bench", action="store_true", help="profile the kernel on a real NPU"
    )
    parser.add_argument(
        "--peak-tflops", type=float, help="dense FP16 peak for the active Cube cores"
    )
    args = parser.parse_args()
    if args.peak_tflops is not None and (
        not math.isfinite(args.peak_tflops) or args.peak_tflops <= 0
    ):
        parser.error("--peak-tflops must be finite and positive")

    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    kernel = batch_gemm(
        args.batch,
        args.m,
        args.n,
        args.k,
        args.block_m,
        args.block_n,
        args.block_k,
        args.cores,
        args.stages,
        args.swizzle_window,
    )
    if args.compile_only:
        print("COMPILE_PASS")
        return

    rng = torch.Generator().manual_seed(0)
    a = (
        torch.randint(
            -4, 5, (args.batch, args.m, args.k), dtype=torch.int8, generator=rng
        )
        .half()
        .div_(8)
    )
    w = (
        torch.randint(
            -4, 5, (args.batch, args.n, args.k), dtype=torch.int8, generator=rng
        )
        .half()
        .div_(8)
    )
    a_npu, w_npu = a.npu(), w.npu()
    actual = kernel(a_npu, w_npu).cpu()
    for i in range(args.batch):
        torch.testing.assert_close(
            actual[i], a[i].float() @ w[i].float().T, rtol=0, atol=0
        )
    print(f"CORRECTNESS_PASS B={args.batch}, M={args.m}, N={args.n}, K={args.k}")

    if args.bench:
        from tilelang.profiler.bench import do_bench_npu

        with TemporaryDirectory(prefix="batch-gemm-profile-") as prof_dir:
            latency_ms = do_bench_npu(
                [lambda: kernel(a_npu, w_npu)], warmup=10, rep=30, prof_dir=prof_dir
            )
        if not math.isfinite(latency_ms) or latency_ms <= 0:
            raise RuntimeError(
                "The NPU profiler did not report a valid kernel duration"
            )
        tflops = 2 * args.batch * args.m * args.n * args.k / (latency_ms * 1e9)
        print(f"{latency_ms * 1e3:.2f} us | {tflops:.2f} TFLOP/s")
        if args.peak_tflops is not None:
            print(
                f"MFU {100 * tflops / args.peak_tflops:.2f}% (peak {args.peak_tflops:g} TFLOP/s)"
            )


if __name__ == "__main__":
    main()
