# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Persistent BNK Batch GEMM for AscendC with FP16/BF16 inputs and FP32 output."""

import argparse
import math

import tilelang
import tilelang.ascend.language as T
import torch


def batch_gemm(
    batch,
    m,
    n,
    k,
    block_m=256,
    block_n=256,
    block_k=256,
    cores=28,
    stages=2,
    window=4,
    split_m=2,
    prefetch=True,
    dtype="float16",
    unit_flag=True,
):
    if min(batch, m, n, k, block_m, block_n, block_k, cores, stages, window) <= 0:
        raise ValueError("Shapes, tile dimensions, core count, stages and window must be positive")
    if block_m % 16 or block_n % 16 or block_k % 32:
        raise ValueError("M/N tiles must be multiples of 16 and K tiles of 32")
    if m % block_m or n % block_n or k % block_k:
        raise ValueError("M, N and K must be divisible by their tile sizes")
    if split_m not in (1, 2) or block_m % split_m or (block_m // split_m) % 16:
        raise ValueError("Split M must leave a multiple-of-16 Cube tile")
    uf_final, uf_partial = (3, 2) if unit_flag else (0, 0)
    tile_m = block_m // split_m
    l0_k = min(block_k, 16384 // max(block_m, block_n)) // 32 * 32
    while l0_k and block_k % l0_k:
        l0_k -= 32
    if not l0_k:
        raise ValueError("M/N tiles are too large for double-buffered L0 operands")

    m_tiles, n_tiles, k_tiles = m // block_m, n // block_n, k // block_k
    tiles_per_batch = m_tiles * n_tiles
    tiles = batch * tiles_per_batch
    cores = min(cores, tiles)
    stages = min(stages, k_tiles)
    window = min(window, m_tiles)
    last_window = (m_tiles - 1) // window

    @T.prim_func
    def batch_gemm_persistent(
        A: T.Tensor((batch, m, k), dtype),
        W: T.Tensor((batch, n, k), dtype),
        C: T.Tensor((batch, m, n), "float32"),
    ):
        with T.Kernel(cores) as cid:
            lhs = T.alloc_l1((block_m, block_k), dtype)
            rhs = T.alloc_l1((block_n, block_k), dtype)
            lhs0_l0 = T.alloc_l0a((tile_m, l0_k), dtype)
            rhs_l0 = T.alloc_l0b((block_n, l0_k), dtype)
            acc0 = T.alloc_l0c((tile_m, block_n), "float32")
            T.annotate_buffer_versions({lhs: 2, rhs: 2, lhs0_l0: 2, rhs_l0: 2})
            if split_m == 2:
                lhs1_l0 = T.alloc_l0a((tile_m, l0_k), dtype)
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

                for kt in T.Pipelined(k_tiles, num_stages=stages, annotations={"enable_offset": prefetch}):
                    with T.Stage(0):
                        T.copy(A[batch_id, row, kt * block_k], lhs)
                        T.copy(W[batch_id, col, kt * block_k], rhs)
                    with T.Stage(1 if prefetch else 0):
                        for sub in T.Pipelined(block_k // l0_k, num_stages=2):
                            # One B load feeds both M halves.
                            T.copy(rhs[:, sub * l0_k : (sub + 1) * l0_k], rhs_l0)
                            if split_m == 2:
                                T.copy(lhs[tile_m:, sub * l0_k : (sub + 1) * l0_k], lhs1_l0)
                                T.gemm(
                                    lhs1_l0,
                                    rhs_l0,
                                    acc1,
                                    transpose_B=True,
                                    clear_accum=(kt == 0) & (sub == 0),
                                    unit_flag_ctrl=T.Select((kt == k_tiles - 1) & (sub == block_k // l0_k - 1), uf_final, uf_partial),
                                )
                            T.copy(lhs[:tile_m, sub * l0_k : (sub + 1) * l0_k], lhs0_l0)
                            T.gemm(
                                lhs0_l0,
                                rhs_l0,
                                acc0,
                                transpose_B=True,
                                clear_accum=(kt == 0) & (sub == 0),
                                unit_flag_ctrl=T.Select((kt == k_tiles - 1) & (sub == block_k // l0_k - 1), uf_final, uf_partial),
                            )

                T.copy(acc0, C[batch_id, row, col], unit_flag_ctrl=uf_final)
                if split_m == 2:
                    T.copy(acc1, C[batch_id, row + tile_m, col], unit_flag_ctrl=uf_final)

    return batch_gemm_persistent.with_attr("l0_tile_k", l0_k)


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
    parser.add_argument("--split-m", type=int, choices=[1, 2], default=2, help="use two half-height L0C slots")
    parser.add_argument("--unit-flag", action=argparse.BooleanOptionalAction, default=True, help="use hardware Cube/Fixpipe handoff")
    parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--compile-only", action="store_true")
    run_mode.add_argument("--bench", action="store_true", help="profile the kernel on a real NPU")
    parser.add_argument("--peak-tflops", type=float, help="dense input-dtype peak for the active Cube cores")
    args = parser.parse_args()
    if args.peak_tflops is not None and (not math.isfinite(args.peak_tflops) or args.peak_tflops <= 0):
        parser.error("--peak-tflops must be finite and positive")

    program = batch_gemm(
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
        args.split_m,
        args.prefetch,
        args.dtype,
        args.unit_flag,
    )
    l0_k = int(program.attrs["l0_tile_k"])
    print(
        f"CONFIG {args.dtype}/BNK -> FP32 B={args.batch} M={args.m} N={args.n} K={args.k}; "
        f"cores={args.cores}; L1K={args.block_k}; L0={args.block_m // args.split_m}x{args.block_n}x{l0_k}; "
        f"split_m={args.split_m}; unit_flag={args.unit_flag}; prefetch={args.prefetch}",
        flush=True,
    )
    kernel = tilelang.compile(program, out_idx=-1, target="ascend", execution_backend="tvm_ffi")
    if args.compile_only:
        print("COMPILE_PASS")
        return

    rng = torch.Generator().manual_seed(0)
    a = torch.randint(-4, 5, (args.batch, args.m, args.k), dtype=torch.int8, generator=rng).to(getattr(torch, args.dtype)).div_(8)
    w = torch.randint(-4, 5, (args.batch, args.n, args.k), dtype=torch.int8, generator=rng).to(getattr(torch, args.dtype)).div_(8)
    a_npu, w_npu = a.npu(), w.npu()
    actual = kernel(a_npu, w_npu).cpu()
    for i in range(args.batch):
        torch.testing.assert_close(actual[i], a[i].float() @ w[i].float().T, rtol=0, atol=0)
    print(f"CORRECTNESS_PASS B={args.batch}, M={args.m}, N={args.n}, K={args.k}")

    if args.bench:
        from tilelang.profiler import do_bench

        latency_ms = do_bench(
            lambda: kernel(a_npu, w_npu),
            backend="msprof",
            _n_warmup=10,
            _n_repeat=30,
            device=a_npu.device,
        )
        if not math.isfinite(latency_ms) or latency_ms <= 0:
            raise RuntimeError("The NPU profiler did not report a valid kernel duration")
        tflops = 2 * args.batch * args.m * args.n * args.k / (latency_ms * 1e9)
        print(f"{latency_ms * 1e3:.2f} us | {tflops:.2f} TFLOP/s")
        if args.peak_tflops is not None:
            print(f"MFU {100 * tflops / args.peak_tflops:.2f}% (peak {args.peak_tflops:g} TFLOP/s)")


if __name__ == "__main__":
    main()
