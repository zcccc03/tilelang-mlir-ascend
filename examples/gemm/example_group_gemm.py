"""Pure BF16 GroupGEMM for the two-expert Ascend 950 tuning case.

X [M,K], W [2,K,N], cumulative group_list [2] -> Y [M,N].
Group boundaries are read on device on every invocation, including empty groups.
FP32 accumulation, BF16 output; no bias, activation, routing or host fallback.
"""

import argparse

import tilelang
import tilelang.ascend.language as T


def group_gemm(total_m=512, n=4, k=256, block_m=64, block_k=256, cores=28):
    if min(total_m, n, k, block_m, block_k, cores) <= 0:
        raise ValueError("Dimensions and core count must be positive")
    if block_m % 16 or block_k % 32 or k % block_k:
        raise ValueError("M tile must be 16-aligned; K tile must be 32-aligned and divide K")
    if block_k != k:
        raise ValueError("The current tuned path requires one full-K reduction tile")
    if n > 16:
        raise ValueError("This small-N tuning kernel supports 1 <= N <= 16")
    if 2 * block_m * block_k > 65536:
        raise ValueError("A tile exceeds L0A capacity")
    # The split may add one partial tile to ceil(M / block_m).
    max_tiles = (total_m + block_m - 1) // block_m + 1
    launch_cores = min(cores, max_tiles)

    @T.prim_func
    def group_gemm_bf16(
        X: T.Tensor((total_m, k), "bfloat16"),
        W: T.Tensor((2, k, n), "bfloat16"),
        Group: T.Tensor((2,), "int64"),
        Y: T.Tensor((total_m, n), "bfloat16"),
    ):
        with T.Kernel(launch_cores) as cid:
            lhs = T.alloc_l1((block_m, block_k), "bfloat16")
            rhs = T.alloc_l1((16, block_k), "bfloat16")
            acc = T.alloc_l0c((block_m, 16), "float32")
            tail_lhs = T.alloc_l1((16, block_k), "bfloat16")
            tail_acc = T.alloc_l0c((16, 16), "float32")
            split = T.cast(Group[0], "int32")
            finish = T.cast(Group[1], "int32")
            T.assume(split >= 0)
            T.assume(split <= total_m)
            T.assume(finish == total_m)
            first_tiles = T.ceildiv(split, block_m)
            total_tiles = first_tiles + T.ceildiv(finish - split, block_m)
            T.assume(total_tiles > 0)
            for wave in T.serial((max_tiles + launch_cores - 1) // launch_cores):
                task = wave * launch_cores + cid
                if task < total_tiles:
                    expert = T.Select(task < first_tiles, 0, 1)
                    start = T.Select(expert == 0, 0, split)
                    end = T.Select(expert == 0, split, finish)
                    local_task = task - T.Select(expert == 0, 0, first_tiles)
                    row = start + local_task * block_m
                    rows = T.min(block_m, end - row)
                    T.assume(rows > 0)
                    T.assume(rows <= block_m)
                    T.copy(W[expert, 0:k, 0:16], rhs, transpose=True)
                    if rows == block_m:
                        T.copy(X[row : row + block_m, 0:k], lhs)
                        T.gemm(lhs, rhs, acc, transpose_B=True, clear_accum=True, unit_flag_ctrl=3)
                        T.copy(acc[:, :n], Y[row : row + block_m, :n], unit_flag_ctrl=3)
                    else:
                        # Bounded M16 tail tasks use zero-based L1 loads.
                        for part in T.serial(block_m // 16):
                            if part * 16 < rows:
                                tail_row = row + part * 16
                                tail_rows = T.min(16, end - tail_row)
                                T.copy(X[tail_row : tail_row + 16, 0:k], tail_lhs)
                                T.gemm(tail_lhs, rhs, tail_acc, transpose_B=True, clear_accum=True, unit_flag_ctrl=3)
                                T.copy(tail_acc[:tail_rows, :n], Y[tail_row : tail_row + tail_rows, :n], unit_flag_ctrl=3, l0_pitch=1)

    return group_gemm_bf16


def reference(x, w, cumulative):
    """Independent CPU FP32 matrix products, rounded once to BF16."""
    import torch

    parts = []
    start = 0
    for expert, end in enumerate(cumulative):
        parts.append(x[start:end].float() @ w[expert].float())
        start = end
    return torch.cat(parts).bfloat16()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument("--cores", type=int, default=28)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    program = group_gemm(block_m=args.block_m, block_k=args.block_k, cores=args.cores)
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)
    if args.compile_only:
        print("COMPILE_PASS")
        return
    import torch
    import torch_npu  # noqa: F401

    generator = torch.Generator().manual_seed(20261126)
    x = (torch.randint(-8, 9, (512, 256), generator=generator).float() / 8).bfloat16()
    w = (torch.randint(-8, 9, (2, 256, 4), generator=generator).float() / 8).bfloat16()
    xn, wn = x.npu(), w.npu()
    groups = torch.tensor([256, 512], dtype=torch.int64, device="npu")
    # Reuse the exact same metadata allocation and compiled kernel.
    for split in (256, 0, 511, 1, 512, 257, 256):
        groups.copy_(torch.tensor([split, 512], dtype=torch.int64))
        actual = torch.full((512, 4), float("nan"), dtype=torch.bfloat16, device="npu")
        kernel(xn, wn, groups, actual)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), reference(x, w, [split, 512]), rtol=0, atol=0)
    print("GROUPGEMM_PASS BF16 (E,M,N,K)=(2,512,4,256), mutable groups, empty groups and tails")


if __name__ == "__main__":
    main()
