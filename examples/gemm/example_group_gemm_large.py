"""BF16 GroupGEMM scaling experiment: packed M, device cumulative groups.

This experiment requires every group boundary to be block_m aligned.
It retains FP32 accumulation and BF16 output, with no fused operations.
"""

import tilelang.ascend.language as T


def group_gemm_large(experts=28, total_m=14336, n=512, k=512, block_m=256, block_n=256, l1_k=256, cores=28):
    if min(experts, total_m, n, k, block_m, block_n, l1_k, cores) <= 0:
        raise ValueError("Dimensions and core count must be positive")
    if block_m % 16 or block_n % 16 or l1_k % 32:
        raise ValueError("M/N tiles must be 16-aligned and K tile 32-aligned")
    if total_m % block_m or n % block_n or k % l1_k:
        raise ValueError("Aligned scaling experiment requires full tiles")
    if 4 * (block_m + block_n) * l1_k > 524288:
        raise ValueError("Double-buffered BF16 L1 tiles exceed 512 KiB")
    if 4 * block_m * block_n > 262144:
        raise ValueError("FP32 accumulator exceeds L0C")
    l0_k = min(l1_k, 16384 // max(block_m, block_n)) // 32 * 32
    if l0_k == 0 or l1_k % l0_k:
        raise ValueError("Invalid double-buffered L0 shape")
    n_tiles = n // block_n
    stages = min(2, k // l1_k)
    prefetch = k // l1_k > 1

    @T.prim_func
    def group_gemm_large_bf16(
        X: T.Tensor((total_m, k), "bfloat16"),
        W: T.Tensor((experts, k, n), "bfloat16"),
        Group: T.Tensor((experts,), "int64"),
        Y: T.Tensor((total_m, n), "bfloat16"),
    ):
        with T.Kernel(cores) as cid:
            lhs = T.alloc_l1((block_m, l1_k), "bfloat16")
            rhs = T.alloc_l1((block_n, l1_k), "bfloat16")
            a0 = T.alloc_l0a((block_m, l0_k), "bfloat16")
            b0 = T.alloc_l0b((block_n, l0_k), "bfloat16")
            acc = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions({lhs: 2, rhs: 2, a0: 2, b0: 2})
            group_start = T.alloc_var("int32", init=0)
            task_base = T.alloc_var("int32", init=0)
            for expert in T.serial(experts):
                group_end = T.cast(Group[expert], "int32")
                T.assume(group_end >= group_start)
                T.assume(group_end <= total_m)
                T.assume(group_end % block_m == 0)
                m_tiles = (group_end - group_start) // block_m
                group_tiles = m_tiles * n_tiles
                # Continue one global round-robin sequence across expert boundaries.
                # Small groups therefore use different physical cores rather than
                # repeatedly assigning all work to cores 0..3.
                first = (cid + cores - task_base % cores) % cores
                for work in T.serial(first, group_tiles, cores):
                    row = group_start + work % m_tiles * block_m
                    col = work // m_tiles * block_n
                    for kt in T.Pipelined(k // l1_k, num_stages=stages, annotations={"enable_offset": prefetch}):
                        with T.Stage(0):
                            T.copy(X[row : row + block_m, kt * l1_k : (kt + 1) * l1_k], lhs)
                            T.copy(W[expert, kt * l1_k : (kt + 1) * l1_k, col : col + block_n], rhs, transpose=True)
                        with T.Stage(1 if prefetch else 0):
                            for sub in T.Pipelined(l1_k // l0_k, num_stages=2):
                                T.copy(lhs[:, sub * l0_k : (sub + 1) * l0_k], a0)
                                T.copy(rhs[:, sub * l0_k : (sub + 1) * l0_k], b0)
                                T.gemm(
                                    a0,
                                    b0,
                                    acc,
                                    transpose_B=True,
                                    clear_accum=(kt == 0) & (sub == 0),
                                    unit_flag_ctrl=T.Select((kt == k // l1_k - 1) & (sub == l1_k // l0_k - 1), 3, 2),
                                )
                    T.copy(acc, Y[row : row + block_m, col : col + block_n], unit_flag_ctrl=3)
                task_base = task_base + group_tiles
                group_start = group_end

    return group_gemm_large_bf16
