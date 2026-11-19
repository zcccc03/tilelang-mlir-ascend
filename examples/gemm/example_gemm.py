import tilelang
import tilelang.language as T


@tilelang.jit
def matmul(A, B, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def ascend_matmul(M, N, K, block_M=128, block_N=128, block_K=32, cores=28):
    """AscendC BF16 GEMM: A[M, K] @ B[N, K].T -> C[M, N] in FP32."""
    import tilelang.ascend.language as T

    n_tiles = N // block_N

    @T.prim_func
    def matmul_ascend(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(cores) as cid:
            lhs = T.alloc_l1((block_M, block_K), "bfloat16")
            rhs = T.alloc_l1((block_N, block_K), "bfloat16")
            acc = T.alloc_l0c((block_M, block_N), "float32")
            for tile in T.Persistent([M // block_M * n_tiles], cores, cid):
                row = tile // n_tiles * block_M
                col = tile % n_tiles * block_N
                for kt in T.Pipelined(K // block_K, num_stages=2):
                    T.copy(A[row, kt * block_K], lhs)
                    T.copy(B[col, kt * block_K], rhs)
                    T.gemm(lhs, rhs, acc, transpose_B=True, clear_accum=(kt == 0))
                T.copy(acc, C[row, col])

    return matmul_ascend


def main_ascend(compile_only=False):
    import torch

    kernel = tilelang.compile(ascend_matmul(1024, 1024, 1024), out_idx=-1, target="ascend", execution_backend="tvm_ffi")
    if compile_only:
        print("COMPILE_PASS")
        return
    import torch_npu  # noqa: F401

    rng = torch.Generator().manual_seed(0)
    a = torch.randint(-4, 5, (1024, 1024), generator=rng).bfloat16() / 8
    b = torch.randint(-4, 5, (1024, 1024), generator=rng).bfloat16() / 8
    actual = kernel(a.npu(), b.npu()).cpu()
    torch.testing.assert_close(actual, a.float() @ b.float().T, rtol=0, atol=0)
    print("AscendC GEMM passed for M=N=K=1024, BF16 to FP32.")


def main(compile_only=False):
    from tilelang.backend.target import determine_target
    from tilelang.env import env

    if determine_target(env.get_default_target(), return_object=True).kind.name == "ascend":
        return main_ascend(compile_only)

    kernel = matmul.compile(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)

    import torch

    a = torch.randn(1024, 1024).cuda().half()
    b = torch.randn(1024, 1024).cuda().half()

    c = kernel(a, b)

    ref_c = a @ b

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    # Get CUDA Source
    print("CUDA Source:")
    print(kernel.get_kernel_source())

    # benchmark
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    # latency = profiler.do_bench()
    print(f"tilelang Latency: {latency}ms")


def run_regression_perf():
    kernel = matmul.compile(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)
    profiler = kernel.get_profiler()
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true", help="compile the AscendC example without running it")
    main(compile_only=parser.parse_args().compile_only)
