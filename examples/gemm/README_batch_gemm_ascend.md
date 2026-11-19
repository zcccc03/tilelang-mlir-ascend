# AscendC Batch GEMM

These examples use the native Ascend 950 backend on `master`. They do not use
the NPUIR backend or its compiler submodule.

Build TileLang using the [Ascend installation guide](../../tilelang/ascend/README.md):

```bash
git submodule update --init --recursive
USE_ASCEND=ON USE_CUDA=OFF python -m pip install -v .
```

Configure CANN and install compatible PyTorch and `torch_npu` packages first.
Run the commands below from the repository root with that TileLang installation.

## Examples

- `example_batch_gemm.py`: BF16 `A[B, M, K] @ B[B, K, N]`, FP32 output;
  defaults to batch 28 and M=N=K=1024 with 28 Cube cores.
- `example_batch_gemm_persistent.py`: FP16 `A[B, M, K] @ W[B, N, K].T`,
  FP32 output; defaults to batch 28 and M=N=K=2048 with 28 Cube cores,
  a two-stage reduction pipeline and alternating swizzle windows.

The BF16 kernel transposes B inside the L1-to-L0B copy, preserving its original
input layout. Both kernels distribute output tiles with `T.Persistent`.
The Python `batch_gemm` factories return a PrimFunc; compile it with
`tilelang.compile(program, out_idx=-1, target="ascend")`.

```bash
# Compile with CANN/Bisheng without allocating input tensors or running the kernel.
python examples/gemm/example_batch_gemm.py --compile-only
python examples/gemm/example_batch_gemm_persistent.py --compile-only

# Check correctness on an Ascend 950 NPU.
python examples/gemm/example_batch_gemm.py
python examples/gemm/example_batch_gemm_persistent.py

# Profile on hardware; report MFU only when a matching dense FP16 peak is supplied.
python examples/gemm/example_batch_gemm_persistent.py --bench
```

`--bench` uses the `msprof` backend with 10 warmup iterations and 30 measured
iterations. `--peak-tflops` must describe the active cores and hardware.

## Tests

```bash
python -m pytest examples/gemm/test_example_batch_gemm.py -q
```

Lowering tests cover the default kernels, a single K tile, a partial core wave,
and a final swizzle window. Hardware tests check both input layouts against
independent FP32 PyTorch references, including fewer tiles than cores and
multiple batches. Hardware tests skip when no Ascend NPU is available.
