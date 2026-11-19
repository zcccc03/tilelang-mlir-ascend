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
For a source checkout, put this repository first on Python's import path:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_DEFAULT_TARGET=ascend
```

`python -c 'import tilelang; print(tilelang.__file__)'` must point to this
checkout, rather than a neighboring `tilelang` repository. These examples
explicitly select `target="ascend"` and `execution_backend="tvm_ffi"`.

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

## The two requested tiles

Both factories support batch 28 and M=N=K=1024 on 28 Cube cores. The ordinary
example accepts shape and tile options:

```bash
python examples/gemm/example_batch_gemm.py --block-m 256 --block-n 256 --block-k 64
python examples/gemm/example_batch_gemm.py --block-m 128 --block-n 128 --block-k 128
python examples/gemm/example_batch_gemm_persistent.py --m 1024 --n 1024 --k 1024 --block-m 256 --block-n 256 --block-k 64
python examples/gemm/example_batch_gemm_persistent.py --m 1024 --n 1024 --k 1024 --block-m 128 --block-n 128 --block-k 128
```

## Profiling and the FP32-to-FP16 copy error

The reported command ran `example_gemm.py`, not a Batch GEMM example.
The generic GPU kernel uses an FP32 register fragment and FP16 output;
its copy implies a conversion that Ascend DMA cannot perform. On an Ascend
target, `examples/gemm/example_gemm.py` now selects a dedicated BF16 GEMM
with L1 inputs, an FP32 L0C accumulator and FP32 GM output. The GPU path stays
available for GPU targets. All three Ascend examples write FP32 results
from L0C, without an FP32-to-FP16 DMA cast.

```bash
TILELANG_DEFAULT_TARGET=ascend python examples/gemm/example_gemm.py --compile-only
TILELANG_DEFAULT_TARGET=ascend python examples/gemm/example_gemm.py
```

Run Batch GEMM profiling from the repository root after correctness succeeds:

```bash
msopprof --warm-up=10 --aic-metrics=Roofline,InstrTimeline --launch-count=1 \
  --output="$PWD/msopprof_out/batchgemm" \
  python examples/gemm/example_batch_gemm.py --block-k 64
```

Omit the kernel-name filter until the actual kernel name is known.
The original command combined a `batch` filter with an unrelated `matmul`
kernel. Compilation failure also leaves no device kernel to profile;
the empty-profiler message is secondary. Hardware correctness and MFU
require execution on an Ascend NPU.
