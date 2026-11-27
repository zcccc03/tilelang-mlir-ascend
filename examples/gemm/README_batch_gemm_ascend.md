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

## Fixed workload

Both examples are specialized for B=28 and M=N=K=2048 on 28 Cube cores.
Like `ascend_matmul`, the factory is `batch_gemm(batch, M, N, K)`.
`main` explicitly sets batch=28 and M=N=K=2048; tuning values remain fixed
inside the factory. Tile counts, swizzle indices and K loop counts are derived
from these named dimensions. There is no shape or tuning CLI.
L1 tiles are 256x256, L0 K is 64, both pipeline stages are fixed at two,
K prefetch is enabled, and the swizzle window is four M tiles.
There are eight M/N/K tiles, 64 output tiles per batch, and 1792 total.
The two full swizzle windows need no tail-width clamp or last-window clamp.

- `example_batch_gemm.py`: BF16 A[B,M,K] and B[B,N,K], FP32 output,
  one 256x256 L0C accumulator (split M=1).
- `example_batch_gemm_persistent.py`: FP16 A[B,M,K] and B[B,N,K],
  FP32 output, two 128x256 L0C accumulators sharing B (split M=2).

Both examples now load B in BNK layout using ND2NZ. The ordinary example
previously used BKN, so its earlier board result does not measure this layout.
The split values remain one and two respectively; the records do not establish
the best split independently for each input layout and dtype.
Both kernels use `T.Persistent`, double-buffered L1/L0 operands and UnitFlag.
Shape, dtype, tile, core-count and pipeline command-line options are removed.

```bash
# Compile without allocating tensors.
python examples/gemm/example_batch_gemm.py --compile-only
python examples/gemm/example_batch_gemm_persistent.py --compile-only

# Run the fixed workload and check FP32 results on a real NPU.
python examples/gemm/example_batch_gemm.py
python examples/gemm/example_batch_gemm_persistent.py

# Persistent hardware benchmark: 10 warmups, 30 repeats, 378 TFLOP/s peak.
python examples/gemm/example_batch_gemm_persistent.py --bench

python -m pytest examples/gemm/test_example_batch_gemm.py -q
```

For board pipeline collection, run either file with the same profiling settings:

```bash
msopprof --warm-up=10 --aic-metrics=Roofline,InstrTimeline --launch-count=1 \
  --output="$PWD/msopprof_out/batchgemm" \
  python examples/gemm/example_batch_gemm.py
```

Hardware tests skip when no Ascend NPU is available.
