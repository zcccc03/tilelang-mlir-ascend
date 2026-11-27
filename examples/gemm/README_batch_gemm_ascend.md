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

## Batch ownership comparison

Both examples use BF16 A[batch,M,K] and B[batch,N,K], FP32 output,
and explicitly set batch=28 and M=N=K=2048 in `main`.
Their factories follow `batch_gemm(batch, M, N, K)` with fixed configuration:

- batch_m=batch_n=batch_k=256; l0_k=64; split_m=1.
- stages=2; K prefetch enabled; swizzle window=4.
- Double-buffered L1/L0 operands and UnitFlag Cube/Fixpipe handoff.

All tile counts, offsets and loop bounds are derived from these dimensions.
The two files differ in task assignment:

| File | Assignment |
| --- | --- |
| `example_batch_gemm.py` | Distributes all batches' output tiles across 28 AICOREs. |
| `example_batch_gemm_persistent.py` | Launches one worker per batch; core ID is batch ID, and that worker traverses every M/N tile of its batch. |

At batch=28 both launch 28 AICORE workers and perform identical useful work.
The batch-owned kernel retains the same local tile traversal and pipeline;
its `T.Persistent` loop has one local worker starting at tile zero.
This is a task-assignment comparison, with matching input layout, dtype,
tiles, split, initialization and correctness reference (`A @ B.T`).

```bash
python examples/gemm/example_batch_gemm.py --compile-only
python examples/gemm/example_batch_gemm_persistent.py --compile-only

python examples/gemm/example_batch_gemm.py
python examples/gemm/example_batch_gemm_persistent.py

python -m pytest examples/gemm/test_example_batch_gemm.py -q
```

Use identical settings for the real-board comparison:

```bash
msopprof --warm-up=10 --aic-metrics=Roofline,InstrTimeline --launch-count=1 \
  --output="$PWD/msopprof_out/distributed" \
  python examples/gemm/example_batch_gemm.py

msopprof --warm-up=10 --aic-metrics=Roofline,InstrTimeline --launch-count=1 \
  --output="$PWD/msopprof_out/batch_owned" \
  python examples/gemm/example_batch_gemm_persistent.py
```

The batch-owned example also retains `--bench` (10 warmups, 30 repeats,
378 TFLOP/s peak for the default 28 active cores). Do not compare that timing
with a different collection protocol for the distributed example.
Hardware tests skip when no Ascend NPU is available.
