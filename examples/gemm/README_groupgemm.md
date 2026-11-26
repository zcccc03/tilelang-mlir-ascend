# BF16 GroupGEMM development case

Case: `(E,total_M,N,K)=(2,512,4,256)`, BF16 X/W/Y, FP32 accumulation, pure grouped matrix multiplication. X and Y are packed by expert; W is `[2,K,N]`. `Group` is a device INT64 cumulative array, nonnegative/monotonic with final value 512. Main case is `[256,512]`.

`example_group_gemm.py` uses M64/full-K tiles on a bounded linear grid. Partial groups use independent M16 GEMMs. Device group boundaries are read on each call; there is no host numerical fallback or bias/activation/routing. Current tuning is limited to small N (1..16) and one full-K tile; only the specified N=4/K=256 case is qualified by this batch.

Development testing uses an independent CPU FP32 reference, nearest-even BF16 output, NaN prefill and guards. The CAModel driver passed seven in-place boundary updates: splits 256,0,511,1,512,257,256. The supplied pytest test requires an actual Ascend NPU and has not yet been run on one.

Local Docker development commands (from repository root, with the existing read-only compiler/TVM dependencies):

```bash
bash tools/groupgemm/run_groupgemm.sh --compile-only
python -I tools/groupgemm/tune_group_gemm.py --variants 64:256:28
```

The tuning driver creates a new run directory each invocation and captures source/binary hashes, build commands, logs and exact-output results. Raw cycle profiling uses `tools/groupgemm/profile_group_gemm.py`; it is a simulator diagnostic, not a real-device benchmark.

First batch: actual local mapping Ascend950PR_9589, 1.65 GHz. Current binary: 3876 model cycles and 1320 executed instructions, 9 launched blocks. Initial M64/K64: 5422 cycles. This 28.51% reduction is local simulator evidence only. No ACLNN speedup or real-NPU performance is claimed. FP8 is planned as E4M3; it is not implemented in this BF16 batch.

Experimental sources, failures and raw profiles are retained in `.workspace/groupgemm/`. Final device validation and delivery remain deferred until requested.

## Large-shape MFU experiment

`example_group_gemm_large.py` adds an aligned-group BF16 path with device cumulative metadata, FP32 accumulation and BF16 output. Current experiment: total_M=14336, N=K=512; either E=28/M_i512 or E=2/M_i7168. Each performs 7,516,192,768 useful FLOPs, matching Batch GEMM [28,512,512,512]. Group boundaries must be block_M aligned in this separate factory.

M256/N256/L1K256/L0K64 tiles and 28 launched Cube blocks were checked with all 7,340,032 outputs plus guard regions. Both independent functional launches and profile launches passed exactly. On local Ascend950PR_9589 at 1.65 GHz: E28 = 46251 cycles / 28.0309 us / MFU32 62.0693%; E2 = 48953 cycles / 29.6685 us / MFU32 58.6434%. Using the same times with a 378 TFLOPS 28-core denominator gives 70.9364% and 67.0210%; these are target normalizations, not target-device measurements.

Frozen driver: tools/groupgemm/scale_group_gemm.py. Raw evidence: .workspace/groupgemm/runs/scale-bf16-r1/. The driver refuses to overwrite that existing run. Full source, output/guard checks, binary hashes and raw database audits are retained in the project records. Final device validation and delivery remain deferred.
