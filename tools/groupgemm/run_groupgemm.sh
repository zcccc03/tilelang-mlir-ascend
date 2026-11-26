#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
dependency=/workspace/code/tilelang-mlir-ascend-testgemm
source /usr/local/Ascend/cann/set_env.sh
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export TVM_IMPORT_PYTHON_PATH="$dependency/3rdparty/tvm/python"
export TL_CUTLASS_PATH="$dependency/3rdparty/cutlass/include"
export TILELANG_CACHE_DIR="$repo/.workspace/groupgemm/cache"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
exec python "$repo/examples/gemm/example_group_gemm.py" "$@"
