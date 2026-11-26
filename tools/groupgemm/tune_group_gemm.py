"""Offline compile and CAModel development tests; not a device benchmark."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
BASE = Path("/workspace/code/tilelang-mlir-ascend-testgemm")
ROOT = REPO / ".workspace/groupgemm/runs" / f"tune-{time.time_ns()}"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["TVM_IMPORT_PYTHON_PATH"] = str(BASE / "3rdparty/tvm/python")
os.environ["TL_CUTLASS_PATH"] = str(BASE / "3rdparty/cutlass/include")
os.environ["TILELANG_CACHE_DIR"] = str(ROOT / "cache")
sys.path.insert(0, str(REPO))
import tilelang

spec = importlib.util.spec_from_file_location("groupgemm_candidate", REPO / "examples/gemm/example_group_gemm.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

LAUNCHER = r"""
#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CHECK(call) do { auto s=(call); if(s!=ACL_SUCCESS){printf("ACL_FAIL %s %d\n",#call,int(s));return 2;} }while(0)
static uint16_t bf16(float f){uint32_t u;memcpy(&u,&f,4);u+=0x7fff+((u>>16)&1);return u>>16;}
static float fp32(uint16_t b){uint32_t u=uint32_t(b)<<16;float f;memcpy(&f,&u,4);return f;}
int main(int argc,char**argv){
  constexpr int M=512,K=256,N=4,E=2,GUARD=32;
  uint16_t x[M*K],w[E*K*N],expected[M*N],actual[M*N+2*GUARD];
  for(int i=0;i<M*K;i++)x[i]=bf16(float((i*3+i/7+i/256)%17-8)/8);
  for(int i=0;i<E*K*N;i++)w[i]=bf16(float((i*7+i/11+i/128)%17-8)/8);
  CHECK(aclInit(nullptr));CHECK(aclrtSetDevice(0));aclrtStream stream;CHECK(aclrtCreateStream(&stream));
  void *dx,*dw,*dg,*dy;
  CHECK(aclrtMalloc(&dx,sizeof(x),ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMalloc(&dw,sizeof(w),ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMalloc(&dg,E*8,ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMalloc(&dy,sizeof(actual),ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMemcpy(dx,sizeof(x),x,sizeof(x),ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK(aclrtMemcpy(dw,sizeof(w),w,sizeof(w),ACL_MEMCPY_HOST_TO_DEVICE));
  const int splits[]={256,0,511,1,512,257,256};
  const int runs=argc>1?1:7;
  for(int step=0;step<runs;step++){
    const int split=splits[step];int64_t group[]={split,512};
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){
      float sum=0;int expert=m<split?0:1;
      for(int k=0;k<K;k++)sum+=fp32(x[m*K+k])*fp32(w[(expert*K+k)*N+n]);
      expected[m*N+n]=bf16(sum);
    }
    for(auto &v:actual)v=0x7fc1;
    CHECK(aclrtMemcpy(dg,E*8,group,E*8,ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK(aclrtMemcpy(dy,sizeof(actual),actual,sizeof(actual),ACL_MEMCPY_HOST_TO_DEVICE));
    uint16_t* output=(uint16_t*)dy+GUARD;
    KERNEL<<<BLOCKS,nullptr,stream>>>(ARGUMENTS);
    CHECK(aclrtSynchronizeStream(stream));
    CHECK(aclrtMemcpy(actual,sizeof(actual),dy,sizeof(actual),ACL_MEMCPY_DEVICE_TO_HOST));
    int bad=0;float maxerr=0;
    for(int i=0;i<M*N;i++){
      if(actual[GUARD+i]!=expected[i]){
        if(bad++<4)printf("MISMATCH split=%d i=%d actual=%g expected=%g\n",split,i,fp32(actual[GUARD+i]),fp32(expected[i]));
        float delta=fp32(actual[GUARD+i])-fp32(expected[i]);
        if(delta<0)delta=-delta;
        if(delta>maxerr)maxerr=delta;
      }
    }
    for(int i=0;i<GUARD;i++)if(actual[i]!=0x7fc1||actual[GUARD+M*N+i]!=0x7fc1){printf("GUARD_FAIL %d\n",i);return 4;}
    printf("CASE split=%d bad=%d maxerr=%g\n",split,bad,maxerr);fflush(stdout);
    if(bad)return 3;
  }
  CHECK(aclrtFree(dx));CHECK(aclrtFree(dw));CHECK(aclrtFree(dg));CHECK(aclrtFree(dy));
  CHECK(aclrtDestroyStream(stream));CHECK(aclrtResetDevice(0));CHECK(aclFinalize());
  printf("NUMERICAL_PASS cases=%d BF16 exact guard=pass\n",runs);return 0;
}
"""


def execute(argv, directory, filename, env, timeout=180):
    started = time.monotonic()
    with (directory / filename).open("w") as log:
        try:
            result = subprocess.run(argv, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            rc = result.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    return {"argv": argv, "exit_code": rc, "wall_seconds": time.monotonic() - started, "log": filename}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="64:256:28")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    camodel = "/usr/local/Ascend/cann/tools/simulator/dav_3510/camodel"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = camodel + ":" + env.get("LD_LIBRARY_PATH", "")
    summary = []
    for variant in args.variants.split(","):
        bm, bk, cores = map(int, variant.split(":"))
        directory = ROOT / f"m{bm}-k{bk}-c{cores}"
        directory.mkdir(exist_ok=True)
        record = {
            "block_m": bm,
            "block_k": bk,
            "requested_cores": cores,
            "case": [2, 512, 4, 256],
            "tilelang_import": tilelang.__file__,
            "source_sha256": hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
        }
        try:
            program = mod.group_gemm(block_m=bm, block_k=bk, cores=cores)
            source = tilelang.lower(program, target="ascend", enable_device_compile=False).kernel_source
            (directory / "kernel.cce").write_text(source)
            (directory / "input.tir").write_text(program.script())
            name, params = re.search(r"void (\w+)\(([^)]*)\)", source).groups()
            pointers = {"X": "(bfloat16_t*)dx", "W": "(bfloat16_t*)dw", "Group": "(int64_t*)dg", "Y": "(bfloat16_t*)output"}
            order = [x.strip().split()[-1].lstrip("*") for x in params.split(",")]
            blocks = min(cores, (512 + bm - 1) // bm + 1)
            defines = f"\n#define KERNEL {name}\n#define BLOCKS {blocks}\n#define ARGUMENTS " + ", ".join(pointers[x] for x in order) + "\n"
            (directory / "launch.asc").write_text(source + defines + LAUNCHER)
            command = [
                "bisheng",
                "launch.asc",
                "-o",
                "launch",
                "--npu-arch=dav-3510",
                "--run-mode=sim",
                "-O2",
                "-g",
                "-std=c++20",
                "-mllvm",
                "-cce-aicore-dcpreload-args=false",
                "-I" + str(REPO / "src"),
                "-lruntime_camodel",
                "-lnpu_drv_camodel",
                "-lascendcl",
                "-lprofapi",
                "-lunified_dlog",
                "-lmmpa",
                "-lascend_dump",
                "-lc_sec",
                "-lerror_manager",
                "-lnpu_drv",
                "-lm",
            ]
            record["build"] = execute(command, directory, "build.log", env)
            if record["build"]["exit_code"]:
                raise RuntimeError((directory / "build.log").read_text()[-6000:])
            subprocess.run(["patchelf", "--set-rpath", camodel, str(directory / "launch")], check=True)
            record["generated_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            record["binary_sha256"] = hashlib.sha256((directory / "launch").read_bytes()).hexdigest()
            record["blocks"] = blocks
            if not args.compile_only:
                record["numerical"] = execute([str(directory / "launch")], directory, "numerical.log", env)
                record["numerical_pass"] = (
                    record["numerical"]["exit_code"] == 0 and "NUMERICAL_PASS" in (directory / "numerical.log").read_text()
                )
                if not record["numerical_pass"]:
                    raise RuntimeError((directory / "numerical.log").read_text()[-6000:])
            print("VARIANT_DONE", variant, json.dumps(record), flush=True)
        except Exception as error:
            record["error"] = str(error)
            print("VARIANT_FAIL", variant, str(error), flush=True)
        (directory / "result.json").write_text(json.dumps(record, indent=2))
        summary.append(record)
        (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    return int(any("error" in x for x in summary))


if __name__ == "__main__":
    raise SystemExit(main())
