"""Compile, check, and profile equal-FLOP BF16 GroupGEMM scaling cases."""

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
ROOT = REPO / ".workspace/groupgemm/runs/scale-bf16-r1"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["TVM_IMPORT_PYTHON_PATH"] = str(BASE / "3rdparty/tvm/python")
os.environ["TL_CUTLASS_PATH"] = str(BASE / "3rdparty/cutlass/include")
os.environ["TILELANG_CACHE_DIR"] = str(ROOT / "cache")
sys.path.insert(0, str(REPO))
import tilelang

spec = importlib.util.spec_from_file_location("large_groupgemm", REPO / "examples/gemm/example_group_gemm_large.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

LAUNCHER = r"""
#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CHECK(call) do { auto s=(call); if(s!=ACL_SUCCESS){printf("ACL_FAIL %s %d\n",#call,int(s));return 2;} }while(0)
static uint16_t bf16(float f){uint32_t u;memcpy(&u,&f,4);u+=0x7fff+((u>>16)&1);return u>>16;}
static float fp32(uint16_t b){uint32_t u=uint32_t(b)<<16;float f;memcpy(&f,&u,4);return f;}
int main(){
  constexpr int M=14336,K=512,N=512,E=EXPERTS,GUARD=256;
  constexpr size_t NA=size_t(M)*K,NB=size_t(E)*K*N,NC=size_t(M)*N;
  auto x=(uint16_t*)std::malloc(NA*2);auto w=(uint16_t*)std::malloc(NB*2);
  auto actual=(uint16_t*)std::malloc((NC+2*GUARD)*2);
  auto expected=(float*)std::calloc(NC,4);
  int64_t group[E];for(int e=0;e<E;e++)group[e]=int64_t(e+1)*(M/E);
  for(size_t i=0;i<NA;i++)x[i]=bf16(float(int((i*3+i/7+i/256)%17)-8)/8);
  for(size_t i=0;i<NB;i++)w[i]=bf16(float(int((i*7+i/11+i/128)%17)-8)/8);
  // Independent CPU FP32 reference, contiguous N loop and one BF16 rounding.
  for(int e=0;e<E;e++)for(int m=e*(M/E);m<(e+1)*(M/E);m++)for(int kk=0;kk<K;kk++){
    float a=fp32(x[size_t(m)*K+kk]);
    for(int nn=0;nn<N;nn++)expected[size_t(m)*N+nn]+=a*fp32(w[(size_t(e)*K+kk)*N+nn]);
  }
  for(size_t i=0;i<NC+2*GUARD;i++)actual[i]=0x7fc1;
  printf("REFERENCE_READY E=%d total_M=%d N=%d K=%d values=%zu\n",E,M,N,K,NC);fflush(stdout);
  CHECK(aclInit(nullptr));CHECK(aclrtSetDevice(0));aclrtStream stream;CHECK(aclrtCreateStream(&stream));
  void *dx,*dw,*dg,*dy;
  CHECK(aclrtMalloc(&dx,NA*2,ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMalloc(&dw,NB*2,ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMalloc(&dg,E*8,ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMalloc(&dy,(NC+2*GUARD)*2,ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMemcpy(dx,NA*2,x,NA*2,ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK(aclrtMemcpy(dw,NB*2,w,NB*2,ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK(aclrtMemcpy(dg,E*8,group,E*8,ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK(aclrtMemcpy(dy,(NC+2*GUARD)*2,actual,(NC+2*GUARD)*2,ACL_MEMCPY_HOST_TO_DEVICE));
  uint16_t* output=(uint16_t*)dy+GUARD;
  KERNEL<<<28,nullptr,stream>>>(ARGUMENTS);
  CHECK(aclrtSynchronizeStream(stream));
  CHECK(aclrtMemcpy(actual,(NC+2*GUARD)*2,dy,(NC+2*GUARD)*2,ACL_MEMCPY_DEVICE_TO_HOST));
  size_t bad=0;float maxerr=0;
  for(size_t i=0;i<NC;i++)if(actual[GUARD+i]!=bf16(expected[i])){
    if(bad++<4)printf("MISMATCH i=%zu actual=%g expected=%g\n",i,fp32(actual[GUARD+i]),fp32(bf16(expected[i])));
    float d=fp32(actual[GUARD+i])-fp32(bf16(expected[i]));if(d<0)d=-d;if(d>maxerr)maxerr=d;
  }
  for(int i=0;i<GUARD;i++)if(actual[i]!=0x7fc1||actual[GUARD+NC+i]!=0x7fc1){printf("GUARD_FAIL %d\n",i);return 4;}
  printf("NUMERICAL_RESULT E=%d count=%zu bad=%zu maxerr=%g guard=pass\n",E,NC,bad,maxerr);fflush(stdout);
  if(bad)return 3;
  CHECK(aclrtFree(dx));CHECK(aclrtFree(dw));CHECK(aclrtFree(dg));CHECK(aclrtFree(dy));
  CHECK(aclrtDestroyStream(stream));CHECK(aclrtResetDevice(0));CHECK(aclFinalize());
  std::free(x);std::free(w);std::free(actual);std::free(expected);
  printf("NUMERICAL_PASS E=%d BF16 exact\n",E);return 0;
}
"""


def execute(argv, directory, log_name, env, timeout=600):
    started = time.monotonic()
    with (directory / log_name).open("w") as log:
        try:
            rc = subprocess.run(argv, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    return {"argv": argv, "exit_code": rc, "host_wall_seconds": time.monotonic() - started, "log": log_name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", default="28,2")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    camodel = "/usr/local/Ascend/cann/tools/simulator/dav_3510/camodel"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = camodel + ":" + env.get("LD_LIBRARY_PATH", "")
    results = []
    for experts in map(int, args.experts.split(",")):
        if 14336 % experts:
            raise ValueError("Equal group size must be integral")
        directory = ROOT / f"e{experts}-m14336-n512-k512"
        directory.mkdir(exist_ok=False)
        record = {
            "experts": experts,
            "total_m": 14336,
            "group_m": 14336 // experts,
            "n": 512,
            "k": 512,
            "useful_flops": 2 * 14336 * 512 * 512,
            "blocks": 28,
            "warmup": 0,
            "launches_per_process": 1,
            "tile": [256, 256, 256],
            "l0_k": 64,
            "tilelang_import": tilelang.__file__,
            "source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        }
        (directory / "candidate.py").write_bytes(Path(module.__file__).read_bytes())
        try:
            print("LOWER_START", experts, flush=True)
            program = module.group_gemm_large(experts=experts)
            source = tilelang.lower(program, target="ascend", enable_device_compile=False).kernel_source
            (directory / "kernel.cce").write_text(source)
            (directory / "input.tir").write_text(program.script())
            name, params = re.search(r"void (\w+)\(([^)]*)\)", source).groups()
            order = [x.strip().split()[-1].lstrip("*") for x in params.split(",")]
            pointers = {"X": "(bfloat16_t*)dx", "W": "(bfloat16_t*)dw", "Group": "(int64_t*)dg", "Y": "(bfloat16_t*)output"}
            defines = (
                f"\n#define KERNEL {name}\n#define EXPERTS {experts}\n#define ARGUMENTS " + ", ".join(pointers[x] for x in order) + "\n"
            )
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
                raise RuntimeError((directory / "build.log").read_text()[-5000:])
            subprocess.run(["patchelf", "--set-rpath", camodel, str(directory / "launch")], check=True)
            record["generated_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            record["binary_sha256"] = hashlib.sha256((directory / "launch").read_bytes()).hexdigest()
            print("NUMERICAL_START", experts, flush=True)
            record["numerical"] = execute([str(directory / "launch")], directory, "numerical.log", env)
            record["numerical_pass"] = (
                record["numerical"]["exit_code"] == 0 and "NUMERICAL_PASS" in (directory / "numerical.log").read_text()
            )
            if not record["numerical_pass"]:
                raise RuntimeError((directory / "numerical.log").read_text()[-5000:])
            print("PROFILE_START", experts, flush=True)
            command = [
                sys.executable,
                "-I",
                str(REPO / "tools/groupgemm/profile_group_gemm_large.py"),
                str(directory / "launch"),
                "--output",
                str(directory / "profiles"),
            ]
            record["profile_process"] = execute(command, directory, "profile-process.log", env, 650)
            if record["profile_process"]["exit_code"]:
                raise RuntimeError((directory / "profile-process.log").read_text()[-5000:])
            profile = json.loads(next((directory / "profiles").rglob("profile.json")).read_text())
            info = {x["KernelInfoAttr"]: x["KernelInfoVal"] for x in profile["databases"][0]["kernel_info"]}
            cycles = info["exec_time_ticks"]
            frequency = info["frequency_ghz"]
            seconds = cycles / (frequency * 1e9)
            tflops = record["useful_flops"] / seconds / 1e12
            record["diagnostic"] = {
                "cycles": cycles,
                "frequency_ghz": frequency,
                "time_us": seconds * 1e6,
                "useful_tflops": tflops,
                "mfu32_percent": tflops / 432 * 100,
                "mfu28_target_percent": tflops / 378 * 100,
                "profile_numerical_pass": profile["numerical_pass"],
                "kind": "simulator_estimate_not_real_device_measurement",
                "kernel_info": info,
            }
            print("CASE_DONE", experts, json.dumps(record["diagnostic"]), flush=True)
        except Exception as error:
            record["error"] = str(error)
            print("CASE_FAIL", experts, str(error), flush=True)
        (directory / "result.json").write_text(json.dumps(record, indent=2))
        results.append(record)
        (ROOT / "summary.json").write_text(json.dumps(results, indent=2))
        if "error" in record:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
