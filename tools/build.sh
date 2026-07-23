#!/usr/bin/env bash
# Build host-side inference + eval binaries into bin/.
# Windows (Git Bash): needs MSYS2 gcc on PATH first:
#   export PATH="/c/msys64/ucrt64/bin:$PATH"
set -e
cd "$(dirname "$0")/.."
mkdir -p bin

EXT=""
EXTRA=""
if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then
  EXT=".exe"
  EXTRA="train/llama2.c/win.c"
fi

CFLAGS="-O3 -march=native -Itrain/llama2.c"

gcc $CFLAGS -o "bin/run$EXT"  train/llama2.c/run.c  $EXTRA -lm
gcc $CFLAGS -o "bin/runq$EXT" train/llama2.c/runq.c $EXTRA -lm
gcc $CFLAGS -DTESTING            -o "bin/eval_bpb_q$EXT"   eval/eval_bpb.c $EXTRA -lm
gcc $CFLAGS -DTESTING -DEVAL_FP32 -o "bin/eval_bpb_f32$EXT" eval/eval_bpb.c $EXTRA -lm

# firmware engine 的 host 驗證程式（需先跑 tools/export_header.py 產生 model_data.h）
# 注意：要與上面相同的最佳化旗標，輸出才會與 runq 逐字一致（FMA 融合影響取樣）
if [ -f firmware/tinyllm_arduino/model_data.h ]; then
  gcc $CFLAGS -Ifirmware -o "bin/host_engine$EXT" firmware/host_test.c -lm
  echo "built: bin/host_engine$EXT"
fi

echo "built: bin/run$EXT bin/runq$EXT bin/eval_bpb_q$EXT bin/eval_bpb_f32$EXT"
