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

echo "built: bin/run$EXT bin/runq$EXT bin/eval_bpb_q$EXT bin/eval_bpb_f32$EXT"
