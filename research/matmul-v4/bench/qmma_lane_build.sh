#!/bin/bash
# Build + run qmma_lane_bench on the 5090 host against a built btx-97 (wip/matmul-v4.7 +
# device-pack) tree. Run ON the GPU host:
#   R=~/git/btx-97 SRCDIR=~/matador-prof/qmma bash qmma_lane_build.sh [reps]
# Requires: docker image with CUDA 13.3 (matador-build:deps-2204-cuda1330), --gpus all,
# and an EXCLUSIVE GPU window (stop the miner first; this is a perf measurement).
set -euo pipefail
R=${R:-/home/vanities/git/btx-97}
SRCDIR=${SRCDIR:-/home/vanities/matador-prof/qmma}
IMG=${IMG:-matador-build:deps-2204-cuda1330}
REPS=${1:-20}

echo "[qmma-build] R=$R SRCDIR=$SRCDIR reps=$REPS" >&2
t0=$(date +%s)
docker run --rm --gpus all -v "$R":/w -v "$SRCDIR":/s -w /s "$IMG" bash -c '
set -euo pipefail
CUDA=/usr/local/cuda
echo "[qmma-build] compiling..." >&2
g++ -O2 -std=c++17 -I/w/src -I$CUDA/include qmma_lane_bench.cpp -o qmma_lane_bench \
  -Wl,--start-group \
    /w/build/lib/libbitcoin_common.a /w/build/lib/libbitcoin_shielded.a \
    /w/build/lib/libbitcoin_db.a /w/build/src/univalue/libunivalue.a \
    /w/build/lib/libbtx_matmul_backend.a /w/build/lib/libbitcoin_consensus.a \
    /w/build/lib/libbitcoin_util.a /w/build/lib/libbitcoin_clientversion.a \
    /w/build/lib/libbitcoinpqc.a /w/build/src/secp256k1/lib/libsecp256k1.a \
    /w/build/lib/libbitcoin_crypto.a /w/build/src/libleveldb.a /w/build/src/libcrc32c.a \
  -Wl,--end-group \
  -lgomp -L$CUDA/lib64 -lcublasLt -lcudart -lcudadevrt -lpthread -ldl -lrt
echo "[qmma-build] compile ok; running bench (reps='"$REPS"')" >&2
./qmma_lane_bench '"$REPS"'
'
echo "[qmma-build] done in $(( $(date +%s) - t0 ))s" >&2
