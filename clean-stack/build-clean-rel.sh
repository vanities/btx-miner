#!/usr/bin/env bash
# build-clean-rel.sh -- build a RELEASE matador-miner straight from clean-stack (#5, Path B).
# Multi-arch, versioned, byte-exact-gated, profiler-free -- the first patch-free release binary.
#   VER=v0.6.4 ARCH="80;86;89;90;120" bash build-clean-rel.sh
# Prereqs: $IMG (CUDA-13.3 + openblas/lapack/curl + CMake >=4.x); $BTX (a built STOCK btx tree, build-stock).
#
# NOTE: stock matador-build:pathb-deps ships CMake 3.22.1, too old for the
# project's CUDA20 language dialect with nvcc 13.3 (configure fails: "CMake does
# not know the compile flags to use to enable CUDA20"). The default IMG below
# layers CMake 4.3.2 onto pathb-deps (CUDA 13.3 toolchain unchanged); build it
# from the checked-in Dockerfile:
#   docker build -t matador-build:pathb-deps-cm4 -f clean-stack/Dockerfile.pathb-deps-cm4 clean-stack
set -euo pipefail
S="${S:-$HOME/git/matador-src/clean-stack}"
BTX="${BTX:-$HOME/git/btx-stock/build}"
IMG="${IMG:-matador-build:pathb-deps-cm4}"   # CUDA 13.3 + CMake 4.3.2 (see NOTE above)
# CUTLASS supplies the int8 GEMM that replaces the static cuBLASLt link. WITHOUT it the build
# silently falls back to linking cuBLASLt and the binary goes from ~95 MB to ~598 MB -- a 500 MB
# difference that is invisible until someone checks the asset size, so the release path REFUSES
# rather than falling back. Header-only; pin the tag (v4.6.1 as of 2026-08-05):
#   git clone --depth 1 --branch v4.6.1 https://github.com/NVIDIA/cutlass.git ~/git/cutlass461
# Set ALLOW_CUBLASLT=1 to deliberately build the big cuBLASLt-linked binary (A/B work).
CUTLASS="${CUTLASS:-$HOME/git/cutlass461}"
ALLOW_CUBLASLT="${ALLOW_CUBLASLT:-0}"
if [ ! -f "$CUTLASS/include/cutlass/cutlass.h" ]; then
  if [ "$ALLOW_CUBLASLT" != "1" ]; then
    echo "FATAL: CUTLASS not found at $CUTLASS -- a release built without it links cuBLASLt" >&2
    echo "       and ships a ~598 MB binary instead of ~95 MB. Clone it (see build-clean-rel.sh)" >&2
    echo "       or set ALLOW_CUBLASLT=1 if that is genuinely what you want." >&2
    exit 1
  fi
  echo "[build] ALLOW_CUBLASLT=1 and no CUTLASS -- building the LARGE cuBLASLt-linked binary"
  CUTLASS=/dev/null
fi
VER="${VER:-v0.6.4}"
ARCH="${ARCH:-80;86;89;90;120}"
# CPUS caps the container (build-on-the-mining-rig etiquette: feeding keeps its cores).
CPUS="${CPUS:-}"
CPUARG=""; [ -n "$CPUS" ] && CPUARG="--cpus=$CPUS"
CUTLASS_MOUNT=""; CUTLASS_ARG=""
if [ -f "$CUTLASS/include/cutlass/cutlass.h" ]; then
  CUTLASS_MOUNT="-v $CUTLASS:/cutlass:ro"
  CUTLASS_ARG="-DMATADOR_CUTLASS_DIR=/cutlass"
fi
docker run --rm $CPUARG -e "ARCH=$ARCH" -e "VER=$VER" -e "JOBS=${CPUS:-$(nproc)}" \
  -e "CUTLASS_ARG=$CUTLASS_ARG" \
  -v "$S":/src -v "$BTX":/btx $CUTLASS_MOUNT -w /src "$IMG" bash -lc '
  rm -rf build-rel
  cmake -B build-rel -S core -DMATADOR_ENABLE_CUDA=ON -DMATADOR_CUDA_ARCH="$ARCH" \
        -DMATADOR_MINER_VERSION="$VER" -DCMAKE_BUILD_TYPE=Release \
        \
        -DCMAKE_CUDA_FLAGS="-Dconsteval=constexpr" -DBTX_ARCHIVE_DIR=/btx \
        $CUTLASS_ARG
  cmake --build build-rel --target rc_probe matador-miner --parallel "$JOBS"
  echo "[byte-exact gate] expect episode golden 5b1bff3c..."; ./build-rel/rc_probe
  strip --strip-unneeded build-rel/matador-miner || true
  mkdir -p dist
  cp build-rel/matador-miner dist/matador-miner
  ( cd dist && sha256sum matador-miner | tee matador-miner.sha256 )
'
echo "built: $S/dist/matador-miner ($VER)"
