#!/usr/bin/env bash
# Build the reference check. Compiles PR #89's UNMODIFIED int8_field.cpp +
# matmul_v4.cpp against thin shadow headers + our KAT-verified SHA-256.
# Usage: BTX_SRC=/path/to/btx-v4/src ./build.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BTX_SRC="${BTX_SRC:-}"
if [[ -z "$BTX_SRC" || ! -f "$BTX_SRC/matmul/int8_field.cpp" ]]; then
  echo "Set BTX_SRC to the PR #89 clone's src/ dir (contains matmul/int8_field.cpp)." >&2
  echo "  git clone --depth 1 -b claude/matmul-v4-design-spec-af23sj https://github.com/btxchain/btx.git" >&2
  exit 1
fi
clang++ -O3 -std=c++20 \
  -I"$HERE/shadow" -I"$BTX_SRC" \
  "$HERE/refcheck.cpp" \
  "$HERE/sha256_impl.cpp" \
  "$BTX_SRC/matmul/int8_field.cpp" \
  "$BTX_SRC/matmul/matmul_v4.cpp" \
  -o "$HERE/refcheck"
echo "built $HERE/refcheck"
