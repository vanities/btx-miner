#!/usr/bin/env bash
# Compile + run the standalone clean-stack unit tests (harness/*_test.cpp).
#
# Most tests are header-only: each pulls in a single miner/*.h (VersionGreater, InDevFeeWindow, ...)
# and links nothing -- no btx, no Docker, no cmake configure -- which keeps this fast enough to run on
# every commit (see install-hooks.sh -> .git/hooks/pre-commit). A test that needs a few vendored core
# sources can declare them with a `// run-tests: <cc args>` line (paths relative to the clean-stack
# root); e.g. miner_format_test pulls in arith_uint256 for the difficulty math.
#
#   ./run-tests.sh           # build + run all harness/*_test.cpp, exit non-zero on any failure
#   CXX=clang++ ./run-tests.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CXX="${CXX:-c++}"
OUT="$HERE/build-tests"        # persistent (gitignored), never /tmp
mkdir -p "$OUT"

log() { echo "[$(date +%T)] [run-tests] $*" >&2; }

shopt -s nullglob
tests=("$HERE"/harness/*_test.cpp)
if [ ${#tests[@]} -eq 0 ]; then
    log "no harness/*_test.cpp found -- nothing to run"
    exit 0
fi

log "compiler=$($CXX --version | head -1)"
fails=0
for src in "${tests[@]}"; do
    name="$(basename "$src" .cpp)"
    bin="$OUT/$name"
    t0=$SECONDS
    # Optional per-test directive: `// run-tests: <extra cc args>` lets a test pull in the minimal
    # core sources/includes it needs. Bare tokens are paths under the clean-stack root; -I dirs too.
    extra=""
    directive="$(sed -n 's|.*// run-tests:||p' "$src" | head -1)"
    for tok in $directive; do
        case "$tok" in
            -I*) extra="$extra -I$HERE/${tok#-I}" ;;
            -*)  extra="$extra $tok" ;;
            *)   extra="$extra $HERE/$tok" ;;
        esac
    done
    if ! "$CXX" -std=c++20 -O1 -Wall -Wextra $extra "$src" -o "$bin" 2>"$OUT/$name.build.log"; then
        log "COMPILE FAIL $name (see $OUT/$name.build.log)"
        cat "$OUT/$name.build.log" >&2
        fails=$((fails + 1))
        continue
    fi
    log "built $name in $((SECONDS - t0))s -> running"
    if "$bin"; then
        log "PASS  $name"
    else
        log "FAIL  $name (exit $?)"
        fails=$((fails + 1))
    fi
done

if [ "$fails" -ne 0 ]; then
    log "$fails test(s) FAILED"
    exit 1
fi
log "all ${#tests[@]} test(s) passed"
