#!/usr/bin/env bash
# h-run.sh -- start matador-miner under HiveOS with the args h-config.sh generated.

cd "$(dirname "$0")" || exit 1
. ./h-manifest.conf
# Test override: MATADOR_HIVE_DIR points at an unpacked package dir, so the whole
# h-config -> h-run -> h-stats chain can be driven off a real rig. h-config.sh and
# h-stats.sh already honoured it; without it here the run leg still looked for the
# manifest's absolute /hive/... config path and could only ever be tested on HiveOS.
[[ -n "${MATADOR_HIVE_DIR:-}" && -e "$MATADOR_HIVE_DIR/h-manifest.conf" ]] && \
    CUSTOM_CONFIG_FILENAME="$MATADOR_HIVE_DIR/matador-miner.conf"

LOG_DIR=$(dirname "$CUSTOM_LOG_BASENAME")
[[ ! -d "$LOG_DIR" ]] && mkdir -p "$LOG_DIR"

if [[ -z "$CUSTOM_CONFIG_FILENAME" || ! -f "$CUSTOM_CONFIG_FILENAME" ]]; then
    echo "ERROR: miner config not found (run h-config.sh first): $CUSTOM_CONFIG_FILENAME"
    exit 1
fi
ARGS=$(< "$CUSTOM_CONFIG_FILENAME")

# One binary: Ampere or newer. ENC_RC is a tensor-core proof of work, so a pre-Ampere
# card cannot mine at any speed. Stop with a clear message rather than burning rig power
# on a binary that will never find a share.
BIN=./matador-miner
CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null)
if [[ -n "$CAPS" ]]; then
    old_gpu=0
    modern=0
    while IFS= read -r cap; do
        cap="${cap// /}"
        [[ -z "$cap" ]] && continue
        major="${cap%%.*}"
        if [[ "$major" =~ ^[0-9]+$ ]]; then
            if (( major < 8 )); then old_gpu=$((old_gpu + 1)); else modern=$((modern + 1)); fi
        fi
    done <<< "$CAPS"
    if (( old_gpu > 0 && modern == 0 )); then
        echo "ERROR: every GPU on this rig is pre-Ampere (compute capability < 8.0)."
        echo "       BTX needs Ampere-or-newer tensor cores. This rig cannot mine."
        exit 1
    elif (( old_gpu > 0 && modern > 0 )); then
        echo "WARNING: mixed GPU generations detected. Using the supported cards only;"
        echo "         pre-Ampere cards (compute < 8.0) will not start. Pin the"
        echo "         supported cards with: --gpus <ids> in Extra config arguments."
    fi
fi

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: miner binary missing or not executable: $BIN"
    exit 1
fi

# --- glibc compatibility -------------------------------------------------------
# The release binary is built on Ubuntu 22.04 and needs glibc >= 2.34 (the release
# where pthread/dl/sem were merged into libc). HiveOS images on an Ubuntu 20.04
# base ship glibc 2.31, where the loader refuses the binary outright:
#     libc.so.6: version `GLIBC_2.34' not found
# That floor cannot be linked away: it comes from the toolchain's own static
# libstdc++/libcrypto, not from our sources.
#
# So the package carries a matching glibc in runtime/ and, ONLY when the host's is
# too old, launches through it. Verified on a real Ubuntu 20.04 userland including
# name resolution, which is why runtime/ also carries libnss_* and libresolv --
# glibc dlopens those at resolve time and a version mismatch breaks DNS, which for
# a miner means it can never reach the pool.
LOADER=()
GLIBC_MIN=2.34
host_glibc=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$')
if [[ -n "$host_glibc" && -x ./runtime/ld-linux-x86-64.so.2 ]]; then
    older=$(printf '%s\n%s\n' "$host_glibc" "$GLIBC_MIN" | sort -V | head -1)
    if [[ "$older" == "$host_glibc" && "$host_glibc" != "$GLIBC_MIN" ]]; then
        echo "host glibc $host_glibc is older than $GLIBC_MIN: using the bundled runtime"
        LOADER=(./runtime/ld-linux-x86-64.so.2 --library-path
                "./runtime:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/usr/local/lib")
    fi
fi

echo "starting: ${LOADER[*]} $BIN $ARGS"

# stdbuf keeps the HiveOS log wrapper fed without buffering.
stdbuf -i0 -o0 -e0 "${LOADER[@]}" "$BIN" $ARGS 2>&1 | tee -a "${CUSTOM_LOG_BASENAME}.log"
echo "Miner has exited"
