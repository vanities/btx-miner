#!/usr/bin/env bash
# h-config.sh -- translate HiveOS flight sheet fields into matador-miner arguments.
#
# The Hive agent exports these before calling this script:
#   CUSTOM_URL          Pool URL(s), e.g. stratum+tcp://btx-us-east.lproute.com:8660
#   CUSTOM_TEMPLATE     Wallet template, already expanded, e.g. btx1qq...abc.myrig
#                       (from %WAL%.%WORKER_NAME%)
#   CUSTOM_PASS         Pool password (usually x)
#   CUSTOM_USER_CONFIG  Extra config arguments, passed through to the miner verbatim
#
# The final argument line is written to $CUSTOM_CONFIG_FILENAME; h-run.sh reads it.

[[ -e /hive/custom ]] && . /hive/custom/matador-miner/h-manifest.conf
[[ -e /hive/miners/custom ]] && . /hive/miners/custom/matador-miner/h-manifest.conf
# Test override: MATADOR_HIVE_DIR points at an unpacked package dir.
[[ -n "${MATADOR_HIVE_DIR:-}" && -e "$MATADOR_HIVE_DIR/h-manifest.conf" ]] && . "$MATADOR_HIVE_DIR/h-manifest.conf" && CUSTOM_CONFIG_FILENAME="$MATADOR_HIVE_DIR/matador-miner.conf"

# Wallet template: "<address>" or "<address>.<worker>". BTX bech32 addresses
# never contain a dot, so the first dot separates address from worker name.
TEMPLATE="${CUSTOM_TEMPLATE:-}"
WALLET="${TEMPLATE%%.*}"
WORKER="${TEMPLATE#*.}"
[[ "$WORKER" == "$TEMPLATE" ]] && WORKER=""
# No worker in the template: fall back to the rig's worker name from HiveOS.
[[ -z "$WORKER" ]] && WORKER="${WORKER_NAME:-}"

if [[ -z "$WALLET" ]]; then
    echo "ERROR: no wallet in flight sheet template (expected %WAL% or %WAL%.%WORKER_NAME%)"
fi

conf="--mode pool"

# Pool URL(s), passed through with the scheme intact: the miner strips a plaintext
# scheme itself (stratum+tcp://, stratum://, or any unrecognized one) and reads TLS
# off an ssl:// / tls:// / stratum+ssl:// / stratum+tls:// prefix, so stripping here
# would silently downgrade a TLS pool to plaintext. Multiple space/comma/semicolon
# separated URLs become the primary pool plus fallbacks.
URLS="${CUSTOM_URL:-}"
URLS="${URLS//;/ }"
URLS="${URLS//,/ }"
for u in $URLS; do
    [[ -z "$u" ]] && continue
    conf+=" --pool $u"
done

conf+=" --payoutaddress $WALLET"
[[ -n "$WORKER" ]] && conf+=" --worker $WORKER"
conf+=" --pool-pass ${CUSTOM_PASS:-x}"

# Fixed base port so h-stats.sh can find the status API (one port per GPU).
conf+=" --api-port ${CUSTOM_API_PORT:-4060}"

# HiveOS owns the install: update by bumping the Installation URL to a newer
# package. The miner still logs when a newer release is available.
conf+=" --no-auto-update"

# Extra config arguments pass through last so they win over the defaults above.
[[ -n "${CUSTOM_USER_CONFIG:-}" ]] && conf+=" $CUSTOM_USER_CONFIG"

echo "$conf"
[[ -n "${CUSTOM_CONFIG_FILENAME:-}" ]] && echo "$conf" > "$CUSTOM_CONFIG_FILENAME"
