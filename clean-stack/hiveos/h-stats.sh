#!/usr/bin/env bash
# h-stats.sh -- report miner stats to the HiveOS agent.
#
# Sets the khs and stats variables (and echoes both, like other custom miner
# packages). On a multi GPU rig matador-miner runs one process per GPU with
# status APIs on consecutive ports (base, base+1, ...), so this script probes
# ports until one stops answering and aggregates everything it found.

[[ -e /hive/custom ]] && . /hive/custom/matador-miner/h-manifest.conf
[[ -e /hive/miners/custom ]] && . /hive/miners/custom/matador-miner/h-manifest.conf
# Test override: MATADOR_HIVE_DIR points at an unpacked package dir.
[[ -n "${MATADOR_HIVE_DIR:-}" && -e "$MATADOR_HIVE_DIR/h-manifest.conf" ]] && . "$MATADOR_HIVE_DIR/h-manifest.conf"

API_HOST="${MATADOR_API_HOST:-127.0.0.1}"
API_PORT="${CUSTOM_API_PORT:-4060}"
MAX_GPUS=16

# Zero stats keep the agent happy when the miner or its API is not up yet.
khs=0
stats='{"total_khs":0,"khs":0,"hs_units":"khs","hs":[0],"temp":[0],"fan":[0],"uptime":0,"ver":"unknown","ar":[0,0],"algo":"btx"}'

if command -v jq >/dev/null 2>&1 && command -v curl >/dev/null 2>&1; then
    summaries="[]"
    for ((i = 0; i < MAX_GPUS; i++)); do
        s=$(curl -s --connect-timeout 1 --max-time 2 "http://${API_HOST}:$((API_PORT + i))/summary" 2>/dev/null)
        [[ -z "$s" ]] && break
        jq -e . >/dev/null 2>&1 <<< "$s" || break
        summaries=$(jq -c --argjson one "$s" '. + [$one]' <<< "$summaries")
    done

    count=$(jq 'length' <<< "$summaries")
    if [[ "$count" -gt 0 ]]; then
        stats=$(jq -c '
            def rate: (.averages."1h".nonce_per_s // 0);
            def r2: (. * 100 | round) / 100;
            {
                total_khs: (([.[] | rate] | add) / 1000 | r2),
                khs:       (([.[] | rate] | add) / 1000 | r2),
                hs_units:  "khs",
                hs:        [.[] | rate / 1000 | r2],
                temp:      [.[] | (.gpu_runtime[0].temp_c // 0)],
                fan:       [.[] | 0],
                uptime:    ([.[] | (.uptime_sec // 0)] | max),
                ver:       (.[0].version // "unknown"),
                ar:        [([.[] | (.shares.accepted // 0)] | add),
                            ([.[] | (.shares.rejected // 0)] | add)],
                algo:      "btx"
            }' <<< "$summaries")
        khs=$(jq -r '.khs' <<< "$stats")
    fi
fi

echo "khs:   $khs"
echo "stats: $stats"
