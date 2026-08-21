#!/usr/bin/env bash
# h-stats.sh -- report miner stats to the HiveOS agent.
#
# Sets the khs and stats variables (and echoes both, like other custom miner
# packages). On a multi GPU rig matador-miner runs one process per GPU with
# status APIs on consecutive ports (base, base+1, ...), so this script probes
# ports and aggregates everything it found.
#
# UNITS. The ENC_RC unit of work is the EPISODE: one episode is one full
# dependent INT8 GEMM chain for ONE NONCE, so episodes/s IS the hashrate in the
# classic sense -- nonces tried per second. It is just that a v4 "hash" is
# enormously more expensive than a SHA256 one, so a healthy 5090 sits near
# 1.2-1.5 H/s rather than tens of MH/s. hs[] is therefore reported in "hs"
# units, and khs/total_khs carry the same figure in the kH/s that HiveOS
# expects (the agent scales it back down to H/s for display).
#
# Do NOT "fix" the small numbers by reporting ep/s as though it were kH/s. That
# is a 1000x lie and it would make every pool-vs-rig comparison wrong.

[[ -e /hive/custom ]] && . /hive/custom/matador-miner/h-manifest.conf
[[ -e /hive/miners/custom ]] && . /hive/miners/custom/matador-miner/h-manifest.conf
# Test override: MATADOR_HIVE_DIR points at an unpacked package dir.
[[ -n "${MATADOR_HIVE_DIR:-}" && -e "$MATADOR_HIVE_DIR/h-manifest.conf" ]] && . "$MATADOR_HIVE_DIR/h-manifest.conf"

API_HOST="${MATADOR_API_HOST:-127.0.0.1}"
API_PORT="${CUSTOM_API_PORT:-4060}"
MAX_GPUS=16

# Zero stats keep the agent happy when the miner or its API is not up yet.
khs=0
stats='{"total_khs":0,"khs":0,"hs_units":"hs","hs":[0],"temp":[0],"fan":[0],"uptime":0,"ver":"unknown","ar":[0,0],"algo":"btx"}'

if command -v jq >/dev/null 2>&1 && command -v curl >/dev/null 2>&1; then
    summaries="[]"
    # Probe every port rather than stopping at the first silent one: a child that is
    # mid-restart used to zero the WHOLE rig's reported hashrate. Refused connections
    # on loopback return immediately, so scanning all of them costs nothing.
    for ((i = 0; i < MAX_GPUS; i++)); do
        s=$(curl -s --connect-timeout 1 --max-time 2 "http://${API_HOST}:$((API_PORT + i))/summary" 2>/dev/null)
        [[ -z "$s" ]] && continue
        # Must parse AND look like a matador summary -- an unrelated service that
        # happens to sit on one of these ports must not land in the aggregate.
        jq -e 'type == "object" and has("averages")' >/dev/null 2>&1 <<< "$s" || continue
        summaries=$(jq -c --argjson one "$s" '. + [$one]' <<< "$summaries")
    done

    count=$(jq 'length' <<< "$summaries")
    if [[ "$count" -gt 0 ]]; then
        stats=$(jq -c '
            # Episodes/s for one GPU. Lead with the live rate so a clock change shows up
            # within a heartbeat (~30s) -- that is what an operator tuning OC is watching.
            # Fall back to the 5m then the 1h window while the first interval is still
            # elapsing, and because a miner older than this package has neither field.
            #
            # This used to read .averages."1h".nonce_per_s. Nothing has emitted
            # nonce_per_s since the v3 solver was removed: v4 counts EPISODES, so the
            # lookup silently returned null, "// 0" turned it into 0, and every HiveOS
            # rig running v4 reported exactly 0 hashrate no matter how well it was mining.
            def rate:
                (.rate.episode_per_s // 0) as $now
                | if $now > 0 then $now
                  else (.averages."5m".episode_per_s // 0) as $m5
                       | if $m5 > 0 then $m5
                         else (.averages."1h".episode_per_s // 0) end
                  end;
            def r(n): (. * pow(10; n) | round) / pow(10; n);
            ([.[] | rate] | add) as $total
            | {
                # khs/total_khs are kH/s by HiveOS contract, so convert -- and keep enough
                # decimals to survive it (2dp would round a whole rig back to 0.00).
                total_khs: ($total / 1000 | r(6)),
                khs:       ($total / 1000 | r(6)),
                # hs[] is per GPU, in the units named by hs_units.
                hs_units:  "hs",
                hs:        [.[] | rate | r(3)],
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
