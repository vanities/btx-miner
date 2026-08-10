#!/usr/bin/env bash
# matador-status.sh — human-readable summary from the local matador-miner API.
set -euo pipefail

url="${MATADOR_API_URL:-http://127.0.0.1:4060/summary}"

python3 - "$url" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=2) as r:
        data = json.load(r)
except Exception as e:
    print(f"matador-miner: API unavailable ({url}): {e}", file=sys.stderr)
    raise SystemExit(2)

shares = data.get("shares", {})
nonces = data.get("nonces", {})
watchdog = data.get("watchdog", {})
thermal = data.get("thermal", {})
gpus = data.get("gpu_runtime", []) or []
pools = data.get("pools", []) or []

accepted = shares.get("accepted", 0)
rejected = shares.get("rejected", 0)
stale = shares.get("stale", 0)
dev = shares.get("dev", 0)
backend = data.get("backend", "?")
mode = data.get("mode", "?")
version = data.get("version", "?")
uptime = data.get("uptime_sec", 0)
watchdog_status = watchdog.get("status", "?")
thermal_status = thermal.get("status", "?")

print(f"matador-miner {version}: {data.get('status', '?')} mode={mode} backend={backend} uptime={uptime}s")
if pools:
    pool = pools[0]
    print(f"pool: {pool.get('label') or 'primary'} {pool.get('host')}:{pool.get('port')} ({len(pools)} configured)")
print(f"shares: accepted={accepted} rejected={rejected} stale={stale} dev={dev}")
print(f"episodes: {nonces.get('rc_episodes', 0)} windows={nonces.get('solve_windows', '?')} rc_active={nonces.get('rc_active', '?')}")
print(
    "watchdog: "
    f"{watchdog_status} reject_streak={watchdog.get('reject_streak', '?')} "
    f"notify_age={watchdog.get('last_notify_age_sec', '?')}s "
    f"accept_age={watchdog.get('last_accept_age_sec', '?')}s"
)
print(
    "thermal: "
    f"{thermal_status} max_temp={thermal.get('max_temp_c')}C "
    f"max_power={thermal.get('max_power_w')}W"
)
for i, gpu in enumerate(gpus):
    vendor = gpu.get("vendor") or "gpu"
    print(
        f"gpu[{i}] {vendor}: util={gpu.get('util_pct')}% "
        f"power={gpu.get('power_w')}W temp={gpu.get('temp_c')}C "
        f"uuid={gpu.get('gpu_uuid')}"
    )
if not gpus:
    print("gpu: unavailable")

warnings = []
warnings.extend(thermal.get("warnings") or [])
if watchdog.get("last_warning"):
    warnings.append(watchdog["last_warning"])
for w in warnings:
    print(f"warning: {w}")
PY
