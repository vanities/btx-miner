#!/bin/bash
# Fast btxd-wedge watchdog (stock v0.33.2 deadlocks unprompted; 2 wedges in 40 min on 2026-08-10).
# Every 30s: probe RPC. On a dead probe, CONFIRM before acting (avoid killing a legit block-
# connection batch): re-probe after 20s AND require that btxd has NOT connected a block recently
# (a busy-connecting node still advances UpdateTip). Only a persistently-dead RPC with a stale
# UpdateTip triggers the SIGTERM-flush recovery (unwedge_node.sh). 5-min cooldown after a remedy.
set -u
C=btx-miner-0332
UNWEDGE=/home/vanities/matador-solo/unwedge_node.sh
PROBE_S=30
STALE_S=180          # no UpdateTip for this long + dead RPC = wedged (not just busy)
COOLDOWN_S=300
log(){ echo "[$(date +%H:%M:%S)] $*"; }

rpc_ok(){ [ -n "$(docker exec "$C" timeout 12 btx-cli -datadir=/data getblockcount 2>/dev/null)" ]; }

last_updatetip_age(){
  # seconds since the most recent UpdateTip line in debug.log (99999 if none/parse fail)
  local ts
  ts=$(docker exec "$C" sh -c "grep UpdateTip /data/debug.log 2>/dev/null | tail -1" | tr -d '\0' | grep -oE '^[0-9T:-]+Z')
  [ -z "$ts" ] && { echo 99999; return; }
  local then now
  then=$(date -d "$ts" +%s 2>/dev/null) || { echo 99999; return; }
  now=$(date -u +%s)
  echo $(( now - then ))
}

log "node watchdog up (probe ${PROBE_S}s, stale ${STALE_S}s, cooldown ${COOLDOWN_S}s)"
last_remedy=0
while true; do
  if rpc_ok; then sleep "$PROBE_S"; continue; fi
  # strike 1: RPC dead -- confirm it is not a momentary hiccup
  sleep 20
  if rpc_ok; then log "RPC blip cleared (transient); no action"; sleep "$PROBE_S"; continue; fi
  # strike 2: still dead. Is the node merely busy connecting blocks?
  age=$(last_updatetip_age)
  if [ "$age" -lt "$STALE_S" ]; then
    log "RPC dead but UpdateTip is ${age}s old (<${STALE_S}s) -- node busy connecting, NOT remedying"
    sleep "$PROBE_S"; continue
  fi
  now=$(date -u +%s)
  if [ $(( now - last_remedy )) -lt "$COOLDOWN_S" ]; then
    log "confirmed-wedge signature but within cooldown ($(( now - last_remedy ))s); waiting"
    sleep "$PROBE_S"; continue
  fi
  log "CONFIRMED WEDGE: RPC dead 30s+, last UpdateTip ${age}s ago -> running unwedge"
  bash "$UNWEDGE" 2>&1 | sed 's/^/  /'
  last_remedy=$(date -u +%s)
  sleep "$PROBE_S"
done
