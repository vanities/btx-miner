#!/bin/bash
# Continuous tip-holder + TRUE-TIP PINNER. On this network the public/seed nodes are frozen at
# the stuck-cluster height (~185570, the self-qual bug) and only a few nodes carry the LIVE chain.
# Our node repeatedly strands itself on dead forks / loses the live peers. So this:
#   EVERY cycle  -> re-pin the curated live-chain nodes (livepeers.txt) via addnode add+onetry,
#                   so the pins survive the ~10-min wedge restarts.
#   periodically -> DISCOVER live nodes (find_live_peers.py: probe address book, keep those above
#                   the stuck cluster) and refresh livepeers.txt.
#   when behind  -> shed stuck/dead-fork peers + drain via mopup.
# HONEST CEILING: if the live-chain nodes stop accepting us, we can still strand; this maximizes
# staying pinned to whoever actually has the true tip.
set -u
C=btx-miner-0332
MOPUP=/home/vanities/matador-solo/mopup.py
DISC=/home/vanities/matador-solo/find_live_peers.py
LIVE=/home/vanities/matador-solo/livepeers.txt
DRIFT=2
PROBE_S=20
ON_TIP=2               # resume mining when behind <= this (on/near the true tip)
OFF_TIP=5              # pause mining when behind >= this (mining stale = orphan + wasted power)
STUCK_CEIL=185600      # peers at/below this height = frozen self-qual cluster -> ban them
GATE_SENT=/home/vanities/matador-solo/.mining_paused_offtip
log(){ echo "[$(date +%H:%M:%S)] $*"; }
cli(){ docker exec "$C" timeout 12 btx-cli -datadir=/data "$@" 2>/dev/null; }

best_header(){ cli getchaintips | python3 -c "
import json,sys
try:
  t=json.load(sys.stdin); c=[x['height'] for x in t if x['status'] in ('headers-only','valid-headers')]
  print(max(c) if c else 0)
except: print(0)" 2>/dev/null; }

pin_live(){   # re-assert persistent connections to the curated live-chain nodes
  [ -f "$LIVE" ] || return
  while read -r a; do
    [ -z "$a" ] && continue
    cli addnode "$a" add >/dev/null 2>&1
    cli addnode "$a" onetry >/dev/null 2>&1
  done < "$LIVE"
}

shed_dead(){   # BAN confirmed stuck-cluster peers (frozen self-qual nodes) so they stay off and
  # stop eating slots the live chain needs; just DISCONNECT the merely-behind (they may catch up).
  local tip="$1"
  cli getpeerinfo | python3 -c "
import json,sys
tip=int('$tip'); STUCK=$STUCK_CEIL
try: p=json.load(sys.stdin)
except: p=[]
for x in p:
  h=x.get('startingheight',0); ip=x['addr'].rsplit(':',1)[0]
  if h<=STUCK: print('BAN', ip)
  elif h<tip-40: print('DROP', x['id'])
" | while read act val; do
    if [ "$act" = BAN ]; then cli setban "$val" add 86400 >/dev/null 2>&1
    elif [ "$act" = DROP ]; then cli disconnectnode "" "$val" >/dev/null 2>&1; fi
  done
}

log "tip-holder + true-tip pinner up (pin every cycle; discover periodically; catch up >=${DRIFT})"
disc_gate=0
while true; do
  tip=$(cli getblockcount)
  if [ -z "$tip" ]; then sleep "$PROBE_S"; continue; fi   # wedge -> watchdog owns it
  pin_live                                                # <-- EVERY cycle: keep live nodes pinned
  disc_gate=$(( disc_gate + 1 ))
  if [ "$disc_gate" -ge 9 ]; then                         # ~every 3 min: rediscover live nodes
    timeout 40 python3 "$DISC" 2>&1 | grep -E "true-tip|live-pin|live nodes" | sed 's/^/  disc: /'
    disc_gate=0
  fi
  best=$(best_header)
  if [ -n "$best" ] && [ "$best" -gt 0 ] 2>/dev/null; then
    behind=$(( best - tip ))
    # MINE-ON-TIP GATE: mining behind the tip only produces orphans + burns 600W. Pause the miner
    # when we drift off (also frees the GPU so the keeper's ExactReplay attestations catch us up
    # FASTER), auto-resume when back on the tip. Hysteresis ON_TIP<=..<OFF_TIP avoids flapping.
    mact=$(systemctl is-active matador-miner 2>/dev/null)
    if [ "$behind" -ge "$OFF_TIP" ] && [ "$mact" = "active" ]; then
      log "OFF-TIP behind=$behind -> PAUSE miner (orphan work; GPU -> keeper catch-up)"
      touch "$GATE_SENT"; systemctl stop matador-miner.service 2>/dev/null
    elif [ "$behind" -le "$ON_TIP" ] && [ "$mact" != "active" ] && [ -f "$GATE_SENT" ]; then
      log "ON-TIP behind=$behind -> RESUME miner"
      rm -f "$GATE_SENT"; systemctl start matador-miner.service 2>/dev/null
    fi
    if [ "$behind" -ge "$DRIFT" ]; then
      log "behind=$behind (tip=$tip best=$best) -> catch-up"
      shed_dead "$tip"
      timeout 90 python3 "$MOPUP" 2>&1 | grep -E "swept|CAUGHT-UP|MISSING" | sed 's/^/  /'
      t2=$(cli getblockcount)
      log "catch-up done: tip $tip -> ${t2:-?}"
    fi
  fi
  sleep "$PROBE_S"
done
