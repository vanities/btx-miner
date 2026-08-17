#!/bin/bash
# One-shot rig health probe. Prints exactly one line: "OK ..." or "ALERT: ...".
# Knows the expected-quiet states (A/B timing window) so it does not false-alarm.
set -u
C=btx-miner-0332
STATE=/home/vanities/matador-solo/.health_state   # last rejected count, for trend

miner_active=$(systemctl is-active matador-miner.service 2>/dev/null)
keeper_active=$(systemctl is-active matador-attest.service 2>/dev/null)
autoab=$(systemctl --user is-active autoab 2>/dev/null || echo inactive)

# Node RPC responsiveness = the wedge canary.
tip=$(docker exec "$C" timeout 12 btx-cli -datadir=/data getblockcount 2>/dev/null)
if [ -z "$tip" ]; then
  echo "ALERT: node RPC unresponsive (getblockcount timed out) -- possible btxd wedge"
  exit 0
fi

# Tip-lag vs the network best header (compute now; the STALE alert only applies if we're MINING).
best=$(docker exec "$C" timeout 12 btx-cli -datadir=/data getchaintips 2>/dev/null | \
  python3 -c "import json,sys
try:
  t=json.load(sys.stdin); c=[x['height'] for x in t if x['status'] in ('headers-only','valid-headers')]
  print(max(c) if c else 0)
except: print(0)" 2>/dev/null)
behind=""
[ -n "$best" ] && [ "$best" -gt 0 ] 2>/dev/null && behind=$(( best - tip ))

# Miner legitimately stopped: (a) mine-on-tip gate paused it off-tip, (b) A/B timing window.
# Checked BEFORE the stale alert -- a gate-paused miner behind the tip is CORRECT, not a fault.
if [ "$miner_active" != "active" ]; then
  if [ -f /home/vanities/matador-solo/.mining_paused_offtip ]; then
    echo "OK miner PAUSED off-tip by gate (would orphan); tip=$tip best=$best behind=${behind:-?} keeper=$keeper_active"
  elif [ "$autoab" = "active" ] || [ -f /home/vanities/matador-solo/.ab_running ]; then
    echo "OK ab-window running (miner intentionally stopped for timing); node tip=$tip keeper=$keeper_active"
  else
    echo "ALERT: matador-miner is $miner_active (not A/B, not gate) -- miner is DOWN; node tip=$tip"
  fi
  exit 0
fi

# Miner IS active -> tip-lag is the ORPHAN canary now. behind>=8 = actively mining stale (the gate
# should have paused it; if we see this, the gate isn't keeping up). 3-7 = drifting note.
lag_note=""
if [ -n "$behind" ] && [ "$behind" -ge 8 ]; then
  echo "ALERT: mining STALE -- tip=$tip is $behind behind best=$best; found blocks ORPHAN (gate should pause)"
  exit 0
fi
[ -n "$behind" ] && [ "$behind" -ge 3 ] && lag_note=" behind=$behind(drifting)"

# Miner health from the latest [stats] line.
line=$(journalctl -u matador-miner -n 40 --no-pager 2>/dev/null | grep "\[stats\]" | tail -1)
eps=$(echo "$line" | grep -oE "ep/s=[0-9.]+" | cut -d= -f2)
rc=$(echo "$line" | grep -oE "rc-active=[0-9]+" | cut -d= -f2)
rej=$(echo "$line" | grep -oE "rejected=[0-9]+" | cut -d= -f2)
up=$(echo "$line" | grep -oE "uptime=[0-9hms]+" | cut -d= -f2)

# WIN CANARY -- the event we're actually here for. accepted>0 in solo mode = we mined a block.
# Also catch an explicit found/submitted line across the whole session (survives miner restarts).
acc=$(echo "$line" | grep -oE "accepted=[0-9]+" | cut -d= -f2)
won_line=$(journalctl -u matador-miner --since "2026-08-10 14:49" --no-pager 2>/dev/null | \
  grep -iE "found\+submitted|block found|\[win\]|solo.*accepted|accepted=[1-9]" | tail -1)
if [ -n "${acc:-}" ] && [ "$acc" -gt 0 ] 2>/dev/null; then
  echo "*** WIN *** accepted=$acc block(s)! tip=$tip | ${won_line##*: }"
  exit 0
fi

# No [stats] line yet but node RPC is healthy = miner just restarted (post-wedge warmup).
# This is the ~30s window after a watchdog recovery; not a fault. Report OK, do not cry wolf.
if [ -z "$rc" ]; then
  echo "OK miner warming up (no stats line yet, post-restart); node tip=$tip keeper=$keeper_active"
  exit 0
fi

# rejected trend
prev_rej=$(cat "$STATE" 2>/dev/null || echo "")
echo "${rej:-0}" > "$STATE" 2>/dev/null
rej_note=""
if [ -n "$prev_rej" ] && [ -n "$rej" ] && [ "$rej" -gt "$prev_rej" ] 2>/dev/null; then
  rej_note=" REJECT+$((rej - prev_rej))"
fi

# keeper alerts in the last ~6 min
kalert=$(journalctl -u matador-attest --since "6 min ago" --no-pager 2>/dev/null | grep -c "KEEPER-ALERT")
kmsg=""
[ "${kalert:-0}" -gt 0 ] && kmsg=$(journalctl -u matador-attest --since "6 min ago" --no-pager 2>/dev/null | grep "KEEPER-ALERT" | tail -1 | sed 's/.*KEEPER-ALERT/KEEPER-ALERT/')

# classify
problems=""
[ "${rc:-0}" != "1" ] && problems="${problems} rc-active=${rc:-?}"
# ep/s=0 is a problem only if the miner has been up a while (not a fresh warmup) and no A/B
if [ -n "$eps" ] && awk "BEGIN{exit !($eps==0)}"; then
  case "$up" in
    *h*|*[0-9]m*) [ "${up%m*}" -ge 3 ] 2>/dev/null && problems="${problems} ep/s=0(up=$up)" || true;;
  esac
fi
[ "${kalert:-0}" -gt 0 ] && problems="${problems} ${kmsg}"

if [ -n "$problems" ]; then
  echo "ALERT:${problems}${rej_note}${lag_note} | miner up=$up ep/s=${eps:-?} tip=$tip"
else
  echo "OK ep/s=${eps:-?} rc=${rc:-?} tip=$tip up=$up keeper=$keeper_active${rej_note}${lag_note}"
fi
