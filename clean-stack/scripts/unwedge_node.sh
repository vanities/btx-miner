#!/bin/bash
# Recover a wedged stock-v0.33.2 btxd WITHOUT the 22-min reindex: SIGTERM the supervised
# btxd (clean chainstate flush), let the in-container supervisor respawn it, then bounce the
# host-side miner + keeper so they re-read the regenerated RPC cookie. Proven 2026-08-10.
set -u
C=btx-miner-0332
echo "[unwedge] $(date +%H:%M:%S) start"
BPID=$(docker exec "$C" sh -c 'cat /data/mining-ops/btxd-supervised.pid 2>/dev/null')
[ -z "$BPID" ] && { echo "[unwedge] no btxd pid file; aborting"; exit 1; }
echo "[unwedge] SIGTERM btxd pid=$BPID (clean flush)"
docker exec "$C" sh -c "kill -TERM $BPID" 2>/dev/null
for i in $(seq 1 30); do
  sleep 5
  docker exec "$C" sh -c "kill -0 $BPID 2>/dev/null" || { echo "[unwedge] btxd exited after $((i*5))s"; break; }
done
# supervisor (live-mining-loop.sh) respawns btxd; wait for RPC to answer
echo "[unwedge] waiting for RPC..."
tip=""
for i in $(seq 1 30); do
  tip=$(docker exec "$C" timeout 10 btx-cli -datadir=/data getblockcount 2>/dev/null)
  [ -n "$tip" ] && { echo "[unwedge] RPC up: tip=$tip after $((i*10))s"; break; }
  sleep 10
done
[ -z "$tip" ] && { echo "[unwedge] RPC still dead after 300s -- ESCALATE (may need docker restart + reindex)"; exit 2; }
echo "[unwedge] bouncing keeper for the new cookie"
sudo -n systemctl restart matador-attest.service
# Respect the mine-on-tip gate: if the miner is paused because we're off-tip, keep it paused
# (post-wedge we ARE off-tip -- the tip-holder resumes it once caught up). Otherwise bounce it.
if [ -f /home/vanities/matador-solo/.mining_paused_offtip ]; then
  echo "[unwedge] miner stays PAUSED (off-tip gate); tip-holder resumes it on-tip"
  systemctl stop matador-miner.service 2>/dev/null
else
  echo "[unwedge] bouncing miner for the new cookie"
  systemctl restart matador-miner.service
fi
sleep 20
echo "[unwedge] miner=$(systemctl is-active matador-miner) keeper=$(systemctl is-active matador-attest) tip=$tip"
echo "[unwedge] done"
