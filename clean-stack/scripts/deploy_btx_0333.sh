#!/bin/bash
# Deploy BTX 0.33.3 (PR#105) by swapping btxd.real/btx-cli.real in the running container.
# Full backup + auto-rollback if btxd fails to come back. Survives wedges (btxd restarts via the
# in-container supervisor use the swapped binary); a full CONTAINER restart would re-provision, so
# this is the "use it now" deploy -- image rebuild is the later durability step.
set -u
C=btx-miner-0332
LB=/data/btx-bin/btx-0.33.2/libexec
NEW=/home/vanities/matador-solo/btx0333-bin
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "=== BTX 0.33.3 (PR#105 bc633f90) deploy ==="
sudo -n systemctl stop matador-nodewatch matador-tipholder 2>/dev/null
systemctl stop matador-miner 2>/dev/null
sudo -n systemctl stop matador-attest 2>/dev/null
sleep 2

log "backup current 0.33.2 binaries"
docker exec $C sh -c "cp -f $LB/btxd.real $LB/btxd.real.0332bak && cp -f $LB/btx-cli.real $LB/btx-cli.real.0332bak" || { log "BACKUP FAILED - abort"; exit 1; }

log "swap in 0.33.3"
docker cp $NEW/btxd $C:$LB/btxd.real
docker cp $NEW/btx-cli $C:$LB/btx-cli.real
docker exec $C sh -c "chmod +x $LB/btxd.real $LB/btx-cli.real"

BPID=$(docker exec $C sh -c 'cat /data/mining-ops/btxd-supervised.pid 2>/dev/null')
log "SIGTERM old btxd pid=$BPID (clean flush); supervisor respawns with 0.33.3"
[ -n "$BPID" ] && docker exec $C sh -c "kill -TERM $BPID"
for i in $(seq 1 40); do sleep 5; docker exec $C sh -c "kill -0 $BPID 2>/dev/null" || { log "old btxd exited after $((i*5))s"; break; }; done

log "waiting for RPC (watching for reindex)..."
tip=""; reindex=0
for i in $(seq 1 48); do
  tip=$(docker exec $C timeout 10 btx-cli -datadir=/data getblockcount 2>/dev/null)
  [ -n "$tip" ] && { log "RPC up after $((i*10))s: tip=$tip"; break; }
  if docker exec $C sh -c "tail -5 /data/debug.log 2>/dev/null" | grep -qiE "reindex|rewinding|Reindexing|Verifying blocks"; then
    [ "$reindex" = 0 ] && log "NOTE: reindex/verify in progress (0.33.3 revalidating; may take a while, NOT a failure)"; reindex=1
  fi
  sleep 10
done

if [ -z "$tip" ] && [ "$reindex" = 0 ]; then
  log "!! btxd did NOT come back and no reindex -> ROLLING BACK to 0.33.2"
  docker exec $C sh -c "cp -f $LB/btxd.real.0332bak $LB/btxd.real && cp -f $LB/btx-cli.real.0332bak $LB/btx-cli.real"
  BPID=$(docker exec $C sh -c 'cat /data/mining-ops/btxd-supervised.pid 2>/dev/null'); [ -n "$BPID" ] && docker exec $C sh -c "kill -TERM $BPID"
  sleep 30
  log "rolled back to 0.33.2"
else
  VER=$(docker exec $C sh -c "/data/btx-bin/bin/btxd -version 2>/dev/null | head -1")
  OPT=$(docker exec $C sh -c "/data/btx-bin/bin/btxd -help-debug 2>/dev/null | grep -c matmulrcenforcementheight")
  log "0.33.3 code LIVE (has-0333-opt=$OPT, ver-string='$VER'); tip=${tip:-reindexing}"
fi

log "restart services (new cookie)"
sudo -n systemctl start matador-attest matador-nodewatch matador-tipholder 2>/dev/null
systemctl start matador-miner 2>/dev/null
sleep 15
log "DEPLOY-DONE miner=$(systemctl is-active matador-miner) tip=$(docker exec $C timeout 10 btx-cli -datadir=/data getblockcount 2>/dev/null)"
