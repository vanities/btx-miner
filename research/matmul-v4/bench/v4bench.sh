#!/usr/bin/env bash
# v4 cross-hardware benchmark orchestrator.
#
# Iterate loop: bring a Vast datacenter GPU up ONCE, then re-run build+bench on
# both the local 5090 (host `pc`) and the Vast GPU in PARALLEL for every v4
# change, and print a per-card / per-$hr / per-joule comparison.
#
#   ./v4bench.sh up H100_SXM     # provision a Vast H100, keep it (saves state)
#   ./v4bench.sh run             # build+bench current c13_bench.cu on 5090 + Vast, compare
#   ./v4bench.sh run --pc-only   # just the 5090 (no Vast)
#   ./v4bench.sh run --vast-only # just the kept Vast GPU
#   ./v4bench.sh report          # re-print the comparison from results/scorecard.csv
#   ./v4bench.sh down            # destroy the kept Vast instance
#   ./v4bench.sh status          # show kept instance + all Vast instances
#
# Safety: only ever destroys the instance ID this script created (results/vast.env).
# The Vast account is SHARED -- never `vastai destroy` anything else.
#
# Env overrides:
#   SWEEP="n:nonces:xof[:c13] ..."  default compares wide tiled-vs-c13 at n=4096,8192
#   V5090_DPH=0.30             5090 rental-equivalent $/hr for per-dollar math (override to your cost)
#   PC=pc                      ssh host for the 5090 rig
#   VAST_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/c13_bench.cu"   # private: supports --c13 (v4_proto.cu is the public harness)
RES="$HERE/results"; mkdir -p "$RES"
SCORE="$RES/scorecard.csv"
VENV="$RES/vast.env"
SWEEP="${SWEEP:-4096:32:wide 4096:32:wide:c13 8192:24:wide 8192:24:wide:c13}"  # n:nonces:xof[:c13]
V5090_DPH="${V5090_DPH:-0.30}"
PC="${PC:-pc}"
VAST_IMAGE="${VAST_IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}"
SSHOPT="-o ControlMaster=no -o ControlPath=none -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15"
VKEY="${VKEY:--i $HOME/.ssh/id_ed25519}"   # ssh key for Vast (must be registered on the account)

say(){ echo "[$(date +%H:%M:%S)] $*" >&2; }
die(){ echo "ERROR: $*" >&2; exit 1; }

# build the ./v4_bench invocation list from SWEEP
sweep_cmds(){ local bin="$1"; for s in $SWEEP; do IFS=: read -r n nn xof c13 <<<"$s"; local f=""; [ "$xof" = wide ] && f="$f --wide"; [ "$c13" = c13 ] && f="$f --c13"; echo "$bin $n $nn$f"; done; }

# ---------------- Vast lifecycle ----------------
vast_up(){
  local gpu="${1:-H100_SXM}"
  [ -f "$VENV" ] && die "an instance is already kept ($(grep CID "$VENV")). down it first."
  say "searching $gpu offers..."
  local off; off=$(vastai search offers "gpu_name=$gpu num_gpus=1 verified=true rentable=true direct_port_count>=1 reliability>0.97" -o dph_total --raw 2>/dev/null \
      | python3 -c "import sys,json;d=json.load(sys.stdin);print(d[0]['id'],round(d[0]['dph_total'],3)) if d else print('')")
  [ -z "$off" ] && die "no $gpu offers"
  local oid dph; read -r oid dph <<<"$off"
  say "creating instance from offer $oid (\$$dph/hr, $gpu)..."
  # NOTE the gotcha: create can return success:false yet still spawn. Reconcile
  # against the instance list by label and destroy any extras.
  vastai create instance "$oid" --image "$VAST_IMAGE" --disk 20 --ssh --direct --label "v4bench-$gpu" --raw >/dev/null 2>&1
  sleep 4
  local ids; ids=$(vastai show instances-v1 --raw 2>/dev/null | python3 -c "
import sys,json;d=json.load(sys.stdin);I=d if isinstance(d,list) else d.get('instances',d.get('results',[]))
print(' '.join(str(i['id']) for i in I if str(i.get('label','')).startswith('v4bench-$gpu')))")
  [ -z "$ids" ] && die "create failed (no instance with label v4bench-$gpu)"
  local cid; cid=$(echo "$ids" | tr ' ' '\n' | head -1)
  for extra in $(echo "$ids" | tr ' ' '\n' | tail -n +2); do say "destroying duplicate $extra"; vastai destroy instance "$extra" -y >/dev/null 2>&1; done
  echo "CID=$cid" > "$VENV"; echo "GPU=$gpu" >> "$VENV"; echo "DPH=$dph" >> "$VENV"
  say "instance $cid provisioning; polling until running (~2-5 min, image pull)..."
  local t=0; while [ $t -lt 480 ]; do
    local st; st=$(vastai show instance "$cid" --raw 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin).get('actual_status'))" 2>/dev/null)
    case "$st" in
      running) say "instance $cid RUNNING"; vast_ssh_setup "$cid"; return 0;;
      exited|offline|unknown) die "instance $cid went $st; run: $0 down; and retry";;
    esac
    sleep 15; t=$((t+15)); say "  ...$st ($t s)"
  done
  die "timeout waiting for $cid to run; run: $0 down"
}

vast_ssh_setup(){
  # ssh-url uses the deprecated v0 API; parse the instance JSON for the direct
  # connection (public_ipaddr + the 22/tcp host port).
  local cid="$1" hp
  hp=$(vastai show instance "$cid" --raw 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin); ip=d.get('public_ipaddr'); ports=d.get('ports') or {}
try: p=ports['22/tcp'][0]['HostPort']
except Exception: p=d.get('ssh_port')
print(ip,p)")
  local host port; read -r host port <<<"$hp"
  echo "HOST=$host" >> "$VENV"; echo "PORT=$port" >> "$VENV"
  say "ssh root@$host:$port"
}

vast_down(){
  [ -f "$VENV" ] || { say "no kept instance"; return 0; }
  . "$VENV"; say "destroying kept instance $CID ($GPU)"; vastai destroy instance "$CID" -y >/dev/null 2>&1; rm -f "$VENV"
}

vast_status(){
  [ -f "$VENV" ] && { say "kept:"; cat "$VENV" >&2; } || say "no kept instance"
  say "all your Vast instances:"
  vastai show instances-v1 --raw 2>/dev/null | python3 -c "
import sys,json;d=json.load(sys.stdin);I=d if isinstance(d,list) else d.get('instances',d.get('results',[]))
[print(f\"  id={i.get('id')} {i.get('gpu_name')} {i.get('actual_status')} label={i.get('label')}\") for i in I] or print('  (none)')"
}

# ---------------- runners (emit backend-tagged CSV rows to stdout) ----------------
# sweep commands as a single line: `BIN a b [--wide] 2>/dev/null | grep ^CSV,; ...`
sweep_line(){ sweep_cmds "$1" | sed 's|$| 2>/dev/null \| grep "^CSV,";|' | tr '\n' ' '; }

run_vast(){
  [ -f "$VENV" ] || { say "no kept Vast instance (run: $0 up <GPU>)"; return 1; }
  . "$VENV"; [ -n "${HOST:-}" ] || { say "instance not ready"; return 1; }
  say "vast[$GPU]: copy + build + bench"
  scp $SSHOPT $VKEY -P "$PORT" "$SRC" root@"$HOST":/root/v4_proto.cu >/dev/null 2>&1 || { say "vast scp failed"; return 1; }
  local cmds; cmds="$(sweep_line /root/v4_bench)"
  ssh $SSHOPT $VKEY -p "$PORT" root@"$HOST" "bash -s" <<EOF 2>/dev/null | sed "s|^CSV,|vast,$DPH,|"
cd /root
nvcc -O3 -arch=native v4_proto.cu -lcublasLt -L/usr/local/cuda/lib64/stubs -lnvidia-ml -o v4_bench 2>/dev/null || { echo BUILDFAIL; exit 1; }
$cmds
EOF
}

run_pc(){
  say "pc[5090]: stop miner + build + bench (miner auto-restarts)"
  scp $SSHOPT "$SRC" "$PC":/home/vanities/v4-proto/v4_proto.cu >/dev/null 2>&1 || { say "pc scp failed"; return 1; }
  local cmds; cmds="$(sweep_line ./v4_bench)"
  ssh $SSHOPT "$PC" "bash -s" <<EOF 2>/dev/null | sed "s|^CSV,|5090,$V5090_DPH,|"
trap 'systemctl start matador-miner.service >/dev/null 2>&1' EXIT
systemctl stop matador-miner.service
for i in \$(seq 1 10); do [ -z "\$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null)" ] && break; sleep 1; done
docker run --rm --gpus all -v /home/vanities/v4-proto:/w -w /w matador-build:pathb-deps-cm4 bash -lc 'nvcc -O3 -arch=sm_120 v4_proto.cu -lcublasLt -L/usr/local/cuda/lib64/stubs -lnvidia-ml -o v4_bench 2>/dev/null || exit 1; $cmds'
EOF
}

do_run(){
  local mode="${1:-both}"
  local tmp; tmp=$(mktemp)
  case "$mode" in
    --pc-only)   run_pc   >>"$tmp" ;;
    --vast-only) run_vast >>"$tmp" ;;
    *)  # parallel
        local a b; a=$(mktemp); b=$(mktemp)
        run_pc   >"$a" & local ppc=$!
        run_vast >"$b" & local pv=$!
        wait $ppc; wait $pv; cat "$a" "$b" >>"$tmp"; rm -f "$a" "$b" ;;
  esac
  # append fresh rows to the scorecard
  [ -s "$SCORE" ] || echo "backend,dph,gpu,n,xof,combine,sha_ms,gemm_ms,comb_ms,nonce_s,watts,joules,int8_pct" > "$SCORE"
  grep -E '^(5090|vast),' "$tmp" >> "$SCORE"
  say "collected $(grep -cE '^(5090|vast),' "$tmp") row(s)"
  rm -f "$tmp"
  report
}

report(){
  [ -s "$SCORE" ] || { say "no results yet"; return 0; }
  python3 - "$SCORE" <<'PY'
import sys,csv
rows=list(csv.DictReader(open(sys.argv[1])))
if not rows: print("no rows"); sys.exit()
# latest run per (gpu,n,xof): last occurrence wins
seen={}
for r in rows: seen[(r['gpu'],r['n'],r['xof'],r.get('combine','tiled'))]=r
R=sorted(seen.values(), key=lambda r:(int(r['n']),r['xof'],r.get('combine','tiled'),r['gpu']))
print(f"\n{'gpu':<22}{'n':>6}{'xof':>6}{'combine':>9}{'nonce/s':>9}{'$/hr':>7}{'nonce/$':>10}{'W':>6}{'J/n':>7}")
print("-"*82)
for r in R:
    dph=float(r['dph']); nps=float(r['nonce_s']); npd=nps*3600/dph if dph>0 else 0
    print(f"{r['gpu'][:21]:<22}{r['n']:>6}{r['xof']:>6}{r.get('combine','tiled'):>9}{nps:>9.0f}{dph:>7.2f}{npd:>10.0f}{float(r['watts']):>6.0f}{float(r['joules']):>7.2f}")
print("\nnonce/$ = nonces per rental-dollar (nonce_s*3600/$hr). combine: tiled = ALU mod-q; c13 = 25 s8 tensor GEMMs. Datacenter 'wins' only if it leads per-card.")
PY
}

case "${1:-}" in
  up)     vast_up "${2:-H100_SXM}" ;;
  run)    shift; do_run "${1:-both}" ;;
  report) report ;;
  down)   vast_down ;;
  status) vast_status ;;
  *) sed -n '2,30p' "$0"; exit 1 ;;
esac
