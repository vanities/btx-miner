#!/usr/bin/env bash
# matador-doctor.sh - one-shot setup dump for triaging "it won't mine" reports.
#
# Tell anyone with a broken rig:  "run this and paste the whole output."
# It collects everything we'd otherwise have to ask for (OS, driver, GPU,
# CUDA compatibility, miner version, resolved backend, config, recent logs,
# pool reachability) and REDACTS payout addresses + pool passwords so the
# output is safe to paste into a public issue.
#
#   bash scripts/matador-doctor.sh                 # local rig
#   curl -fsSL <raw-url>/scripts/matador-doctor.sh | bash
#
# Env:
#   MATADOR_API_URL   summary endpoint (default http://127.0.0.1:4060/summary)
#   MATADOR_BIN       path to the binary (default: auto-detect on PATH/.local/usr)
#
# Best-effort by design: every probe is guarded, a missing tool is noted not fatal.
set -uo pipefail

API_URL="${MATADOR_API_URL:-http://127.0.0.1:4060/summary}"
have() { command -v "$1" >/dev/null 2>&1; }
sec()  { printf '\n========== %s ==========\n' "$*"; }
kv()   { printf '  %-22s %s\n' "$1" "$2"; }

# Redact secrets so the dump is safe to paste publicly:
#  - bech32 payout addresses (btx1...) -> btx1...<redacted>
#  - stratum/RPC passwords on CLI or in JSON/conf
redact() {
  sed -E \
    -e 's/(btx1[0-9a-z]{6})[0-9a-z]+/\1...<redacted>/g' \
    -e 's/((-p|--password|--rpcpassword)[ =])[^ ]+/\1<redacted>/g' \
    -e 's/("?(password|rpcpassword|pass)"?[[:space:]]*[:=][[:space:]]*"?)[^",}[:space:]]+/\1<redacted>/g'
}

ts="(time unavailable)"; have date && ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null)"

# ---- resolve the binary -----------------------------------------------------
BIN="${MATADOR_BIN:-}"
if [ -z "$BIN" ]; then
  for c in matador-miner "$HOME/.local/bin/matador-miner" /usr/local/bin/matador-miner; do
    if have "$c" || [ -x "$c" ]; then BIN="$c"; break; fi
  done
fi

# ---- pull a few core facts up front (drives the verdict) ---------------------
summary_json=""
if have curl;   then summary_json="$(curl -fsS --max-time 3 "$API_URL" 2>/dev/null)"; fi
if [ -z "$summary_json" ] && have python3; then
  summary_json="$(python3 - "$API_URL" <<'PY' 2>/dev/null
import sys,urllib.request
try:
    print(urllib.request.urlopen(sys.argv[1],timeout=3).read().decode())
except Exception:
    pass
PY
)"
fi

json_get() { # crude scalar extractor: json_get <key>
  printf '%s' "$summary_json" | grep -oE "\"$1\"[[:space:]]*:[[:space:]]*\"?[^\",}]+" | head -1 \
    | sed -E "s/\"$1\"[[:space:]]*:[[:space:]]*\"?//"
}

version="$(json_get version)"
[ -z "$version" ] && version="$(json_get wrapper_version)"
running="no"; [ -n "$summary_json" ] && running="yes"

# Fall back to the startup banner in the service journal for version when not running.
if [ -z "$version" ] && have journalctl; then
  version="$(journalctl --user -u matador-miner -n 400 --no-pager 2>/dev/null | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' | tail -1)"
  [ -z "$version" ] && version="$(journalctl -u matador-miner -n 400 --no-pager 2>/dev/null | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' | tail -1)"
fi
[ -z "$version" ] && version="unknown"

# Which CUDA toolkit / driver floor does THIS version need?
ver_ge() { [ "$(printf '%s\n%s\n' "${1#v}" "${2#v}" | sort -V 2>/dev/null | head -1)" = "${2#v}" ]; }
req_cuda="13.0"; req_driver="580"; req_note=""
if [ "$version" = "unknown" ]; then
  req_note=" (version unknown; assuming current = CUDA 13)"
elif ver_ge "$version" "0.5.0"; then
  req_cuda="13.0"; req_driver="580"     # v0.5.0+ main build links CUDA 13.x
else
  req_cuda="12.0"; req_driver="525"     # v0.4.x and earlier link CUDA 12.x
fi

# ---- NVIDIA driver / GPU facts ----------------------------------------------
nv_driver=""; nv_cuda=""; nv_present="no"
if have nvidia-smi; then
  nv_present="yes"
  nv_driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  nv_cuda="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
fi

# ---- THE VERDICT (the punchline, printed early) -----------------------------
printf 'matador-doctor   %s\n' "$ts"
sec "VERDICT"
verdict="inconclusive"
if [ "$nv_present" = yes ] && [ -n "$nv_cuda" ]; then
  if awk -v d="$nv_cuda" -v r="$req_cuda" 'BEGIN{exit !(d+0 < r+0)}'; then
    verdict="FAIL: driver too old for this build"
    printf '  X  INCOMPATIBLE DRIVER\n'
    printf '     driver %s exposes CUDA %s max; matador %s needs CUDA %s (driver >= %s).%s\n' \
           "${nv_driver:-?}" "$nv_cuda" "$version" "$req_cuda" "$req_driver" "$req_note"
    printf '     The CUDA runtime cannot initialize -> GPU stays at 0%%, ep/s=0 (often SILENT).\n'
    printf '     FIX (pick one):\n'
    printf '       1) Run on a host/driver >= %s (CUDA >= %s). On Vast, filter offers by CUDA version.\n' "$req_driver" "$req_cuda"
    printf '       2) Install the CUDA-12 build that runs on your driver:\n'
    printf '          VERSION=v0.4.6 curl -fsSL https://raw.githubusercontent.com/vanities/matador-miner/main/install.sh | bash\n'
  else
    verdict="OK: driver satisfies this build's CUDA requirement"
    printf '  OK driver %s exposes CUDA %s >= required CUDA %s (build %s). Driver is NOT the problem.\n' \
           "${nv_driver:-?}" "$nv_cuda" "$req_cuda" "$version"
    if [ "$running" = yes ]; then
      printf '     Miner is up; check the ep/s + backend sections below for a non-driver cause.\n'
    fi
  fi
elif [ "$nv_present" = no ] && [ "$(uname -s)" = Linux ]; then
  verdict="FAIL: nvidia-smi not found"
  printf '  X  nvidia-smi NOT FOUND. No NVIDIA driver visible (or not exposed to this container).\n'
  printf '     If on Vast/Docker, the instance must pass the GPU through (nvidia runtime).\n'
else
  printf '  ?  Non-NVIDIA host or driver not detectable; see sections below.\n'
fi

# ---- platform / OS ----------------------------------------------------------
sec "PLATFORM"
kv "uname" "$(uname -a 2>/dev/null)"
if [ -r /etc/os-release ]; then kv "distro" "$(. /etc/os-release 2>/dev/null; printf '%s' "${PRETTY_NAME:-?}")"; fi
if have getconf; then kv "glibc" "$(getconf GNU_LIBC_VERSION 2>/dev/null)"; fi
if [ "$(uname -s)" = Darwin ] && have sw_vers; then kv "macOS" "$(sw_vers -productVersion 2>/dev/null)"; fi
if have nproc; then kv "cpus" "$(nproc 2>/dev/null)"; fi
if [ -r /proc/cpuinfo ]; then kv "cpu" "$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ //')"; fi
if [ -r /proc/meminfo ]; then kv "mem_total" "$(grep -m1 MemTotal /proc/meminfo 2>/dev/null | awk '{print $2/1024/1024 " GiB"}')"; fi

# ---- miner binary -----------------------------------------------------------
sec "MATADOR BINARY"
kv "version" "$version"
kv "running (API up)" "$running"
kv "needs" "CUDA $req_cuda+ / driver $req_driver+"
if [ -n "$BIN" ]; then
  kv "path" "$BIN"
  have file && kv "file" "$(file -b "$BIN" 2>/dev/null | cut -c1-100)"
  # Running --help is a cheap smoke test: it catches glibc mismatches
  # ("requires glibc 2.XX") and a non-executable / wrong-arch binary.
  help_err="$("$BIN" --help 2>&1 >/dev/null)"; help_rc=$?
  kv "--help exit" "$help_rc"
  [ -n "$help_err" ] && printf '  --help stderr:\n%s\n' "$(printf '%s' "$help_err" | sed 's/^/    /' | head -8)"
else
  kv "path" "NOT FOUND on PATH / ~/.local/bin / /usr/local/bin"
fi

# ---- GPU detail -------------------------------------------------------------
sec "GPU"
if [ "$nv_present" = yes ]; then
  kv "nvidia driver" "${nv_driver:-?}"
  kv "driver CUDA max" "${nv_cuda:-?}"
  kv "gpu count" "$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')"
  printf '  GPUs (index, name, compute_cap, mem, util, power):\n'
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader 2>/dev/null | sed 's/^/    /' || printf '    (query failed)\n'
  printf '  Compute processes (is anything actually on the GPU?):\n'
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
    | sed 's/^/    /' | head -10
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q . \
    || printf '    (none -- GPU is idle; the miner is NOT running a CUDA kernel)\n'
elif have rocm-smi; then
  kv "amd rocm" "present"
  rocm-smi --showproductname --showdriverversion 2>/dev/null | sed 's/^/    /' | head -12
elif [ "$(uname -s)" = Darwin ] && have system_profiler; then
  system_profiler SPDisplaysDataType 2>/dev/null | grep -E 'Chipset|Vendor|Metal|Cores' | sed 's/^/    /' | head -8
else
  printf '  no NVIDIA/AMD/Apple GPU tooling found\n'
fi

# ---- live miner state -------------------------------------------------------
sec "MINER STATE (/summary)"
if [ "$running" = yes ]; then
  if have python3; then
    python3 - "$summary_json" <<'PY' 2>/dev/null | redact
import json,sys
try:
    d=json.loads(sys.argv[1])
except Exception as e:
    print("(unparseable summary)"); raise SystemExit
def g(*p):
    x=d
    for k in p:
        x=(x or {}).get(k) if isinstance(x,dict) else None
    return x
print(f"  status={d.get('status')} mode={d.get('mode')} backend={d.get('backend')} uptime={d.get('uptime_sec')}s")
print(f"  episodes={g('nonces','rc_episodes')} windows={g('nonces','solve_windows')} rc_active={g('nonces','rc_active')}")
print(f"  shares: accepted={g('shares','accepted')} rejected={g('shares','rejected')} stale={g('shares','stale')}")
for i,gpu in enumerate(d.get('gpu_runtime') or []):
    print(f"  gpu[{i}] util={gpu.get('util_pct')}% power={gpu.get('power_w')}W temp={gpu.get('temp_c')}C")
for w in (g('thermal','warnings') or []): print(f"  warn: {w}")
PY
  else
    printf '%s\n' "$summary_json" | redact | head -40
  fi
else
  kv "API" "unreachable at $API_URL (miner not running, or API disabled / different port)"
fi

# ---- config (redacted) ------------------------------------------------------
sec "CONFIG (redacted)"
cfg_found="no"
for p in "${MATADOR_CONFIG:-}" ./matador.json "$HOME/.config/matador/matador.json" /etc/matador/matador.json; do
  [ -n "$p" ] && [ -r "$p" ] || continue
  cfg_found="yes"; printf '  file: %s\n' "$p"
  redact < "$p" | sed 's/^/    /' | head -60
  break
done
[ "$cfg_found" = no ] && printf '  no matador.json found (config may be all CLI flags -- see service unit below)\n'

# systemd ExecStart often holds the real (flag-based) config.
if have systemctl; then
  for scope in "--user" ""; do
    es="$(systemctl $scope cat matador-miner 2>/dev/null | grep -E '^ExecStart=' | head -1)"
    [ -n "$es" ] && { printf '  service ExecStart (%s):\n' "${scope:-system}"; printf '%s\n' "$es" | redact | sed 's/^/    /'; break; }
  done
fi

# ---- recent logs (redacted) -------------------------------------------------
sec "RECENT LOG LINES (redacted)"
log_dumped="no"
if have journalctl; then
  for scope in "--user" ""; do
    out="$(journalctl $scope -u matador-miner -n 60 --no-pager 2>/dev/null)"
    [ -n "$out" ] && {
      printf '  journalctl %s -u matador-miner (key lines):\n' "${scope:-system}"
      printf '%s\n' "$out" | grep -iE '\[backend\]|\[solver\]|\[solve\]|cuda|error|warn|ep/s|rc-active|insufficient|driver' \
        | tail -25 | redact | sed 's/^/    /'
      log_dumped="yes"; break
    }
  done
fi
[ "$log_dumped" = no ] && printf '  no service journal found (running in foreground/Docker? capture stdout/stderr instead)\n'

# ---- pool reachability ------------------------------------------------------
sec "POOL / NODE REACHABILITY"
endpoint="$(printf '%s' "$summary_json" | grep -oE '"host"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 | sed -E 's/.*"host"[^"]*"([^"]+)"/\1/')"
port="$(printf '%s' "$summary_json" | grep -oE '"port"[[:space:]]*:[[:space:]]*[0-9]+' | head -1 | grep -oE '[0-9]+$')"
if [ -n "$endpoint" ] && [ -n "$port" ]; then
  if timeout 5 bash -c "exec 3<>/dev/tcp/$endpoint/$port" 2>/dev/null; then
    kv "tcp $endpoint:$port" "reachable"
  else
    kv "tcp $endpoint:$port" "UNREACHABLE (firewall / wrong host / pool down)"
  fi
else
  kv "endpoint" "not found in /summary (miner down or solo) -- skipping"
fi

sec "END"
printf 'Paste everything from "matador-doctor" to here. Verdict: %s\n' "$verdict"
