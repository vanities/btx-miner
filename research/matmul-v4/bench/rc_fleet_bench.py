# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
RC fleet bench: run the byte-exact ENC_RC episode solver on a card and report
throughput AND MEASURED BOARD POWER. Committed harness -- do not fork a one-off.

    uv run research/matmul-v4/bench/rc_fleet_bench.py --local
    uv run research/matmul-v4/bench/rc_fleet_bench.py --gpu RTX_3090 --arch 86
    uv run research/matmul-v4/bench/rc_fleet_bench.py --gpu B200 --arch 100
    uv run research/matmul-v4/bench/rc_fleet_bench.py --all

POWER IS ALWAYS SAMPLED. A rate without watts is half a number: it cannot say
whether the card was saturated or whether the bench was measuring launch
overhead. nvidia-smi power.limit (the nameplate cap) is NOT a measurement --
an ep/kWh computed from it is a spec-sheet ratio wearing a lab coat.

Sampling gotchas baked in (each one cost a real mistake):
  * Sample DURING the run in a separate process at 10 Hz. Reading NVML after
    cudaEventSynchronize returns the idle floor because the GPU has already
    clocked down (this once produced a bogus 88 W).
  * Filter to busy samples (>IDLE_W) so pre/post idle tails do not drag the
    mean down.
  * Run N episodes back to back and discard a warm-up: one cold episode
    includes cuBLASLt handle creation plus the M11 upload and is too short to
    sample cleanly.

Vast gotchas baked in:
  * `vastai show instances` now returns HTTP 410 -- use `show instances-v1`.
  * Take the ssh endpoint from the instance JSON (ssh_host/ssh_port).
    Parsing `vastai ssh-url` returns EMPTY and then fails as a fake ssh timeout.
  * Allow 20 min to reach `running`: the CUDA devel image is several GB and the
    pull dominates. Filter offers on inet_down so the pull is not the bench.
  * ALWAYS destroy on exit, including on failure. Account is SHARED: only ever
    destroy the instance this script created.
"""

import argparse
import json
import re
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent.parent / "clean-stack" / "core" / "cuda" / "rc_gpu_episode.cu"
RESULTS = HERE / "results"

IDLE_W = 150.0          # samples below this are idle tail, not work
IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"
BUILD_IMAGE = "matador-build:pathb-deps-cm4"     # pc-local, has CUDA 13.3
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15", "-o", "ControlMaster=no", "-o", "ControlPath=none"]

NOISE = re.compile(r"NVIDIA Deep Learning|NGC-DL|Copyright|CUDA Version|governed by|"
                   r"copy of this license|developer\.nvidia|Container Toolkit|docs\.nvidia|"
                   r"Driver was not|^=+$|^\s*$")

# Runs on the target box. Builds, samples power at 10 Hz throughout, prints a
# __RESULT__ json line.
# NB: @-placeholders, NOT str.format -- this payload embeds a python heredoc full
# of { } braces and .format() chokes on them (KeyError on the first dict key).
PAYLOAD = r"""
set -u
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader | head -1
echo "[build] arch=sm_@ARCH@"
if [ "@MXFP4@" = "1" ]; then
  # native MXFP4 needs the a-variant gencode (compute_120a / compute_100a); sm_86 etc CANNOT
  # build the mma.kind::mxf8f6f4 instruction -- a build failure here IS the exclusion result.
  echo "[mxfp4] gencode compute_@ARCH@a,code=sm_@ARCH@a"
  nvcc -O3 -gencode arch=compute_@ARCH@a,code=sm_@ARCH@a "@MXSRC@" -o /tmp/fp4c 2>/tmp/nvcc_err.txt
  if [ ! -x /tmp/fp4c ]; then
    echo "__MXFP4__ {\"native_mxfp4\": false, \"reason\": \"arch sm_@ARCH@ cannot compile mma.kind::mxf8f6f4\"}"
    head -3 /tmp/nvcc_err.txt
    exit 0
  fi
  /tmp/fp4c 20000 20 4 2>&1 | tee /tmp/fp4.txt | grep -vE "^CSVFP4"
  python3 - /tmp/fp4.txt <<'PYX'
import sys,json
line=[l for l in open(sys.argv[1]) if l.startswith("CSVFP4")]
if line:
    _,name,sms,i8,f8f4,mxf4,best=line[0].strip().split(",")
    print("__MXFP4__ "+json.dumps({"native_mxfp4":True,"name":name,"sms":int(sms),
        "int8_tops":float(i8),"mxf8f6f4_tops":float(f8f4),"mxf4_tops":float(mxf4),
        "best_ratio":float(best)}))
PYX
  exit 0
fi
nvcc -O3 -arch=sm_@ARCH@ --default-stream per-thread "@SRC@" -lcublasLt -o /tmp/rc_solver_bin 2>&1 | grep -E "error" | head -5
echo "[build] exit=${PIPESTATUS[0]}"
[ -x /tmp/rc_solver_bin ] || { echo '__RESULT__ {"error":"build failed"}'; exit 1; }

echo "--- GOLDEN GATE ---"
timeout 900 /tmp/rc_solver_bin 2>&1 | grep -E "digest|PHASES|MISMATCH"

PW=$HOME/.rc_power_samples.csv; rm -f "$PW"
nvidia-smi --query-gpu=power.draw,clocks.sm,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits -lms 100 > "$PW" 2>/dev/null &
SAMP=$!
sleep 1
if [ "@STREAM@" = "1" ]; then
  echo "--- STREAMING PENALTY (coupled bank: derive vs read) ---"
  /tmp/rc_solver_bin 2 8192 20 2>&1 | tee /tmp/rc_stream.txt
  python3 - /tmp/rc_stream.txt <<'PYX'
import re,sys,json
t=open(sys.argv[1]).read()
def g(p):
    m=re.search(p,t); return float(m.group(1)) if m else None
print("__STREAM__ "+json.dumps({
  "derive_ms": g(r"derive one page \(STREAMED path\) :\s*([\d.]+) ms"),
  "read_ms":   g(r"read   one page \(RESIDENT path\) :\s*([\d.]+) ms"),
  "penalty":   g(r"STREAMING PENALTY\s*:\s*([\d.]+)x"),
}))
PYX
  exit 0
fi
echo "--- PRODUCTION x@REPS@ mode=@MODE@ (1=base, 3=datacenter/profile-2; concurrency sweep: @CONC@) ---"
for Q in @CONC@; do
  echo "[conc Q=$Q]"
  timeout 1800 /tmp/rc_solver_bin @MODE@ @REPS@ $Q 2>&1 | tee /tmp/rc_run_$Q.txt | grep -E "GPU episode|!!"
done
# the LAST Q is what the parsed result reports; the per-Q lines above carry the sweep
cp /tmp/rc_run_$Q.txt /tmp/rc_run.txt
sleep 1
kill $SAMP 2>/dev/null || true

python3 - "$PW" /tmp/rc_run.txt @IDLE@ <<'PYEOF'
import json, re, sys, statistics
pw_path, run_path, idle_w = sys.argv[1], sys.argv[2], float(sys.argv[3])
p, sm, ut, tp = [], [], [], []
for line in open(pw_path):
    f = [x.strip() for x in line.split(',')]
    if len(f) < 4: continue
    try: p.append(float(f[0])); sm.append(float(f[1])); ut.append(float(f[2])); tp.append(float(f[3]))
    except ValueError: continue
busy = [(a,b,c,d) for a,b,c,d in zip(p,sm,ut,tp) if a > idle_w]
txt = open(run_path).read()
def grab(pat, cast=float):
    m = re.search(pat, txt)
    return cast(m.group(1)) if m else None
out = {
    "episode_ms": grab(r"GPU episode\s*:\s*([\d.]+) ms"),
    "episodes_s": grab(r"\(([\d.]+) episodes/s\)"),
    "digest":     grab(r"digest ([0-9a-f]+)", str),
    "t_opgen":    grab(r"MX operand gen \(GPU\):\s*([\d.]+) s"),
    "t_gpu_work": grab(r"actual GPU work\s*:\s*([\d.]+) s"),
    "t_merkle":   grab(r"phase3 GPU Merkle\s*:\s*([\d.]+) s"),
    "samples_total": len(p), "samples_busy": len(busy),
}
if busy:
    out |= {
        "power_mean_w":   round(statistics.fmean(x[0] for x in busy), 1),
        "power_median_w": round(statistics.median(x[0] for x in busy), 1),
        "power_max_w":    round(max(x[0] for x in busy), 1),
        "sm_clock_mean":  round(statistics.fmean(x[1] for x in busy)),
        "util_mean_pct":  round(statistics.fmean(x[2] for x in busy), 1),
        "temp_max_c":     round(max(x[3] for x in busy)),
    }
print("__RESULT__ " + json.dumps(out))
PYEOF
"""


def build_payload(arch, reps, src, conc="1", stream="0", mxfp4="0", mxsrc="", mode="3"):
    """@-placeholder substitution: the payload embeds a python heredoc whose braces
    would break str.format()."""
    return (PAYLOAD.replace("@MODE@", str(mode))
                   .replace("@ARCH@", str(arch))
                   .replace("@REPS@", str(reps))
                   .replace("@SRC@", src)
                   .replace("@CONC@", conc)
                   .replace("@STREAM@", stream)
                   .replace("@MXFP4@", mxfp4)
                   .replace("@MXSRC@", mxsrc)
                   .replace("@IDLE@", str(IDLE_W)))


def write_markdown(rows, reps, stamp):
    """Raw per-run JSON is gitignored (noisy); this scorecard IS tracked, matching the
    convention of clean-stack/bench/vast/results-*/. Regenerate from a saved run with
    --from-json so rebuilding a scorecard never becomes a scratch one-liner."""
    path = HERE / "rc-fleet-scorecard.md"
    L = ["# ENC_RC production episode: fleet scorecard", "",
         f"`rc_fleet_bench.py`, {reps} episodes/arm, warm-up excluded. Generated {stamp}.",
         "",
         "Power sampled at 10 Hz **during** the run; `ep/kWh` uses **measured mean draw**, "
         "never the nameplate `power.limit`.", "",
         "| card | arch | ep/s | ms | W mean | W max | util% | MHz | ep/kWh | $/hr | ep/$ | digest |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r or r.get("episodes_s") is None:
            continue
        eps, w = r["episodes_s"], r.get("power_mean_w")
        ekwh = f"{(eps*3600)/(w/1000.0):.0f}" if w else "n/a"
        epd = f"{(eps*3600)/r['dph']:.0f}" if r.get("dph") else "n/a"
        L.append(f"| {r['card']} | {r['arch']} | {eps:.3f} | {r['episode_ms']:.1f} | "
                 f"{w or 'n/a'} | {r.get('power_max_w') or 'n/a'} | {r.get('util_mean_pct') or 'n/a'} | "
                 f"{r.get('sm_clock_mean') or 'n/a'} | {ekwh} | {r.get('dph','n/a')} | {epd} | "
                 f"`{(r.get('digest') or '?')[:16]}` |")
    digs = {r.get("digest") for r in rows if r and r.get("digest")}
    L += ["", f"**Digest agreement across architectures:** "
              f"{'YES `' + digs.pop() + '`' if len(digs)==1 else 'MISMATCH ' + str(digs)}",
          "", "A rate without watts cannot say whether the card was saturated or whether the",
          "bench was measuring launch overhead. Both columns or neither."]
    path.write_text("\n".join(L) + "\n")
    return path


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, **kw)


def vast(*args, raw=True):
    cmd = ["vastai", *args] + (["--raw"] if raw else [])
    r = sh(cmd)
    if raw:
        try:
            return json.loads(r.stdout)
        except Exception:
            return None
    return r.stdout


def parse_mxfp4(text):
    for line in text.splitlines():
        if line.startswith("__MXFP4__"):
            try: return json.loads(line[len("__MXFP4__"):].strip())
            except Exception: return None
    return None


def parse_stream(text):
    for line in text.splitlines():
        if line.startswith("__STREAM__"):
            try: return json.loads(line[len("__STREAM__"):].strip())
            except Exception: return None
    return None


def parse_result(text):
    for line in text.splitlines():
        if line.startswith("__RESULT__"):
            try:
                return json.loads(line[len("__RESULT__"):].strip())
            except Exception:
                return None
    return None


def show(text):
    for line in text.splitlines():
        if not NOISE.search(line) and not line.startswith("__RESULT__"):
            print("   " + line.rstrip())


def run_local(reps, host="pc", price=0.313, conc="1", stream="0", mxfp4="0", mode="3"):
    """pc's 5090. Stops the miner for a single-stack measurement, always restarts it."""
    print(f"[5090] local via {host} (miner stopped for single-stack, restarted on exit)")
    # walker = omarchy's desktop launcher; holds a tiny GPU context, not a workload.
    foreign = sh(["ssh", *SSH_OPTS, host,
                  "nvidia-smi --query-compute-apps=process_name --format=csv,noheader "
                  "| grep -v matador-miner | grep -v /usr/bin/walker | grep -v '^$' || true"]).stdout.strip()
    if foreign:
        print(f"[5090] ABORT, foreign GPU process: {foreign}")
        return None
    # the solver source is mounted into the build image from the pc-side checkout.
    # matador-src-main is the CURRENT main worktree (matador-src-lt was the stale
    # LT-era detached worktree -- the S= default trap from CLAUDE.md).
    payload = build_payload(120, reps, "/src/clean-stack/core/cuda/rc_gpu_episode.cu", conc, stream, mxfp4, "/src/research/matmul-v4/bench/fp4_int8_ceiling.cu", mode)
    script = f"""
trap 'systemctl start matador-miner.service' EXIT
systemctl stop matador-miner.service
sleep 3
docker run --rm --gpus all -v ~/git/matador-src-main:/src -w /src {BUILD_IMAGE} bash -c {shlex.quote(payload)}
"""
    r = sh(["ssh", *SSH_OPTS, host, "bash -s"], input=script, timeout=3600)
    show(r.stdout)
    res = (parse_mxfp4(r.stdout) if mxfp4 == "1"
           else parse_stream(r.stdout) if stream == "1" else parse_result(r.stdout))
    if res:
        res |= {"card": "RTX 5090", "arch": "sm_120", "dph": price}
    return res


def run_vast(gpu, arch, reps, offer_id=None, price=None, conc="1", stream="0", mxfp4="0", mode="3"):
    inst = None
    try:
        if offer_id is None:
            offers = vast("search", "offers",
                          f"gpu_name={gpu} num_gpus=1 verified=true rentable=true "
                          f"direct_port_count>=1 reliability>0.98 inet_down>2000",
                          "-o", "dph_total") or []
            if not offers:
                print(f"[{gpu}] no offers"); return None
            offer_id, price = offers[0]["id"], offers[0]["dph_total"]
        print(f"[{gpu}] offer {offer_id} @ ${price:.3f}/hr, sm_{arch}")

        created = vast("create", "instance", str(offer_id), "--image", IMAGE,
                       "--disk", "30", "--ssh", "--direct")
        inst = (created or {}).get("new_contract")
        if not inst:
            print(f"[{gpu}] CREATE FAILED: {created}"); return None
        print(f"[{gpu}] instance {inst}")

        for i in range(120):                      # 20 min: multi-GB image pull
            st = (vast("show", "instance", str(inst)) or {}).get("actual_status", "")
            if st == "running":
                print(f"[{gpu}] running after {i*10}s"); break
            if st in ("exited", "unknown", "offline"):
                print(f"[{gpu}] TERMINAL STATE {st}"); return None
            if i and i % 12 == 0:
                print(f"[{gpu}] still {st} at {i*10}s")
            time.sleep(10)
        else:
            print(f"[{gpu}] TIMEOUT waiting for running"); return None

        j = vast("show", "instance", str(inst)) or {}
        host, port = j.get("ssh_host") or j.get("public_ipaddr"), j.get("ssh_port")
        if not host or not port:                  # ssh-url parsing returns empty; diagnose loudly
            print(f"[{gpu}] NO SSH ENDPOINT; ssh/port keys: "
                  f"{ {k: v for k, v in j.items() if any(t in k for t in ('ssh','port','ipaddr'))} }")
            return None
        print(f"[{gpu}] ssh {host}:{port}")

        for i in range(42):
            if sh(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", "true"]).returncode == 0:
                print(f"[{gpu}] ssh ready after {i*10}s"); break
            time.sleep(10)
        else:
            print(f"[{gpu}] TIMEOUT ssh never ready"); return None

        if sh(["scp", *SSH_OPTS, "-P", str(port), "-q", str(SRC),
               f"root@{host}:/root/rc_gpu_solver.cu"]).returncode != 0:
            print(f"[{gpu}] SCP FAILED"); return None
        if mxfp4 == "1":
            sh(["scp", *SSH_OPTS, "-P", str(port), "-q", str(HERE / "fp4_int8_ceiling.cu"),
                f"root@{host}:/root/fp4_int8_ceiling.cu"])

        payload = build_payload(arch, reps, "/root/rc_gpu_solver.cu", conc, stream, mxfp4, "/root/fp4_int8_ceiling.cu", mode)
        r = sh(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", "bash -s"],
               input=payload, timeout=3600)
        show(r.stdout)
        res = (parse_mxfp4(r.stdout) if mxfp4 == "1"
               else parse_stream(r.stdout) if stream == "1" else parse_result(r.stdout))
        if res:
            res |= {"card": gpu, "arch": f"sm_{arch}", "dph": price}
        return res
    finally:
        if inst:
            print(f"[{gpu}] destroying {inst}")
            sh(["vastai", "destroy", "instance", str(inst), "-y"])


def scorecard(rows):
    print("\n" + "=" * 108)
    print("ENC_RC PRODUCTION EPISODE -- MEASURED (power sampled at 10 Hz during the run)")
    print("=" * 108)
    hdr = (f"{'card':10} {'arch':8} {'ep/s':>7} {'ms':>8} {'W mean':>7} {'W max':>7} "
           f"{'util%':>6} {'MHz':>6} {'ep/kWh':>8} {'$/hr':>7} {'ep/$':>8}  digest")
    print(hdr); print("-" * 108)
    for r in rows:
        if not r or r.get("episodes_s") is None:
            continue
        eps, w = r["episodes_s"], r.get("power_mean_w")
        ep_kwh = (eps * 3600) / (w / 1000.0) if w else None
        ep_d = (eps * 3600) / r["dph"] if r.get("dph") else None
        print(f"{r['card']:10} {r['arch']:8} {eps:7.3f} {r['episode_ms']:8.1f} "
              f"{(w or 0):7.1f} {(r.get('power_max_w') or 0):7.1f} "
              f"{(r.get('util_mean_pct') or 0):6.1f} {(r.get('sm_clock_mean') or 0):6d} "
              f"{(ep_kwh or 0):8.0f} {(r.get('dph') or 0):7.3f} {(ep_d or 0):8.0f}  "
              f"{(r.get('digest') or '?')[:16]}")
    print("-" * 108)
    print("ep/kWh uses MEASURED mean board power, not the nameplate limit.")
    digs = {r.get("digest") for r in rows if r and r.get("digest")}
    print(f"digest agreement across arches: {'YES ' + digs.pop() if len(digs) == 1 else 'MISMATCH ' + str(digs)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true", help="pc's 5090 via ssh+docker")
    ap.add_argument("--host", default="pc")
    ap.add_argument("--gpu", help="Vast gpu_name, e.g. RTX_3090 / B200")
    ap.add_argument("--arch", type=int, help="sm arch for --gpu (86=Ampere, 100=B200, 120=5090)")
    ap.add_argument("--offer", type=int, help="specific Vast offer id")
    ap.add_argument("--price", type=float, help="$/hr override")
    ap.add_argument("--all", action="store_true", help="5090 local + 3090 + B200")
    ap.add_argument("--reps", type=int, default=30, help="episodes per arm (default 30)")
    ap.add_argument("--from-json", help="rebuild the tracked scorecard from a saved run")
    ap.add_argument("--mxfp4-ceiling", action="store_true",
                    help="native MXFP4 vs INT8 tensor throughput per card (needs compute_Xa "
                         "gencode; sm_86/Ampere CANNOT build it -- that failure IS the exclusion)")
    ap.add_argument("--stream-penalty", action="store_true",
                    help="coupled-bank derive-vs-read per card (5090 streams a >32GB bank, "
                         "B200 holds it resident); the coupled compute is mode-invariant and "
                         "cancels, so this ratio IS the 5090-vs-B200 coupled comparison")
    ap.add_argument("--episode-mode", default="3", choices=["1", "3"],
                    help="solver dims: 3 = DATACENTER profile-2 (mainnet activation shape, "
                         "default), 1 = epoch-0 base")
    ap.add_argument("--conc", default="1",
                    help="concurrent episodes in flight; space-separated sweeps a range, "
                         "e.g. --conc '1 2 4 8'. A saturated card LOSES throughput here "
                         "(5090 measured -13%% at Q=2); a wide, underfed one should gain.")
    a = ap.parse_args()

    if a.from_json:
        d = json.loads(Path(a.from_json).read_text())
        scorecard(d["rows"])
        print(f"\nwrote {write_markdown(d['rows'], d.get('reps','?'), d['utc']).name}")
        return

    if a.mxfp4_ceiling:
        rows = [run_local(1, a.host, a.price or 0.313, mxfp4="1"),
                run_vast("B200", 100, 1, mxfp4="1"),
                run_vast("RTX_3090", 86, 1, mxfp4="1")]   # expected: 3090 fails to build (no native MXFP4)
        rows = [r for r in rows if r]
        print("\n" + "=" * 78)
        print("NATIVE MXFP4 vs INT8 TENSOR THROUGHPUT (the datacenter consumption lever)")
        print("=" * 78)
        print(f"{'card':16} {'native?':>8} {'INT8 TOPS':>10} {'MXFP4 TOPS':>11} {'MXFP4:INT8':>11}")
        for r in rows:
            nm = r.get("native_mxfp4")
            if nm:
                best = max(r.get("mxf8f6f4_tops",0), r.get("mxf4_tops",0))
                print(f"{r.get('name','?')[:16]:16} {'YES':>8} {r['int8_tops']:>10.0f} "
                      f"{best:>11.0f} {r['best_ratio']:>10.2f}x")
            else:
                print(f"{'(sm_86 Ampere)':16} {'NO':>8} {'--':>10} {'--':>11} "
                      f"{'excluded':>11}")
        print("\nAmpere (3090) has NO native MXFP4 path: under a packed-MXFP4 bank it must")
        print("dequant to INT8 per page use -- a tax the Blackwell cards skip.")
        return

    if a.stream_penalty:
        if a.gpu:                       # single card (e.g. --gpu RTX_3090 --arch 86)
            rows = [run_vast(a.gpu, a.arch, 1, a.offer, a.price, stream="1")]
        else:                           # default trio
            rows = [run_local(1, a.host, a.price or 0.313, stream="1"),
                    run_vast("B200", 100, 1, stream="1")]
        rows = [r for r in rows if r]
        print("\n" + "=" * 84)
        print("COUPLED BANK: 5090 STREAMS a >32GB bank vs B200 holds it RESIDENT")
        print("(coupled compute is mode-invariant and cancels; this IS the coupled comparison)")
        print("=" * 84)
        print(f"{'card':10} {'derive/page (streamed)':>24} {'read/page (resident)':>22}")
        for r in rows:
            print(f"{r['card']:10} {r.get('derive_ms',0):>19.3f} ms {r.get('read_ms',0):>17.3f} ms")
        d5 = next((r for r in rows if '5090' in r['card']), None)
        b2 = next((r for r in rows if 'B200' in r['card']), None)
        if d5 and b2:
            # 5090 must stream (pays derive); B200 resident (pays read). 768 page-uses/episode.
            s5, sb = d5['derive_ms'] * 768, b2['read_ms'] * 768
            print(f"\ncoupled bank phase / episode (768 page-uses):")
            print(f"  5090 STREAMED : {s5:8.1f} ms")
            print(f"  B200 RESIDENT : {sb:8.1f} ms")
            print(f"  => B200 does the coupled bank phase {s5/sb:.1f}x faster than a 5090 that must stream")
            print(f"     (per rented dollar, 5090 ${d5['dph']}/hr vs B200 ${b2['dph']}/hr: "
                  f"5090 is {(sb*b2['dph'])/(s5*d5['dph']):.2f}x the cost-efficiency on this phase)")
        return

    rows = []
    if a.all:
        rows.append(run_local(a.reps, a.host, mode=a.episode_mode))
        rows.append(run_vast("RTX_3090", 86, a.reps, mode=a.episode_mode))
        rows.append(run_vast("B200", 100, a.reps, mode=a.episode_mode))
    elif a.local:
        rows.append(run_local(a.reps, a.host, a.price or 0.313, a.conc, mode=a.episode_mode))
    elif a.gpu:
        if a.arch is None:
            sys.exit("--gpu needs --arch")
        rows.append(run_vast(a.gpu, a.arch, a.reps, a.offer, a.price, a.conc, mode=a.episode_mode))
    else:
        sys.exit("pick --local, --gpu, or --all")

    rows = [r for r in rows if r]
    if not rows:
        sys.exit("no results")
    scorecard(rows)

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS / f"rc-fleet-{stamp}.json"
    out.write_text(json.dumps({"utc": stamp, "reps": a.reps, "rows": rows}, indent=2))
    # never overwrite the tracked scorecard with a failed run
    if any(r.get("episodes_s") for r in rows):
        md_path = write_markdown(rows, a.reps, stamp)
        print(f"\nwrote {out.name} (raw, ignored) and {md_path.name} (tracked)")
    else:
        print(f"\nwrote {out.name}; NO valid rows, tracked scorecard left untouched")


if __name__ == "__main__":
    main()
