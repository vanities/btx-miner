# Overclocking the 5090 rig (matador-miner)

Operational guide + live findings for GPU tuning on `pc` (host `omarchy`, RTX 5090).
All knobs here are **software** (NVML) levers applied by the miner itself. Frame gains as
**nonce/s** and **nonce-per-joule** — the card is hard-capped at 600 W, so this is an
efficiency game, not a "add more watts" game.

Last measured: 2026-06-25.

---

## TL;DR

- **Current OC:** `clk_offset = +250 MHz`, `power_limit = 600 W` (max), **memory stock**.
  Set in `~/.config/matador-miner/config.json`, applied via NVML at every miner start
  (runs as root for this). Look for `[gpu-tune] GPC clock offset = 250 MHz` in the journal.
- **The card is POWER-bound, not clock-bound.** The only active throttle is **SW Power Cap**
  (600 W). HW/SW *Thermal* Slowdown are **Not Active**, counters `0 µs`. It is not throttling
  on heat — it is throttling on the 600 W ceiling.
- **"nonce/s goes down over time" is a thermal-margin boost-bin step-down, not a regression.**
  Observed live 2026-06-25 (same process, no restart/update): flat ~39.3k from 08:51-12:55,
  then a clean **step down to ~37.1k at 13:00** that held flat through 14:10 — a ~6% drop
  while core temp (84 C), power (600 W) and scan MN/s were all unchanged. Cause: the GPU's
  thermal margin (`GPU T.Limit Temp`) shrank to **6 C** as afternoon ambient rose, so the
  boost algorithm dropped a clock bin to protect the limit. The **fan was only at 60%** (not
  maxed) - the card chose to drop clocks rather than spin up. So it is environmental
  (ambient/thermal-margin at the 600 W cap), the same family as `34k <-> 33k is environmental`,
  one offset-band higher. Not a code change.
- **"Are we pushing the clock too hard?" — at +300, yes.** Live A/B from the journal:

  | clk_offset | avg nonce/s | max    | reject ratio | effective (accepted) | verdict |
  |-----------:|------------:|-------:|-------------:|---------------------:|---------|
  | **+200**   | 38,011      | 39,152 | **0.000%**   | 38,011               | safe, ~2% slow (short/noisy sample) |
  | **+250**   | 38,884      | 41,249 | **0.195%**   | **38,808**           | **sweet spot (current)** |
  | **+300**   | 39,206      | 41,079 | **1.074%**   | 38,786               | **too hard** — rejects eat the gain |

  +300 is fastest on *raw* nonce/s but its reject rate is **5.5x** higher: ~1% of shares are
  marginally-wrong digests the pool throws out. After subtracting rejects, **+250 ≈ +300 on
  accepted throughput** but +250 wastes far fewer shares. +250 is correct. (Note the temp
  confound actually *strengthens* this: +250's average spans cool overnight hours, +300's is
  a warm afternoon — yet +250 still matches it on effective output.)

---

## How the miner applies OC (mechanism)

NVML loaded at runtime via `dlopen("libnvidia-ml.so.1")` — the release binary keeps **zero
link-time NVML dependency**; the library is only touched when a knob is set. These are the
exact calls LACT / lolMiner make; no third-party code. **Requires root** (NVML control is
privileged). **Never fatal** — any failure logs `[gpu-tune] ... FAILED` and mining continues.

Source: `clean-stack/miner/matador-miner.cpp` -> `ApplyGpuTuning()` (~L4312).

| Knob | NVML call | config key / CLI / env | Notes |
|------|-----------|------------------------|-------|
| **clk_offset** (GPC V/F offset, MHz) | `nvmlDeviceSetGpcClkVfOffset` | `clk_offset` / `--clk-offset` / `BTX_GPU_CLK_OFFSET` | Shifts the whole V/F curve up: same watts buy more clock. Clamped to the driver's allowed range (logs `(clamped, range lo..hi)` if hit; +250 is **not** clamped). `0` = stock. |
| **power_limit** (W) | `nvmlDeviceSetPowerManagementLimit` | `power_limit` / `--power-limit` / `BTX_GPU_POWER_LIMIT` | Applied **first** so the offset has the full budget. Clamped to `[min,max]` (575..600 on this card). `0` = leave as-is. |
| **lock_gpu_clock** (pin core MHz) | `nvmlDeviceSetGpuLockedClocks` | `lock_gpu_clock` / `--lock-gpu-clock` / `BTX_GPU_LOCK_CLOCK` | Pins min=max core clock for consistency / to cap boost-hunting. **Currently unused (0).** A lever for *variance*, not *level*. |

**No memory-clock offset knob is wired** — only the GPC (core) offset. A memory OC would need
LACT or a new `nvmlDeviceSetMemClkVfOffset` path in the miner. See Open Levers.

---

## Where it's configured / change history (answers "where did we change it")

- **File:** `~/.config/matador-miner/config.json` on `pc`. Re-read and re-applied on every
  (re)start of the `matador-miner.service` **system** unit (`/etc/systemd/system/matador-miner.service`,
  runs as root). The `--user` unit is dead/disabled — ignore it.
- **History is preserved** as `~/.config/matador-miner/config.json.bak-*` snapshots, and as
  `[gpu-tune] GPC clock offset = N MHz` lines in `journalctl -u matador-miner`.

Timeline (pc local time):

| When | clk_offset | evidence |
|------|-----------|----------|
| Jun 24 16:55 | stock (none) | `config.json.bak-preclk` |
| Jun 24 16:58 → 21:35 | **+300** | `config.json.bak-clk300` (21:28); churny start incl. one +4 s restart, then ~4 h stable |
| Jun 24 21:35 → Jun 25 08:26 | **+250** | hourly restarts = auto-update, not crashes |
| Jun 25 08:26 → 08:50 | **+200** (brief test) | `config.json.bak-clk250` (08:26) |
| Jun 25 08:50 → now | **+250** (current) | `config.json` (08:50) |

The ~hourly restarts in every era are **auto-updates** (`auto_update: true`,
`update_channel: prerelease`, checks every 300 s) — confirmed: every systemd stop logs
`Deactivated successfully`, and the crash grep (`Failed`/`signal`/`Xid`/`NVRM`/`CUDA error`/
`core-dump`) is **empty**. So +300 did **not** crash; it failed the only test that matters —
**reject rate**.

---

## Why nonce/s drifts down over time (the core question)

It is **not** the SW Power Cap throttle and **not** thermal *slowdown* (those counters are
`0 µs`). It is the **boost algorithm protecting its thermal margin** at the 600 W cap:

```
afternoon ambient ↑  ⇒  GPU T.Limit margin ↓  ⇒  boost drops a clock bin  ⇒  nonce/s steps down
```

Live trace 2026-06-25 (pid 16190, no restart, power/temp flat):

```
12:00–12:55   ~39,000–39,800   flat, high
13:00          37,612          ← step down ~6%
13:00–14:10   ~36,700–37,400   flat, low
```

- It is a **step, not a smooth sag** — it drops a boost bin and holds. Core temp (84 C),
  power (600 W) and scan MN/s were unchanged across the step, so it is not load, not power
  cap, not a code change.
- At the time of the step: `GPU T.Limit Temp = 6 C` (only 6 C of margin to the throttle
  point) and **fan at 60%** (not maxed) — the card chooses to shed clock rather than spin up.
- Boost also hunts ~150 MHz sample-to-sample (saw 2602 ↔ 2752 MHz), so add ±1-2k jitter on
  top of the step.
- **Levers for the step-down** (distinct from the offset *level*): (a) a **lower clk_offset**
  runs cooler internally → bigger margin → may not trip the downshift (the experiment below,
  and your hypothesis); (b) a more aggressive fan curve holds the margin in software; (c)
  `lock_gpu_clock` pins a flat clock so it can't step. The offset sets *which band* you sit
  in; the thermal margin decides *whether you keep the top of that band* through the day.

So the "39 -> 37" you see is the boost dropping a bin when the afternoon eats the thermal
margin — environmental, expected, but **possibly avoidable with a cooler-running offset.**

---

## How to measure (telemetry)

1. **Logger** (running on pc): `~/gpu-oc-telemetry.sh` -> appends `~/gpu-oc-telemetry.log`
   every 30 s. Read-only/additive, safe to re-run (it no-ops if already running).
   CSV columns:
   `ts_iso,temp_c,sm_mhz,mem_mhz,power_w,util_pct,pcap_active,nonce_s,scan_mns,cum_avg,roll10_avg,n`
   — `roll10_avg` is the 10-sample (5-min) rolling mean; `pcap_active` confirms the power cap
   is the binding constraint. Correlate `temp_c` vs `nonce_s` to see the drift directly.
2. **Journal:** `journalctl -u matador-miner | grep '\[stats\]'` -> `nonce/s`, `scan …MN/s`,
   `acc/rej/stale`, `batch`, `async`.
3. **Miner HTTP API:** `127.0.0.1:4060` (config `api.enabled: true`).
4. **Per-offset replay** (attribute nonce/s + rejects to the offset that was live):

   ```bash
   journalctl -u matador-miner -o short-unix --no-pager | awk '
     /GPC clock offset =/ { for(i=1;i<=NF;i++) if($i=="offset"){cur=$(i+2)} next }
     /\[share\] ACCEPTED/ { acc[cur]++ }  /\[share\] REJECTED/ { rej[cur]++ }
     match($0,/nonce\/s=[0-9]+/){ s=substr($0,RSTART,RLENGTH);sub(/nonce\/s=/,"",s);
       sum[cur]+=s; n[cur]++ }
     END{ for(k in sum) printf "off=%s avg=%d rej=%.3f%%\n",
       k, sum[k]/n[k], (rej[k]+0)*100/(acc[k]+rej[k]+0) }'
   ```

---

## A/B methodology (how to test an offset safely)

1. **Snapshot config:** `cp config.json config.json.bak-<tag>`.
2. Edit `clk_offset`, then `sudo systemctl restart matador-miner` (config-only, ~10 s
   downtime — ABM-cheap; no rebuild). Confirm the new `[gpu-tune]` line.
3. **Let it reach steady-state temp (~10-15 min at ~83 C) before reading.** Cold numbers
   lie — a fresh OC always looks faster for the first few minutes.
4. Compare in the **same thermal state**: `roll10_avg` nonce/s **and** reject ratio vs
   baseline. **Optimize `effective = avg × (1 − rej_ratio)`, never raw nonce/s.**
5. **Gate on rejects.** Per the project's #1 rule, keep live `rej` near 0. Treat
   `rej_ratio > ~0.3%` as "too hard" and back off — that is the OC corrupting the math, the
   live-hardware analogue of failing the byte-exact digest gate.

OC is a runtime/hardware knob, so the byte-exact `*_probe` tools and `ab_env.sh` perf harness
(`reference: A/B toolchain on pc`) don't apply directly — the journal reject ratio is the
correctness gate here.

---

## Staged experiment: does a lower offset hold its clock through the afternoon? (ready to run)

Hypothesis (2026-06-25): +250 trips the 13:00 boost-bin step-down because its aggressive V/F
runs the silicon hot enough to eat the thermal margin. A **lower offset runs cooler -> keeps
more margin -> may hold a flat clock all day**, netting more accepted nonce/s over a full
cycle despite a lower cold-start peak. Counter-point: the step-down is leakage/ambient driven
and may happen at any offset — only a multi-hour run settles it. We also lack clean +200/+225
data (the only +200 window was 45 restart-churned samples).

Staged on `pc` (live `config.json` left at +250 — **nothing changed until you apply**):
- `~/.config/matador-miner/config.json.clk225`  (+225)
- `~/.config/matador-miner/config.json.clk200`  (+200)
- `~/.config/matador-miner/config.json.bak-clk250-baseline`  (revert target)

```bash
# apply +225 for a clean multi-hour run (config is user-owned; only the restart needs sudo)
cp ~/.config/matador-miner/config.json.clk225 ~/.config/matador-miner/config.json
sudo systemctl restart matador-miner
# ...let it run >=2 h across the afternoon; the logger captures temp+nonce/s...
# revert
cp ~/.config/matador-miner/config.json.bak-clk250-baseline ~/.config/matador-miner/config.json
sudo systemctl restart matador-miner
```

Read the result from `~/gpu-oc-telemetry.log`: compare the **roll10_avg trajectory** (does it
step down at ~13:00 like +250 did?) and the **reject ratio** (should be ~0). Win = higher
*effective* accepted nonce/s over the whole window, or a flatter line at ~0 rejects.

## Open levers

- **Memory OC (WIRED + TESTED 2026-06-25, shipped v0.7.11): real but small.** Added
  `nvmlDeviceSetMemClkVfOffset` to the miner as `mem_clk_offset` / `--mem-clk-offset` /
  `BTX_GPU_MEM_CLK_OFFSET` (mirrors clk_offset: clamp via `GetMemClkMinMaxVfOffset` +
  revert-on-shutdown). Live A/B on pc (clk +250/fan 80/600W; offset:effective is ~2:1 MHz):
  mem0 baseline ~38.7-38.9k, +500(14051)=39.0k, **+1500(14551)=39.1k = peak ~+0.7-0.9%, rej 0**,
  +3000(15301)=38.6k = rolled back to baseline. The +3000 rollover (non-overlapping below +1500,
  and the latest arm yet lowest -> not drift) is the clean proof mem-clock moves throughput.
  Mechanism: GDDR7 **on-die ECC** throttles effective bandwidth past the knee WITHOUT bad shares
  (rej 0 even at +3000), so mem-OC is asymmetric-safe (overshoot self-limits to ~baseline, does
  not corrupt). Small because the digest is mostly genbase-compute-bound (74%); only the ~26%
  matmul portion is BW-sensitive. **Deployed at a conservative +1000** (14301, on the up-slope,
  below the knee). Config-tunable; +1500 is the measured peak if you want the last ~0.2%.
- **`lock_gpu_clock`.** If *variance* (not level) is the complaint, pin the core to trade the
  cool-start peak for a flat line. Already wired; just set the config key and A/B.
- **Re-sweep the edge.** +250 is good; try +260 / +275, gating strictly on `rej < 0.3%` and
  measuring effective throughput. +300 is the known-too-hard ceiling.
- Keep reporting everything as nonce/s **and** nonce-per-joule (600 W fixed).

---

## Quick reference (current state, 2026-06-25)

```
GPU         : RTX 5090, driver 595.71.05
power       : 600 W limit (575 default / 600 max), drawing ~600 W
clocks      : SM ~2602 / 3090 max MHz,  mem 13801 / 14001 MHz
temp / util : ~84 C, 100%, pstate P1, fan 60% (headroom unused)
margin      : GPU T.Limit = 6 C to throttle  (afternoon shrinks this -> boost steps down)
throttle    : SW Power Cap = Active  (thermal slowdown = 0 µs)
OC          : clk_offset +250 MHz, power_limit 600 W, mem stock, lock off
nonce/s     : 39.3k morning -> 37.1k afternoon (boost-bin step at ~13:00), scan ~315 MN/s
shares      : rej ~0.2% at +250 (healthy); +300 was 1.07% (too hard)
```
