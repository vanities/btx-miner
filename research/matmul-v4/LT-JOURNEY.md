# v4.4-LT (ENC_BMX4C_LT) — the full engagement journey

One doc for everything we did on numair's datacenter-favoring successor PoW: his
changes, our scaffold, our optimizations, how each benchmark number moved, and the
current verdict. Companion to `RESULTS.md` (raw v4.0-4.2 runs, ends 2026-07-17) and
`README.md` (the original PR #89 investigation). This file picks up the LT chapter
(2026-07-18 onward).

- **Proposal:** `btxchain/btx` PR #89, author `numair`. Profile `ENC_BMX4C_LT` (enum 4).
- **Status:** **inert on all public nets** (`nMatMulDRLTHeight = INT32_MAX`); regtest = 100.
  Public activation **NO-GO**. Native MXFP4 is a fail-closed stub. This is a *future*
  successor, not live.
- **Byte-exact golden (n=64, nonce=0xdeadbeef):**
  `db1136f2974d45d9757262978ab074ef53ba54c368df9829f565ee2d26da0da9`
- **The game:** byte-exactness proves the math; A/B proves the speed. Our kernel
  optimizations stay matador-exclusive — we do not hand them to numair.

---

## TL;DR — the verdict in one screen

numair's stated goal: make the successor PoW favor datacenter cards (B200) over
consumer cards (5090) by ~4× so a datacenter operator out-earns a 5090 fleet. **On
his own harness that goal fails, and the direction depends on pipeline depth.**

| Regime | measured | B200 : 5090 | winner |
|---|---|---|---|
| numair harness, **pre-batching** (unfed) | 5090 3.59 / B200 2.85 | **0.79×** | 5090 |
| numair harness, **resident-batched** @ `d4f5577` (verified pair) | 5090 **4.36** / B200 **2.68** | **0.62×** | 5090 (1.63×) |
| **our deep ring** (perf-model, many Q* windows pipelined) | 5090 381 / B200 840 | **2.2×** | B200 |
| his gate needs | — | **≥ 4×** | — |
| per rental-$ (any depth) | 5090 owned ~$0.4 vs B200 $6.25/hr | **~6–20×** | 5090 |

**Reading:** the raw B200:5090 ratio *flips with how deep the digest is pipelined*.
At the protocol's own window (Q*=256) the workload under-occupies the B200's 208 SMs
and the SHA-ALU-bound solve favors the 5090's clocks → consumer wins 1.63×. Only when
a miner pipelines *many* Q* windows deep does the B200's memory let it hide the 32 MiB
digest latency and pull ~2.2× ahead — still nowhere near 4×. **Per rental-dollar the
5090 wins decisively at every depth. That economics gap is the durable claim.**

Notably, numair's resident-batching fix (added to *help* the datacenter card) lifted
the 5090 +21% and left the B200 flat, **widening** the shallow-depth gap. He
independently reached the same "it's launch/host topology, not silicon" conclusion we
did — from an nsys trace (~169 µs GPU compute inside ~11.2 ms/nonce).

**His own design doc now concedes our thesis.** `doc/btx-matmul-v4.4-exact-accel-lanes.md`
@ `d4f5577` states, in his words: *"B200 ≈ 2× RTX 5090 … not the 4–6× dense INT8 peak
ratio"*; *"FP4 does not tilt datacenter further; consumer Blackwell's FP4/INT8 ratio was
higher than B200's"*; *"cuBLASLt does not serve OCP MXFP4."* All three are our findings,
now in his documentation. His original goal was a 4–6× datacenter advantage; his own doc
now caps the realistic advantage at ~2× and rules out FP4 as a way to extend it.

> Caveat that keeps this from being final: the 0.62× pair is real, fully-verified
> silicon (right tip, `imma_s8s8s32` tensor path, resident-Q* batch engaged) but
> **telemetry-only** — numair's own note says it "cannot certify a silicon-comparable
> rate; rerun without `--telemetry-only`." The certified rate + a depth sweep
> (Q*=128/256/512 + multi-window) would map exactly where the B200 crosses over.

---

## numair's changes — timeline (what he shipped)

| commit | what | our read |
|---|---|---|
| — | `ENC_BMX4C_LT` profile (enum 4), deep tile m=n/2=2048 | the successor shape; golden db1136f2 |
| — | **MatExpand** operand-gen: G/H/W M11 projectors → Y=G·W → B32=Y·H → per-element ChaCha20 PRF Extract → Bhat | SHA/PRF-heavy; this is the cost center |
| `6c714b0` | **MX-block Extract overhaul**: retired ChaCha for MXFP4 E2M1 block-scale Extract, ~32× PRF dilution (1 SHA scale / 32-elem block) | dilutes the PRF cost; not native MX yet |
| `770031e` | w = 128 → 1024 | wider sketch |
| `1ca87fb` | **K-aligned scale axes** | **adopts our FP4 finding** — and re-opens the mxf4 lever (below) |
| `9ca94e70a6` | **resident LT window batching**: on-device W-gen + on-device SHA256d digest, **one sync per batch** (was ~53 launches + H2D/D2H + sync *per nonce*) | the "feed the card" fix; == our ring idea |
| `9bd11af2ed` | raw telemetry interface: `--lt-raw-only` / `--lt-raw-full-window` (our exact bench flags, as compat aliases) | folded in "Vanities/JD/CUDA/Mac feedback" |
| `d4f55774` | review-gap hardening: G2/G3 accept only one clean B200-vs-5090 same-revision campaign; `source_revision` stamped in every report JSON | strict gate; no cherry-picking |

Constant across all of it: **native MXFP4 is a fail-closed stub** (`cutlass_mxfp4`,
"once the pinned recipe lands"), all heights inert, activation NO-GO.

**Update (late 07-19, tip `df12c2a` — "require native MX on peak silicon by default"):**
the stub era is ending. New `matmul_v4_lt_mx_native.cu/.h` (CUDA + HIP + Metal stubs)
wire a real CUTLASS MXFP4 path: E2M1 nibble packing + Vec32-UE8M0 scales in K-blocks
of 32 — structurally the recipe we validated. Policy is aggressive: *"Blackwell must
run qualified native MXFP4/FP8"*; the exact-INT8 path **declines by default** on
sm_10x/sm_12x (escape: `ALLOW_EXACT_MX_FALLBACK`), and his log literally prints
"ACTION REQUIRED … do not ship silent INT8-only rates as peak silicon evidence."
Consequences for us: (a) future `--mode numair` runs on Blackwell must qualify native
MX or set the fallback env; (b) if this ships working, the reference gets the FP4 2×
on those stages and our matador-exclusive mxf4 edge there evaporates (ratio-neutral
across Blackwell, per our earlier finding). **And: the tip does not compile** —
he put host-side `LogPrintf` into the `.cu` TU (`DiagnoseLtPeakMxPathOnce`,
`mx_native.cu:838–851`); nvcc reports `LogPrintFormatInternal` no-instance for all
four calls. Verified on pc twice: fails stock **and** with the `-Dconsteval=constexpr`
shim — so it's not flag-shimmable; the template simply isn't viable from a CUDA TU in
his tree. **The only fix is editing his source** (move the diagnostics to a `.cpp`).
He has no nvcc (Mac) and his CentOS CI never starts (billing lock), so he cannot see
it and cannot fix-verify it. Fleet harness now pins (`--numair-ref`, commit `81dad53`)
so a moving/broken tip can't kill a campaign again.

---

## Our work — the deployment scaffold (committed, byte-exact)

Branch `v4/remove-bmx4c-scaffold`. Pivoted from the retired BMX4-C shape to LT.

| commit | what |
|---|---|
| `02a1549` | **removed the BMX4-C deploy scaffold** (1395 lines); base `digest_probe` still gates `1dad86f1…` on mac + pc-CUDA + 11/11 tests |
| `f8916c8` | **LT module** `core/matmul/matmul_v4_lt.{h,cpp}`: ComputeTemplateHash, DeriveTaggedSeed, LtTemplateSeeds{seed_g,seed_h,seed_wa,seed_u,seed_v}, LtNonceSeeds{seed_wb,sigma}. `core/cuda/lt_accel.{h,cu}` (LtComputeChat) + `harness/lt_probe.cpp` — **byte-exact db1136f2 verified on pc CUDA** |
| `4fcfc07` | **generic height switch**: `MatMulSolveVariant{Base,DRLT}` + `ActiveMatMulSolveVariant(params,height)` keyed on `nMatMulDRLTHeight` (INT32_MAX = inert; agnostic, reusable) |
| `f81d50d` | **gated `SolveMatMulLT` grind**, routed into the height switch, behind `g_lt_mx_extract_ready = false` (dead until the MX-extract port lands) |

Byte-exact gates in play: base matmul `digest_probe = 1dad86f1…`; LT
`lt_probe = db1136f2…`; share validity `share_verify_probe --selftest = e18bbc72…`.

---

## Our optimizations — how the number moved (all byte-exact db1136f2)

Compute-only bench (`bench/lt_bench.cu`, operands pre-filled — measures the GEMM /
extract / combine, *not* the digest):

| step | 5090 nonce/s | delta | commit |
|---|---|---|---|
| naive (tensor combine) | 347 | — | `1f77329` |
| + **tensor-B32** (3-limb balanced-base-256 int8 split of Y, `|Y|<2^18`, one big INT8 GEMM) | 425 | **+22.5%** | `3e43b8a` / ported `a84cb0f` |
| + **3-base-128 combine** (LT P,Q are 18-bit → 3 base-128 limbs replace 4 base-64, 0.56× GEMM) | 553 | **+30%** | `3eb41b4` / ported `604f674` |
| + w=1024 + MX-extract (perf *model*) | 858 | — | `8f2f0d0` |

Both real optimizations verified byte-exact against db1136f2. tensor-B32 helps the
B200 more (it crushes tensor); the combine helps the 5090 more.

---

## The benchmark evolution — and the correction

Three harnesses, increasing honesty. The lesson: **the compute-only number was a
mirage; the real solve is digest-bound.**

1. **Compute-only** (`lt_bench.cu`): 347 → 858 nonce/s. *But this measures ~0.2% of
   the real solve.* Perf-model, ~100× optimistic.
2. **End-to-end** (`lt_e2e_bench.cu`, `--mode lte2e`): adds real per-nonce operand-gen
   (device SHA256 `mant_stream`) + the **32 MiB SHA256d S4 digest** (`Chat[m×m]` u64).
   Finding (`52c6555`): the **S4 digest dominates — ~787 ms/nonce single serial chain.**
   Hidden with a digest **ring** (Little's law): 1.3 → **381 nonce/s** (266×), then
   **memory-capped at 896 chains (28 GiB) on the 5090.** Single-ring beat a
   double-buffered overlap (`e6489ac`) — halving chains cost more parallelism than
   overlap bought. (Dead-end kept as documented: 4-way interleaved digest 139 vs 381 —
   collapsed occupancy.)
3. **numair's authoritative** (`matmul-v4-report`, fleet `--mode numair`, `90f7865`):
   builds his own harness on a rental and runs it. This is his gate metric and the
   real rate. Our perf-model 381/840/858 are upper bounds — good for *ratio direction*,
   not absolutes.

---

## Open optimizations — the remaining levers

| lever | status | note |
|---|---|---|
| **S4 digest** (32 MiB SHA256d, 787 ms/nonce — *dominates everything*) | ring hides it, memory-capped 896 on 5090 | deeper ring (less mem/chain) or multi-GPU; this is where the B200 crosses over |
| **fused operand-gen / coalesced scatter** | proven +30% on BMX4-C, **not ported to LT** | recompute-in-scatter kills the scratch round-trip |
| **mxf4 GEMM recipe** (matador-exclusive, ~5% end-to-end, hand-PTX 1.98× vs INT8) | **blocker structurally gone; low-priority** | was "geometrically dead" (E8M0 along free axis, not K). numair's accel-lanes doc @ `d4f5577` now states **"LT has its own already-contraction-aligned B layout"** and calls a one-pass native MXFP4 projection "structurally possible" (still fail-closed / not wired). So our recipe now *applies*, but native MXFP4 is ours to hand-wire and the gain is SHA-floor-bounded (~5%). Bank it; port only when his Extract freezes. |
| **byte-exact MX-extract port** | deferred | his layout changed *yesterday* (batching); porting now = re-porting on his next push |
| 3-base-128 combine | done, at floor | 18-bit P,Q → can't go below 3 limbs |

### The optimization map — from reading his resident-batched path (`matmul_v4_lt_accel.cu` @ `d4f5577`)

Where his ~59 s/window (256/4.36) on the 5090 goes, structurally:

| his code | structure | our lever (matador-exclusive) |
|---|---|---|
| `LaunchDeviceDigestBatch` | **one thread per candidate, `kThreads=64`** → Q\*=256 = 4 blocks = **4 SMs active, 166 idle**; each thread serially SHA256d's the full **m·m = 33.5 MB** Chat (8.6 GB/window through 4 SMs, per-thread strided) | likely THE dominant phase and THE reason the B200 loses (4 SMs can't use HBM; 5090 clock wins serial SHA). Ours: many more chains resident + **interleaved Chat layout** (lane i reads chain i → coalesced, full BW), and ultimately **fuse combine→SHA** so Chat never round-trips DRAM (removes the 33.5 MB/nonce store+load AND the VRAM cap on chains) |
| `DeviceGemmS8S8Tiled` (+ CUDA-graph replay) | **naive SIMT** one-thread-per-output `for k` inner product — the *fallback* when cuBLASLt IMMA declines (`TryLaunchLtImmaGemmS8S8Device` is the real path) | our tensor-B32 3-limb + 3-base-128 Karatsuba combine already beat this class byte-exact; if his fallback ever serves, it's 10–50× slow |
| `DeviceScanMantissaCounts` | **`<<<1,1>>>` single-thread prefix scan** per operand-gen call | trivial parallel scan; small but free |
| one `cudaStream_t` for everything | **single stream** — per-slot kernels execute strictly in order; no operand-gen/GEMM/digest overlap (HIP got a multi-stream ring; CUDA didn't) | multi-stream + multi-window pipelining (our ring result: 266× on the digest phase) |

Sanity check on the numbers: IMMA GEMMs at these shapes are milliseconds/slot — they can't be the 59 s. The 4-SM digest phase over 8.6 GB at the ~100 GB/s four SMs can pull ≈ 60–90 s. **The digest-launch geometry alone explains his measured rate and the B200 inversion.**

**MEASURED — theory confirmed on pc silicon (2026-07-19, our 5090, his binary @ `d4f5577`, CUDA 13.3):**

- Rate: **3.587 nonce/s** @ Q\*=256 (matches the Vast class → pc is a valid instrument)
- Power during solve: **153 W of 600 W; memory util 1%** — the card is ~75% dark while
  kernel-residency reads 97%
- Window scaling **flat**: 3.566 @ Q\*=128 vs 3.587 @ Q\*=256 (digest time scales with
  candidates → nonce/s invariant, exactly as the geometry predicts)
- **nsys (Q\*=128): `DeviceSha256dSketchBatch` = 98.5% of all GPU kernel time**
  (35.1 s of 35.6 s; 33 launches ≈ 1.06 s each). Next: the `<<<1,1>>>` scan at 0.5%.
  The GEMMs ran as real cuBLASLt `nvjet_sm120` IMMA kernels — 0.2%, irrelevant.
  His solver is a one-kernel problem, and it's the kernel our levers attack.
- Instrument notes: profile artifacts live on pc in `~/btx-d4f5577/` (`lt_w128.nsys-rep`,
  `rep_pc_*.log`, `util_*.csv`); the NGC container's nsys lacks its importer — install
  `nsight-systems-<ver>` in the ephemeral container to import the `.qdstrm`.

**Strategic caveat (why we hold the deep ones):** LT is inert and numair is *actively
reshaping* it (resident batching landed 2026-07-19). Every shape change breaks
byte-exactness and forces a re-port. The smart play is to keep the scaffold + bench
tracking his tip and bank the shape-robust *ideas* (tensor-B32, 3-base-128, ring, mxf4
recipe — all held), and do the heavy MX-extract + mxf4 work only once the shape freezes
or a height is set. The one lever worth a *timely* look is the mxf4 re-check, because
his K-aligned-scale adoption just removed its blocker.

---

## Reproduce

```bash
# byte-exact LT gate (must print db1136f2…)
./build-*/lt_probe

# compute-only LT bench (our optimizations)
#   research/matmul-v4/bench/lt_bench.cu

# end-to-end LT bench (real S4 digest, the ring)
python3 clean-stack/bench/vast/bmx4c_fleet_bench.py --mode lte2e --cards 5090,b200 --ring 384

# numair's authoritative harness on a rental (his gate metric)
python3 clean-stack/bench/vast/bmx4c_fleet_bench.py --mode numair --cards 5090,b200 --n 4096 --window 256
#   each rental's full report is saved to <workdir>/joblog_<iid>.log — verify
#   HEAD=, "resolved backend : cuda", "device identity", and the telemetry line there.
```

---

## Current state (2026-07-19)

- Scaffold + benches **committed** on `v4/remove-bmx4c-scaffold`; **not yet pushed**.
  `results-2026-07/lt-*` + `run*-b200*` output dirs are untracked.
- Latest numair tip benched: `d4f55774` (no newer commits/comments as of 22:52 UTC).
- Next to settle the ratio: **certified** run (drop `--lt-raw-only`) on both cards at
  one revision + a Q*=128/256/512 and multi-window depth sweep.
- Live artifact (benchmark ledger, private): claude.ai/code/artifact/7ac59fbf-996b-444e-b494-ee667a57c0ac
