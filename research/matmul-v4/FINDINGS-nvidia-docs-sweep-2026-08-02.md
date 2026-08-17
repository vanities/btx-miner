# NVIDIA-docs sweep for v4.7 episode levers — 2026-08-02 (evening)

Question asked: after the 08-02 board closure (~825 ms free-boost, GEMM/extract/format/restructure/HW
all measured-closed), what does the CURRENT NVIDIA documentation set (CUDA 13.x, cuBLASLt 13.3,
PTX ISA 9.3, Blackwell whitepapers/tuning guide) still offer that we have NOT tried? Three doc
sweeps (CUDA runtime, cuBLASLt, sm_120 micro-arch) + one code map of `rc_gpu_episode.cu`.

Status: Tier-1 items MEASURED same day (see "Measured outcomes" at the bottom). Tier-2 items
remain hypotheses.

## The code-map fact that frames everything

The episode is a single fully-serialized stream with **~280 synchronous pageable `cudaMemcpy`s per
episode**, of which only 4 are algorithmically required:

- 136 × 32 B PRF-key H2D — `rc_gpu_episode.cu:1187` (one per `gpu_extract`; 34/round × 4 rounds)
- 144 × 96 B opgen seed+midstate H2D — `rc_gpu_episode.cu:1505` (36/round × 4)
- 4 × 32 B round-root D2H — `rc_gpu_episode.cu:1832` — **the only consensus-required host
  round-trips** (round r+1's seed chain needs round r's Merkle root on host: `:1666`)

All 70 intra-round uploads are derivable at round start from `seed_r` (host SHA chain, ~105
hashes/round, µs-class). Each sync pageable memcpy is a full pipeline drain + submit/wait
(~5-15 µs). No pinned memory, no async copies, no events, no graphs, no PDL anywhere in the RC
path. Kernels run on the legacy NULL stream; cuBLASLt on `cudaStreamPerThread` (mutually syncing).
~1,200-1,400 kernel launches/episode.

Glue-overhead ceiling: 280 × ~10 µs + ~1,300 launches × ~2-5 µs ≈ **4-9 ms ≈ 0.5-1% of 825 ms**.
That bounds Tier 1+2 below. The only items that touch KERNEL time are the SASS/compiler checks.

## Tier 1 — build/check first (cheap, doc-backed)

1. **Round-batched pinned upload (kills 276 of 280 sync copies).** At round start, pack all 34
   PRF keys + 36 seed/midstate blocks (~4.5 KB) into one pinned staging buffer, single
   `cudaMemcpyAsync`, kernels take an offset index instead of re-uploading into the single pooled
   `dprf`/`dsb` buffers. Alternative per docs for tiny one-shot reads: zero-copy write-combined
   mapped pinned memory (`cudaHostAllocMapped|WriteCombined` — Best Practices explicitly blesses
   read-once patterns). `cudaMemcpyBatchAsync` (12.8) is the WRONG tool — it batches copies that
   are dispatchable together, ours are currently interleaved; batching at round start makes it
   moot anyway. Expected ~0.3-1% alone; PREREQUISITE for PDL/graphs (sync memcpys break both).
   Option B (equal effect, more build): derive intra-round keys on-device in one tiny kernel from
   an uploaded `seed_r` — eliminates uploads entirely; host SHA is not the cost, so only worth it
   if it simplifies graph capture.

2. **Persist the GEMM autotune cache to disk.** cuBLASLt 13.3 §3.3.5, exact quote:
   `cublasLtMatmulAlgo_t` "can be trivially serialized and later restored for use with the same
   version of cuBLAS library". Serialize per-shape tuned algo (+ workspace size), key on
   `cublasLtGetVersion()` + GPU name + driver; restore + `cublasLtMatmulAlgoCheck()` at startup,
   re-tune on mismatch. Kills the autotune share of the **cold 1.38 s vs warm 0.85 s** delta —
   irrelevant to warm mining, DIRECTLY relevant to `matador-verify` one-shot share validation
   (the pool/validator lane we just shipped to shib+byron). Today's cache
   (`g_algo_cache`, rc_gpu_episode.cu:878) is in-process only; the env heuristics cache
   (`CUBLASLT_HEURISTICS_CACHE_CAPACITY`) is RAM-only — no disk mechanism exists in the library.

3. **Zero-cost SASS pre-checks (the "what can the compiler see" pattern):**
   - **256-bit loads** `ld.global.v8.b32` (PTX 8.8+, doc says "sm_100 or higher" — ambiguous
     whether 12.0 counts; one 5090 practitioner reports vector loads capped at 128-bit).
     `cuobjdump -sass` for `LDG.E.256` vs a 2×128 split. If real: fewer LSU ops + full
     sector-pair per instruction on the L2-sector-bound extract. Expect dead; 2-min check.
   - **IADD vs IADD3 audit in SHA/ChaCha kernels.** CC 12.0 doubled 2-operand `IADD`, `ISETP`,
     `IMNMX`, `vote.ballot` to 128/clk (unified INT32/FP32 pipes) while **LOP3/SHF/PRMT/IADD3/IMAD
     stay 64/clk** — NVIDIA staff confirm ptxas often emits IADD3 with a dead third operand,
     forfeiting the 2x pipe. Grep SASS for IADD3 with RZ third operand in the hot kernels; where
     the mix is LOP3/SHF-saturated keep IADD3 (64×2 adds), where a dependent 2-op add chain
     stalls, IADD co-issues free. Re-audit each CUDA point release (ptxas expected to improve).
     This is the SAME lever class as the IMAD→IADD3 genbase win, one generation later.
   - **CompileIQ (CUDA 13.3)**: per-kernel evolutionary compiler-flag autotune, NVIDIA claims up
     to 15% on GEMM/attention-class kernels. Point it at `k_extract_tiles` + the two opgen
     kernels; costs machine time, zero code.

## Tier 2 — bounded A/Bs after Tier 1 (~0.2-0.6% class each, may overlap)

4. **PDL (Programmatic Dependent Launch).** Supported on sm_120 (`griddepcontrol` = sm_90+;
   confirmed live: 13.3 known-issue CUB-10409 names sm_12x GEMM kernels calling
   `cudaGridDependencySynchronize`). **cuBLAS GEMM kernels are already PDL-enabled since 13.0 U1
   ("some kernels", no knob, automatic)** — the missing half is OUR kernels: launch extract/opgen
   via `cudaLaunchKernelEx` + `cudaLaunchAttributeProgrammaticStreamSerialization`, add
   `cudaTriggerProgrammaticLaunchCompletion()` / `cudaGridDependencySynchronize()`. Win = launch
   gap + prologue-under-tail overlap; GEMM tail waves at M=16384 underfill 170 SMs, so the
   extract prologue riding the GEMM tail is the interesting part, not the µs gap. CUB-10409
   caveat (device alpha/beta WAR hazard) doesn't apply — we use host-pointer-mode constants.

5. **Per-round CUDA graph** (needs #1 first — pageable sync memcpys are not graph-legal).
   ~305 launches/round → one instantiated graph relaunched with updated params
   (`cudaStreamBeginRecaptureToGraph`, new in 13.3, is the cheap re-capture path; capture AFTER
   autotune so the tuned algo bakes in — heuristic query during capture errors, 13.3 known
   issue). 12.6+ straight-line graph launch is ~2.5 µs + ~1 ns/node; saves ~1-2.5 ms/episode of
   CPU submit + jitter immunity. NOTE: the −4.3% CUDA-graphs dead entry was the v1 DIGEST path —
   different workload, does not pre-empt this per the no-assumptions rule; but expectations
   bounded by the glue ceiling. Graphs + PDL compose (programmatic edges inside graphs).

6. **L2 persistence window + carveout hint on extract.** Doc-supported on Blackwell (tuning
   guide: "similar to Ampere"; `cudaAccessPolicyWindow`, `evict_last`). 5090 L2 = 96 MB (GB202
   full die 128 MB). Honest physics: this is a residency tool and extract is L2-sector-
   THROUGHPUT-bound (84% L2, 24% DRAM, mem-OC neutral), so expect ~0 — one cheap A/B max.
   Companion free check: force `cudaFuncAttributePreferredSharedMemoryCarveout = 0` on extract
   (full 128 KB L1; 12.0 carveouts are {0,8,16,32,64,100} KB) — the 91.5% L1 hit rate is
   load-bearing and the default carveout choice is driver-heuristic.

## Closed by this sweep (ledger these — do not re-chase)

- **INT4 IMMA as a 2x-INT8 format lever: DEAD on sm_120.** Not in the 12.x tensor-core type
  table; whitepaper omits it; PTX `.s4` mma still compiles but is EMULATED (measured ~4.6x
  slower than INT8 mma, community cycle-level bench). Per-SM INT8 IMMA rate is IDENTICAL to Ada
  (2048 ops/SM/clk from whitepaper TOPS math) — the 5090's GEMM edge is SM-count × clock only.
  Closes the last cheap-format hope, consistent with the FP4 closure.
- **cuBLASLt int8 epilogue fusion beyond alpha/beta: documented-unsupported and actively
  enforced** (13.3 matmul table: "Epilogue is not supported" for both int32-out and int8-out;
  CUB-10066 fixed the accidental acceptance; the 32F-scale loophole CUB-10067 is slated for
  removal). beta=1 residual fusion IS the ceiling of this lane. Also: with scaleType R_32I,
  alpha/beta may only be 0 or 1 (CUB-8873) — nothing else to fold.
- **Atomic-sync tile handoff (GEMM output consumed before GEMM completes): REMOVED in CUDA 13.0**,
  and was Hopper-FP8-fast-accum-only even when alive. PDL is the surviving (coarser) mechanism.
- **Grouped/batched INT8 GEMM: type tables exclude INT8** (grouped = FP8/16/BF16/NVFP4 on
  cc 10.x/11.0 only), and our GEMMs are a dependent chain regardless.
- **TMA / cp.async.bulk for extract: dead.** `.tile::gather4` is sm_100a/f-only (NOT sm_120),
  multicast absent/degraded on GB202, and a 5090 measurement has bulk-prefetch LOSING to `__ldg`
  by ~5%. The direct L1-cached read stays the optimum (consistent with the 6 dead restructures).
- **Clusters/DSMEM:** max-8 clusters exist on 12.0 but no multicast + DSMEM = a staging pattern
  we already measured dead. No sector-bandwidth gain possible.
- **`cudaMemcpyBatchAsync`:** wrong shape for interleaved copies; superseded by #1.
- **Workspace:** 13.3 recommends ≥32 MiB for sm_12x; we run 256 MiB (`BTX_RC_GEMM_WS_MB`) —
  already covered, no action.
- **Green contexts (SM partitioning):** supported on sm_120 (runtime API since 13.1, cuBLAS
  support 13.3) — not useful intra-episode (dependent chain), but it IS the right mechanism if
  the parked conc/pipelining lever ever re-opens on B200-class hardware: hard SM fencing instead
  of stream interleaving. L2 is still shared, so the extract-vs-GEMM L2 fight remains.

## Sources (primary)

cuBLAS 13.3 docs (matmul type tables §3.4.17, algo serialization §3.3.5, heuristics cache
§3.1.2, workspace §2.4.8) · CUDA 13.0-13.3 release notes (PDL in cuBLAS 13.0 U1; CUB-8873/8874/
10066/10067/10409; CompileIQ + `cudaStreamBeginRecaptureToGraph` in 13.3) · CUDA C++ Programming
Guide (PDL §, compute-capability tables: 12.0 = 48 warps/SM, 128 KB L1, carveouts {0..100} KB) ·
Best Practices Guide Table 5 (12.0 integer throughput: IADD/ISETP/IMNMX/ballot 128, LOP3/SHF/
IADD3/IMAD 64, SAD halved to 32) · PTX ISA 9.3 (cp.async.bulk sm_90+; gather4 sm_100a/f;
256-bit ld "sm_100 or higher"; sm_120a extras = FP4 plumbing + setmaxnreg) · RTX Blackwell PRO
whitepaper (GB202 128 MB L2 full-die / 96 MB on 5090; unified INT32/FP32 cores; INT8 TOPS →
2048 ops/SM/clk = Ada rate; no INT4 in supported list) · NVIDIA forum "Blackwell Integer"
(mjoux: IADD3-with-dead-operand forfeits the doubled pipe) · community cycle-level 5090
characterizations (zartbot; Alpin) for L2 near/far slices (79/180 cyc), s4-mma emulation, and
the 128-bit vector-load cap claim.

## Measured outcomes (same day, 5090, locked 2600, miner stopped, container --gpus all)

1. **Disk tune cache: BANKED. Cold episode 1590.6 -> 1054.4/1081.5 ms with a warm tune file
   (−34% cold; −535 ms).** Warm episode unaffected (973-980 ms across all arms). Implementation:
   serialize `g_algo_cache` (GemmAlgoKey + cublasLtMatmulAlgo_t, raw POD) with a header keyed on
   cublasLtGetVersion + GPU name + workspace + tune level; AlgoCheck-validate each disk algo at
   first use; tmp+rename atomic write. Gate `BTX_RC_TUNE_CACHE` (default ON), path override
   `BTX_RC_TUNE_CACHE_PATH` (default `~/.cache/matador-miner/rc_gemm_tune.bin`). Toy golden
   5b1bff3c PASS with cache cold, warm, and disabled; mode-1 digest 42e74cd6542d86bd unchanged.
   This is the validator/one-shot lane win (matador-verify cold share audits).

2. **Round-batched upload: DEAD in every form, default OFF.** Window 1: full batching −1%
   (980.2/981.7 vs 970.5/971.1). Window 2, three arms x2 interleaved, monotone: off 958.9/968.9,
   PRF-only 967.8/973.5, full 979.3/983.8 ms. Byte-exact everywhere. Since even the PRF-only arm
   (zero added host work at the boundary) loses, the boundary-serialization hypothesis is only
   half the story — the deeper finding is that **the ~280 sync pageable copies were never a
   cost**: the launch queue runs deep enough that the GPU never starves during a drain. This
   DISCONFIRMS the 0.5-1% "glue" estimate for this pipeline and pre-disproves the CPU-submit
   share of the CUDA-graphs lever (Tier-2 #5); PDL tail-overlap (device-side mechanism) stays
   open. Code kept behind BTX_RC_BATCH_UPLOAD for future graph/PDL work and B200-class hosts.

3. **256-bit loads ARE REAL on sm_120** (pre-check, zero A/B cost): ptxas 13.3 accepts
   `ld.global.v8.b32` for sm_120 and emits `LDG.E.ENL2.256` (probe: pc
   ~/matador-prof/ld256_probe.cu). The "128-bit cap on consumer Blackwell" practitioner claim is
   WRONG. This re-opens a narrow extract lever: the raw-matrix reads (i32 tiles, currently
   128-bit class) could issue as 256-bit loads = half the LSU ops, guaranteed full sector-pair
   pulls, on the kernel that is L2-SECTOR-bound. Needs its own A/B; the six dead restructures do
   not pre-empt this (it changes the LOAD WIDTH, not the layout/staging).

4. **IADD-vs-IADD3 audit: already banked by the toolchain.** SASS of all six hot kernels shows
   ZERO IADD3-with-RZ-third-operand; ptxas 13.3 already emits plain 2-op IADD (546-1380/kernel)
   where the doubled 128/clk cc-12.0 pipe applies. Nothing to hand-fix; re-check only on
   toolchain bumps. SHF (64/clk funnel shift) dominates opgen SASS ~4k/kernel — SHA rotation
   remains the structural bound, as known.

## Window 3 (same evening): Tier-2 measured — PACKMIX and PDL both DEAD

- **256-bit PACKMIX: +10.9%** (1068.0/1068.0 vs base 963.3/963.3, locked 2600 x2). Seventh
  extract kill. LDG.E.256 works on sm_120 but the per-attempt loads it replaced were ~free L1
  hits; the register pack costs more than cached traffic. The instruction stays in the toolbox
  for some future kernel whose loads actually miss.
- **PDL extract: +29%** (1239.4/1246.6). Eighth extract kill. The early-launched secondary's
  CTAs run the pre-wait SHA then spin at cudaGridDependencySynchronize, stealing SMs from the
  GEMM's remaining waves. PDL fits small secondaries behind a draining primary — not a
  full-device extract behind a multi-wave GEMM. PDL+PACKMIX intermediate (1126.1/1128.6).
- Both byte-exact (toy golden 5b1bff3c all four gate combos; mode-1 digest 42e74cd6 all arms).
  Code REMOVED per repo convention; dead-comment in k_extract_tiles is the pointer.

**Sweep closure: the entire glue/overlap class is now measured dead on the 5090 at P1 dims**
(batch upload −1%, PRF-only staging −0.7%, PACKMIX −10.9%, PDL −29%). The single banked win is
the disk tune cache (−34% cold). Unrun remainders: L2-persistence window (expect ~0), CompileIQ
(machine-time, park). The honest post-sweep verdict matches the pre-sweep board: warm episode
time on this card moves only via FORMAT (closed), consensus changes, or wider silicon.

## Window 4 (late evening): Tier-2 remainder — all NEUTRAL, sweep checklist closed

Five arms, locked 2600, x2 interleaved, byte-exact everywhere: base 907.8/908.9 vs
carveout=maxL1 907.9/910.8 vs L2-persisting-window 909.8/913.2 vs L2-streaming-window
910.4/912.7 vs whole-binary ptxas -O2 910.1/914.2 ms. Spread <=0.5% incl. in-window drift.
As predicted: residency policy cannot move a sector-throughput bound, the default carveout was
already right, and -O2/-O3 is settled. Research gates dropped (never committed).

Sweep scoreboard, final: 1 banked (disk tune cache, −34% cold, shipped in v0.9.1), 4 dead
(batch upload, PRF staging, PACKMIX, PDL), 4 neutral (carveout, L2 persist, L2 stream,
ptxas -O2), 2 pre-closed by SASS audit (IADD already optimal, LDG.E.256 available-but-useless
here), 1 external remainder (CompileIQ — separate NVIDIA download, ACF via nvcc
--apply-controls, Blackwell+; assessment pending).

## Window 5: CompileIQ — DEAD (second independent null). SWEEP FULLY CLOSED.

Prior art found mid-session: **CompileIQ was already run 2026-07-03 on the scan TU** (research/
EXPERIMENTS.md) — ptxas advanced-controls space ICEs (C7907) on our 24 MB unrolled PTX; the NVVM
space ran 80 clean evals and its +1.56% best did NOT reproduce. That session's lesson (lock
clocks during fitness benching) was applied here.

Episode-TU campaign (3 gen x 6 pool = 18 evals, locked 2600, digest tripwire per candidate):
- Search best **879.9 ms** vs baseline 897.6/900.0 (-2.0% nominal, gen 2).
- Interleaved validation x3: base 895.3/899.3/903.5 vs ACF **931.9/935.0/935.4 = +4.0% SLOWER.**
- The GA's winner is not merely noise — it is a real regression that single-eval fitness could
  not see. Tripwire never fired (18/18 byte-exact, incl. candidates at 1.14-1.38 s).

Structural reason: ~44% of the episode is inside cuBLASLt (precompiled — an ACF cannot reach it),
and our own kernels are dependency-chain / L2-sector bound rather than schedule-bound, which is
what NVVM controls actually move. Two TUs, two nulls: **the compiler-config axis is closed.**

### Final sweep scoreboard (2026-08-02)
| outcome | items |
|---|---|
| BANKED (shipped v0.9.1) | disk-persisted cuBLASLt tune cache — cold episode 1590 -> 1054 ms (-34%) |
| DEAD (measured) | round-batched upload (-1%), PRF-only staging (-0.7%), 256-bit PACKMIX (+10.9%), PDL extract (+29%), CompileIQ ACF (+4.0%) |
| NEUTRAL (measured) | L2 persisting window, L2 streaming window, max-L1 carveout, ptxas -O2 |
| CLOSED by docs/SASS | INT4 IMMA (emulated), int8 epilogue fusion, atomic-sync handoff (removed 13.0), grouped int8 GEMM, TMA/gather4, IADD3 hand-fix (ptxas already optimal) |
| AVAILABLE, unused | LDG.E.256 on sm_120 (real; dead in extract, keep for a future miss-bound kernel) |

One win, everything else measured shut. The warm episode on this card moves only via FORMAT
(closed), consensus changes, or wider silicon — the same verdict the morning board reached,
now with the whole current NVIDIA feature surface checked against it.
