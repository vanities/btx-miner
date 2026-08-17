# v4 prototype — raw measured runs

All from PR #89 branch `claude/matmul-v4-design-spec-af23sj` @ tip `e2339e8`.
Date: 2026-07-15.

## RTX 5090 (pc `omarchy`, sm_120, single-stack — matador-miner stopped for the run)
`v4_proto.cu` built in `matador-build:pathb-deps-cm4` (CUDA 13.3),
`nvcc -O3 -arch=sm_120 v4_proto.cu -lcublasLt`, run under `--gpus all`.

Correctness (alongside live miner, n=256):
```
verify: cuBLASLt-vs-scalar Chat mismatches = 0 / 1024 OK
```

Perf (single-stack):
```
----- n=2048 -----
 1 operand-gen(SHA)   0.617  74.5%   9.4 M SHA-256/nonce
 2 INT8 GEMMs         0.023   2.8%   2.15e+09 MAC
 3 Fq combine         0.188  22.7%
 TOTAL                0.828         1207 nonce/s     SHA/GEMM = 26.6x   SHA=15.29 GH/s

----- n=4096 (LAUNCH DEFAULT) -----
 1 operand-gen(SHA)   2.414  62.9%   37.7 M SHA-256/nonce
 2 INT8 GEMMs         0.062   1.6%   1.72e+10 MAC
 3 Fq combine         1.361  35.5%
 TOTAL                3.837          261 nonce/s      SHA/GEMM = 38.8x   SHA=15.64 GH/s

----- n=8192 (max dim) -----
 1 operand-gen(SHA)   9.606  47.5%   151.0 M SHA-256/nonce
 2 INT8 GEMMs         0.502   2.5%   1.37e+11 MAC
 3 Fq combine        10.128  50.1%
 TOTAL               20.236           49 nonce/s      SHA/GEMM = 19.1x   SHA=15.72 GH/s
```
Read: INT8 tensor GEMM = 1.6–2.8 % across the entire legal dimension range. As n
grows, work shifts SHA→combine (both non-tensor), never toward the tensor GEMM.

## RTX 5090 — legacy vs WIDE XOF sweep (v4_bench, NVML power)
The `--wide` XOF models the proposed fix (one SHA-256 per ~30 elements instead of
per element). Key result: **widening the XOF is necessary but NOT sufficient** —
the bottleneck moves to the mod-q combine (also integer ALU, not tensor cores),
and the INT8 tensor GEMM never exceeds ~4.5% in any configuration.

| n | XOF | operand-gen | INT8 GEMM | combine | nonce/s | power | J/nonce |
|---|---|---|---|---|---|---|---|
| 4096 | legacy | 62.9% | **1.6%** | 35.5% | 261 | 598 W | 2.29 |
| 4096 | wide | 23.5% | **3.7%** | 72.8% | 534 | 280 W | 0.52 |
| 8192 | legacy | 46.9% | **2.5%** | 50.6% | 49 | 242 W | 4.95 |
| 8192 | wide | 13.8% | **4.5%** | 81.7% | 82 | 393 W | 4.81 |

**Structural reason:** the sketch enforces only `2n²m` INT8 MACs (not the full
`n³`), because the optimal miner evaluates `(U·A)(B·V)` directly (spec §0.7-(3)).
On tensor cores that is ~0.06 ms at n=4096, a rounding error next to the XOF (∝n²)
and the mod-q combine (`nm²` of 61-bit modmul on integer ALU). No tile size `b`
fixes it: combine/GEMM time ratio ≈ `363/(2b)`, so INT8 only dominates the combine
at `b>181` (which makes the GEMM itself vanish). Making INT8 dominant appears to
require the full-C profile (enforce `n³`), which the sketch was chosen to avoid
(64+ MiB payload, bigger verify). This is a tension between cheap verification
(sketch) and INT8-compute-bound-ness (full-C). Combine kernel here is naive; even
optimized it stays ALU-bound and non-tensor, so the qualitative conclusion holds.

## Cross-hardware: H100 SXM (Vast) vs RTX 5090 — the datacenter test
Same v4_bench, both bit-exact to the reference (H100 emit digest = `d5796a09…9230`,
identical to 5090 + CPU: cross-arch determinism, sm_90 vs sm_120, CONFIRMED).

| gpu | n | xof | INT8% | nonce/s | $/hr | nonce/$ | W | J/nonce |
|---|---|---|---|---|---|---|---|---|
| **5090** | 4096 | legacy | 1.6% | **261** | 0.30 | 3,132,000 | 598 | 2.29 |
| H100 | 4096 | legacy | 3.4% | 92 | 1.74 | 190,661 | 116 | 1.26 |
| **5090** | 4096 | wide | 3.7% | **534** | 0.30 | 6,408,000 | 280 | 0.52 |
| H100 | 4096 | wide | 5.5% | 151 | 1.74 | 311,972 | 135 | 0.89 |
| **5090** | 8192 | wide | 4.5% | **82** | 0.30 | 984,000 | 393 | 4.81 |
| H100 | 8192 | wide | 5.7% | 21 | 1.74 | 43,059 | 268 | 12.87 |

**The H100 is ~2.8x SLOWER per card and ~16–30x worse per rental-dollar.** v4 does
not just fail to make datacenter win — it makes datacenter *lose*. Reason: 94%+ of
the work is SHA + mod-q combine (integer/memory), where the 5090's higher clocks
beat the H100; the INT8 tensor slice the H100 dominates is <6%.

**Honest kernel caveats (the direction is robust, exact ratios are not):**
- INT8 GEMM measured *slower* on H100 (0.37 vs 0.06 ms at n=4096) — a cuBLASLt
  tall-skinny INT8 algo artifact; with a tuned config the H100 should win that
  slice. But it is <6% of work, so it cannot flip the result.
- The mod-q combine is a naive one-thread-per-output kernel. The H100's HBM
  (3.35 TB/s vs 5090 1.8) could help a *bandwidth-tuned* combine and close some of
  the gap. This is the biggest caveat and the top TODO for a fair datacenter run.
- H100 draws only 116 W here (underutilized by the naive kernels), so its lower
  J/nonce is an idle artifact, not superior perf/watt on real tensor work.
- **Kernel-independent, robust claim:** the INT8 tensor GEMM is <6% of per-nonce
  work (cuBLASLt is ~optimal for it), so the datacenter INT8 advantage structurally
  cannot dominate under the sketch. Widening the XOF and tuning the combine do not
  change that ceiling.

### Tuned combine (the fair test) — datacenter STILL loses
Replaced the naive one-thread-per-output combine with a shared-memory tiled mod-q
GEMM (coalesced HBM loads; byte-exact — F_q add is associative, `--emit` still
`d5796a09…`, verify 0/1024). This helped the H100 MORE than the 5090 (its HBM now
gets used) and raised the INT8 share to 12-13%:

| gpu | n | xof | INT8% | nonce/s | nonce/$ |
|---|---|---|---|---|---|
| **5090** | 4096 | wide | 4.7% | **670** | 8.0M |
| H100 | 4096 | wide | 12.4% | 339 | 0.70M |
| **5090** | 8192 | wide | 5.9% | **107** | 1.28M |
| H100 | 8192 | wide | 13.2% | 48 | 0.10M |

Combine tuning lifted the H100 **2.2x** (151→339 at n=4096 wide) vs the 5090's
1.25x (534→670): the HBM caveat was real and is now addressed. The datacenter part
**still loses ~2x per card and ~8-12x per rental-dollar**, and the INT8 slice tops
out at ~13% even on the H100. (Power/J columns are unreliable: NVML sampled once
per nonce post-sync catches clock ramp-down; rely on nonce/s + nonce/$.)

Harness: `bench/v4bench.sh` (up/run/report/down) + `bench/results/scorecard.csv`.

## Apple M4 Max (16 threads, CPU — CommonCrypto hardware SHA)
`v4_microbench_mt.cpp`, `clang++ -O3 -std=c++17`.
```
----- n=2048 -----  SHA 104ms 55.3% / GEMM 73ms 38.5% / combine 12ms 6.3% = 5.3 nonce/s
----- n=4096 -----  SHA 389ms 43.3% / GEMM 425ms 47.3% / combine 84ms 9.4% = 1.1 nonce/s
```
GEMM share high ONLY because M4 has no INT8 tensor path (scalar ALU matmul).
Aggregate CC_SHA256 rate ~0.10 GH/s is one-shot-call-overhead-bound; a batched
NEON-crypto SHA would be much faster (→ pushes back to SHA-bound). Per v4 spec,
M1–M4 are verify-only; M5+ needed for competitive INT8 mining.

## Single-thread CPU (M4 Max, one core) — `v4_microbench.cpp`
```
n=512   operand-gen 95.7% / GEMM 2.3% / combine 2.0%   (590 K SHA/nonce)
n=1024  operand-gen 91.8% / GEMM 4.3% / combine 3.9%   (2.36 M SHA/nonce)
```
Exact per-nonce SHA count at n=4096 = 37,748,736 (= 2n² + 2mn, m=n/8).

## C-13 limb combine — byte-exact on GPU, real speedup, but does NOT flip datacenter (PRIVATE)
`c13_bench.cu --c13`: mod-q combine as 25 s8->s32 tensor GEMMs (base-128 limbs of
biased P',Q') + O(m^2) reconstruct. Emit `--c13 --wide` bit-exact to the reference
(`70c4d85d` @n256, `fccde138` @n4096) on both 5090 (sm_120) and H100 (sm_90).

Flip-test (nonce/s, wide XOF, tiled combine -> C-13 combine):

| | 5090 tiled | 5090 C-13 | H100 tiled | H100 C-13 |
|---|---|---|---|---|
| n=4096 | 496 | **544** | 286 | 275 |
| n=8192 | 89 | **161** | 44 | 64 |

**C-13 is a genuine speedup (5090 n=8192: 89 -> 161, 1.8x) BUT does NOT flip the
per-card ranking: the 5090 still beats the H100 ~2-2.5x with C-13.** H100 n=8192
+C-13 split: operand-gen 21.7% / main GEMMs 17.8% / c13-combine 60.6% -> 15.5ms
vs the 5090's 6.2ms. The H100 is slower at every stage: operand-gen (SHA/memory,
5090 clocks win), the tall-skinny main GEMMs, AND the small m*m*n c13 GEMMs
(cuBLASLt underperforms on these shapes; 25 sequential launches).

=> Even after BOTH fixes (wide XOF + C-13 combine-on-tensor-cores), v4 does not
favor datacenter. The sketch's stage shapes (SHA-ish operand-gen + small/skinny
GEMMs) all reward high-clock consumer cards. Caveat: kernels are not maximally
tuned per-arch (esp. the H100's 25 sequential small GEMMs); a batched/fused H100
path would narrow but the ~2x gap + operand-gen floor make a per-card flip
unlikely. STRATEGIC: consumer wins v4 robustly; C-13 is our private speed edge.

## f50f0f8 (numair's XOF fix) — VERIFIED with our harness
In response to the review, numair replaced the per-element oracle with a wide
counter-mode stream: `SHA256(seed_le || domain || LE64(block))`, all 32 bytes
rejection-sampled in stream order (`ExpandBalancedS8Stream`, domain 's'=0x73 /
'q'=0x71). refcheck on HIS code, n=4096:
- operand-gen SHA/nonce: **38,499,212 -> 1,203,157 (32x fewer)**, ratio 0.0319.
- `ComputeSketchOptimal == ComputeSketch`: still BYTE-IDENTICAL (determinism kept).
- Our GPU harness now reproduces his new derivation **bit-exact**: `--emit` n=4096
  digest = `fccde1381c3c270e062d66c99b1696659c07b7f87cd0a0565c2c894f36d9cf6c`
  == reference; `--verify --wide` 0/4096. (Bit-exact GPU wide stream = count ->
  thrust exclusive_scan -> scatter, matching stream order under rejection.)

Post-fix the mod-q combine is the dominant non-tensor stage (open item -> C-13
limb-decomposition). Harness contributed as btxchain/btx PR #90.

## Reference check — PR #89's OWN code, instrumented (`refcheck/`)
Compiles the **unmodified** `int8_field.cpp` + `matmul_v4.cpp` from the PR against
thin shadow headers + a KAT-verified SHA-256 with a `Finalize` counter, and drives
`ExpandOperand`/`ComputeSketch*` directly (no header plumbing). Run on M4 Max:
```
SHA-256 KAT ("abc"): PASS

n=256   operand-expansion SHA-256 calls = 150,375   (2n^2+2mn=147,456, ratio 1.0198)
        ComputeSketchOptimal (U*A)(B*V) == ComputeSketch full-C U*(A*B)*V : BYTE-IDENTICAL

n=4096  operand-expansion SHA-256 calls = 38,499,212  (~38.5 M; 2n^2+2mn x 1.0199)
        vs 1.72e10 INT8 MACs  =>  38.5 M SHA-256 per 1.72e10 MACs
        (single-core generic SHA expansion wall time = 6180 ms for ONE nonce)
```
=> The SHA-bound mechanism is confirmed **on the reference's own code**, not a
reimplementation: one SHA-256 per matrix element, ~38.5 M per nonce at n=4096.
The 1.02 factor over 2n²+2mn is the ~2% rejection-sampling retries.

## Byte-exact cross-check — GPU prototype == reference digest (`--emit`)
Fixed shared seeds (seed_a=0.., seed_b=64.., seed_u=128.., seed_v=192.., sigma=32..).
`refcheck emit <n>` (CPU, their code) vs `v4_proto --emit` (5090: device-SHA
operand-gen → cuBLASLt INT8 → mod-q combine → LE64 serialize → SHA256d digest):

| n | A[0..7] | CHAT[0] | digest H(sigma‖Ĉ) | match |
|---|---|---|---|---|
| 256 | -56 65 -93 3 37 -107 52 116 | 27325006274 | `e2b873d6a41ceca766783f69a73ae91970344842b6ce6a2992188019297ffc75` | ✅ |
| 4096 | -56 65 -93 3 37 -107 52 116 | 18054099993133 | `d5796a095238ccd0fcde8ba7c5e0507dbc933989fc74484366d168ea049f9230` | ✅ |

=> The GPU prototype is **byte-exact to the reference at the production dimension
n=4096**. The 5090 timing split (INT8 = 1.6 %) is therefore of the real v4 work.

**Remaining for a full consensus test (deploy-time):** header→seed derivation
(`matmul_pow` DeriveOperandSeed/DeriveSigma from a real `CBlockHeader`) + actual
block acceptance (`submitblock` on a regtest node built from the PR). That is O(1)
SHA plumbing around the now-proven core; it does not affect the finding.

## BMX4-C (v4.2) native FP4 on the RTX 5090 — DOES NOT DISPATCH (`fp4_t24.cu`)
BMX4-C v4.2 hangs its whole datacenter tilt on a block-scaled FP4 tensor path
(`CUDA_R_4F_E2M1` operands + `VEC32_UE8M0` per-32 scales), gated on an M-t24
exactness check: does the FP4 tensor core accumulate EXACTLY to t=24 bits (so the
mod-q reduction is deterministic across silicon), or round early (t≈14, Hopper
style). Their own measurement tool cannot answer it — `matmul-v4-report` prints
"device native kernel wired: no"; its mt24 verdict is CPU-reference only.

`fp4_t24.cu` mirrors their backend's exact recipe (`RunMxf4Gemm` +
`RunMxf4Qualification`, from `src/cuda/matmul_v4_bmx4_accel.cu`) as a standalone
probe and runs it on 5090 silicon. Built in the matador-build container
(CUDA 13.3 image, cuBLASLt 13.5.1), `nvcc -arch=sm_120 -lcublasLt`, run
single-stack on the 5090:

```
device: NVIDIA GeForce RTX 5090  sm_120  cuBLASLt 130501
[control] INT8 tensor GEMM (CUDA_R_8I, COMPUTE_32I) 512x512x4096: AVAILABLE
[availability] cuBLASLt block-scaled FP4 (CUDA_R_4F_E2M1 + UE8M0):
   512x512x4096     full-precision    : none
   512x512x4096     fast_accum(round) : none
   1024x1024x8192   full-precision    : none
   1024x1024x8192   fast_accum(round) : none
   256x256x2048     full-precision    : none
   256x256x2048     fast_accum(round) : none
   32x32x1864128    full-precision    : none
   32x32x1864128    fast_accum(round) : none
```

Every shape, both accumulation modes: `cublasLtMatmulAlgoGetHeuristic` returns
ZERO block-scaled FP4 algorithms. The INT8 control through the identical heuristic
path returns AVAILABLE, so this is FP4-specific, not a harness bug: the 5090 has
a working INT8 tensor path; cuBLASLt exposes no block-scaled FP4 GEMM for it (latest
toolkit).

**Consequence.** The reference's committed FP4 recipe FAILS CLOSED to INT8 on the
5090 — exactly the fallback numair's C-1' note describes. So under the reference
backend, BMX4-C on the 5090 runs on INT8 tensor cores, identical to v4.1 (alphabet
≤48 fits INT8 at full rate, no tax). The M-t24 exactness gate is un-measurable on
the 5090 through cuBLASLt because there is no FP4 GEMM to run.

**What this does and does not prove.**
- PROVEN: the reference miner's FP4 path does not dispatch on consumer Blackwell
  (5090) via cuBLASLt today; it degrades to INT8. matador's fleet position is
  therefore UNCHANGED by v4.2's FP4 aspect vs v4.1.
- NOT proven that the 5090 *hardware* lacks FP4 — it has FP4 tensor cores; only the
  cuBLASLt block-scaled GEMM path is missing. A hand-written PTX `mma.kind::mxf4`
  kernel could reach it, but (a) that is a real lift and the reference would still
  not use it, and (b) M-t24 exactness stays unproven on ANY FP4 silicon.
- The FP4 lever, as specified, is a datacenter-Blackwell (B200, sm_100) lever, NOT
  something consumer Blackwell or Hopper (H100 = INT8-only, no FP4) gets from the
  standard library. The datacenter-vs-consumer tilt requires B200-class silicon AND
  FP4 beating INT8 by enough after $/hr and joules AND M-t24 holding on B200 FP4.

## v4.1 batched-sketch GATE — is the matmul the bottleneck on the 5090? (`c13_bench --gate`)
Before building an FP4 GEMM, we answered the reference's own gate
(`bench/matmul_v4_stage_bench.cpp` §K.2a-WT): what fraction of per-nonce MARGINAL
wall-time is tensor work at the v4.1 batched-sketch shape. Added `--gate` to
`c13_bench.cu`: S0 (A,U,V + P=U*A) amortized out, per-nonce S1b (expand B) / S2
(Q=B*V, tensor) / both combine paths (S3' ALU-direct vs S3b C-13 limb-tensor). Ran
on the 5090 at n=4096, m=n/kTileB=1024, sweeping window Q. Rock-stable across Q=8..64:

| stage | ms/nonce | % | throughput |
|---|---|---|---|
| S1b expand B (SHA/int) | 0.415 | 26% | |
| S2 Q=B*V (tensor)      | 0.069 |  4% | 495 INT8 TOPS |
| S3' combine ALU-direct | 3.53  |  - | (integer; NOT chosen) |
| S3b combine C-13 tensor| 1.106 | 70% | 190 INT8 TOPS  <- 5090 picks this (3.2x < ALU) |
| **tensor share (S2+S3b)** | | **74%** | majority => matmul IS the bottleneck |

Two findings and a catch:
1. **Gate PASSES on share (74%).** Unlike v4.0 single-nonce (matmul was 1.6%), the
   v4.1 C-13 limb-combine makes the per-nonce work tensor-DOMINATED. The 5090 runs
   the combine on the tensor cores (C-13 1.11 ms beats ALU-direct 3.53 ms by 3.2x).
   So FP4, which speeds the tensor stages, is on the critical path here.
2. **THE CATCH — tensor UTILIZATION is only ~25% of INT8 peak** (S2 59%, C-13 combine
   23%), far below the reference's own 60% gate and matching their reviewer anchor
   "0.40x at b=8." The dominant C-13 combine is OVERHEAD/BANDWIDTH-bound (25 limb
   GEMMs + split/rowsum/colsum/reconstruct passes), NOT MAC-bound.
3. **So naive FP4 (2x on all tensor) projects +59% nonce/s, but REALISTIC is ~+11%** —
   FP4 only halves the MAC portion, and the combine is mostly overhead, not MACs.

**Consequence / where the lever actually is.** The bottleneck is not raw tensor
throughput; it is the combine's low utilization. Fixing that (fuse the limb GEMMs,
cut the split/reconstruct passes) is a bigger, FORMAT-AGNOSTIC win (helps INT8 today,
~2x headroom to peak) AND it is what makes FP4 worth more than +11%. The same ~25%
utilization strands the datacenter's raw-tensor advantage too: a B200's extra TOPS
do not convert at 25% util. That is precisely the reference's stated worry ("the
model has been wrong twice; only these measurements settle it"). On the 5090 the
gate is: tensor-bound YES, tensor-SATURATED no. FP4 is worth pursuing but the combine
utilization is the higher-leverage lever, and unlocks FP4's value.

CAVEATS: S4 (device digest of the m*m sketch) excluded (host SHA unrepresentative) —
including it lowers the tensor share. C-13 here is matador's 5-limb / 25-GEMM version;
the reference's 4-limb / 16-GEMM combine would run a bit hotter (fewer GEMMs), so its
utilization is somewhat higher but still below the 60% gate.

## FUSED C-13 combine — 25 GEMMs -> ONE GEMM: +29% combine, +17% end-to-end (PRIVATE)
The 25 limb-pair GEMMs `G[i][j] = Pl_i * Ql_j` are one block-outer-product in disguise.
Stack P-limbs vertically (`c13_split` already emits (5m x n)) and Q-limbs HORIZONTALLY
(new `c13_split_qh`, (n x 5m)); then ONE (5m x n)*(n x 5m) INT8 GEMM computes the whole
5x5 grid, and `c13_reconstruct_f` reads block (i,j) at rows [i*m..], cols [j*m..].
Identical products, identical reconstruct math — byte-exact gate in `--gate` proves all
three paths agree (ALU vs 25-GEMM vs FUSED: 0 mismatches) before any timing counts.
5090, n=4096, m=1024, Q=32:

| combine path | ms | INT8 TOPS |
|---|---|---|
| S3' ALU-direct        | 3.523 | (integer) |
| S3b C-13 25-GEMM      | 1.119 | 192 |
| **S3f C-13 FUSED 1-GEMM** | **0.871** | **247 (1.29x)** |

Per-nonce marginal: 1.591 -> **1.358 ms = 629 -> 736 nonce/s (+17%)**, tensor share 69%.
The 5120x5120x4096 GEMM runs ~0.43 ms at S2-class ~511 TOPS; the remaining ~0.44 ms is
the elementwise split/rowsum/colsum/reconstruct passes (bandwidth) + launch overhead —
the next (diminishing) target if we push further. This fusion transfers directly to the
reference's 4-limb combine (16 GEMMs -> one (4m x n)*(n x 4m)) and to BMX4-C's batched
structure.

## FP4 vs the v4.1 combine — MATHEMATICALLY DEAD (and why BMX4-C exists)
The C-13 limbs are base-128 digits (0..127). E2M1's exact-integer alphabet is
{0,±1,±2,±3,±4,±6} — a 7-bit digit does NOT fit, so the limb GEMMs can never run in FP4
as-is. Re-decomposing to an FP4-fitting digit set caps at base-5 (digits 0..4; base-7
invalid — digit 5 unrepresentable): 14 limbs -> 196 limb-pair GEMMs = **7.8x the MACs
against FP4's 2x rate = 3.9x net SLOWER. Dead.**
- v4.1 FP4 ceiling is therefore S2 only: 1.358 -> 1.325 ms = **+2.5% nonce/s.** The
  earlier "+53-59% if FP4 2x's all tensor" projection was a fiction for v4.1.
- **This is precisely the problem BMX4-C solves:** the M11 alphabet {0,±1,±2,±3,±4,±6}
  IS E2M1's exact-integer subset — v4.2 makes the OPERANDS FP4-native so the dominant
  GEMM rides FP4 with no limb trick. The datacenter-FP4 thesis lives entirely in the
  v4.2 restructure (their claimed 93% batched tensor share), not in v4.1 — and per the
  MXFP4-vs-NVFP4 finding above, TODAY it dispatches for nobody via cuBLASLt; custom
  kernels (which the 5090 can also run) are the only road in.

## Next measurement that would actually settle M-t24: run this same `fp4_t24.cu`
unchanged on a B200 (Vast, sm_100) where cuBLASLt is expected to expose block-scaled
FP4 — Probe 1 (all-+3 rail, K=1,864,128, expect 2²⁴−64) reads the accumulator width.

### B200 (sm_100, Vast) — cuBLASLt ALSO exposes no MXFP4, on 12.8.4 AND 13.0
Ran `fp4_t24.cu` unchanged on a rented B200 (driver 580.126.09), twice:
```
NVIDIA B200  sm_100  cuBLASLt 120804 (image 12.8.1-devel): MXFP4 none / INT8 AVAILABLE
NVIDIA B200  sm_100  cuBLASLt 130000 (image 13.0.0-devel): MXFP4 none / INT8 AVAILABLE
```
Both toolkits, all 4 shapes, both accum modes: ZERO MXFP4 algorithms for the reference's
exact recipe (CUDA_R_4F_E2M1 + VEC32_UE8M0 + COMPUTE_32F). So it is NOT toolkit age —
the newest cuBLASLt on a datacenter B200 still won't dispatch BMX4-C's MXFP4.

### ROOT CAUSE — cuBLASLt's FP4 GEMM is NVFP4-only; BMX4-C is MXFP4
cuBLASLt DOES ship a block-scaled FP4 engine (NVIDIA's own sample is `LtNvfp4Matmul`).
Adding an NVFP4 contrast probe (E2M1 + `VEC16_UE4M3`, 16-element blocks) settles it on
the 5090:
```
device: NVIDIA GeForce RTX 5090  sm_120  cuBLASLt 130501
[control] INT8 : AVAILABLE
 MXFP4 (E2M1 + VEC32_UE8M0, 32-elt UE8M0 scale) : NONE        <- BMX4-C's format
 NVFP4 (E2M1 + VEC16_UE4M3, 16-elt UE4M3 scale) : AVAILABLE   <- what cuBLASLt serves
```
cuBLASLt's FP4 GEMM exists but only for **NVFP4** (UE4M3 scales, 16-element blocks).
**BMX4-C uses MXFP4** — E8M0 power-of-2 block scales, 32-element blocks — so the
reference's `RunMxf4Gemm(... VEC32_UE8M0 ...)` requests a configuration cuBLASLt has no
kernel for, on every Blackwell tested. The "no algorithm -> fall to INT8" is a real
FORMAT gap (MXFP4 vs NVFP4), not a bug in our harness and not toolkit age.

**Consequence.** As written (cuBLASLt backend), BMX4-C's FP4 path dispatches for NOBODY —
5090 and B200 both silently run INT8. The advertised "datacenter earns more via FP4" does
not materialize through cuBLASLt for the BMX4-C (MXFP4) format on any card. Real MXFP4
requires a hand-written or CUTLASS kernel — which `fp4_mma_sm120.cu` shows the 5090 runs
directly (below). So the FP4 lever is custom-kernel-gated and consumer-reachable, not a
datacenter-exclusive cuBLASLt freebie.

## The 5090 FP4 cores ARE reachable — hand-written PTX MMA (`fp4_mma_sm120.cu`)
cuBLASLt has no FP4 for sm_120, but the hardware does. Public reverse-engineering
(florianmattana.com FP4-attention-sm120; NVIDIA devforum block-scale threads) shows
sm_120 exposes `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32
.row.col.f32.e2m1.e2m1.f32.ue8m0` through inline PTX. `fp4_mma_sm120.cu` drives it
directly. Two gotchas that gate it:
- **`-gencode arch=compute_120a,code=sm_120a`** (plain `-arch=sm_120` -> ptxas
  "instruction not supported on .target sm_120"). This is the trap in the
  llama.cpp #19662 / cutlass #3227 issues.
- **kind::mxf8f6f4 container packing:** each E2M1 nibble sits in bits 5..2 of an
  8-bit lane (`nibble<<2`), so +3 (nibble 0x5) is byte 0x14, not 0x05.

Built in the matador-build container (CUDA 13.3), run on the 5090. Layout-agnostic
uniform +3 rail (every A/B element +3, every UE8M0 scale 2^0 -> each m16n8k32 MMA
adds 32·9 = 288 to every output; N chained MMAs -> 288·N):
```
device: NVIDIA GeForce RTX 5090  sm_120
  N=1       d0=       288.0   expect       288.0   OK   (one MMA = 32*(3*3))
  N=100     d0=     28800.0   expect     28800.0   OK
  N=58254   d0=  16777152.0   expect  16777152.0   OK   (2^24-64 rail)
  N=58261   d0=  16779168.0   expect  16779168.0   OK   (just over 2^24)
```
Every case exact. **The 5090 executes block-scaled FP4 tensor MMA correctly and
accumulates the rail exactly.** The exact 16,777,152 at magnitude 2²⁴ rules out any
reduced-precision accumulate (a t≤16 accumulator would round intermediate multiples
of 32 and miss the target); consistent with the full FP32 the `.f32` instruction
specifies. (This uniform rail advances by 288 = 32·9, whose 2-adic tail proves
t≥19, not a clean t=24 — the fine 9-increment discrimination is cuBLASLt's Probe 1
on the B200. But t≥19 already excludes every reduced accumulator; real silicon is
either reduced (excluded) or full FP32.)

**Strategic consequence.** BMX4-C's FP4 lever is NOT datacenter-exclusive. The
reference miner's cuBLASLt backend fails closed to INT8 on the 5090, but a custom
kernel (matador's lane) reaches the 5090's FP4 cores directly and computes correct
block-scaled FP4 matmul. So if v4.2's FP4 path is actually faster than INT8 for the
tiny-matmul PoW, the "datacenter earns more" tilt is defeatable on our own consumer
fleet by hand-writing FP4 — exactly the kind of edge matador already cultivates.
The open question is throughput (FP4 vs INT8 for this K) and whether the alphabet's
INT8-full-rate fit already captures most of it; that is a bench, not a capability
gap. This microkernel is the reachability proof, not yet a competitive GEMM (no
tiling / shared-mem staging / real fragment layout — those come from Florian's repo
or CUTLASS sm_120 if we pursue it).

## DEPLOY PIPELINE (--pipe): overlap is ~EMPTY, S4 digest is the REAL hidden stage (2026-07-17)
PR #89 moved to solver-evolution Stage 2d (tip `1c32221`): golden `4e192d8b…` UNCHANGED
(our byte-exact anchor holds), ENC-BMX4C-D REINSTATED (b=2/m=2048/32 MiB segregated
proof) on the strength of our B200-vs-5090 measurement, and — the load-bearing detail —
the winner check is `H(sigma||Chat) <= target` where Chat serializes to **32 MiB (D) /
8 MiB (C) per nonce**, a chained SHA256d the reference computes **on the CPU**
(`matmul_v4_bmx4_batch.cpp:96`). Every prior `--gate` number excluded this stage (S4).

`--pipe` (c13_bench.cu) measures the deploy path honestly: hoisted template P-limb
split, double-buffered expand||tensor overlap on separate streams (cub scan, no
thrust-malloc serialization), and a device chained-SHA S4 (zero-copy: SerializeSketch
LE64 == device Chat bytes; golden-verified end-to-end, `DIGEST(dev-S4) = 4e192d8b… MATCH`).
5090, single-stack, n=4096:

| lever | D (b=2, m=2048) | C (b=4, m=1024) |
|---|---|---|
| old --gate marginal (S4 excl.)     | 1.724 ms (580/s) | — |
| A seq + hoisted P-split            | 1.689 ms (592/s) | 0.754 ms (1327/s) |
| B + expand‖tensor overlap          | 1.633 ms (612/s) **+3.4%** | 0.753 ms **+0.0%** |
| C S4 digest amortized (deep ring)  | 1.13 ms/nonce    | 0.43 ms/nonce |
| **D SUSTAINED incl. S4**           | **1.803 ms (555/s)** | **0.833 ms (1200/s)** |

Findings, in order of importance:
1. **S4 was the landmine, not the combine.** A naive digest (128 chains in flight) collapses
   D to **151 nonce/s (−74%)** — the serial 524K-block SHA chain is latency-bound (~0.9-1.2 s
   per chain) and 128 chains = 4 warps = idle GPU (246 W). The fix is Little's law, not ALU:
   **chains-in-flight ≈ rate × latency** → 768 chains / 15 GiB sketch ring at D (512 / 2.1 GiB
   at C) + a chain kernel rewritten from byte-staged local-memory blocks to aligned uint32
   window loads + `__byte_perm` BE assembly (1.33x lower chain latency). Then S4 co-runs for
   **−6.4% (D) / −9.6% (C)** instead of −74%.
2. **The reference CPU-hosts S4: ~17 ms/nonce per SHA-NI core at D (~4 at C).** A reference-
   backend miner needs ~10 dedicated cores just to digest at 5090-D rate; our device S4 makes
   this a structural matador edge, byte-exact to the golden.
3. **Expand‖tensor overlap is measured-EMPTY (+3.4% D, 0% C)** — both kernel classes saturate
   SMs, so streams time-slice; same "no idle exists" mechanism as the v3 nsys finding. The
   ~+33% overlap ceiling from the 07-16 handoff is DISPROVEN at production shape. Remaining
   +3.4% comes from edge effects; not worth solver complexity beyond the 2-slot version.
4. Hoisting the template-scoped P-limb split: 1.724 → 1.689 (**+2%**, free, deploy-relevant).
5. VRAM is now a real deploy constraint at D: 15.1 GiB ring on a 32 GiB card is fine; a 16 GiB
   card (5080) caps chains → rate. Next digest lever if needed: more per-chain latency cuts
   (rolling 16-word schedule, 2 chains/thread ILP) shrink the ring proportionally.

Harness: `c13b_pipe 4096 768 --pipe --b 2 --kd 768` / `… 512 --pipe --b 4 --kd 512`.
CSVPIPE rows in bench/results/. Co-run power: 473 W (D) / 284 W (C) — not power-capped.

## SOLVER-READY: full header->digest compute path, GPU sampler, 3/3 goldens (2026-07-17)
`c13_bench --solve` runs the EXACT compute chain that drops into the miner's
`SolveMatMulBmx4C`, entirely on the 5090, from a `CBlockHeader` to `H(sigma||Chat)`:
  1. HOST seed derivation (validated §H.4/§A.2 port): ComputeMatMulHeaderHash ->
     ComputeTemplateHash (nonce=0, seeds null) + full-header-hash + DeriveSigma
     (SHA256d) -> DeriveTaggedSeed(tag||hash||which) for SA/SB/SU/SV. Byte-exact to
     matmul_v4_bmx4.cpp (the derived seeds reversed == the c13_bench golden hex seeds).
  2. GPU M11/E8M0 sampler (NEW kernels, byte-exact port of host bmxA/bmxB/bmxProj):
     - bmx_mant_count/scatter: wide counter-mode SHA XOF (domain 'm'=0x6D), M11 E2M1
       nibble decode {0,+/-1,+/-2,+/-3,+/-4,+/-6} with REJECTION sampling (11/16),
       low-nibble-first, compacted via preallocated cub::DeviceScan (NOT thrust --
       thrust's per-call malloc serializes the stream).
     - bmx_scale: E8M0 2-bit scale stream (domain 'e'=0x65), no rejection, direct
       index map (block i/128, byte (i%128)/4, shift ((i%128)%4)*2).
     - bmx_dq_A/bmx_dq_B: Dequant mu<<e, scale per-32 along COLS (A) / ROWS (B).
     - SeedBytesLE reversal in-kernel (reference convention): the host bench does NOT
       reverse internally, so the byte-exact gate feeds the host the reversed seed.
  3. P=U*A, Q=B*V (cuBLASLt INT8) -> bmx4_split/split_qh -> ONE fused 4-base-64 GEMM
     -> bmx4_reconstruct (the 1.78x deploy combine, 4d8ca25).
  4. device chained SHA256d over the raw device Chat (SerializeSketch LE64 == device
     bytes, zero-copy); sigma hashed as sigma.data() (SHA256d output order, NOT reversed).

RESULT (5090, single-stack):
```
  n=256 nonce=1  sampler mism A=0 B=0 U=0 V=0  digest=4e192d8b...4f54a9f9  MATCH
  n=256 nonce=2  sampler mism A=0 B=0 U=0 V=0  digest=91fe8b67...25f4aeed  MATCH
  n=128 nonce=1  sampler mism A=0 B=0 U=0 V=0  digest=c9492380...ade67b7c  MATCH
  => 3/3 goldens matched (header->digest, GPU sampler, ready for SolveMatMulBmx4C)
```
These are the reference's OWN three golden vectors (matmul_v4_bmx4_tests.cpp
golden_digests), reproduced end-to-end on GPU. The sampler is independently gated
byte-for-byte against the host reference (A/B/U/V = 0 mismatch) so the digest match
is not a coincidence of a compensating error.

TWO bugs found and fixed on the way (both mine, both would have bitten the miner port):
- Combine buffers (dPlimb/dQlh/dG) were gated on `c13||bmx4c`; --solve set neither ->
  bmx4_split wrote a NULL pointer. memcheck could not localize it (null-deref in a
  library-launched context) and the fault MOVED as syncs were added -- the tell was
  a moving illegal-access, not a fixed-address OOB. This also explains the earlier
  "n=128 fused GEMM fail": same null buffer, not a cuBLASLt small-shape limit.
- sigma over-reversed into the S4 digest header (fed reverse(sigma.data())). Fixed to
  feed sigma.data() (SHA256d output order) directly -> instant 0/3 -> 3/3.

STILL TO WIRE (miner side, tasks 3-5): port these host derivations + GPU kernels into
a clean-stack CUDA module, replace the fail-closed SolveMatMulBmx4C stub (nonce loop +
target compare + payload/share_sink), a bmx4c_probe golden gate, and regtest block
acceptance against the PR-89 node. The COMPUTE is now proven; the rest is plumbing.
