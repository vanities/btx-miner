# BTX MatMul v4 — investigation (PR #89)

Source of truth for the proposal: `btxchain/btx` PR #89
(`RFC: MatMul v4 proof-of-work — design spec + reference implementation`,
author `numair`, opened 2026-07-15, branch `claude/matmul-v4-design-spec-af23sj`).
Local clone of that branch used for this work:
`<scratchpad>/btx-v4/` (re-clone with the branch above).

**Status of the proposal:** RFC, mainnet **disabled** (`nMatMulV4Height =
INT32_MAX`). Enabled on **testnet at height 200,000** and **regtest at height
100** (overridable). Merge = Gate A (disabled); mainnet activation (Gate B) is
gated on CUDA + Apple-M5 on-hardware determinism PASS + audit + testnet burn-in.

---

## TL;DR — the headline

v4 replaces the v3 SHA-lottery-gated matmul with **one dense INT8 `s8×s8→s32`
matrix multiply per nonce**, on tensor cores, verified in O(n²) by Freivalds
over `q = 2⁶¹−1`. The *stated goal* is to make **datacenter INT8 accelerators
(H100/B200) win** and de-rate consumer/CMP cards.

**As written, the reference does not achieve that goal.** The per-nonce operands
`A,B` (n×n) and projectors `U,V` are expanded by **one full SHA-256 per matrix
element** (`SampleBalancedS8FromOracle`, `src/matmul/int8_field.cpp:60`), using
only `hash[0]`. At the launch dimension n=4096 that is **~37.7 million SHA-256
compressions per nonce**, versus ~1.7×10¹⁰ INT8 MACs for the actual matmul.

> **Verified on the reference's own code** (`refcheck/`, not a reimplementation):
> compiling the *unmodified* `int8_field.cpp` + `matmul_v4.cpp` with a SHA counter
> and running `ExpandOperand` at n=4096 fires **38,499,212 SHA-256 calls per nonce**
> (= 2n²+2mn × 1.02, the 1.02 = rejection retries). Their `ComputeSketchOptimal`
> also proves **byte-identical** to the full-C path. See `RESULTS.md`.

Measured on the real hardware (this repo's benches, below):

| platform | operand-gen (SHA) | **INT8 matmul** | Fq combine (mod q) | nonce/s |
|---|---|---|---|---|
| **RTX 5090** (has INT8 tensor), n=4096 | **62.9 %** | **1.6 %** | 35.5 % | 261 |
| M4 Max CPU (no INT8 tensor), n=4096 | 43.3 % | 47.3 %¹ | 9.4 % | 1.1 |

¹ 47 % only because the M4 has **no INT8 tensor path** so the matmul runs on
scalar ALU (~1000× slow). On any device that actually *has* the INT8 hardware
v4 targets (5090 here, Apple M5), the matmul collapses to ~1–2 % and the work is
SHA-bound.

**Conclusion:** on the exact hardware v4 is meant to reward, the INT8
tensor-core GEMM is **1.6 % of per-nonce work — tensor cores idle 98.4 %**. The
bottleneck resource is **SHA-256 throughput** — precisely the resource v4 set out
to demote. Consumer/CMP-class fast-SHA silicon stays competitive or dominant; the
H100/B200 INT8 advantage (which only touches that 1.6 %) is washed out. The
design's own §A.2 ("expansion … is subdominant to the per-nonce GEMM") is an
accounting error: it compares operand **bytes** (n²) to GEMM **MACs** (n²m) and
never prices the ~10⁴× cost gap between a SHA compression and an INT8 MAC.

This is a legitimate, load-bearing review point for an RFC that explicitly asks
for it, **and** it is strategically decisive for matador (see below).

---

## Update log — PR #89 collaboration + C-13 (2026-07-15)

- **numair confirmed the finding and fixed the XOF** (`f50f0f8`): wide counter-mode
  stream, **32× fewer SHA** (verified on his own code — RESULTS.md). Determinism intact.
- **Harness contributed as btxchain/btx PR #90** (`contrib/matmul-v4/stage-bench/`,
  PUBLIC, neutral): the §K.2a-WT wall-time check, bit-exact to the reference digest.
- **C-13 (limb-decomposition combine) prototyped PRIVATELY** (`bench/c13_bench.cu`,
  `bench/c13_proof.cpp`): the mod-q combine as 25 s8→s32 tensor GEMMs, byte-exact
  (5090 + H100). A **real speedup for our miner** (5090 n=8192: 89→161 nonce/s) but it
  **does NOT flip the per-card ranking** — the 5090 still beats the H100 ~2.5× at
  C-13 + n=8192 (their planned activation config). Even *both* fixes (wide XOF + C-13)
  leave v4 consumer-favoring: operand-gen (SHA/memory) + small/skinny GEMM shapes
  reward high-clock consumer cards. Told numair straight (PR #89 reply) with the
  not-yet-optimized caveat; **kept the optimized kernel private** (matador edge).
- **Orchestrator**: `bench/v4bench.sh` (`up`/`run`/`report`/`down`) — parallel 5090 (pc)
  + Vast datacenter GPU, wide + c13 sweep, per-card/$/joule report. Uses the private
  `c13_bench.cu` internally; the public PR ships only `v4_proto.cu`.

## Measurements (reproducible)

### `bench/v4_microbench.cpp` — single-thread CPU work-shape (any machine)
Mirrors the consensus derivation exactly (SHA-per-element) + the optimal-miner
GEMMs `P=U·A`, `Q=B·V` + the mod-q combine. Proves the split; the 37.7 M SHA
count is exact (derivation-defined).
```
clang++ -O3 -std=c++17 v4_microbench.cpp -o v4_microbench   # mac (CommonCrypto)
./v4_microbench 1024      # → operand-gen 91.8 %, GEMM 4.3 %, combine 3.9 %
```

### `bench/v4_microbench_mt.cpp` — full-chip CPU (M4 Max)
```
./v4_microbench_mt 4096    # → 388ms SHA / 425ms scalar-GEMM / 84ms combine = 1.1 nonce/s
```

### `bench/v4_proto.cu` — matador-v4 GPU hot-loop (5090, sm_120)
Does the **whole** hot path on-GPU (unlike the PR backend, which expands operands
on the host): device-SHA operand-gen kernel + cuBLASLt INT8 `P=U·A`,`Q=B·V` +
mod-q combine kernel, per-stage CUDA-event timing. `--verify` diffs cuBLASLt vs a
scalar GEMM (correctness).
```
# build in the matador-build:pathb-deps-cm4 container (CUDA 13.3):
nvcc -O3 -arch=sm_120 v4_proto.cu -lcublasLt -o v4_proto
./v4_proto 256 4 --verify   # → 0 mismatches OK   (cuBLASLt INT8 == scalar, byte-exact)
./v4_proto 4096 32          # → operand-gen 62.9% / INT8 1.6% / combine 35.5% = 261 nonce/s
```
Raw pc run (single-stack, miner stopped) in `RESULTS.md`.

**Caveats / honesty:**
- The GPU SHA kernel is a first cut (~15.6 GH/s effective). A tuned SHA (matador's
  scan pipeline does ~2× this) would *raise* nonce/s and *lower* the SHA share —
  but the INT8 GEMM stays ~1.6 %; SHA would have to speed up ~40× to stop
  dominating, which is impossible (SHA is fixed work). Thesis is robust.
- The mod-q combine (35 %) is a naive one-thread-per-output kernel; it is
  optimizable to sub-ms. That would push the SHA share *up*, not down.
- Byte-exactness: the prototype's digest now matches the **reference** byte-for-byte
  at n=256 AND n=4096 (`--emit` vs `refcheck emit`; see `RESULTS.md`). Full
  end-to-end **node** consensus (header→seed derivation + `submitblock` on regtest)
  remains the deploy-time step — O(1) SHA plumbing around the proven core.

---

## Testing against our node — can we? (answer: regtest yes, public testnet not yet)

- **Public testnet:** v4 is set to activate at **height 200,000** (`chainparams.cpp:567`,
  n=4096). But no *released* BTX node has v4 — it lives only in this unmerged PR.
  So the live public testnet will not validate v4 blocks until PR #89 merges, a
  build ships, testnet operators upgrade, **and** testnet passes 200k. Our pc node
  is a mainnet `v0.33.0` container and has none of this. → **not testable on the
  live testnet today.**
- **Local regtest / private 2-node net (the real near-term path): yes.** Build the
  node from PR #89 and run the bundled `test/functional/feature_matmul_v4_activation.py`
  — it mines across the fork on regtest (v4 height 8, n=128), proves a corrupted
  sketch is rejected and the honest one accepted. Then stand up a regtest node at
  n=4096 with a low `-regtestmatmulv4height`, point the prototype at it via
  GBT/`submitblock`, and gate on **byte-exact digests vs the reference** before any
  perf claim (the matador way). This also gives us the real activation ASERT-rescale
  behavior to study.

See `TESTING.md` for the exact build + regtest recipe (to be filled as we run it).

---

## Difficulty behavior (v3 vs v4) — and a benchmarking win

Confirmed in `SolveMatMulV4` (`src/pow.cpp:4092`): v4 fixes `matmul_dim =
nMatMulV4Dimension` (4096, not difficulty-scaled), runs the **full** digest on
every nonce (no pre-hash gate), and accepts iff `digest ≤ target`. So:

| | v3 | v4 |
|---|---|---|
| pre-hash gate | yes (ε=18, block-target-keyed) | **none** |
| per-nonce work | cheap SHA scan; matmul on ~1/2¹⁸ | **full GEMM every nonce, fixed n** |
| nonce/s vs difficulty | **∝ 1/net-diff** (gate spacing widens) | **flat** — hardware constant (~261/s on 5090) |
| block/share rate | falls with difficulty | ∝ 1/difficulty (standard PoW) |

**Benchmarking implication (important for our A/B method):** in v3, difficulty
drift confounds a nonce/s A/B (nonce/s ∝ 1/net-diff), which is *why* the
freeze-job bridge pins net-diff. In **v4, nonce/s is difficulty-invariant** — a
lever A/B measured as nonce/s does **not** need net-diff pinned to be comparable
(share/block *cadence* still ∝ 1/difficulty, so pin diff only if measuring that).
v4 lever A/Bs are inherently cleaner: measure a card's nonce/s once, it holds at
any difficulty. And difficulty (ASERT) tracks delivered work rate, which is ~98%
SHA — so network difficulty in v4 effectively tracks aggregate SHA throughput,
not INT8 (the finding, seen through the difficulty lens).

## What this means for matador (strategy)

Two scenarios, both good — but opposite prep:

1. **If v4 ships with the per-element SHA XOF (as written):** matador is *extremely*
   well positioned. v4 mining = fast GPU SHA operand-gen (matador's scan pipeline is
   already the field's fastest GPU SHA) + a tiny INT8 GEMM (cuBLASLt/CUTLASS) + a
   mod-q combine. The 5090 stays competitive-or-better vs H100. matador-v4 ≈ the
   v3 skillset re-pointed. The scan-kernel SHA work we've banked (padfold, sigma0,
   pairshare) is **directly reusable** on the operand-gen kernel.
2. **If the XOF is fixed to a wide/stream form (many bytes per SHA, ~32–64× fewer
   hashes):** SHA drops to ~0.3–0.6 ms/nonce and the INT8 GEMM (~0.2 ms) becomes
   comparable — *then* datacenter INT8 starts to matter, and matador's edge shifts
   to INT8 GEMM efficiency + operand-bandwidth. Still squarely in matador's wheelhouse
   (we already run the digest matmul), but a different tuning target.

**The pivot is entirely the operand-XOF cost — and it's an open RFC.** Worth a
technical comment on PR #89 (the SHA-bound measurement), both because it's correct
and because *which way they resolve it decides who wins v4.*

### Carries over vs rewrite
| matador component | v4 status |
|---|---|
| GPU SHA-256 scan pipeline (scan kernels) | **Reusable** — becomes the operand-gen kernel (the 63 % bottleneck) |
| Digest matmul infra / cuBLAS-style GEMM plumbing | **Reusable** — becomes `P=U·A`, `Q=B·V` INT8 GEMMs |
| Feeding / host-overlap / job pipeline | **Reusable** — same stratum/GBT loop |
| v3 pre-hash-gate scan logic, low-rank, transcript | **Dead** — v4 removes the ε-gate and low-rank entirely |
| mod-q (2⁶¹−1) combine kernel | **New** — small, but currently 35 %; a real optimization lever |
| Byte-exact digest gate (`digest_probe`) | **New v4 digest** — `H(sigma‖Ĉ)`; needs a v4 KAT vs the reference node |

---

## Files
- `bench/v4_proto.cu` — GPU per-stage harness (bit-exact wide stream, `--emit`/`--verify`/`--wide`). PUBLIC (== PR #90).
- `bench/c13_bench.cu` — **PRIVATE**: `v4_proto.cu` + `--c13` limb-decomposition combine (25 s8 tensor GEMMs). Never contribute.
- `bench/c13_proof.cpp` — **PRIVATE**: CPU proof that the C-13 limb combine is byte-exact to the mod-q combine.
- `bench/v4bench.sh` — cross-hardware orchestrator (5090 + Vast, wide/c13 sweep). PRIVATE.
- `bench/v4_microbench.cpp` / `_mt.cpp` — CPU work-shape proofs (single-thread / full-chip M4).
- `refcheck/` — instruments the reference's OWN code (SHA count, optimal==full-C, bit-exact digest).
- `RESULTS.md` — raw measured runs (CPU, 5090, H100, C-13 flip-test).
- `FINDINGS-shareable.md` — public review write-up (== the fork doc / PR #90 context).
- `pr-comment.md` / `pr-reply-c13.md` / `pr-harness-README.md` — public PR prose (posted).
