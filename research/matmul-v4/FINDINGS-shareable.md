# MatMul v4 (BTX PR #89): the reference operand XOF makes mining SHA-256-bound

**Status:** independent technical review of `btxchain/btx` PR #89 (RFC).
**Date:** 2026-07-15.
**Scope:** performance and mining-incentive analysis of the reference implementation as written. This is not a consensus-safety claim: the design is deterministic and, as measured below, bit-exact across CPU and GPU.

---

## Summary

MatMul v4's stated goal is to make a dense INT8 tensor-core matrix multiply the sole per-nonce unit of work, so that datacenter accelerators (H100/B200) win on cost-per-block and commodity/consumer cards are de-rated.

As implemented in the PR, mining does not become matmul-bound. It becomes **SHA-256-bound**. The per-nonce operands are expanded at a rate of **one SHA-256 compression per matrix element**, which at the launch dimension n=4096 is about **37.7 million SHA-256 per nonce**, versus roughly 1.7e10 INT8 MACs for the matmul the design is meant to reward.

Measured on an RTX 5090, at n=4096, the per-nonce cost splits as:

| stage | share of per-nonce time | hardware it uses |
|---|---|---|
| operand expansion (SHA-256) | **62.9%** | integer / SHA units |
| **INT8 matmul (the intended work)** | **1.6%** | **tensor cores** |
| mod-q sketch combine | 35.5% | integer ALU |

The INT8 tensor-core GEMM is about **1.6% of per-nonce work; the tensor cores are idle roughly 98% of the time.** Because the binding resource is SHA-256 throughput (where consumer and prior-generation mining cards are strong) rather than INT8 TOPS (where datacenter parts are strong), the intended hardware inversion does not occur with the reference as written. The good news: the cause is a single, localized implementation choice in the operand expansion, and it is straightforward to change while preserving the design's determinism.

---

## What v4 sets out to do

From the design spec (section 0.2): "v4 makes the dense matrix multiplication the sole, unavoidable, per-nonce unit of work, executed on low-precision tensor cores, and scales the problem so that throughput is bounded by tensor-FLOPS." The v3 SHA-256 pre-hash "epsilon gate" is removed so that "SHA is limited to seed derivation and final sealing," and every nonce is meant to pay one dense INT8 GEMM.

The verification path is elegant and, as far as these tests go, correct: an O(n^2) Freivalds check over the independent prime q = 2^61 - 1, with the default payload a compressed sketch Chat = U * C * V. The optimal miner evaluates Chat = (U * A)(B * V) directly at about 2 n^2 m MACs, which is still a dense INT8 GEMM of the same hardware profile.

---

## The finding

The operands A, B (each n x n) and the sketch projectors U (m x n) and V (n x m), with m = n/8, are expanded from their seeds by a per-element oracle:

- `matmul::v4::ExpandOperand` / `ExpandProjector` loop over every element index and call `int8_field::SampleBalancedS8FromOracle(seed, index)`.
- `SampleBalancedS8FromOracle` (`src/matmul/int8_field.cpp`) computes a full `SHA-256(seed_bytes[32] || index_le[4])` and consumes **only the first output byte** (`hash[0]`), rejection-sampled into a balanced s8 value in [-125, 125]. On rejection (about 2% of draws, byte >= 251) it hashes again with an appended retry counter.

So each of the roughly 2 n^2 + 2 m n operand elements costs at least one SHA-256 compression. At n=4096 that is `2*4096^2 + 2*512*4096 = 37,748,736` compressions per nonce, before rejection retries.

Note that even the PR's CUDA backend (`src/cuda/matmul_v4_accel.cu`) performs this expansion on the host CPU and copies the operands to the device; only the three GEMMs run on the GPU. Its own header comment states: "The operand derivation ... [is] done on the HOST ... Only the three GEMMs run on the GPU."

---

## Evidence

All of the following is reproducible; the harnesses are described at the end.

### 1. Instrumented run of the reference's own code

Compiling the **unmodified** `int8_field.cpp` and `matmul_v4.cpp` from the PR, with a counter wired into the SHA-256 `Finalize`, and driving `ExpandOperand` / `ExpandProjector` directly:

```
SHA-256 KAT ("abc"): PASS

n=256    operand-expansion SHA-256 calls = 150,375     (2n^2+2mn = 147,456; ratio 1.020)
         ComputeSketchOptimal (U*A)(B*V) == full-C U*(A*B)*V : BYTE-IDENTICAL

n=4096   operand-expansion SHA-256 calls = 38,499,212  (2n^2+2mn x 1.020)
         vs 1.72e10 INT8 MACs for the matmul
```

The 1.020 factor over 2n^2 + 2mn is the roughly 2% rejection-sampling retries. This is the reference's own logic; the counter only observes it.

### 2. Measured stage split on an RTX 5090

An independent GPU implementation that performs the whole hot path on-device (a SHA-256 operand-generation kernel, cuBLASLt INT8->INT32 GEMMs for P = U*A and Q = B*V, and a mod-q combine kernel), with per-stage timing:

```
n=2048   SHA 74.5% | INT8 GEMM 2.8% | combine 22.7%   -> 1207 nonce/s
n=4096   SHA 62.9% | INT8 GEMM 1.6% | combine 35.5%   ->  261 nonce/s
n=8192   SHA 47.5% | INT8 GEMM 2.5% | combine 50.1%   ->   49 nonce/s
```

The INT8 tensor-core GEMM stays at 1.6 to 2.8% across the entire legal dimension range (the spec bounds n to [4096, 8192] on production nets). As n grows, work shifts from SHA to the mod-q combine, both of which run on integer units, not tensor cores; it does not shift toward the tensor GEMM.

### 3. Byte-exact validation of the measurement

To confirm the GPU implementation measures the real v4 work and not an approximation, it reproduces the reference digest for fixed shared seeds:

| n | digest H(sigma \|\| Chat), reference vs GPU |
|---|---|
| 256 | `e2b873d6a41ceca766783f69a73ae91970344842b6ce6a2992188019297ffc75` (identical) |
| 4096 | `d5796a095238ccd0fcde8ba7c5e0507dbc933989fc74484366d168ea049f9230` (identical) |

The GPU path (device SHA operand-gen, cuBLASLt INT8, mod-q combine, LE64 serialize, SHA256d digest) is bit-identical to the CPU reference at the production dimension.

### 4. The design's own timing estimate is consistent

`ACTIVATION.md` item B2d budgets an "Operand XOF regen timing envelope (15-35 ms)." That is the **verifier** regenerating A and B once for the Freivalds check. A miner performs the same expansion on **every nonce**, so 15 to 35 ms per nonce implies a per-card throughput on the order of tens of nonces per second dominated by SHA, consistent with the measurements above.

---

## Why it matters

The design intent is that difficulty tracks INT8 tensor-FLOPS, so that a datacenter part with roughly 2.4x the INT8 throughput of a consumer card earns proportionally more. But if the INT8 GEMM is about 1.6% of per-nonce cost, that 2.4x advantage applies to 1.6% of the work and is washed out. The remaining roughly 98% is SHA-256 plus a 61-bit modular combine, both of which run on general integer units where high-clock consumer cards (and, for the SHA portion, retired ASIC-class mining cards) are competitive or better.

In other words, the reference as written re-creates a SHA-256-bound proof of work, which is the property v4 set out to remove.

---

## Root cause

Two things combine:

1. **The XOF emits one usable byte per SHA-256.** `SampleBalancedS8FromOracle` hashes a fresh `(seed, index)` preimage per element and keeps only `hash[0]`. A SHA-256 compression produces 32 bytes; 31 of them are discarded.

2. **The cost model in the spec counts bytes, not time.** Section A.2 argues expansion "is subdominant to the per-nonce GEMM cost" by comparing operand **bytes** (order n^2) to GEMM **MACs** (order n^2 m). But one SHA-256 compression and one INT8 MAC differ in hardware throughput by roughly four orders of magnitude, so an n^2 term of SHA compressions dominates an n^2 m term of INT8 MACs at the relevant sizes. The per-element SHA is the cost, and it was not priced.

---

## Remediation (design-level, determinism-preserving)

The fix is to raise the number of usable bytes produced per SHA-256, so that operand generation stops dominating:

- **Widen the XOF.** Derive the operand stream in a counter / squeeze mode (for example `SHA-256(seed || block_counter)` yielding 32 bytes per compression, rejection-sampled from that stream), rather than one hash per element. This cuts operand-gen compressions by roughly 32x (more with a wider primitive), which on the measurements above brings SHA operand-gen down to the same order as the INT8 GEMM. Determinism is fully preserved: it is still an exact-integer, byte-reproducible XOF; only the hash-to-byte ratio changes. It does require making the rejection-sampling index-to-stream mapping canonical, which is a small, well-understood consensus detail.

- **Then re-check the balance.** A wide XOF is necessary but, on its own, only brings SHA and the GEMM to the same order at n=4096. For the intended datacenter-favoring ordering to actually hold, the dense INT8 GEMM must clearly dominate per-nonce cost, so it is worth choosing the XOF-to-compute ratio (and possibly n, or the sketch tile b) so that the matmul is the majority of the work after the XOF fix.

- **Note the combine stage.** The mod-q sketch combine (P*Q over q = 2^61 - 1) is roughly 35% of per-nonce time here and runs on integer ALU, not tensor cores, so it does not favor datacenter parts either. It is also amenable to optimization, but as a general-integer workload it should be counted on the "does not advantage datacenter" side of the ledger when calibrating.

None of this affects the verification design (Freivalds over q), the digest form, or the determinism guarantees; it is confined to how operand bytes are produced and to difficulty calibration.

---

## Update: widening the XOF is necessary but not sufficient

Building a cross-hardware benchmark (both XOF variants, per-stage timing, board power) surfaced a second issue. Widening the XOF removes the SHA bottleneck, but the per-nonce cost then moves to the mod-q sketch combine (`Chat = P*Q` over `q = 2^61-1`), which also runs on integer ALU, not tensor cores. The INT8 tensor GEMM never rises above about 6% of per-nonce work in any configuration measured (n = 4096 or 8192, legacy or wide XOF, on an RTX 5090 and an H100).

Measured stage split (RTX 5090, n=4096):

| XOF | operand-gen (SHA) | INT8 matmul | mod-q combine |
|---|---|---|---|
| legacy (per element) | 62.9% | 1.6% | 35.5% |
| wide | 23.5% | 3.7% | 72.8% |

The structural reason: the sketch enforces only `2*n^2*m` INT8 MACs (the optimal miner evaluates `(U*A)(B*V)` directly, per section 0.7-(3)), which on tensor cores is about 0.06 ms at n=4096, a rounding error next to the XOF (order `n^2`) and the mod-q combine (`n*m^2` 61-bit modular multiplies on integer ALU). No commit tile `b` fixes it: the combine-to-GEMM time ratio scales like `363/(2b)` on this hardware, so INT8 only overtakes the combine at `b > 180`, where the GEMM itself has shrunk to nothing. Making the INT8 GEMM dominant appears to require the full-C profile (enforce the full `n^3` product), which the sketch was chosen to avoid for payload and verification cost. That is a real tension between cheap verification (sketch) and being INT8-compute-bound (full-C).

Cross-hardware check (H100 SXM vs RTX 5090, both bit-exact to the reference digest, which also confirms cross-architecture determinism across sm_90 and sm_120):

| card | n=4096 legacy | n=4096 wide |
|---|---|---|
| RTX 5090 (consumer) | 261 nonce/s | 534 nonce/s |
| H100 SXM (datacenter) | 92 nonce/s | 151 nonce/s |

With these first-cut kernels the H100 is about 2.8x slower per card, because most of the work is on the integer and memory paths where the 5090's higher clocks win, and the INT8 slice the H100 would dominate is too small to matter. Caveats worth stating up front: the combine kernel is naive, and the H100's higher HBM bandwidth could recover some of the gap with a bandwidth-tuned combine; the tall-skinny INT8 GEMM was also not tuned per architecture. The kernel-independent point still stands, since cuBLASLt is near-optimal for the GEMM and it is still under 6% of the work: the datacenter INT8 advantage cannot express while the enforced tensor work stays at `2*n^2*m`.

Net: a wide XOF is still worth doing, since it removes an ASIC-friendly SHA bottleneck, but on its own it does not make v4 compute-bound on INT8 or make datacenter parts win. That likely needs a different work-binding than the sketch.

## Behavior as difficulty rises

v4's per-nonce work is independent of difficulty. The mining loop (`SolveMatMulV4`, `src/pow.cpp`) fixes the matrix dimension to the consensus constant `nMatMulV4Dimension` (4096), computes the full digest for every nonce with no pre-hash gate, and accepts a nonce if and only if `digest <= target`. Difficulty, via `nBits` to target, changes only the acceptance threshold, not the amount of work per nonce.

Consequences:

- **Work rate (nonce/s) is flat versus difficulty.** Every nonce costs the same fixed operand expansion plus GEMM plus combine, so a card's nonce/s is a stable hardware constant (about 261/s for the RTX 5090 measured here) at any difficulty. Only the number of nonces per block changes.
- **Block-find rate scales as 1/difficulty**, exactly as in standard Bitcoin proof of work: expected nonces per block is proportional to difficulty while the work rate stays constant.

This is a cleaner difficulty profile than v3, where the pre-hash epsilon gate is keyed to the block target, so the rate at which the matmul path executes (and share cadence) falls as difficulty rises. v4 has no such coupling.

The relevant implication for the design's goals: difficulty (ASERT) tracks the delivered work rate, and the work rate is about 98% SHA-256 operand expansion. So network difficulty in v4 effectively tracks aggregate **SHA-256** throughput, not aggregate INT8 tensor throughput. That is the same finding as above, viewed through the difficulty lens: the quantity difficulty rises with is SHA, not INT8.

Finally, absolute nonce counts are not comparable across the fork: a v3 nonce is one cheap SHA behind the gate, a v4 nonce is a full matmul, so the network's headline nonce/s metric drops by several orders of magnitude at activation even as delivered compute rises. The spec acknowledges this metric rebasing.

## Reproduction

Two small harnesses (about 400 lines total, no BTX build required):

1. **Reference SHA count and optimal-vs-full-C check.** Compile the PR's unmodified `int8_field.cpp` and `matmul_v4.cpp` against thin shadow headers plus a KAT-verified SHA-256 with a `Finalize` counter; drive `ExpandOperand` / `ComputeSketch*` directly. Reports the per-nonce SHA count and confirms `ComputeSketchOptimal == ComputeSketch`.

2. **GPU stage split and byte-exact digest.** A CUDA program that runs the whole hot path on-device (SHA operand-gen kernel + cuBLASLt INT8 GEMMs + mod-q combine), with per-stage CUDA-event timing and an `--emit` mode that reproduces the reference digest for fixed seeds. Built with `nvcc -arch=sm_120 ... -lcublasLt` (CUDA 13.3).

Measurements above are from an RTX 5090 (single process) and an Apple M4 Max (CPU work-shape control).

---

## Caveats and scope

- This is a performance and incentive finding, not a correctness or consensus-safety bug. The v4 arithmetic is deterministic and bit-exact across the CPU reference and an independent GPU implementation, as validated above.
- The absolute nonce/s figures come from a first-cut GPU implementation; a tuned miner would raise throughput and lower the SHA share somewhat. It would not change the conclusion: SHA operand-gen would need to become roughly 40x cheaper to stop dominating, and since it is fixed per-element work, the only way to achieve that is to change the operand XOF in consensus (the remediation above), not to optimize the miner.
