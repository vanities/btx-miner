# FP4-on-S2 verdict: BMX4-C's E8M0 scale blocks run along the WRONG axis for block-scaled tensor hardware (2026-07-18)

**One line:** the reference's E8M0 scales vary **per-element along the contraction (K)** in BOTH
consensus GEMMs, which no block-scaled tensor instruction can express — so the hand `kind::mxf4`
operand GEMM (validated below, byte-exact at the tile level) **cannot be fed BMX4-C's actual
operands under current consensus bytes**. FP4-on-S2 ceiling under the current spec: **0%**
(supersedes the earlier "+4.7% FP4 rides S2" line, which reasoned from the mantissa alphabet only
and never checked the scale-block axis).

## The geometry (from the reference @ a5b3588, verbatim semantics)

The consensus GEMMs (`src/matmul/matmul_v4.cpp`):

- `ComputeProjectedLeft`:  `P[a][k] = sum_i U[a][i] * A[i][k]`  -> contraction = `i` = **A's rows**
- `ComputeProjectedRight`: `Q[k][c] = sum_j B[k][j] * V[j][c]`  -> contraction = `j` = **B's columns**

The scale applications (`src/matmul/matmul_v4_bmx4.cpp`, `ExpandOperandA/B`):

| operand | scale layout (code) | blocks run along | contraction axis | scale along K |
|---|---|---|---|---|
| A | `scale[i*(n/32) + k/32]`, n x (n/32) | columns (free) | rows | **per-element** ✗ |
| B | `scale[(k/32)*n + j]`, (n/32) x n | rows (free) | columns | **per-element** ✗ |

MX-format hardware block scale (the entire mxf4/mxf8f6f4/tcgen05/cuBLASLt family) is by
definition **one UE8M0 per 32 elements along K**. Per-element-along-K scaling is exactly the
thing block formats cannot do. Therefore neither operand of either GEMM is expressible.

## It looks like an axis-swap BUG in the reference, not a design choice

1. **The reference's own comments claim the blocked axis IS the contraction** ("blocks along
   columns (contraction)" on A, "blocks along rows (contraction)" on B). Both claims are false
   in the code. The comments document the intent; the indexing delivers the transpose.
2. **The two layouts are exactly swapped.** For hw-FP4:
   - B (mma A-operand in Q=B*V) needs per-row x per-32-along-K = `scale[k][j/32]` = n x (n/32)
     — **the layout A currently has**.
   - A (mma B-operand in P=U*A) needs per-32-along-K x per-col = `scale[i/32][k]` = (n/32) x n
     — **the layout B currently has**.
   Swap the two indexings and BOTH consensus GEMMs become hardware-MX-native, matching the
   comments.
3. **Bug mechanism (near-certain):** both expand loops name their *free* axis `k` — the same
   letter the GEMM loops use for their *output* index — and block along the variable named `k`
   believing it is the contraction. Classic index-naming slip.
4. **The design clearly intends hw-FP4:** M11 is the E2M1 exact-int subset by construction, the
   design validates the M-t24 accumulator-exactness bounds (2304n / 288n / 1024n, all < 2^24 =
   the f32-tensor-accumulator exactness condition), and the reference backend carries an mxf4
   qualification harness (`RunMxf4Qualification`). The reference never *wired* a native FP4
   kernel ("device native kernel wired: no"), so the axis swap was never hit.

## No workaround exists under current bytes (exhaustive)

- **Fold scale into elements:** scaled values reach ±48; E2M1 tops out at ±6. ✗
- **Fold into the other operand (per output-row-band):** moves the same per-element-K scale onto
  V's rows (still K). ✗ (Also 128 per-band V copies = memory explosion.)
- **Masked split by scale value (s in {0..3}):** 4 full GEMMs = 4x MACs at 2x rate = net 2x
  LOSS. Binary split (2^s = 2^s0 * 4^s1): intermediate values 8,12 exceed E2M1. ✗
- **E4M3 (FP8) elements:** all scaled values {0..48} are exact in E4M3, but FP8 rate ≈ INT8
  rate on sm_120 → 0 gain. ✗
- **Residual decomposition (choose hw scales freely, represent B as sum of E2M1 planes):**
  needs ≥4 planes to cover the 48/12 dynamic spread inside a 32-block → ≥2x MACs. ✗
- **Scale-sorted column permutation:** s depends on (row-band, column); one permutation per
  band scrambles every other band. ✗

## What IS banked and ready (commit a8cec29)

The hardware layer is fully validated on the 5090, byte-exact vs an integer reference:
- `fp4_mxf4_tile_test.cu`: m16n8k64 `.e2m1` A/B fragment layout (int4-double-K derivation) +
  **f32 accumulator byte-exactness on varied M11 operands** (128/128).
- `fp4_mxf4_scale_test.cu`: SFA layout for `scale_vec::2X` nailed empirically — contributing
  threads `tig in {0,1}` (tig0 -> row gID, tig1 -> row gID+8), sA low 2 bytes = the two
  32-K-block UE8M0 scales, all other bytes MUST be 0x7F (a 0x00 byte = 2^-127 annihilates the
  tile). **This test's scale semantics (per-row x per-32-K) are exactly the swapped-B layout**
  — i.e. it is already the end-to-end demo that a corrected operand feeds the hardware
  byte-exactly.

**If the reference swaps the two scale indexings (a one-commit, pre-activation consensus fix
that changes goldens), the +4.7%-class FP4 lever comes back to life, and this recipe drops
straight in.** Until then: S2 stays INT8 (cuBLASLt), which is optimal under current bytes.

## Sequencing note

v4 is **pre-activation** — this is the cheap moment for the upstream fix. Whether/when to
report the axis swap to the reference author is a strategic call (it reveals we are building
FP4 kernels; it also hands every sm_120 competitor the same door the day they hand-write mxf4
— which none has today, and cuBLASLt will not give them).

## UPDATE (same day): the swap is CONFIRMED CORRECT on silicon, end-to-end

`fp4_mxf4_swapped_e2e.cu` builds the POST-FIX operands (B scale = `sc[k][j/32]`, per-row x
per-32-along-K; V raw) at the REAL S2 shape (Q[4096x1024] = B[4096x4096]*V, K=4096) and runs
both paths on the 5090:

- ground truth: validated row-major cuBLASLt INT8 on the dequantized s8 operands
- FP4: hand tiled `kind::mxf4` GEMM (packed E2M1 nibbles + hw SFA scales, f32 accum)

**BYTE-EXACT MATCH: 4,194,304 / 4,194,304 elements** (also seed 2, and off-shape n=2048
m=512). So the one-line swap makes BMX4-C's S2 *exactly* computable by consumer-Blackwell FP4
tensor cores — the design's advertised FP4-native property becomes literally true.

Perf (same run): naive global-load mxf4 kernel = 0.363 ms / 94.6 TOPS vs cuBLASLt INT8
0.088 ms / 390 TOPS. That naive floor is expected (no smem staging / double-buffering);
the mxf4 issue-rate ceiling is 1914 TOPS (2x INT8's 969), and cuBLASLt itself only achieves
~40% of the INT8 ceiling at this shape — a tuned mxf4 kernel at the same ~40% efficiency
would land ~766 TOPS > the INT8 path's 390-516, i.e. the S2 win is plausible post-fix but
requires a real (smem/double-buffered) kernel. Build that ONLY if the consensus swap lands.
