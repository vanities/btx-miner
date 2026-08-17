# Replacing the static cuBLASLt link with CUTLASS (2026-08-05)

Goal: the shipped binary is 598 MB, of which ~530 MB is a statically linked cuBLASLt carrying
2,785 precompiled kernels over 11 archs plus ~293 MB of dispatch tables and CASK descriptors --
for a call surface of exactly ONE int8 GEMM shape family. Replace it with CUTLASS compiled for
only the configs we use.

**Outcome: size achieved, speed NOT achieved.** Binary 598 -> 95 MB, HiveOS package 403 -> 47 MB
(-88%), every correctness gate green, episode **-2.5%**. Shipped? NO -- see the verdict.

## Measured results

| | binary | HiveOS pkg | episode | gates |
|---|---|---|---|---|
| v0.9.2 shipped | 598,337,464 | 403 MB | baseline | pass |
| nvprune only | 516,630,456 | 322 MB | unchanged | pass |
| CUTLASS, cuBLASLt unlinked | 95,228,472 | 47 MB | **-2.5%** | pass |

Gates: `digest_probe` 1dad86f1..., `rc_gpu_accel_probe` 5b1bff3c..., production episode digest
42e74cd6542d86bd identical on both backends, `rc_gemm_i8_probe` bit-equality on every shape.

## Per-shape, against TUNED cuBLASLt (profile-1 dims)

| shape | m,n,k | calls/ep | best CUTLASS | gap |
|---|---|---|---|---|
| FFN X.W_up | 16384, 16384, 4096 | 64 | 256x128x64 s4 | **+0.1 .. +1.5%** |
| FFN H.W_down (beta=1) | 16384, 4096, 16384 | 64 | 128x256x64 s4 swz8 | **-6.8 .. -7.5%** |
| attn Q.Kt | 512, 786432, 128 | 4 | 64x256x64 s4 swz8 | -0.5 .. -3.4% |
| attn S.V | 512, 128, 786432 | 4 | streamk 64x128x64 s4 | -24 .. -34% |

FFN H.W_down is essentially the entire remaining gap: -7% x 64 calls x 3.4 ms = ~17 ms of a
912 ms episode.

## What moved the number

- **Benchmarking against the right baseline.** The probe passed `algo=nullptr` (NVIDIA's default
  heuristic) while the episode runs `BTX_RC_GEMM_TUNE` (+4.4%, 2026-07-31). The bench claimed
  -2.7% on configs the episode measured at -9%. **Any cuBLASLt comparison must tune the baseline.**
- **Threadblock swizzle.** `GemmIdentityThreadblockSwizzle<8>` took FFN down from -15.8% to -7.1%.
  Pure L2-reuse ordering, no effect on results. The single biggest win of the campaign.
- **Split-k for attn S.V.** m=512 n=128 k=786432 yields ONE threadblock under a 128x256 tile:
  one SM grinding a 786k contraction, 169 idle. Split-k (64 slices) and stream-k both fix it.
- **Tuning under the real beta.** Selecting under beta=0 and applying to the beta=1 call picks
  for the wrong bottleneck.
- **min-of-N with a cross-candidate warm-up.** The first tuner used 3 back-to-back reps with no
  warm-up and picked wrong on BOTH FFN shapes -- clock ramp swamped the 5-10% it was resolving.

## What did NOT move it (do not re-try without new information)

- **Native RowMajor B (no transpose).** Does not compile: CUTLASS has no int8 TensorOp
  instantiation for `LayoutB=RowMajor` (`DefaultMmaCore` rejects it). This is `mma.m16n8k32`
  requiring a k-major B fragment, not a CUTLASS gap. cuBLASLt hits the same wall and solves it
  the same way internally.
- **Stream-K on FFN down.** Disproved the wave-quantization theory outright. The grid is
  128x16 = 2048 tiles / 170 SMs = 12.05 waves, and the tail arithmetic (7.3%) matched the deficit
  almost exactly -- but eliminating the tail changed nothing. The number was a coincidence.
- **Deeper K-tile (TBK=128).** Blocked by hardware: sm_120 caps shared memory near 99 KB/block
  and 128x256x64 s4 at 98 KB is already at the ceiling. Every TBK=128 config that fits is worse.
- **CUTLASS 4.6.1 vs 3.5.1.** Identical. Consumer Blackwell has no tcgen05; its int8 path IS the
  Ampere-style `mma.sync` we already emit. The sm_120 collectives are for block-scaled narrow
  precision (nvfp4/mxfp8), not int8.
- **`ScaleType::NoBetaScaling`.** The beta=1 deficit is the C *load* (268 MB), not the multiply.
  cuBLASLt reads C too and still gets beta=1 free (3.406 vs 3.426 ms).
- **Skipping `can_implement` per call.** 136 GEMM calls/episode; removing the host-side validation
  changed nothing measurable.
- **beta=0 + separate `k_add_resid`.** Worse than beta=1 on the CUTLASS path too (939.9 vs 931.0 ms),
  so the existing residual-in-GEMM fusion stays.

## The beta=1 split

Same shape, beta=0 vs beta=1: **-3.1% vs -6.8%**. So 3.7 points of the FFN-down deficit are the
epilogue and ~3.1 are the mainloop. cuBLASLt pays neither.

## The transpose

INT8 IMMA needs both operands k-contiguous. A(m,k) row-major is; B(k,n) row-major is not, so B^T
is materialised per call. The vectorised 64x64 kernel (16-byte loads AND stores, byte shuffle in
smem) moves 134 MB in 0.087 ms = **~1.54 TB/s = 5090 memory bandwidth**. It cannot be made faster,
only unnecessary. Total cost ~11.9 ms/episode (1.3%).

Eliminating it means generating the sigma-derived weights pre-transposed. NOT viable as written:
the generator's writes are coalesced by construction (output offsets come from a prefix sum over
the mantissa scan), so a transposed store would scatter them and cost more than the 0.087 ms it
saves. Swapping the generator's (rows, cols) is NOT an option -- it changes the VALUES, not just
the layout, because the scale block structure is keyed on `cols`.

## Verdict

CUTLASS reaches or beats tuned cuBLASLt on three of four episode shapes. On FFN H.W_down it is
~7% behind and I could not close it from the configuration space CUTLASS exposes. -2.5% of episode
throughput is -2.5% of mining revenue once ENC_RC activates, permanently, against a one-time 356 MB
download saving. **That is the wrong trade for a miner** -- rig operators download once and mine
for months.

Ship the nvprune build (322 MB, zero speed cost, gates green). Keep this branch
(`build/cublaslt-prune`) warm. Reasons to revisit: a CUTLASS release whose sm_120 int8 collectives
beat the Ampere path; profile-2 activation (`share_ep=true` generates weights ONCE per episode, so
the transpose hoists and the shape mix changes); or fusing the residual add into the following
Extract, which would delete both the `k_widen8_32` pass and the beta=1 C read (~536 MB/layer) --
but that is an episode-level change that helps cuBLASLt equally, so it offsets the gap rather than
closing it.
