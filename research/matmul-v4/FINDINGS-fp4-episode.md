# FP4 for the v4.6 ENC_RC episode — the real picture (2026-07-23)

**One line:** naive block-FP4 (feed consensus operands straight to FP4 tensor cores) is
axis-dead, BUT the branch's **Ozaki K-panel + native-MXFP4 MMA** route is byte-exact and is the
real question. Whether it *wins* is a per-panel FP4:INT8 throughput bet that differs by card —
undecided for the B200 until measured, and that is exactly the path numair is building.

## What's measured (RTX 5090, sm_120, cuBLASLt 13.3.0.1)

`cublaslt_fp4_ceiling.cu` — INT8 vs NVFP4 vs MXFP4 (the episode's actual format), TOPS:

| shape | INT8 | NVFP4 (VEC16_UE4M3) | MXFP4 (VEC32_UE8M0) |
|---|---|---|---|
| square 8192³ | 672 | 1463 (2.18×) | **no cuBLASLt algo on sm_120** |
| ffn-up  (M=87552 K=4096 N=16384) | 887 | 1395 (1.57×) | no algo |
| ffn-down (M=87552 K=16384 N=4096) | 937 | 1491 (1.59×) | no algo |

Two facts from this: (a) cuBLASLt on the **5090 has no MXFP4 kernel** — the consensus format is
only reachable on sm_120 via the branch's **hand-QMMA** path
(`mma.sync…kind::mxf8f6f4.block_scale…m16n8k32…e2m1.e2m1.f32.ue8m0`, gated to sm_120a). (b) Even
the raw NVFP4 headroom at the episode's fat-M shapes is only ~1.6×, not the 2.18× of a saturated
square.

## Episode time split (DC dims, RC_PROFILE, sync-serialized)

- phase-2 **GEMM (int8 IMMA): ~64%** of phase-2 — corrects the earlier "Extract is #1 lever" claim.
  Already near the card's INT8 tensor peak (887 TOPS at ffn-up).
- phase-2 **Extract-MX: ~36%** (our own ChaCha+M11 kernel; overlaps the GEMM stream).
- operand-gen ~1.0 s, Merkle ~0.03 s — negligible.

So the GEMM is the dominant block and is what FP4 would accelerate — which is why the Ozaki route
matters despite the naive path being dead.

## Why naive block-FP4 is dead, and how Ozaki routes around it

`ExpandMxDequantInt8` pre-multiplies the scale into int8 (`out = mu · 2^scale[row][col/32]`) with
values reaching ±48. Feeding that straight to FP4 fails: (1) ±48 ∉ E2M1 (±6); (2) the weight
operand's E8M0 scale is per-element along the contraction, which no block-FP4 instruction can
express (the base-matmul `FINDINGS-fp4-scale-axis.md` wall). The branch's Ozaki path avoids both:
it splits K into panels where `2304·chunk < 2^24` (episode: ffn-up K=4096 = **1 panel**, ffn-down
K=16384 = **3–4 panels**, phase-1 S·V K=786432 = **~192 panels**) and re-packs each panel's
operands as (E2M1 nibble, UE8M0 block scale) so the `m16n8k32` tile's `scale_vec::1X` reconstructs
the exact value — then recombines panels with exact int64 weights. Bit-exact to the int64 oracle
by construction, gated behind a self-qual suite (`IsRcOzakiMxfp4Qualified`).

## The break-even (why it's a per-card bet, not a foregone win)

Ozaki does *N panels* of FP4 MMA per int8 GEMM. It wins only if **FP4_rate > N × INT8_rate** on
that card, panel-count N by GEMM:

- ffn-up (N=1): wins if FP4 > 1× INT8 — i.e. *any* FP4 speedup helps. On the 5090 the hand-QMMA
  MXFP4 ceiling (from the earlier fp4_int8 hand-PTX probe) was ~2× → a real win candidate here.
- ffn-down (N=3–4): needs FP4 > 3–4× INT8 — a loss on the 5090 (~2× ceiling), plausibly a win
  only where FP4 ≫ INT8.
- phase-1 S·V (N~192): never wins on panel arithmetic; but it's a small share of episode FLOPs.

**The B200 is the open case.** On datacenter Blackwell INT8 is *sub-peak* while FP4/FP8 run at the
5th-gen tensor peak, so the FP4:INT8 ratio can be much higher than the 5090's ~2× — which is
exactly the regime where even N=3–4 panels net-win. cuBLASLt MXFP4 (VEC32_UE8M0 / tcgen05) may
also be *served* on sm_100 where it is not on sm_120. Both are unmeasured (Vast credit exhausted
2026-07-23); the branch's `SM100_CUBLASLT` Ozaki backend is a fail-closed stub, and numair has an
Opus agent implementing it.

## What we're doing on available hardware

Building the branch's **sm_120a native MXFP4** path (Recipe 2) on our 5090 and running its
capability + qualification test — the consumer-Blackwell analog of the B200 path. It answers
"does native-FP4 Ozaki qualify byte-exact, and does the N=1 ffn-up GEMM actually beat int8 on
Blackwell we have," which is the direct template for the B200 backend.

## Bottom line for the thesis

FP4 is NOT ruled out for the episode — the Ozaki route is byte-exact and could win the N=1 GEMM.
But it does not obviously rescue the B200 per-dollar case: the win is bounded by panel count and
the FP4:INT8 ratio, the 5090 can run the same route, and the B200 must beat a 14× price gap, not
just an int8 GEMM. The honest status is **undecided until a B200 measurement**, not "dead."

---

## 2026-08-02: MXFP4-vs-INT8 ceiling — my "4x dead" was WRONG, but result INCONCLUSIVE

Re-examined the "MXFP4 is 4x heavier" claim (project-pr97-cuda-test). It was WRONG as a floor:
- **Our GEMM operands are natively MXFP4** (Extract output = mu x 2^e, mu in E2M1 {0,±1,±2,±3,±4,±6},
  e per-32-block scale = UE8M0). So representation needs ONE plane, not 4. numair's
  `DecomposeInt8Base4Planes` 4-plane cost was his GENERAL-int8 approach, not our floor.
- Real cost = accumulation exactness only. FP4 tensor accumulates in FP32 (2^24 exact-int limit);
  |product| <= 2304, so exact needs ceil(K/7281) panels: **ffn-up K=4096 -> 1 pass; ffn-down
  K=16384 -> 3 panels; phase-1 small-K -> 1 pass.** NOT a blanket 4x. Plausibly a WIN on up-proj
  + phase-1 (1 pass at FP4 speed), a loss on down-proj (3 panels).

**Whether net-positive hinges on the dense FP4:INT8 throughput ratio on the 5090 (GB202) — which I
COULD NOT MEASURE.** microbench `research/matmul-v4/bench/mxfp4_vs_int8_ceiling.cu`:
- INT8 baseline clean: ffn-up ~2.5ms, ffn-down ~2.5ms, qk-small(K=128) 1.33ms (M=N=16384).
- **cuBLASLt 13.3 FP4 matmul: heuristic status 7 (INVALID_VALUE) for these layouts, BOTH
  block-scaled and plain.** The generic cuBLASLt FP4 path is not accessible with straightforward
  row/col descriptors on sm_120 GB202 — likely needs a specific layout ORDER / alignment / swizzled
  scale format, or isn't exposed this way. Did NOT resolve (stopped to preserve mining uptime).

**FP4 IS available on this GPU** — numair's hand-written QMMA kernel `rc_ozaki_mxfp4_mma_gemm`
self-quals (SM120_MMA, linked_sm120a=1 qualified=1). So the RIGHT way to get the number is to time
HIS kernel raw (pre-packed operands, exclude host packing), not fight cuBLASLt.

**VERDICT: OPEN, not dead.** Decision rule for a future dedicated session: ffn-down FP4 wins iff
3*t_fp4 < ~2.5ms (FP4 >= 3x INT8); ffn-up/phase-1 win iff t_fp4 < INT8 (any speedup). Cheap decisive
next step = time numair's QMMA kernel at our shapes. Do it on-device only (host packing was HIS bug,
we already GPU-ified packing in the withdrawn PR#99). Until measured, neither "dead" nor "win" is
proven — the earlier "4x dead" memory line is a KNOWN-WRONG floor and should not pre-empt the test.

---

## 2026-08-02: MEASURED — numair's QMMA lane is ~500x SLOWER than INT8; existing-kernel FP4 CLOSED

The dedicated session ran. New instrument `bench/qmma_lane_bench.cpp` (+ `qmma_lane_build.sh`):
links the btx-97 tree (wip/matmul-v4.7 @ 540fb25 + constexpr fix + our device-side panel packing
a109ab8), calls `TryLaunchRcOzakiMxfp4GemmS8S8Int64Device` directly (device pointers, stream-
ordered, NO host packing, NO H2D in the timed loop), MX-valid operands (mu in E2M1 x 2^e per
32-block along K), exactness spot-checked vs CPU int64 dot, native-tensor launch counters read
to prove QMMA actually served the calls. 5090, CUDA 13.3, container, miner stopped, reps=10.

| shape | cuBLASLt INT8 | QMMA lane (SM120_MMA) | ratio | panels observed |
|---|---|---|---|---|
| ffn-up  M=16384 N=16384 K=4096 | 3.117 ms | 1645.9 ms | **528x slower** | 1/call (12 launches / 12 calls) |
| ffn-down M=16384 N=4096 K=16384 | 2.978 ms | 1431.3 ms | **481x slower** | 3/call (36 launches / 12 calls) |

Exactness: 4/4 sampled entries match the int64 oracle on both shapes (the lane is CORRECT — and
the panel arithmetic from the 08-02 analysis is confirmed by the launch counters: K=4096 -> 1,
K=16384 -> 3).

End-to-end confirmation, same day: numair's own harness (`--base-production --backend cuda`),
same-window A/B. `BTX_RC_ACCEL_POLICY=native` (MXFP4 lane, WITH device packing) could not finish
3 episodes in 1200 s (rc=124; consistent with 128 FFN GEMMs x ~1.5 s each). `portable` control:
11.20 s/episode (p1=0.743 p2=8.903 p3=1.486, cv 1.6%). NOTE the control read 11.20 s in this
window vs 8.54 s in the Jul-30 campaign — clock-state difference between sessions; same-window
arms only, as ever.

**What this closes and what stays open:**
- CLOSED: FP4 via numair's existing SM120 QMMA kernel. It is a correctness vehicle, not a speed
  vehicle — a naive warp-MMA (no smem staging/pipelining) running ~500x off cuBLASLt INT8 and
  ~1000x off the card's FP4 ceiling. No policy/packing fix rescues it; the kernel itself is the tax.
- STILL OPEN (unchanged from 07-23): FP4 via a from-scratch CUTLASS-grade kernel. The hand-PTX
  issue-rate ceiling (mxf4 1914 vs INT8 969, 1.98x) still stands: at ceiling, ffn-up (1 panel)
  could run ~2x the INT8 GEMM and ffn-down (3 panels) still LOSES (3 x 0.5 = 1.5x INT8). Episode
  ceiling ~15% (ledger candidate: "MXFP4 ffn-up"). That is a large build for a bounded win —
  priority unchanged, below cheaper levers.
- Upstream note for numair (if we report): with device packing his lane is still ~500x off; the
  qualification gate should time the lane against dense INT8 before NativePreferred may select it.

## 2026-08-02 (later): the road to a WINNING FP4 kernel, scouted

- Our own hand kernel `fp4_gemm_tiled_v2` (smem-staged, register-blocked, no TMA/double-buffer)
  at 16384x4096x4096, locked 2600: **5.46 ms vs cuBLASLt INT8 0.93 ms = 0.17x** (v1 was 0.04x).
  Break-even at ffn-up needs ~5.9x more; the ~2x ceiling needs ~12x. Hand-rolling that is a
  CUTLASS-reimplementation project — wrong route.
- => Scouting CUTLASS's own SM120 block-scaled FP4 kernels (examples/79_blackwell_geforce_gemm,
  arch 120a) at our shapes. If CUTLASS FP4 beats 1x INT8 at ffn-up dims, the episode lever is
  real (~15% ceiling) and the build is an integration, not kernel R&D. If even CUTLASS loses to
  INT8 at our fat-M shapes, FP4-for-the-episode is DEAD on consumer Blackwell, full stop.
- Context numbers: GEMM kernel-selection axis is exhausted (deep config search neutral), so FP4
  format change is the ONLY remaining GEMM lever; GEMM is ~46% of the episode.

## 2026-08-02 FINAL: FP4-for-the-episode is DEAD on consumer Blackwell — measured + structural

**The structural fact my 08-02 morning correction MISSED: the W operand cannot ride FP4 in one
plane.** ExpandMx blocks W's scales per (row, col/32) — along **d_ff = the N axis**. Block-scaled
FP4 hardware scales along **K**. For a fixed output column, W's exponent varies per K element;
a mixed-exponent 16/32-block cannot be refactored into E2M1 x 2^e_blk (mu would need up to 48).
The only exact escape is exponent-plane splitting: W = sum of 4 planes (one per e in 0..3), each
uniformly scaled -> **>= 4 FP4 GEMMs on the W side**, x the ceil(K/7281) accumulation panels.
So exact FP4 pays only where FP4:INT8 >= 4. (X/activations ARE 1-plane — their extract blocks
run along K — but the GEMM needs both operands.) This is FINDINGS-fp4-scale-axis coming back;
the morning's "1 exact pass for ffn-up" was right for X and wrong for W.

**The measured ceiling (CUTLASS 79a warp-specialized NVFP4, verified Passed on a host-checked
shape, locked 2600, same-window INT8):**

| shape | CUTLASS NVFP4 | cuBLASLt INT8 | ratio |
|---|---|---|---|
| ffn-up 16384x16384x4096 | 1.601 ms | 2.721 ms | 1.70x |
| ffn-down 16384x4096x16384 | 1.550 ms | 2.732 ms | 1.76x |

1.7x << 4x. Exact ffn-up = 4 planes / 1.70 = **2.4x SLOWER than INT8**; ffn-down = 12 passes /
1.76 = **6.8x slower**. Not a kernel-quality problem: this is NVIDIA's own best consumer-Blackwell
FP4 kernel. VERDICT: **CLOSED-DEAD on the 5090 class.** The episode GEMM stays INT8 IMMA, which
this card runs at ~84% of peak with the autotune.

Still open elsewhere: the B200 (FP4:INT8 can plausibly exceed 4x there since DC-Blackwell INT8 is
sub-peak) — numair's original bet, unaffected by this closure, and it CUTS AGAINST consumer cards
only if his exact-pack overhead is also solved (his current kernel is 500x off, see above).
Ladder of record: naive FP4 dead (scale axis) -> Ozaki-exact viable-but-unmeasured -> his kernel
500x off -> hand-tiled v2 0.17x -> CUTLASS ceiling 1.7x < required 4x. Every rung measured.
