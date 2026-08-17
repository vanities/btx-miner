# v4.6 ENC_RC episode solver — optimization experiment ledger (5090, DC dims)

Read before retrying ANY episode-solver optimization. Baseline instrument:
`bench/rc_gpu_solver.cu` (toy golden `5b1bff3c…`, DC digest `1cc5709d…`), mode 3
(datacenter profile-2 dims), reps>=3, miner stopped, container `matador-build:deps-2204-cuda1330`.
Baseline wall: **11.8-12.2 s/episode** (run-to-run spread ~3%; A/B arms back-to-back in one
session). Phase split (RC_PROFILE): opgen 1.05 s + GEMM 7.2 s (64% int8-IMMA peak) +
Extract 3.7 s + Merkle 0.03 s — fully dependent serialized chain, no timeline gaps.

## Dead (measured, byte-exact-gated — do not re-chase)

| date | lever | result | why |
|---|---|---|---|
| 07-23 | extract `launch_bounds(256,5)` occupancy | -20% | compute-bound kernel, spills |
| 07-23 | cuBLASLt algo cache (heuristic-best per shape) | neutral | heuristic already stable |
| 07-23 | fused-apply extract (residual in-extract) | -19% | divergent scattered writes |
| 07-23 | conc=2 episodes at DC | OOM | re-verified 07-24 (cudaMalloc @863) |
| 07-24 | extract SHA 16-word ring schedule (64 regs) | -11% | ring serializes the schedule (ILP loss > occupancy gain) |
| 07-24 | extract `launch_bounds(256,4)` (64 regs, keeps w[64] ILP) | -12% | 3rd kill on the occupancy axis: 70 regs / 46% occ is a LOCAL OPTIMUM |
| 07-24 | cuBLASLt workspace 256 MB -> 1 GiB | neutral | heuristic not workspace-starved |
| 07-24 | build with `--default-stream per-thread` | -25% | flag reroutes every bare launch/sync; monolith pays even single-threaded |
| 07-24 | PRF key as by-value kernel param | ~-20% | addressing a by-value param spills it to LOCAL memory; wrecks the unrolled SHA. Keep the global-pointer dprf form |
| 08-02 | Extract scale-SHA midstate (precompute w[0..13]+round-13 registers, resume per tile) | ~-1% | byte-exact but the 23-u32 ctx loads + register pressure in k_extract_tiles cost more than 14 skipped rounds; the SAME trick wins in opgen (1-block hash + kills byte staging). Code removed; comment left at k_extract_tiles |
| 08-02 | **Extract ILP restructure** (mix-nibble precompute via int4 loads + scale-SHA + ChaCha issued as adjacent straight-line chains; rejection loop reduced to xor/lookup) | **+3.5% SLOWER** (locked 2600 MHz x2 interleaved: 877.1/878.9 OFF vs 907.1/909.3 ON) | extended live ranges (mixn+ks+SHA schedule co-resident) = register pressure > ILP win. FOURTH dead axis on this kernel (occupancy x3, ring, midstate, ILP): the short-live-range legacy order IS the local optimum. Extract's ~17% ALU efficiency is real but unexploitable by restructure — next attack needs ncu stall evidence first |
| 08-02 | atomic-compaction mantissa write (skip the count pass) | dead by construction | emission order must be the deterministic per-block prefix-sum order; atomics randomize offsets = digest fork. Not an A/B candidate, ever |
| 08-02 | memory-clock VF offset +1000 (the base-chain knee), core locked 2600 | neutral (850.0/852.7 vs 851.0/853.4, x2 interleaved) | physically consistent: episode DRAM sits at 24% and L2 rides the CORE domain — VRAM speed is not a limiter. Standalone NVML tool `bench dir nvml_memoc.c` (dlopen, SetMemClkVfOffset) works from the privileged container; offset reset to 0 after. Last hardware knob on this card: CLOSED |
| 08-02 | warp-cooperative smem staging of extract raw reads (coalesced 4 KiB/warp span, transposed, conflict-free loop reads) | **+8% SLOWER** (locked x2: 862.0/864.6/865.5 vs 933.2/934.2) | directly targeted the ncu-measured L2-sector limiter and STILL lost: the 32 KiB/block carveout shrinks the unified L1 (91.5% hit rate was doing the work) and staging stores eat 32-way bank conflicts. SIXTH dead extract restructure. The kernel is L2-sector-bound with a consensus-fixed layout; the direct L1-cached read IS the optimum. Extract axis CLOSED unless the layout itself changes |
| 07-24 | **phase-2 row-strip stream pipelining (RC_STRIPS=2/4)** | **+34% slower** | overlap premise false: GEMM + extract are both DEVICE-FILLING, so streams interleave, never co-run; per-strip GEMMs re-stream the full W operand (traffic x S) and thrash L2 reuse; M/S GEMMs less efficient. Byte-exactness of the strip construction PROVEN (toy golden passes at S=2 with global-row crypto offsets) — it is the PERF that is dead, on this card, at these dims |
| 08-10 | **W-pair opgen prefetch ring** (per-layer W_up/W_dn expanded on a dedicated cudaStreamNonBlocking side stream during GEMM+extract, event-handshaked ring of 2-8 slots, pinned-staged async ctx uploads, own scratch pool; NO --default-stream flag, NO GEMM slicing, NO spin — built to dodge every prior overlap kill) | **DEAD +20%** (locked 2600, 15 reps/arm: ring-only 1061.7 vs base 881.0/882.2; ring=2 1050.6, ring=6 1003.5, CUDA_DEVICE_MAX_CONNECTIONS=32 and MANT_CACHE=0 no help) | NINTH overlap-class kill; byte-exact all arms (toy golden + 42e74cd6). The device-filling physics held for opgen too: the side stream's full-device expansion grids ALTERNATE with the main chain at kernel granularity instead of co-running — extract's 3 CTAs/SM at 72-80 regs leave no register room for a 56-reg opgen CTA (the ptxas headroom argument was wrong: 64K-55.3K = 10.2K < the 14.3K a 256-thread opgen CTA needs), only GEMM windows could host one — and the per-layer ready-waits turn each alternation into a bubble. Scheduling cannot hide opgen on this card. Code kept compiled + gated (`BTX_RC_OVERLAP`, default 0) for wider silicon. The autopsy ncu run found the REAL opgen story (see 08-10 banked win) |

Also verified: no serialization gaps to exploit (phase walls sum to total); pool-carved
cuBLASLt workspaces fail with status=14 (need 256B alignment — use direct cudaMalloc).

## Banked wins

| date | lever | result | notes |
|---|---|---|---|
| 08-10 | **scale-SHA fixed block-2 schedule + pad-spine fold** (`BTX_RC_SCALE_W2C` + `BTX_RC_FOLD_SPINE`, both default ON: block 2 of the 67-byte scale preimage is a pure function of bl1=(bj>>8)<96 at consensus dims, so its 48-word schedule is a constexpr table (kW2Sched[96][48], static_assert-pinned, __constant__, warp-uniform); the Merkle fold takes host-computed all-pad subtree roots P_l as the right child where real width runs out -- half the padded tree never hashes, k_fill_pad gone) | **~-1%** (-6 to -9 ms; locked pairings 706.8->701.0 and 718.4->709.7; all-off validity ≈ A; extract regs held at 80) | the schedule-ALU estimate (~-2.8%) did not fully materialize -- part of the scale SHA hides under other stalls. Byte-exact: toy golden x4 gate combos + 42e74cd6. Shipped v0.9.14. NOTE the W2C table's in-kernel guard (bj<24576) keeps foreign dims exact via the computed path |
| 08-10 | **word-assembled leaf hashing** (`BTX_RC_LEAF_VEC` default ON: k_hash_leaves byte-copied every leaf through DSha's staging buf -- ncu 1024 load sectors PER LEAF, 67.1M sector-ops per 65,536-leaf emit. v2 assembles SHA message words from aligned u32 loads with a PRMT byte window (0x7012) absorbing the 1-byte 0x00 tag offset; second SHA in registers) | **-0.8%** (-5.8 ms consistent in both adjacent pairings: 715.8->709.8, 721.7->716.2 locked; vec-off validity 719.7 ≈ A) | byte-exact ON/OFF toy golden + 42e74cd6. Shipped v0.9.13 |
| 08-10 | **CUTLASS disk tune cache restored** (the cuBLASLt-era -34%-cold lever was silently compiled out at the backend switch; TunePick = 3 ints, persisted keyed on GPU name + candidate-set sizes, bounds-checked load, tmp+rename save; same env surface BTX_RC_TUNE_CACHE/_PATH, file rc_gemm_tune_cutlass.bin) | **COLD one-shot 2719.9 -> 752.4 ms (-72%)**; warm unchanged | THE VALIDATOR-LANE fix: matador-verify / rc_attest one-shots were re-timing the GEMM candidate set every process. One-shot replay now ~0.75 s, warm ~0.70 s (vs shib 7.18 s, numair CUDA replay 32.6 s). Shipped v0.9.13 |
| 08-10 | **extract tile-store packing** (`BTX_RC_EXTRACT_PACK`, default 2: the final store loop wrote each 32-byte tile as 32 single-byte stores — warp lanes 32 B apart = one sector op per lane per instruction, ncu-measured EXACTLY 32 store sectors/tile + ~22 L2 write sectors of partial-sector amplification, 403M store sector-ops in the S-extract launch alone. PACK=2 = two int4 stores, PACK=1 = eight u32 stores, PACK=0 = legacy. PACK is a TEMPLATE param so each variant keeps its own SASS/registers — all six instantiations hold 72/80 regs exactly, occupancy unchanged) | **-6.4%** (locked 2600, 15 reps: A 768.1/776.3 vs pack=2 719.9/725.5; pack=1 723.5; pack=0 validity 774.3 ≈ A). Cumulative 08-10: 876.8 -> 722.7 locked = **-17.6%**, ep/s 1.140 -> 1.389 | The tile span is 16-byte aligned by construction (base = i*cols + bj*32, every consensus cols %32==0). THE LESSON PAIRED WITH THE SIX DEAD RESTRUCTURES: every prior extract attack hit the LOAD/compute side (the local optimum); the STORE side was never the accused, and it was transaction-bound the whole time. Same disease+cure as the 08-10 opgen win. Byte-exact: toy golden all PACK modes, 42e74cd6 all arms. Shipped v0.9.12 |
| 08-10 | **opgen store coalescing** (ncu-guided, `BTX_RC_OPGEN_COALESCE` default ON: scale stream packs 4 codes/u32 and emits 8 int4 stores per block instead of 128 byte stores; mantissa hash cache goes word-PLANAR — hc32[plane*nb+b], 8 u32 planes — so warp lanes write/read consecutive words per plane instead of eight 32B-strided u32s; mantissa write packs interior stores 4 bytes wide behind the unchanged rejection walk, scalar head/tail). Companion `BTX_RC_QKT_DIRECT` (default ON): K as expanded is (n_ctx,d_head) row-major = exactly the k-contiguous B^T the IMMA path needs for Q·K^T — feed dK directly, deleting the naive dKt transpose AND the CUTLASS-internal transpose that reproduced dK bit-for-bit | **-11.0%** (locked 2600, 15 reps/arm: base 871.6/876.2 vs 775.2/781.7 defaults; coalesce-only 781.5, qkt-only 872.4 = ~-1%; all-gates-off validity arm 876.6 ≈ base). Free-boost guard arm **-9.1%** (874.9 -> 795.0). ep/s 1.147 -> 1.290 locked | THE FINDING (ncu on the shipped binary, the first stage re-split since the CUTLASS switch): P1 profile splits GEMM 409 / Extract 250 / **opgen 180** / rest ~38 ms of 877, and opgen is L2-STORE-TRANSACTION-bound, not SHA-bound — k_scale_stream ran 82% L2 issuing one sector op per single-byte store (3.15M for a K/V operand); k_mant_count issued 18.6M store sectors writing the hash cache 32B-strided (every lane its own sector, every instruction). Measured opgen was ~2.4x its SHA cost; the gap was store transactions. Byte-exact by construction (identical values, positions, emission order; only internal-scratch layout + store width change): toy golden all gate combos incl. ring-on, 42e74cd6 all arms. Shipped v0.9.11 |
| 08-02 | **residual-in-GEMM via beta=1** (k_widen8_32 preloads X into dy32; down-proj runs D = H·Wdn + C, replacing the k_add_resid round trip; autotune times beta!=0 candidates into a throwaway pool buffer so the preloaded C survives) | **-0.5%** (locked interleaved x2: 849.4/852.6/854.2 OFF vs 845.9/848.5 ON) | small because beta=1 adds a C READ inside the GEMM the estimate missed — traffic nets to a wash; the win is one less launch + better overlap. Gate BTX_RC_RESID_BETA default ON; tune cache keyed by beta. int32 exactness: |X|<=48, GuardInt32Bound |
| 08-02 | **COLD vs WARM replay measured** (warm-up episode timed: ctx + cuBLASLt handle + autotune sweeps + 3 GB arenas) | cold one-shot **1.38 s** (1376.9-1387.2 across 5 fresh processes), warm steady **~0.85 s/ep** | THE validator numbers: quote sustained ~0.85 s/share for a long-lived pool validator, 1.4 s one-shot in-process (plus process/CUDA-init wall for CLI). vs shib's 7.18 s on a 5070Ti |
| 08-02 | **M11 tables as register immediates** (ncu-guided: `__constant__` dM11/c_m11 arrays indexed by DATA-DIVERGENT nibbles serialize into per-unique-index replays, ~128 lookups per tile/block across extract + both opgen kernels; acceptance -> 16-bit mask `(0xF4F5>>nib)&1`, values -> one u64 of 4-bit two's-complement nibbles, constexpr-derived from kM11 + static_asserts) | **-2.3%** (locked 2600 interleaved x2: 882.2/886.3/886.4 OLD vs 864.1/865.7 NEW) | the FIFTH extract attack and the first ncu-guided one — the win was never occupancy/ILP, it was constant-cache divergence. Companion neutral change rides along: ChaCha keystream emitted as 16 register words instead of a local ks[64] byte array (measured flat alone, simplifies the loop). ncu also showed the big extract at 84% L2 / 24% DRAM / 46% occupancy, 70 regs |
| 08-02 | **mantissa margin tightening + opgen SHA midstate** (nb over-provision 1.25x -> mean+16 sigma, ~20% fewer count blocks; opgen stream SHA resumes from a host-precomputed round-7 checkpoint — seed words w[0..7] are constant per operand — and skips the DSha byte staging) | **+2.1%** (interleaved A/B x2: 846.4 vs 864.3 expected, 851.5 vs 870.2 expected, drift-interpolated from 3 OFF anchors 860.4/868.1/874.3) | gates `BTX_RC_MANT_TIGHT`, `BTX_RC_SHA_MIDSTATE` (default ON); digest 42e74cd6 all arms, toy golden passes all gate combos. blk<2^24 word-layout guard falls back to the generic hash. DC-dims (mode 3) confirmation: all three 08-02 opgen levers together = **-1.8%** (11731.6/11559.1/11818.0 A/B/A, digest 1cc5709d) — smaller share because Config W hoists most opgen at DC. |
| 08-02 | **mantissa hash cache in opgen** (k_mant_count stores its 32-byte SHA digest; k_mant_write_dequant reloads it instead of re-hashing — every mantissa stream block was hashed TWICE) | **+2.9%** (A/B/A 897.2 / **874.2** / 901.6 ms, mode 1 reps 10/10/5, 5090 stock 575 W) | gate `BTX_RC_MANT_CACHE` (default ON, =0 reverts in-binary); digest 42e74cd6542d86bd identical all arms; toy golden 5b1bff3c passes ON and OFF. Cost: ~32 B/block extra traffic (~60-90 MB per large operand, pooled). |
| 08-02 | **disk-persisted GEMM tune cache** (cublasLtMatmulAlgo_t is documented "trivially serializable" — cuBLASLt 13.3 §3.3.5; per-shape tuned algos POD-serialized with a header keyed on cublasLtGetVersion+GPU name+workspace+tune level, AlgoCheck-validated at first use, tmp+rename atomic write) | **COLD episode 1590.6 -> 1054.4/1081.5 ms (−34% cold, −535 ms); warm episodes unchanged** (973-980 ms all arms) | gates `BTX_RC_TUNE_CACHE` (default ON), `BTX_RC_TUNE_CACHE_PATH` (default `~/.cache/matador-miner/rc_gemm_tune.bin`; empty HOME = silently skipped). Toy golden 5b1bff3c passes cache-cold/warm/disabled; mode-1 digest 42e74cd6542d86bd. This is the VALIDATOR lane win (matador-verify one-shot share audits pay the tune every process); the warm mining loop never sees it. Header mismatch (driver/lib bump, ws change) = ignore + overwrite; AlgoCheck failure = loud re-tune, never trust the file. From the 08-02 NVIDIA-docs sweep (research/matmul-v4/FINDINGS-nvidia-docs-sweep-2026-08-02.md) |
| 08-02 | **round-batched pinned upload arena** (all 34 PRF keys + 36 opgen seed/midstate ctx blocks staged into one pinned buffer at round start, ONE cudaMemcpyAsync replaces the ~70/round sync pageable copies; `BTX_RC_BATCH_UPLOAD` 1=full, 2=PRF-only) | **DEAD, monotone slower** (locked 2600 x2 interleaved: off 958.9/968.9 vs PRF-only 967.8/973.5 vs full 979.3/983.8 ms; window 1 full-vs-off confirmed −1%) | byte-exact all arms (toy golden + 42e74cd6). THE FINDING: the ~280 sync pageable memcpys/episode are NOT a cost on this card — the launch queue runs deep enough that the GPU never starves during a sync-copy drain, and even the zero-added-host-work PRF-only arm loses, so work moved to the idle round boundary is a pure loss. This DISCONFIRMS the launch/copy-glue overhead class (~0.5-1% est.) for this pipeline: host submit is not on the critical path -> CUDA-graph CPU-submit savings are likewise pre-disproven here (PDL tail-overlap is a different, device-side mechanism and stays open). Code kept gated OFF: graphs/PDL need graph-legal copies, and B200-class hosts may pace differently |
| 08-02 | **256-bit PACKMIX extract** (four ld.global.v8.b32 preloads of the tile's 32 raw i32, mix nibbles pre-folded into two u64 REGISTERS, rejection loop reads a register funnel-shift instead of a per-attempt global load — LDG.E.256 verified real on sm_120 first, probe pc:~/matador-prof/ld256_probe.cu) | **DEAD +10.9%** (locked 2600 x2: 1068.0/1068.0 vs base 963.3/963.3) | SEVENTH extract kill, byte-exact. The per-attempt raw loads were ~free L1 hits (91.5%); the pack pays register pressure for traffic that was already cached. LDG.E.256 itself stays a verified-available instruction (the "128-bit consumer cap" claim is WRONG) — dead HERE, not everywhere. Code removed, dead-comment in kernel |
| 08-02 | **PDL extract** (cudaLaunchKernelEx + programmatic-stream-serialization on cudaStreamPerThread = the GEMM's stream, so pairs form; prf-only prologue [key unpack + 2-block scale SHA] hoisted before cudaGridDependencySynchronize to overlap the GEMM tail; sm_120 cuBLAS kernels PDL-enabled since 13.0 U1) | **DEAD +29%** (1239.4/1246.6 vs 963.3/963.3; PDL+PACKMIX 1126.1/1128.6) | EIGHTH extract kill, byte-exact. Failure mode: the secondary's early CTAs occupy SMs running the SHA then SPIN at the dependency wait, crowding out the producer GEMM's remaining waves — PDL fits SMALL secondaries behind a draining primary, not a full-device grid behind a multi-wave GEMM. With this + batch-upload dead, the ENTIRE glue class from the 08-02 NVIDIA-docs sweep is measured dead at P1 dims on the 5090: the episode keeps the GPU saturated; estimated launch/copy overhead was never real. Remaining sweep items: L2-persistence window (unmeasured, expect ~0 — residency tool vs a sector-throughput bound) and CompileIQ compiler autotune (unrun, machine-time). Code removed, dead-comment in kernel |
| 08-02 | **L2 access-policy window on extract raw input** (cudaAccessPolicyWindow on the legacy stream, per-call; arm C = PERSISTING + 48 MiB set-aside, arm D = STREAMING/evict-first; max window 128 MB on the 5090) + **extract carveout=maxL1** (cudaFuncAttributePreferredSharedMemoryCarveout=0) + **ptxas -O2 whole-binary** | **ALL NEUTRAL** (locked 2600 x2: base 907.8/908.9 vs carveout 907.9/910.8 vs persist 909.8/913.2 vs stream 910.4/912.7 vs ptxasO2 910.1/914.2 — spread <=0.5%, mild in-window drift) | byte-exact all arms (toy golden all combos + 42e74cd6). Physics as predicted: residency tools cannot move a sector-THROUGHPUT bound (extract reads each line ~once, L1 covers repeats), the driver's default carveout choice was already right, and ptxas -O2 vs -O3 is settled. Research gates NOT kept (uncommitted, dropped). With this, the 08-02 NVIDIA-docs sweep checklist is FULLY measured except CompileIQ (external download, parked pending its install/workflow assessment) |
| 08-02 | **CompileIQ ACF autotune on the episode TU** (NvccSearchSpace 13.3 GA, 3 gen x 6 pool = 18 evals + baseline pair, fitness = mode-1 warm ms, digest 42e74cd6 as a per-candidate fail-closed tripwire, locked 2600 per the 07-03 lesson) | **DEAD — the GA "winner" is a REGRESSION.** Search best 879.9 ms vs baseline 897.6/900.0 (-2.0% nominal), but interleaved validation x3: base 895.3/899.3/903.5 vs ACF **931.9/935.0/935.4 = +4.0% SLOWER** | second independent CompileIQ null (after 07-03 on the scan TU, where +1.56% likewise failed to reproduce) — the GA selects on single noisy evals, so its best is winner's curse even WITH locked clocks; here the selected config is actively worse. Mechanism healthy: 18/18 candidates byte-exact (tripwire never fired; several were 1.14-1.38 s = much slower but still correct), compiles 13-29 s. Structural reason it can't win big here: ~44% of the episode is cuBLASLt GEMM (precompiled — no ACF reach), and our kernels are dep-chain/L2-bound, not schedule-bound. **Compiler-config axis CLOSED on both TUs.** Artifacts pc:~/matador-prof/ciq/{best_episode_config.bin,ciq_results_episode.csv}, runner ciq_extract.py + ciq_validate.sh |
| 07-31 | **per-shape cuBLASLt algo AUTOTUNE** (timed sweep of up to 16 heuristic candidates, cudaEvent 5-iter probe, cached per descriptor tuple) | **+4.4%** (A/B/A 925 -> 884 ms on the RC episode GEMM block, 5090) | commit c59eb9b, default ON, gate `BTX_RC_GEMM_TUNE=0` reverts in-binary; companion `BTX_RC_GEMM_WS_MB` (ff8dd1a, 256 MiB) widens the candidate set. NOT the same lever as the 07-23 "algo cache" dead entry: that one cached the heuristic's FIRST pick (neutral); the win comes from TIMING the candidates — the heuristic's ranking is wrong at these shapes. Byte-exact (every int8 IMMA algo yields identical int32; episode golden 5b1bff3c unchanged). RE-CONFIRMED 08-02 same-window A/B (mode 1, reps 5, 575 W stock limit): 898.6 ms ON vs 937.0 ms OFF = +4.3%, digest `42e74cd6542d86bd` both arms. Phase split at profile-1: GEMM ~398 ms/ep (44%), Extract ~232 ms/ep (26%), opgen/upload/phase1 ~270 ms/ep. |

Related closure 08-02: FP4 via numair's existing SM120 QMMA kernel measured ~500x SLOWER than
cuBLASLt INT8 at both FFN shapes (`bench/qmma_lane_bench.cpp`; full numbers in
`FINDINGS-fp4-episode.md`). The "MXFP4 ffn-up" live candidate stays open ONLY as a from-scratch
CUTLASS-grade build (~15% episode ceiling) — never via his kernel.

## 08-10 evening re-profile (v0.9.15 code, locked 2600, fresh instrument ~/ep-bench3)

Fresh RC_PROFILE + ncu after the store-transaction campaign (the 180 ms opgen figure was stale):
**episode 695.2 ms warm** (1.438 ep/s locked; COLD one-shot 742 ms with the disk tune cache).
Split: **GEMM 442 ms (63.6%) / phase-2 extract 170 ms (24.5%) / opgen 60 ms (8.6%) / rest 18 ms
(2.6%: S/Z extracts ~13 + phase-1 GEMMs + leaves + Merkle)**. Kernel-level (ncu detailed, ns):
up-extract 2.10 ms x64, dn-extract 0.69 x64, mant_write 0.28 x144, mant_count 0.10 x144
(SM 94% = SHA-saturated), scale_stream 8 us, W transposes 78-88 us (DRAM 80% = at bandwidth),
leaf-hash 0.145 ms x68 at 22.8% occupancy (grid 256 = latency-bound), Merkle ~2 ms.

- **GEMM AXIS CLOSED (measured, do not re-open without new silicon/CUTLASS):** ncu on the live
  binary: FFN up (256x128 s4) **SM 89.9%**, FFN down **SM 95.1-95.3%** -- the tuner's disk-cached
  pick for the down shape is now **stream-k 128x256** (Kernel2/GemmUniversal, grid 2040 = 12x170
  persistent), which closed the 08-05 "-7% down" gap on its own. Whole-episode dual-backend A/B
  (one binary, both backends, in-container, locked): tuned cuBLASLt 756.5 ms vs CUTLASS 768.9 ms
  = **Lt +1.6%**, all of it schedule-level. Effective int8 throughput ~640-750 TOPS = 70-83% of
  the 905-TOPS 2600 MHz peak per shape. The "FFN-down mainloop archaeology" NEXT item is MOOT.
- **Extract re-confirmed closed with stall evidence** (the precondition the 6-kill note asked
  for): up-extract **L2 Cache Throughput 85.1%** / dn 86.7%, DRAM 37%, SM 55%, occ 46% at the
  50%-theoretical (80-reg) ceiling, No-Eligible 62%. L2-sector-transit floor ~1.7 ms of the
  2.10 ms: within ~20% of physics with a consensus-fixed layout. Seventh restructure NOT advised.
- Residue candidates surfaced (see rows below once A/B'd): pow2 div->shift in
  k_mant_write_dequant (u64 divide per ACCEPTED NIBBLE, ~55/thread) + k_extract_tiles (one
  div+mod per tile); leaf emits on a side stream (BTX_RC_LEAF_ASYNC, fire-and-forget until
  phase 3 -- the one overlap shape with no mid-round main-chain wait).

## 08-10 residue-candidate A/B (game-contaminated, MIN-of-N + within-round paired method)

Measured with the Steam game sharing the GPU (user opted in). Method: contention only ADDS
latency and a code change only REMOVES work, so per-round back-to-back base->cdiv->cdiv+la
deltas cancel the ~11% game drift (each arm ~8 s apart; the last-measured arm even pays a drift
penalty). 10 rounds x 3 arms x 10 reps, locked 2600, miner+keeper stopped. Log ~/ep-bench3/measure4.log.

| candidate | within-round paired mean | sign test | verdict |
|---|---|---|---|
| **pow2 div->shift** (`k_mant_write_dequant` + `k_extract_tiles` index math, branch 09f97b3) | **-0.13% (~-1 ms)** | 8/10 faster but ~noise, grows with drift | **FLAT -- disconfirmed.** The write-pass divide is NOT the bottleneck: k_mant_write is SM 70%, memory/SHA-reload-bound, not ALU-div-bound (ncu 08-10). Removing ~55 divides/thread buys nothing. Byte-exact + harmless; not worth a release on its own. |
| **BTX_RC_LEAF_ASYNC** (side-stream leaf emits, default OFF) | **-0.4% (~-3 ms)** | 9/10 faster; measured last so drift works against it => floor | **SMALL REAL WIN**, but 0.4% under 11% contamination is at the resolution edge. CONFIRM on a clean window before flipping default / releasing. |

## 08-10 CROSS-NONCE pipelining: MEASURED DEAD (-19%/-21%) -- the 10th overlap kill, now CROSS-episode

Solo mining makes independent-nonce concurrency natural (we pick nonces), so this was the last
open structural lever -- the ledger's live-candidate #3. Tested at LIVE base dims (mode 1), flag
build (`--default-stream per-thread`) with `splitk_workspace()` made thread_local (the fix that
unhung it). BYTE-EXACT GATE PASSED: conc=2 digest = 42e74cd6542d86bd, isolated (concurrency is
byte-correct). Aggregate throughput (locked 2600, 90s/run cap, best-of-2):

| conc | aggregate ep/s | vs serial |
|---|---|---|
| 1 | 1.301 | -- |
| 2 | 1.051 | **-19%** |
| 3 | 1.029 | **-21%** |

**Concurrency HURTS: two nonces contend so hard each slows >2x, dropping aggregate throughput.**
The device-filling physics holds even for INDEPENDENT episodes -- at base dims one episode already
saturates the 5090, so a second just steals SMs from the first's GEMM instead of usefully filling
its extract/opgen gaps. Big clean effect, well above game noise. **10th overlap-class kill** (9
within-episode + this cross-episode one): scheduling/overlap is dead on this card, full stop.
Two ledger assumptions were overturned along the way (both moot now): conc=2 is memory-feasible at
base dims (9.3 GiB, the OOM was DC dims), and the `--default-stream per-thread` -25% tax is NOT
real for v4 code (flag-c1 1.301 == nonflag 1.288). Binary pc:~/ep-bench3/rc_bench_pts2; the
thread_local splitk fix is correct and worth upstreaming if concurrency is ever revisited on wider
silicon (B200-class, where util headroom is real -- the only regime this could pay).

**With cross-nonce dead, the episode is at its practical floor on the 5090:** GEMM closed (90-95%
SM), extract closed (L2-bound + stall evidence), opgen closed (coalescing + SHA-saturated + div
disconfirmed), overlap closed (10 kills). Only sub-1% shaves remain (leaf-async ~0.4%); the next
real gains need new silicon, a consensus/algo change, or the OC edge (operational, not code).

Absolute numbers were 768-782 ms (vs 695 ms clean) = the ~11% game tax, which is exactly why
only the paired deltas are trustworthy. auto_ab.sh re-armed for a clean confirmation of
leaf-async. Neither shipped: pow2-div flat, leaf-async needs the clean number first.

**Board status after this:** GEMM closed (95%/90% SM), extract closed (L2-bound, stall evidence),
opgen closed (store-coalescing + mant_count SHA-saturated + write-pass div disconfirmed), leaves
partially hideable (~0.4%). The code-optimization vein is largely exhausted after the +25%
store-transaction campaign; the remaining real angles are all structural or operational, not
kernel micro-opts -- see below.

## 2026-08-11: clean-GPU confirm -- cdiv FLAT (banked), leaf-async FLAT (default stays OFF), and the clock-sag discovery

Full-idle window (miner + node + keepers deliberately down, cold box): warmup then interleaved
A/B/A (4 base arms), then a 6-pair base-vs-leafasync alternation with per-arm clocks.sm/temp
logging. Byte-exact tripwire `42e74cd6` printed on every arm. Log: pc:~/ep-bench3/auto_ab.log.

- **`-lgc` does NOT hold the operating point on this 5090.** Even pinned `-lgc 2600,2600` the SM
  clock ran 2332 -> 2205 MHz as the die went 65 -> 80 C: the lock is a CEILING; sustained episode
  load thermal-sags ~5% below it within ~2 min. That sag IS the monotonic "drift" seen in every
  prior locked window (08-10's 679 -> 700 ms base ramp included), and it is the same order as the
  effects being measured. A/B PROTOCOL FIX going forward: load-warm to thermal plateau (~60-90 s)
  before the first arm, pin at a clock the card can SUSTAIN (~2100-2200 on this cooling), log
  clocks.sm + temp per arm, and interleave pairs so residual sag cancels within pairs.
- **cdiv (pow2 div->shift in k_mant_write_dequant + k_extract_tiles): FLAT** in all three windows
  (08-10 dirty; 08-11 interleave, drift-corrected -0.3%, within the +/-0.3% base scatter).
  Byte-exact + up-or-flat => BANKED per house rule: strictly fewer instructions in two hot
  kernels, zero cost today, may pay on other dims/silicon.
- **leaf-async: FLAT at thermal plateau** (pairs 4-6, temps stable 77-80 C: +0.7 / -0.5 / 0.0 ms),
  -0.25% at free boost (single guard pair, within noise). Never worse in any clean window -- the
  08-10 "-0.4% slower" read was sag pollution, not signal. NOT a clean win => default stays OFF;
  `BTX_RC_LEAF_ASYNC=1` stays as the env knob for wider silicon.

Plateau numbers (pinned, sagged ~2210 MHz): 687-694 ms; free boost 689-691 ms (~1.45 ep/s).
Board unchanged: the episode stays at its practical 5090 floor; the remaining levers are the OC
edge re-sweep (+270/275, operational, needs a live multi-hour rej<0.3% gate) and solo keep-rate
(orphan) economics -- not kernels.

## Live candidates (unbuilt, in value order)

1. **Miner solve-loop integration of the GPU solver** — SolveMatMulRCEpisode still runs the CPU
   oracle; wiring the GPU episode context is ~200x at DC dims. The one that matters for mining.
2. ~~MXFP4 ffn-up (N=1 K-panel)~~ **CLOSED-DEAD 08-02**: W's MX scales block along N, not K ->
   exact FP4 needs 4 exponent planes on the W side -> pays only if FP4:INT8 >= 4x. Measured
   ceiling with CUTLASS 79a (verified, locked clocks): 1.70x/1.76x at ffn-up/down vs cuBLASLt
   INT8. Exact ffn-up would be 2.4x SLOWER. Full ladder in FINDINGS-fp4-episode.md. Do not
   re-open on consumer Blackwell; the B200 (INT8 sub-peak) is the only regime where it can pay.
3. **Cross-NONCE episode pipelining** — different from row strips: two independent episodes
   ping-ponged (needs ~1.6x buffers, not 2x — conc=2's OOM was full duplication). Same
   device-filling physics likely caps the gain at hiding opgen (~8%); measure before building far.
   NOTE 08-02: the conc path needs the `--default-stream per-thread` build, which is a measured
   -25% single-episode (Dead table) — so conc=2 must beat 1.33x on ITS OWN binary to net-win vs
   the monolith. At 5090's 98-99% util the premise is weak; B200 (73.8% util) is where it pays.
4. ~~Deeper timed GEMM algo search~~ CLOSED 08-02: full config-space search (all algo ids x tile
   ids x stages x split-K x CTA swizzle via AlgoGetIds/AlgoCheck, 128 timed configs) = NEUTRAL vs
   the timed-heuristic winner (locked 2600: 878.9/880.6 T1 vs 879.7/880.4 T2). The heuristic's
   candidate list already contains the best config at our shapes. Kept as opt-in
   `BTX_RC_GEMM_TUNE=2` (default 1). Kernel-selection axis EXHAUSTED: nullptr -> heuristic-timed
   (+4.4%) -> full space (+0). Only a FORMAT change (FP4) moves the GEMM now.
5. Batched per-round PRF-key upload — gpu_extract does a synchronous 32 B H2D per call
   (~136/episode at profile-1); one per-round batched upload removes ~130 stream syncs. Expected
   ~ms-class; cheap to build alongside the next opgen touch.

## Instrument note (2026-07-24)

numair's `run-full-benchmark.py`/`matmul-v4-rc-harness` **times the CPU reference pass**
(`RecomputeResidentCurriculumReference(..., gemm={})`); `MineRCEpisode` (the backend pass) is
never timed. Its "episode numbers" are host-CPU numbers on every machine — do NOT use it for
speed work; use it only for correctness/self-qual and verify-carrier budget. Our bench is the
speed instrument.
