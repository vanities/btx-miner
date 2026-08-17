# DRAFT — PR #89 comment: ran the tool on Blackwell + build/measurement gaps
# Voice: Adam. No em-dashes, no AI attribution. Post FIRST (answers the direct ask).
# Shares only: build fixes, B1 PASS, tool limitations. NO fused combine, no matador kernels.

---

Ran `measure-hardware.sh cuda` on a 5090 at PR head `c324361b`. Good news and a wall.

**The good news: with the build fixed, your device path verifies bit-exact on Blackwell.** `resolved backend: cuda [compiled=yes available=yes admissible=yes]`, `device identity: imma_s8s8s32_tensor_path:sm_120`, and B1 passes: the CUDA batched digests are byte-identical to the CPU reference over the nonce window (`backend_used_device=true`, windows 1/0/0). That is your B1 determinism gate, green on sm_120. We had already cross-checked the same digest byte-for-byte on sm_90 and sm_120 back in July, so cross-architecture determinism holds.

**Five things stand between a clean checkout and that result.** Your audit already flagged the GPU build as unverifiable without a toolchain, so here is the exact list, all confirmed at head:

1. `src/CMakeLists.txt` omits `cuda/matmul_v4_bmx4_accel.cu` from the `BTX_ENABLE_CUDA_EXPERIMENTAL` sources, so the BMX4-C backend never links.
2. nvcc rejects `src/primitives/transaction.h` (the two `CTransaction(deserialize_type, ...)` ctors) without an `#ifndef __CUDACC__` guard.
3. `BUILD_UTIL` defaults to `${BUILD_TESTS}` and the script passes `-DBUILD_TESTS=OFF`, so `matmul-v4-report` never configures and the script dies at its own build step. Needs `-DBUILD_UTIL=ON`.
4. The script's `CUDA_ARCH` default is `75;80;89;90`. No sm_100 or sm_120, so Blackwell runs JIT'd sm_90.
5. **The one that actually matters.** Even with 1 through 4, the tool reports `compiled=no reason=disabled_by_build` and silently runs the CPU stub. `CudaEligibility()` lives in `backend_capabilities_v4.cpp`, which compiles into `bitcoin_common`, but `BTX_ENABLE_CUDA_EXPERIMENTAL` is set `PRIVATE` to `btx_matmul_backend` only. So the classifier takes the `#else -> DisabledByBuild()` branch and the whole measurement runs on the CPU reference while looking like it ran. One line fixes it: `target_compile_definitions(bitcoin_common PRIVATE BTX_ENABLE_CUDA_EXPERIMENTAL=1)` inside the CUDA block. This is worth a hard look, because it silently invalidates every B2g/B2b/M-t24 number the tool would produce on any operator's machine unless they happen to notice the `compiled=no` line.

**The wall: the tool cannot produce B2g at the dimension the ordering question lives at.** Two reasons, both independent of the fixes above:

- The `stages` block (S0..S4 per-stage split) is timed on the **CPU reference** by design. At n=1024 that is already 8 to 12 seconds per window; at n=4096 it does not finish in a bounded run (we killed it twice past 30 minutes). So the datacenter-ordering measurement, which you rightly anchor at large n and Q>=32, is exactly where the CPU lane makes the tool unusable.
- The device batched path that B1 exercises is real but unoptimized: `backend_nonce_per_s=142` at n=1024, `tensor_util_pct≈0`. That is one window at a time with host round-trips, so B2b throughput and any implied utilization measure your dispatch harness, not the silicon.

Net: the tool is now a correct **determinism** instrument on real GPUs (B1 is trustworthy), but not yet a **performance** instrument. To settle B2g/B2b you need the per-stage timers reading the on-device stacked-window path (the S2 GEMM and the S3 combine on tensor units), not the CPU reference, and a batched dispatch that keeps the device busy across the window.

**What we can tell you about the ordering anyway,** from direct on-device instrumentation at n=4096 (your S-boundaries, INT8 path, one 5090):

- The per-nonce marginal IS tensor-dominated once batched: S1b expand-B ~26%, S2 (B*V) ~4%, S3 (P*Qstack limb-tensor combine) ~70%. So §K.2a-WT tensor-majority holds, the C-13 combine carries it.
- But the tensor **utilization** is only ~25% of the card's INT8 peak. The combine is a stack of medium limb-pair GEMMs plus the elementwise decompose/recombine passes, which is bandwidth and launch bound, not MAC bound. That is the same signal as your 0.40x anchor, and it is why a datacenter part's extra TOPS do not convert: at 25% utilization the workload does not saturate even a 5090. The ordering question is really a utilization question, and right now nobody's tensor units are full.

Happy to re-run the fixed tool on an H100 for the pair once the per-stage path reads the device, since today it would just time the H100's CPU.
