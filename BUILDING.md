# Building matador-miner

One target, one flow. ENC_RC (v4) is the only algorithm, and `linux-x86_64` is the only asset
lane — the macOS/Metal and pre-Ampere legacy builds were retired along with the v3 solver
because neither hardware path can run an ENC_RC episode at all.

| target | flow | script |
|--------|------|--------|
| **main** sm_80;86;89;90;120 | clean-stack | `clean-stack/build-clean-rel.sh` |
| HiveOS package | wraps the built binary | `clean-stack/build-hiveos-pkg.sh` |

## Two gates EVERY build must pass

1. **Byte-exact.** `rc_probe` must print
   `digest=5b1bff3c835b1c8e7816a2cccb181eb2fc30a99d97a971d73108c52a8238acd4`
   — the ENC_RC episode golden. On a CUDA build also run `rc_gpu_accel_probe`, which must report
   `cpu==golden 1, gpu==golden 1, gpu==cpu 1`: the CPU oracle alone cannot catch a GPU backend
   that has drifted. `rc_coupled_probe` (golden `a4bb0cc4`) gates the coupled-V3 leg.
   This proves the PoW MATH.

   > The old v3 gate — `digest_probe` printing `1dad86f1…` — is gone with the solver it described.
   > Any doc, script, or memory still quoting `1dad86f1` is stale.

2. **A-B perf (NON-NEGOTIABLE).** Byte-exact does NOT mean same runtime speed; this repo learned
   that the hard way when a byte-exact clean-stack build ran 28% slower than its predecessor.
   ALWAYS A-B against the prior binary in the **SAME GPU state**. The health metric is **`ep/s`
   on the `[stats]` line + board power**. One episode is a full dependent INT8 GEMM chain for one
   nonce (~825 ms on a 5090 at production dims), so `ep/s` is fractional — around 1.2/s per card.
   `rc-active=0` means the RC height never latched and nothing is being mined; that is a
   configuration fault, not a slow build.

   **Lock the clocks for any episode-level A/B.** Free-boost spread (~24 ms) is larger than most
   lever effects, so an unlocked comparison cannot resolve them: `nvidia-smi -lgc 0,2600`, then
   A/B/A. Give a deploy decision one boost-clock guard arm, since locked clocks shift the
   GEMM/extract balance.

## Toolchain (do NOT change)

- **CUDA 13.3.33**. Deps image: `matador-build:pathb-deps-cm4`.
- **Arch:** `sm_80;86;89;90;120` (Ampere/Ada/Hopper/Blackwell). Each emits
  `--generate-code=arch=compute_X,code=[compute_X,sm_X]` (PTX + native SASS). The 5090 runs
  native sm_120. Pre-Ampere is not built: ENC_RC is tensor-core-bound.
- **Optimization:** `-O3 -DNDEBUG` (from `CMAKE_BUILD_TYPE=Release`). Release is REQUIRED — the
  CMake default is `-O0`, and the probes carry their own `-O3`, so a probe can look fine while
  `matador_core` is unoptimized.

## Flags (`build-clean-rel.sh`)

```
cmake -B build-rel -S core -DMATADOR_ENABLE_CUDA=ON -DMATADOR_CUDA_ARCH="80;86;89;90;120" \
  -DMATADOR_MINER_VERSION="$VER" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_FLAGS="-Dconsteval=constexpr" -DBTX_ARCHIVE_DIR=/btx
```

`-S core` is REQUIRED (else "no CMakeLists in /src"). Needs a built stock btx tree at `$BTX`
(`~/git/btx-stock/build`) — the miner links the stock UNPATCHED btx archives for the node
plumbing (GBT/tx/address/shielded/consensus) and `matador_core` wins the duplicate symbols by
link order.

## cuBLASLt pruning

Stock `libcublasLt_static.a` is 799 MB and carries SASS for every arch NVIDIA supports.
`MATADOR_PRUNE_CUBLASLT` (ON by default) runs `nvprune` down to `MATADOR_CUDA_ARCH` at configure
time. This removes DEVICE CODE ONLY, for archs we never emit kernels for, so no instruction that
executes changes and the golden gate is unaffected.

GOTCHA: `nvprune`'s `--arch` does NOT accumulate. Passing it repeatedly warns "incompatible
redefinition for option 'arch'" and silently keeps only the LAST value, pruning a multi-arch
build down to one arch. Multiple archs REQUIRE repeated `-gencode`.

## Profiling

`RC_PROFILE=1` is the profiling knob. It turns on per-stage attribution inside the episode
backend (`core/cuda/rc_gpu_episode.cu`) and prints a breakdown:

```
phase1+2 | MX operand gen (GPU) | H2D upload | GEMM | Extract | D2H | absorb | final | phase3
```

The stage timers need ~80 `cudaDeviceSynchronize()` calls per episode to be attributable at
all (kernels are async), so they are pure overhead on the fast path and stay opt-in. Never
benchmark with `RC_PROFILE` set — profile with it, then measure without it.

It is reached through the bench entry in `rc_gpu_episode.cu`, not the miner: the miner prints
throughput (`ep/s`), not a stage breakdown.

There is no separate PROF build any more. `MATADOR_PROF_ENABLED`, `BTX_MATMUL_PROFILE`,
`[pipeline-prof]` and `build-h1prof.sh` were removed with the v3 pipeline whose stages they
timed — the flags had become no-ops that produced a binary identical to release while
advertising themselves as a profiling build.

## Historical lesson worth keeping

The v0.6.4/0.6.5 regression (~22.5k vs ~31k, byte-exact either way) was a stubbed batch-prepare
in the host FEEDING path, not the kernels — cuobjdump SASS was byte-identical, and CUDA version,
`-Dconsteval=constexpr` and `-S core` were all wrong suspects. The lesson generalizes past v3:
**byte-exact proves the MATH, not the speed.** When a correct build is slow, diff the host-side
feeding path and watch power, not the digest.
