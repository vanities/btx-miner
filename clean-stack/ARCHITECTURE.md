# Matador — clean-stack miner architecture

Our own stack for the BTX **ENC_RC (v4)** PoW, **extracted** from the btx node codebase instead of
patched into it. Byte-exact solver (non-negotiable — wrong digest = rejected share), clean
ownership, a real base to build fleet/multi-GPU on.

The miner links the stock UNPATCHED btx archives for the node plumbing and provides its own
solver in `matador_core`. Byte-exact against the episode golden `5b1bff3c`, mines rej=0. Release
build = `build-clean-rel.sh`, gated by the `rc_probe` target.

## The core/miner split

```
clean-stack/
  core/                  byte-exact PoW + backends (the part that must never change semantically)
    matmul/              matmul_v4_rc{,_coupled}   ENC_RC episode + coupled-V3 CPU oracles
                         matmul_pow                header hash + DeriveSigma (the RC preimage root)
                         pow_solve                 seeds, DeriveTarget, DGW/ASERT retarget,
                                                   SolveMatMul -> RC episode / coupled solve loops
                         backend_capabilities      which backend is compiled/usable/active
                         power_sampler             [power-prof] board energy (PROF builds)
    cuda/                rc_episode_accel.cu       GPU episode backend (byte-exact vs the oracle)
                         rc_coupled_accel.cu       GPU coupled-V3 backend (device-derived bank)
                         rc_gpu_episode.cu         the shared episode kernel chain
                         rc_gemm_i8{,_cutlass}     the INT8 tensor-core GEMM backends
    vendor/              minimal btx util closure (NOT the node) — see "Vendoring" below
    test/                ported node parity tests for the surviving surface (dgw, header,
                         subsidy, block_capacity). Not wired into this CMake; the live gates
                         are the rc_* probes and run-tests.sh.
  miner/                 orchestration (pool client, config, pipeline, idle-gate, hub/fleet).
                         ONE translation unit (matador-miner.cpp, no LTO) split into focused,
                         #included section headers so the hot pool path keeps cross-function
                         inlining while the file stays navigable (see the MODULE MAP at the top):
                           matador-miner.cpp  pool-mining core: Config, StratumClient, RunPoolLoop, main
                           solo_mining.h      solo RPC/GBT client, coinbase+block assembly, solve+submit
                           poolcore.h         minebtx poolcore compute-core protocol + RC job dispatch
                           config_parse.h     config-file / CLI parsing + RPC-auth
                           status_api.h       read-only HTTP status API + GPU/thermal JSON + watchdog
                           updater.h          startup auto-update (GitHub release -> swap -> re-exec)
                           gpu_tuning.h       optional NVML clock/power/fan tuning + revert
                           gpu_telemetry.h    NVML [gpu] heartbeat telemetry
                           mgpu_stats.h       multi-GPU rig roll-up ([stats-all])
                           cursor_persist.h   nonce-cursor persistence (resume past restart)
                           miner_format.h     pure difficulty/rate/duration formatters
                           miner_log.h        structured scope-tagged logging (mlog)
                         (plus the unit-tested leaves: version_compare/update_gate/
                         devfee_window/endpoint_parse/log_tee.) Links `core` instead of the node.
  harness/               rc_* golden + solve-loop probes, and the standalone *_test.cpp units
                         that run-tests.sh compiles on every commit.
  scripts/               dev scripts (byron stratum probe, dexbtx poolcore bridge).
```

**The seam:** the miner calls `SolveMatMul()` (or `SolveMatMulRCEpisode` directly, on the
poolcore path) and never sees a kernel. `matmul_v4_rc.h` declares both the CPU oracle and the GPU
entry points; `RCEpisodeGpuAvailable()` picks between them at runtime, and the GPU result is
byte-exact to the oracle by gate (`rc_gpu_accel_probe`).

## What was removed at the v4 cut

ENC_RC activated on mainnet, so the v3 matmul solver came out whole rather than being left as
dead weight. Gone: the grind pipeline (GPU pre-hash scan, batched digest, CPU scan-ahead,
parallel solver, `accelerated_solver`), its supporting math (`field`, `matrix`, `noise`,
`transcript`, `freivalds`, `derivesigma_avx2/512`), the v3 CUDA accelerators
(`matmul_accel.cu`, `oracle_accel.cu`, `cuda_context`, `cuda_scheduler`), the entire Metal
backend, the external HIP/AMD sidecar bridge, and the v3-only consensus verify paths
(Freivalds carrier, product-committed digest, phase1/phase2, the sigma pre-hash gate) with
their peer-verification budget plumbing.

Two deliberate exceptions:

- **`core/vendor/consensus/params.h` keeps its dead v3 fields.** Layout is ABI:
  `SelectParams()`/`Params()` come from the prebuilt btx archives, so the node builds
  `Consensus::Params` with ITS layout and hands it across. Removing a field shifts everything
  after it — including `nMatMulRCHeight`.
- **`core/vendor/pow.h` keeps the retarget surface.** Solo mining and the difficulty readout
  still need DGW/ASERT and `DeriveTarget`.

## Vendoring (why this is a thin layer, not half of Bitcoin-core)

The core references only a **small, self-contained** slice of btx node utilities:
`uint256`, `CSHA256`/`CHash256`, `Span`, `CBlockHeader`, serialize, prevector, attributes,
`crypto/{common,sha256,ripemd160}`, `compat/{endian,byteswap}`, `util/{strencodings,string}`.
The **real `crypto/sha256.cpp`** (with the SSE4/AVX2/SHA-NI and ARM-SHA2 paths) is vendored as
the CPU byte-exact reference — do NOT reimplement it.

## Build strategy

Standalone CMake, **no node deps** (no boost, secp256k1, zmq, wallet, qt):

- `matador_core` library: `matmul/` + `vendor/` + the CUDA backend when enabled.
- CUDA (sm_80;86;89;90;120): built on `pc`, CUDA 13.3, image `matador-build:pathb-deps-cm4`.
  `build-clean-rel.sh`. This is the only shipped lane.
- The miner links `matador_core` FIRST plus the stock UNPATCHED btx archives; `--start-group`
  resolves their circular deps and `--allow-multiple-definition` lets `matador_core`'s vendored
  duplicates win by link order.

## The byte-exact gate

- `rc_probe` — ENC_RC episode golden `5b1bff3c` (CPU oracle).
- `rc_gpu_accel_probe` — GPU == CPU == golden. The CPU oracle alone cannot catch GPU drift.
- `rc_coupled_probe` — coupled-V3 golden `a4bb0cc4`.
- `rc_solve_probe` / `rc_coupled_solve_probe` — the solve-loop contract (grind, target compare,
  winner == oracle).
- `./run-tests.sh` — the standalone `harness/*_test.cpp` units; fast enough for a pre-commit hook
  (`install-hooks.sh`).

**Every change must keep these green** + live pool `rej=0`. That is how we prove a refactor
didn't change a single digest bit.

## Notes / risks

- **Byte-exactness is the whole game.** Keep reduction and accumulation orders identical; the
  episode is a dependent INT8 GEMM chain and a reassociation is a fork.
- This is a private stack (kernels = secret sauce). Public repo stays binary-only.
- Source of truth = `vanities/matador-src`, the `clean-stack/` tree.
