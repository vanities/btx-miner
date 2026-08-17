# PR#89 rapid-react runbook

How to go from "numair pushed" to "measured + byte-exact in our code" in hours, not days.
Everything here is hard-won recipe — do not reconstruct from memory.

## 0. Triage (always first)
```
research/matmul-v4/watch-pr89.sh          # classify the delta
```
Then act by the highest flag it raised (see §5 playbook). `--accept` advances the pin once reviewed.

## 1. Build his matmul targets on pc (NOT test_btx)
`test_btx` is blocked by an unrelated Boost `multi_index` incompatibility in `txmempool.h`. The
matmul targets build fine and don't drag in the node:
```
ssh pc; cd ~/btx-pr89
git worktree add -f ~/btx-react refs/remotes/pr89-head      # clean tree at the head
docker run --rm -v ~/btx-react:/src -w /src matador-build:pathb-deps-cm4 bash -c '
  cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DBUILD_GUI=OFF -DBUILD_TESTS=ON -DENABLE_WALLET=OFF
  cmake --build build --target btx_matmul_backend btx-matmul-cost-bench -j12'
```
- `btx-matmul-cost-bench` = his puzzle-cost bench (JSON timings; `product_digest_dense` is a
  reference path, NOT the tuned native kernel — do not read it as FP4 throughput).
- `libbtx_matmul_backend.a` = link target if we want to call his kernels
  (`TryLaunchRcOzakiMxfp4GemmS8S8Int64Device`) from our own harness.
- sm_120a native MXFP4 build (his closure ask): add `-DBTX_ENABLE_CUDA_EXPERIMENTAL=ON
  -DBTX_CUDA_ARCHITECTURES=120 -DBTX_CUDA_SM120_MXFP4_NATIVE=ON`; the verbose nvcc line must carry
  `-gencode=arch=compute_120a,code=sm_120a` and no literal `SHELL:` token.

## 2. Dump ground-truth intermediates (the method that makes byte-exact work first-try)
Don't write blind and diff the final digest — bisect against pinned checkpoints. Add a dumper test
to his coupled/rc test file, build test_btx IF it builds (else a standalone linking the backend .a),
and capture: sigma, bank_root_seed, first bank page, lobe seeds, per-barrier roots, final digest.
For the coupled path sigma == our uncoupled sigma `86c171d7...` for `MakeCoupHeader(42)` (same
fields), and `ComputeMatMulHeaderHash` is a single SHA over LE-serialized header fields — so sigma
and bank_root_seed can be RECOMPUTED, no dump needed (see rc_coupled_solver.cpp). Only dump what
you can't recompute.

## 3. Reproduce byte-exact in our oracle
- Uncoupled episode: `research/matmul-v4/bench/rc_gpu_solver.cu` (v4.6 fused-FFN golden
  `5b1bff3c`; byte-exact CPU+GPU on the 5090; mode 3 = datacenter profile-2 shape), clean-stack
  `core/matmul/matmul_v4_rc.{h,cpp}` + `harness/rc_probe.cpp` (same golden).
- Coupled: `rc_coupled_solver.cpp` (V1 golden `7a7ce106`, still valid upstream) and
  `rc_coupled_solver_v3.cpp` (V3, golden `a4bb0cc4` BYTE-EXACT: COUP_*_V3 tags, M-row GEMM,
  u64-wrap mix, exchange rounds). Clean-stack: `core/matmul/matmul_v4_rc_coupled.*` +
  `harness/rc_coupled_probe.cpp`. Production exchange_rounds=4 path needs a dumped ground truth
  (no upstream golden).
- STABLE primitives that transfer across every rewrite (v4.4-LT -> V1 -> V2 -> V3): SHA256(d),
  ChaCha20, M11 table, MX expand (`expand_mx_dequant_i8`), MX Extract (`extract_mx_tile_i64`),
  butterfly mix, Fisher-Yates perm, bank commitment, barrier root, digest fold. Only the ASSEMBLY
  (which primitive, what order, what dims, which tag version) changes. Reuse the primitives verbatim.
- WATCH: `uint256::GetHex()` prints `.data()` REVERSED; single-vs-double SHA (BarrierRoot &
  BankCommitment are `Sha256dBytes`/double, most tags are single). These cost real time each build.

## 4. Bench (committed tools, all parameterized)
- `research/matmul-v4/bench/rc_fleet_bench.py` — per-card episode + power + `--stream-penalty`
  + `--mxfp4-ceiling`. `--all` = 5090 local + 3090 + B200 vast.
- `research/matmul-v4/bench/v3_adversarial_primitives.cu` + `v3_adversarial_model.py` — the M=128
  per-page GEMM + regen + Q-batch economics. Re-run with new dims when PROFILE changes.
- `research/matmul-v4/bench/fatm_gemm_scaling.cu` — GEMM throughput vs M (the fat-M / batch curve).
- Vast: minimal spot checks (~$0.15 B200). Always destroy-on-exit; `show instances-v1` (v0 is 410).

## 5. Playbook — what to do per classifier flag
| flag | action |
|---|---|
| **ACTIVATION** | RC/coupled going live on a public net. HIGHEST. Confirm the height + which profile. Build the competitive V3 solver NOW (fill the v3 scaffold, validate vs dumped ground-truth), wire into clean-stack, get the SolveMatMul RC branch off fail-closed. This is real money. |
| **GOLDEN** | Byte-exactness moved. Dump the new ground-truth (§2), reproduce in the oracle (§3), re-pin our golden. If it's a V3 toy golden freezing, the V3 scaffold is now worth finishing. |
| **PROFILE** | Dims/tags changed. Re-run the economics (§4 v3_adversarial + fatm) at the new shape; update the 5090-vs-B200 finding. |
| **ARBITER** | Proof now gates consensus. Read the arbiter wiring; re-check whether our shares would verify. |
| **ECONOMICS** | He touched our lane / may have answered the finding. Read the diff'd doc/code; update or defend our numbers. |
| **GKR-ONLY** | Crypto hardening. Low priority. Keep monitoring; no action. |

## 6. Standing guardrails (his + ours)
Heights INT32_MAX until we SEE one set; never push to his PR; our work stays in matador-src as
research/inert; keep the 5090 mining between benches (ABM — `systemctl` trap on every bench).
