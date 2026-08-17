# RC miner integration — readiness spec & turnkey plan (2026-07-23)

What it takes to mine v4.6 ENC_RC the moment a height activates, extracted from the consensus
reference (btx PR#89 `src/pow.cpp` `SolveMatMulV4RC` / `SolveMatMulV4RCCoupled` +
`CheckMatMulProofOfWork_RC`). This is the exact contract our `SolveMatMul` RC branch must satisfy.

## The PoW predicate (the easy part — we have it)

Per nonce, mirroring `SolveMatMulV4RC` (pow.cpp ~6188):

1. `SetDeterministicMatMulSeeds(header, params, height, pmtp)` — pin seed_a/seed_b per nonce. **have**
2. `sigma = DeriveSigma(header)`. **have**
3. `mined = episode_digest(sigma, params_rc)` via `ComputeEpisodeDigest`. **have (byte-exact 5b1bff3c)**
4. If `UintToArith256(mined) > effective_target` → loser, `--max_tries`, `++nonce`, continue.
5. Winner: CPU-reseal `RecomputeResidentCurriculumReference` (our oracle), abort on mismatch,
   set `block.matmul_digest = resealed`. **have**

`params_rc = ResolveRCEpisodeParams(params, height)` selects dims by `nMatMulRCProfile`
(1=base, 2=datacenter). We have `DatacenterEpisodeParams()` / `ProductionEpisodeParams()`; the
resolver is a trivial `profile==2 ? Datacenter : Base` switch.

Coupled leg (`SolveMatMulV4RCCoupled`, pow.cpp ~6113) is the same shape over
`TryMineRCCoupledBatch` / `RecomputeCoupledPuzzleReference` — we have the byte-exact V3 oracle
(`ComputeCoupledDigestV3`, a4bb0cc4), needs `RCBankTemplateHash(header)` (recomputable from the
header, present in our research solver).

## The HARD BLOCKER (mainnet, profile 2) — the succinct carrier prover

Mainnet defaults `nMatMulRCProfile = 2`. Under profile 2, `CheckMatMulProofOfWork_RC`
(pow.cpp:4166) **REJECTS the block unless a valid Freivalds sampled carrier is present**
(pow.cpp:4216 `RCFreivaldsCarrierStoreGet` → no carrier = `finish(false)`). The winner must, at
solve time (pow.cpp:6270), build:

- `ProveWinnerEpisodeV7(block, params_rc, height, target, resealed)` → the v7 GKR/FRI succinct
  proof (the G1–G5 in-circuit arithmetization).
- `BuildFreivaldsSampledCarrier(proof, ...)` → the relay-optimized sampled carrier (λ sampled
  layers' bytes + tile-tree openings), stored by header hash for the verifier + P2P RCCARRIER relay.

**We do NOT have `ProveWinnerEpisodeV7` or the carrier builder.** This is the deep succinct-proof
crypto stack — the same work numair assigned to a dedicated agent. Without it, every profile-2
block we mine fails closed at verify. So:

> **The miner cannot be made mainnet-ready for RC by us alone. The blocker is the profile-2
> succinct-carrier prover, which is external (numair's agent) — not a wiring gap.**

The verifier also binds the full 9-field episode shape (rounds, d_head, n_q, n_ctx, L_lyr,
d_model, b_seq, T_leaf, **d_ff**) and λ = `kRCFreivaldsSampleCount` to consensus constants, so the
carrier must carry exactly the consensus dims — no cheaper substitute.

## Second blocker: the pool/stratum RC work format is undefined

Our deployed miner mines via **pool (stratum)**. numair has not published how RC work is handed
out or what an RC share looks like. The SOLO/GBT path is derivable (mine against our own node), but
the production pool path waits on upstream. Solo is enough for regtest end-to-end validation.

## What IS ready / done (controllable)

- **Delivery**: auto-update prerelease, `min_version_age_s=0` → new release deploys in ~5 min.
- **Byte-exact oracles**, CI-gated: episode `5b1bff3c`, coupled V3 `a4bb0cc4` (pc CUDA + M4 Max + cross-arch).
- **Loud activation alarm**: `SolveMatMul` RC branch logs "MATMUL RC ACTIVE ... mining HALTED"
  (60s throttle) instead of silently idling.
- **Fast GPU episode solver**: 12.0 s/episode DC-dims on a 5090 (bench; not yet wired into SolveMatMul).
- **Activation tripwires**: watch-pr89.sh (ratification gate / height-assign / ASERT-16422 sites).

## Turnkey plan when the blockers clear

1. **Now (pre-staging, safe):** wire the episode + coupled digest solve loop into `SolveMatMul`
   (steps above), gated behind the RC height (inert on mainnet). Profile 1 / regtest / toy → mines
   valid blocks. Profile 2 → computes the digest but fails closed at the carrier with a loud log
   (explicit plug-in point). Validate end-to-end on regtest: `btxd -regtest
   -regtestrcunifiedheight=200 -regtestrctoydims=1`, mine → node accepts.
2. **When the prover lands** (upstream ships `ProveWinnerEpisodeV7`/carrier, or we build it): plug
   it into the winner branch; profile-2 blocks become valid.
3. **When the pool RC format lands**: wire RC job parse + share submission; integrate the GPU
   solver for production-dim throughput.
4. Cut a prerelease → auto-deploys in ~5 min.

## Bottom line

Digest solving is a solved problem for us (byte-exact, fast). Mainnet RC mining is blocked on two
external deliverables — the profile-2 succinct-carrier prover and the pool RC work format — neither
of which we can complete alone today. The controllable readiness (oracles, alarm, delivery,
tripwires, this spec) is maxed; the remaining work is turnkey the moment those land.
