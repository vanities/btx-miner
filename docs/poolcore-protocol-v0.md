# Matador poolcore protocol v0 (draft for Atticus)

Transport: **supervised subprocess, JSON-lines** — one JSON object per line, UTF-8.
Wrapper→core on **stdin**, core→wrapper on **stdout** (stdout carries protocol messages ONLY;
all logs are on stderr). Launch: `matador-miner --poolcore`.

Conventions:
- All byte fields are **lowercase hex strings** (no `0x`).
- `schema_version` (integer) is a versioned contract: breaking changes bump it, additive
  fields do not. Current: `1`.
- `profile` is a **string**, not a number, to avoid enum drift vs upstream:
  - `"btx-live"` — the current live BTX PoW. Fully functional path; use this to test the
    whole wrapper↔core↔pool loop end-to-end against the live chain TODAY.
  - `"enc-rc-v46"` — the v4.6 ENC_RC episode profile. Interface identical; challenge derivation is
    versioned (`derivation` field) because upstream has not finalized parameters. When they
    land, only the derivation version changes — wrapper code does not. Dispatches to the byte-exact
    episode solve loop; DEFAULT ON since miner v0.9.2 (`BTX_RC_ENABLE_EPISODE_SOLVE=0`
    force-disables; miners ≤ v0.9.1 instead required `=1` to grind and refused jobs otherwise).
    `"enc-rc-v47"` is the accepted v4.7 relabel — same fields, same solver, label echoed back.
  - `"enc-rc-v46-coupled"` — the v4.6 ENC_RC coupled (V3) profile: the episode leg bound to the
    ~48 GiB bank puzzle (the strictest fork; it supersedes bare `enc-rc-v46` where both are active).
    Same job shape (`header_hex`/`target_hex`/`height`/`parent_mtp`). Dispatches to the byte-exact
    coupled solve loop; GATED off by default (`BTX_RC_ENABLE_COUPLED_SOLVE=1` to grind). On a winner
    it emits a `block_winner` solution plus a `witness_version:"enc-rc-coupled-v0"` witness carrying
    the nonce-independent `bank_template_hash` (the pool re-derives the bank from it).
- Every core→wrapper message carries `ts_unix_ms`.

## Startup

Core emits immediately on launch:
```json
{"type":"hello","protocol":"poolcore-v0","schema_version":1,"solver_version":"v0.8.x","ts_unix_ms":0}
```
Wrapper should gate on `protocol`/`schema_version` before proceeding.

## Wrapper → core

```json
{"type":"init","devices":[],"backend":"auto","extra":{}}
{"type":"job","job_id":1,"profile":"btx-live","derivation":"v0","challenge":{"header_hex":"...","height":123456,"parent_mtp":1784900000},"target_hex":"...","clean":true}
{"type":"stats"}
{"type":"preempt","job_id":1}
{"type":"shutdown"}
```

- `init.devices`: list of solver indices to bind (empty = all).
- `job.challenge` is an object keyed by `derivation` version. For `btx-live` v0 it is the
  80-byte header template (`header_hex`, nonce field zeroed) plus `height` (block height the
  job targets) and `parent_mtp` (parent block's median-time-past, unix seconds) — both are
  consensus inputs the solver requires; the pool has them from its own node/template. For
  `enc-rc-v46` the v0 shape will be proposed alongside the first RC test vectors. `clean:true`
  means drop the current job (new block) — equivalent to preempt+job in one message.
- `stats`: poll (suggested cadence 5 s).

## Core → wrapper

```json
{"type":"init_ok","schema_version":1,"solver_version":"v0.8.x","devices":[
  {"solver_index":0,"gpu_uuid":"GPU-xxxx","pci_bus_id":"00000000:01:00.0",
   "name":"NVIDIA GeForce RTX 5090","backend":"cuda"}],"ts_unix_ms":0}

{"type":"job_ack","job_id":1,"profile":"btx-live","ts_unix_ms":0}

{"type":"solution","job_id":1,"kind":"share","gpu_uuid":"GPU-xxxx",
 "nonce_hex":"...","digest_hex":"...","profile":"btx-live","payload":{},"ts_unix_ms":0}

{"type":"stats","schema_version":1,"devices":[
  {"gpu_uuid":"GPU-xxxx","pci_bus_id":"00000000:01:00.0","solver_index":0,
   "nps":0.0,"backend":"cuda","solver_version":"v0.8.x","errors":[]}],"ts_unix_ms":0}

{"type":"preempt_ok","job_id":1,"ts_unix_ms":0}
{"type":"shutdown_ok","ts_unix_ms":0}
{"type":"error","what":"...","detail":"...","ts_unix_ms":0}
```

Notes:
- **Solutions are pushed immediately** on find (not poll-only) — lowest share latency.
- `gpu_uuid` is the NVML UUID (join key), `pci_bus_id` the fallback, on both `DeviceBinding`
  and stats rows. Non-NVIDIA backends (Metal) will carry a stable platform id plus a
  `uuid_kind` tag — flagged, not silently different.
- `nps` = 10-second rolling mean per device (agreed definition).
- `errors[]`: device drop / kernel fault / determinism-fault strings. Determinism behavior is
  **fail-closed**: a device failing the digest self-check is dropped and reported; matador
  never submits an unverified solution.

## Proving-witness export (first-class requirement)

For every `kind:"block_winner"` solution on `enc-rc-v46`, the core ALSO emits a `witness`
message (`witness_version:"enc-rc-v0"`: header, nonce, digest, round_roots) sufficient for an
external prover to reconstruct the full episode execution, on the line immediately after the
solution, with the same `job_id`/`nonce_hex`/`digest_hex` triple. Field encodings are FROZEN
(2026-07-27) in `poolcore-witness-v0.md`: every `*_hex` is lowercase hex, no `0x`, fixed
width, big-endian display for hash-like values; nonce is 16 hex chars; round_roots are
round-ordered 0..R-1, 64 hex chars each. v1 (sampled-openings export, prover skips
re-execution) freezes jointly when upstream's carrier format goes final.

## Share vs block-winner payloads (important)

- `kind:"share"`: `nonce_hex` + `digest_hex` — verifiable against the share target. Available
  for both profiles from day one.
- `kind:"block_winner"` (reserved): for `enc-rc-v46` profile-2, consensus block acceptance
  additionally requires the sampled-carrier proof, whose format/prover is **not finalized
  upstream**. The field exists in the schema now so nothing breaks later; the pool should NOT
  assume per-share Freivalds payloads — shares are digest-vs-target, the carrier is a
  block-winner-only artifact.

## Dispatch status by build (IMPORTANT — supersedes any "stubbed" note)

- **s1 scaffold (HISTORICAL ONLY — no longer shipped)**: dispatch was STUBBED — `job_ack`
  carried `note:"dispatch_not_wired_scaffold"` and no `solution` was emitted. This line is
  retained purely so old s1 logs can be interpreted; it does NOT describe any current build.
- **s2 and later (ALL CURRENT BUILDS)**: **`btx-live` dispatch is LIVE.** A
  `{"profile":"btx-live"}` job produces real `{"type":"solution","kind":"share",...}` messages
  the moment a share is found (validated against a live-chain getblocktemplate job).
  `enc-rc-v46`/`enc-rc-v47` dispatch + witness export are also wired, fail-closed until
  explicitly enabled.

## v4.7 phase mapping (upstream PR#97, 2026-07-30)

Upstream relabeled v4.6-v3 as **v4.7** and published a four-epoch transition. The poolcore
mapping:

- **Profile strings**: `enc-rc-v47` is accepted as of this revision and is the preferred label;
  `enc-rc-v46` remains a permanent alias (same job fields, same solver, same digest bytes).
  Whichever string the pool sends is echoed back verbatim in `solution`/`witness` emissions.
- **Epoch A (Phase 1, ExactReplay authority)**: jobs run the **Profile 1** episode shape
  (rounds=4, L=16, b_seq=16384, T_leaf=1024, dim 4096) — the miner's launch default. NOTE:
  `btx-live` is the CURRENT chain's v3 base PoW, not the Phase-1 vehicle; at the Epoch A
  activation height the network switches to the RC episode workload and the pool switches its
  jobs from `btx-live` to `enc-rc-v47`. The share/solution/witness message flow is identical
  across both, by design.
- **Witness in Epoch A**: the miner emits `witness` (enc-rc-v0: header/nonce/digest/
  round_roots) with every `block_winner` from Epoch A onward, even though the carrier is not
  consensus-relevant until Epoch B — this is the zero-risk shadow-proof burn-in window for the
  pool's prover.
- **Epoch D (Profile 2 datacenter shape)** activates at a separate height and will get a
  distinct profile string when upstream freezes its parameters; it is NOT selectable via
  `enc-rc-v47`.

## enc-rc-v47 job semantics (2026-08-03 freeze additions)

Normative details pinned for the Byron adapter freeze (all verified against the sealed
upstream activation tip `909aa703`, mainnet `nMatMulRCHeight = 181894`):

- **`header_hex` stays the 80-byte CLASSIC template** (160 hex chars): version(4 LE) @0,
  prevhash(32, internal order) @4, merkleroot(32, internal order) @36, nTime(4 LE) @68,
  nBits(4 LE) @72, bytes 76..79 ignored (legacy 32-bit nonce slot; the solver owns the
  nonce). The 182-byte serialized BTX header (nNonce64 @76, matmul_digest @84,
  matmul_dim @116, seed_a @118, seed_b @150 — all ints LE) is the ON-CHAIN form the POOL
  assembles; it never crosses this wire.
- **`matmul_n` (optional) now applies to enc-rc jobs too.** Consensus at RC heights
  requires header `matmul_dim == nMatMulV4Dimension` (4096 mainnet) and the field is in
  BOTH the V3 seed preimage and the sigma preimage. The core stamps it before deriving
  seeds: pool-sent `matmul_n` wins; absent/0 falls back to the consensus RC dim (4096;
  env `BTX_MATMUL_RC_HEADER_DIM` overrides for foreign nets). Builds before 2026-08-03
  ground RC jobs at dim 0 — their RC digests were unacceptable to any node; pools must
  not mix pre-freeze cores into an RC deployment.
- **Targets**: `target_hex` is the POOL SHARE target — 64 hex chars, big-endian display
  (most-significant byte first), `uint256::FromHex`. Admission comparison is INCLUSIVE:
  `digest <= target`. The consensus BLOCK target is derived core-side from the job's own
  `nBits` (inside `header_hex`); the pre-hash sigma gate is RETIRED at v4 heights, so
  digest-vs-target is the entire lottery on both tiers.
- **Solution kinds**: every nonce whose episode digest meets `target_hex` is pushed
  immediately as `kind:"share"`. If the digest ALSO meets the nBits block target
  (inclusive) the solution is `kind:"block_winner"` and the `witness` line follows.
  Witness is WINNER-ONLY (it costs one extra episode to materialize round_roots).
- **VarDiff**: there is no separate set-target message — send a NEW `job` with the new
  `target_hex` (job replacement aborts the in-flight solve; the nonce cursor restarts at
  0). Retarget on template refresh, or accept losing the in-flight nonce's partial
  episode (~1 s Profile-1 on a 5090). No reconnect exists in this transport.
- **Nonce space**: the core starts every job at nonce64 = 0. Worker uniqueness is the
  pool's job via per-worker templates (extranonce in the merkle root), exactly like
  stratum.

## `matmul_n` is part of the `btx-live` job

`matmul_n` is an OPTIONAL field of the `btx-live` job's `challenge` object (the consensus matmul
dimension for the height). If absent or `0`, the core falls back to the consensus default for the
job's `height` (`nMatMulDimension`). Include it when your template carries it; omit it to accept
the height default. It is NOT part of the 80-byte `header_hex` (which stays the classic header);
it is a separate consensus input the pool already has from its node/template.
