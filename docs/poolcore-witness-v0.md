# Matador ENC_RC proving-witness export (witness-v0) + prover answers

From: Vanities (Matador) - 2026-07-26
Re: Atticus's prover/carrier questions. This is now a first-class requirement in the
integration contract, alongside telemetry. Versioned like everything else.

## The structural fact that makes this tractable

The ENC_RC episode is **fully deterministic from (header, nonce, consensus params)**: every
operand, every round stream, every Merkle tree, and the digest re-derive from that tuple.
Consequence: the COMPLETE minimal witness for a winning nonce is small and Matador holds all
of it at the moment of the win. A prover with the witness can regenerate any intermediate it
wants; the only question is how much pre-materialized data we hand over so the prover can skip
re-executing the episode.

## witness-v0 (available with the enc-rc-v46 profile drop, stub testable before that)

Emitted automatically alongside any `kind:"block_winner"` solution:

```json
{"type":"witness","witness_version":"enc-rc-v0","job_id":9,
 "header_hex":"...","nonce_hex":"...","digest_hex":"...",
 "round_roots":["...64 hex chars per round..."],
 "profile":"enc-rc-v46","derivation":"v0","ts_unix_ms":0}
```

### Encodings (FROZEN, 2026-07-27 - answers to the pool's pin-downs)

1. **`nonce_hex` (witness) == `nonce_hex` (solution): same field name, same value, same
   encoding.** The earlier draft's `nonce64` name is retired. Encoding: the miner's 64-bit
   nonce as exactly **16 lowercase hex chars, big-endian display, zero-padded** (e.g.
   `00000000075bcd15`). A witness and its paired solution carry byte-identical strings.
2. **`digest_hex` identity: yes.** Witness and block-winner solution carry the exact same
   64-lowercase-hex string: the header `matmul_digest` in big-endian display (the node's
   `GetHex()` convention - the same orientation every digest in this project is quoted in).
3. **`job_id` pairing: yes, guaranteed.** The witness always carries the same poolcore
   `job_id` as its paired `block_winner` solution and is emitted on the very next line after
   it. The correlation triple `(job_id, nonce_hex, digest_hex)` always matches between the
   pair; treat any mismatch as a core bug, not an ambiguity to resolve.
4. **`round_roots`:** ordered **round 0 → R-1**, exactly the order the episode digest absorbs
   them (`digest = SHA256d(tag || roots[0..R-1])`, so the prover can re-derive `digest_hex`
   from this array as a self-check). Count == the profile's round count (datacenter profile:
   8). Each entry is exactly **64 lowercase hex chars (32 bytes), big-endian display**, same
   convention as `digest_hex`.

General rule, applies to every `*_hex` field in poolcore-v0: lowercase hex, no `0x`,
fixed width for fixed-size values, big-endian display for hash-like values.

That tuple is sufficient for any conforming prover to reconstruct the full execution. It is
also tiny (sub-KB), so it survives any transport and can be archived per block forever.

## witness-v1 (after upstream freezes the carrier format)

The carrier's sampled openings are deterministic given (header, digest, round_roots, block
target) - the Fiat-Shamir seed fixes WHICH tiles/leaves get opened. So at solve time Matador
can compute the exact opening set and export only that: sampled tiles + Merkle authentication
paths, megabytes not gigabytes. The solver already materializes every byte the openings touch
(the round streams pass through our GPU Merkle); v1 is a retention hook plus serialization,
not new computation. We will spec v1 jointly the week upstream's carrier format goes final -
it CANNOT responsibly freeze earlier, because upstream's anti-grind redesign (FVT: terminal
round handling) is explicitly going to change what a valid carrier opens.

## Direct answers

1. **What goes in `block_winner.payload`?** Under witness-v0: nothing extra - the witness
   message above rides alongside, and the prover (yours) builds the carrier from it. If/when
   in-solver proving exists (see 5), a finished carrier could ride in `payload` as an
   alternative, negotiated then.
2. **Does Matador prove internally or emit the witness?** Matador emits the witness and keeps
   proving OPEN. You build and run the prover on top of it for this deployment.
3. **Witness/trace export early, even stubbed?** Yes - witness-v0 above is implementable
   against the current upstream head (round_roots + digest are already computed per episode).
   It ships with the enc-rc-v46 profile drop; the message shape is frozen now so your prover
   scaffolding can start immediately.
4. **Compute/memory profile of carrier generation:** honest read from our review of the
   current upstream head: the winner-side proving path as written is CPU-shaped and its own
   code marks it over-budget/parked at consensus dims (measured: ~12.6 s at TOY dims against
   a 2.0 s budget, proof ~3.2 MiB) - upstream must land the redesign before anyone sizes
   hardware seriously. Our expectation for what survives: episode re-execution (GPU-shaped -
   our solver does it in ~12 s at datacenter dims on a 5090) plus Merkle/hash work over the
   opened set (also GPU-friendly). That fits a GPU fleet, not a big-memory-box - but treat
   any sizing as provisional until the FVT redesign lands. We track the upstream prover daily
   and will flag the moment the shape changes.
5. **Commercial:** witness export is IN-SCOPE for Matador as part of the solver contract - no
   separate fee, it is part of what the 1.0% covers. The prover business is YOURS to own for
   this deployment; we take no cut of prover-side revenue. Symmetric reservation: Matador
   stays free to ship its own prover in other deployments or to offer an optional in-solver
   proving mode later - offered, never forced, and never a change to the witness interface
   you build on. Same non-exclusivity logic as the solver itself.

## Near-term items from your reply

- **Binary + share_verify_probe:** packaging both for you now (poolcore scaffold binary +
  the CPU probe with its self-test vector).
- **btx-live dispatch ETA:** days, not weeks - the solver seam is mapped and the protocol
  fields it needs (height, parent_mtp) are already in the doc.
- **Byte-level encoding lock:** yes, btx-live first. Propose edits directly on
  poolcore-protocol-v0.md.
- **WSL2 + CUDA passthrough:** expected to work (standard CUDA userland, no exotic driver
  calls; the NVML identity query falls back cleanly) but NOT yet validated by us - we will
  run it before calling it supported, and it goes in the compatibility table either way.
- **Metal `uuid_kind`:** `{"uuid_kind":"metal-registry-id","gpu_uuid":"<IORegistry GPU id,
  stable across reboots on the same machine>"}`; NVIDIA rows carry `"uuid_kind":"nvml"`
  implicitly (absent = nvml). Exact Metal id source documented with the Metal build.
