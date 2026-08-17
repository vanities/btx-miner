# v4.6 B200 ↔ RTX 5090 — numair's canonical harness, matched run (2026-07-23)

Both cards rented on Vast, **identical environment** (`nvidia/cuda:12.8.1-devel-ubuntu22.04`,
Ninja Release build of PR#89 tip **`d0287a1`**, `BUILD_TESTS=ON`, `BTX_ENABLE_CUDA_EXPERIMENTAL=ON`),
run through `contrib/matmul-v4/measure-enc-rc-v46.sh` per the PR's community ask. Instances were
reclaimed by the provider minutes after the runs completed; all artifacts below were pulled first.
Total rental cost ≈ $4.

## Machines

| leg | GPU | UUID | driver | power limit | arch | $/hr (Vast, 2026-07-23) |
|---|---|---|---|---|---|---|
| A | NVIDIA B200 (183 GiB) | GPU-ebc7056c… | 595.71.05 | 1000 W | sm_100 | 6.494 |
| B | RTX 5090 (32 GiB) | GPU-bf1c0574… | 580.82.09 | 600 W | sm_120 | 0.456 |

## 1. `cuda-episode-tests` — byte-exactness (the consensus-split check)

| card | result |
|---|---|
| B200 (sm_100) | **PASS** — `*** No errors detected`, exit 0 |
| RTX 5090 (sm_120) | **PASS** — `*** No errors detected`, exit 0 |

Both accelerators match the int64 CPU reference on `rc_dc_cuda_episode*`. No consensus-split
signal on either card.

## 2. `cpu rc --coupled-v3-ci` — the difficulty-calibration input

Digest **identical across both machines and all four residency modes**:
`744fd3dfda6a58ddcd95474a9895cd2c6b17c2f1c2591848fc631eed78dea6a9`
(`modes_digest_match=true`, `mine_matches_cpu=true` on both). Full JSONs in this directory.

Per-mode nonce/s (NB: this entry is the **CPU backend** — numbers reflect each instance's host
CPU, recorded as calibration context, not a GPU comparison; host CPU model strings were lost
when the instances were reclaimed):

| mode | B200 host | 5090 host |
|---|---|---|
| SequentialLobes | 130.9 | 124.3 |
| Checkpointed | 59.9 | 64.1 |
| Streamed | 102.4 | 67.7 |
| Resident | 131.4 | 125.1 |

## 3. `verify-carrier` (900 ms relay budget) — validator leg

- B200 host: **PASS** (`No errors detected`, exit 0); the printed `total_ms` line was lost to a
  log-truncation bug in our driver before the instances were reclaimed.
- 5090 host: benchmark errored (exit 201) on that rented host — a host-CPU/RAM artifact of the
  cheap rental, not a GPU datapoint. Verify-carrier is a **validator CPU** measure; the clean
  numbers for this leg should come from real validator hardware (numair's M4 Max 330 ms GO
  reference; our pc CPU when it is back on the network).

## 4. Production-dim GPU economics (the actual B200-vs-5090 question)

`measure-enc-rc-v46.sh` has no GPU production-dim throughput entry today (§2 above is CPU-backend
by design; the full GPU matrix lives in `doc/btx-matmul-v4.5-v3-b200-5090-measurement-protocol.md`
and is not yet runnable turnkey). The matched GPU-vs-GPU numbers at the v4.6 **datacenter episode
dims** (profile-2: rounds=8, L=24, b_seq=87552, T_leaf=4096, Config W + row-block X0) come from
our independent byte-exact solver — golden-gated against the frozen toy vector `5b1bff3c` on each
card in the same run, digest `1cc5709d…` agreeing across sm_86/sm_100/sm_120:

| card | episode | measured power | ep/kWh | $/hr | ep/$ |
|---|---|---|---|---|---|
| RTX 5090 | 12.26 s | 546.7 W mean | 540 | 0.313 | **943** |
| B200 | 8.39 s | 580.8 W mean (89.5% util) | 738 | 6.494 | 66 |
| RTX 3090 | 128.3 s | 217.5 W | 132 | 0.113 | 256 |

Applying the protocol's own **Economic GO** test
(`B200_rate/5090_rate > B200_$/5090_$` with margin):

- rate ratio **1.46×** vs price ratio **6.494/0.456 ≈ 14.2×** (or 20.7× at the 5090's owned
  $0.313 electricity-inclusive figure) → **Economic GO: FAILED**, reason = *rent gap + tensor
  underuse* (B200 at 89.5% util; conc>1 untested and could not close a 10× gap).

The 3090 result supports the *other* half of the design claim: at v4.6 datacenter dims the 3090
is 10.5× slower than a 5090 — the "age out pre-2025 tails" goal is measured and real. What is
NOT supported by measurement is any B200-per-dollar advantage over a flagship consumer card.

## Caveats

- Our solver is a clean-room reimplementation; its absolute ep/s is our own optimization state
  (his GPU path may differ in either direction). The digest gates prove the WORK is identical.
- Rental prices are dated inputs (protocol's own caveat), sampled 2026-07-23 on Vast.
- B200 conc>1 (multi-episode overlap) untested at these dims; it lifts B200 utilization but
  cannot plausibly close a 10–14× per-$ gap.
