# V4.5 V3 adversarial economics: matched 5090 vs B200 (measured 2026-07-21)

His V3 profile: M=128 (rows_per_lobe), K=N=8192, 1536 pages, 51 GiB packed, 12 TiMAC/nonce.

## Measured primitives (per-page GEMM M=128 K=N=8192, int8)
| card | per-page GEMM | compute/nonce (1536) | edge |
|---|---|---|---|
| RTX 5090 (sm_120) | 0.0333 ms | 51.2 ms | 1.00x |
| B200 (sm_100) | 0.0242 ms | 37.2 ms | **1.38x** |

**B200 is only 1.38x a 5090 on this workload** -- M=128 is tall-skinny, so the B200's extra
tensor width is idle. Its raw edge is nowhere near its ~20x rent premium.

## Per-dollar, matched (B200 26.9 nonce/s resident, 5090 batched), rent 20.2x
| batch Q | 5090 n/s | rate B200/5090 | per-$ 5090 vs B200 | winner/$ |
|---|---|---|---|---|
| 1 | 1.12 | 24.0x | 0.8x | B200 |
| 8 | 6.40 | 4.2x | 4.8x | **5090** |
| 32 | 12.9 | 2.1x | 9.7x | **5090** |
| 128 | 17.3 | 1.6x | 13.0x | **5090** |
| 1024 | 19.2 | 1.4x | 14.4x | **5090** |

Across his whole sweep at Q=256: 51 GiB -> 13.8x, 64 -> 13.5x, 80 -> 13.3x, 96 -> 13.1x. The
5090 wins per dollar by ~13x at EVERY bank size he proposes.

## Verdict (his Workstream J/K)
Intended screenshot outcome (B200-over-5090 economics) = **FAILED** at the V3 profile.
Root causes, all measured:
1. nonce-null (template-scoped) bank -> batching amortises the streaming/regen penalty to ~0;
   he cannot cap Q in consensus.
2. M=128 tall-skinny -> B200's tensor width unused, only 1.38x raw compute edge.
3. ~20x rent premium the 1.38x edge cannot overcome.
batchable <=> amortisable: the property he needs for HeadersShareBankTemplate is the one that
erases the moat. Bigger banks do not fix it (regen amortises; compute edge stays 1.38x).

## CORRECTION (fat-M): the realistic operating point, measured on both cards
A rational miner batches nonces into a fat-M GEMM (his own batched-sketch: `P·[B1V|…|BQV]`).
Measured peak int8 throughput vs M (K=N=8192):
| M | 5090 TMAC/s | B200 TMAC/s | B200 edge |
|---|---|---|---|
| 128 (tall-skinny) | 256 | 335 | 1.38x |
| 16384 (fat, Q=128) | 450 | 1986 | **4.4x** |

Fat-M gives the B200 its tensor width -> edge grows 1.38x -> **4.4x**. So the earlier "13x per-$"
(which used the M=128 rate for BOTH cards) was WRONG -- a real miner batches. Corrected at the
fat-M operating point (compute 5090 28.7ms/nonce, B200 6.5ms/nonce; regen amortised):
| bank GiB | Q | 5090 n/s | B200 n/s | rate | per-$ 5090/B200 |
|---|---|---|---|---|---|
| 51 | 128 | 28.4 | 154 | 5.4x | 3.7x |
| 96 | 512 | 31.6 | 154 | 4.9x | 4.1x |

**CORRECTED VERDICT: 5090 still wins per dollar, by ~3-4x (not 13x), because the B200's 4.4x raw
edge at fat-M is still far under the ~20x rent ratio.** FAILED stands; honest margin ~3-4x.
Nuance: on ENERGY (J/nonce) the B200 is ~3x more efficient (6.5ms x 1000W vs 35ms x 575W) -- so
owned+electricity economics differ from rental. But the "rent a B200 to out-mine consumers"
threat model (numair's primary GO criterion) FAILS: 5090 wins rental $/nonce ~3-4x.
