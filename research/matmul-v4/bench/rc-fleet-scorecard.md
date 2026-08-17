# ENC_RC production episode: fleet scorecard

`rc_fleet_bench.py`, 3 episodes/arm, warm-up excluded. Generated 20260723T165050Z.

Power sampled at 10 Hz **during** the run; `ep/kWh` uses **measured mean draw**, never the nameplate `power.limit`.

| card | arch | ep/s | ms | W mean | W max | util% | MHz | ep/kWh | $/hr | ep/$ | digest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RTX 5090 | sm_120 | 0.082 | 12197.7 | 544.8 | 586.6 | 98.1 | 2506 | 542 | 0.313 | 943 | `1cc5709d2fbd4be8` |

**Digest agreement across architectures:** YES `1cc5709d2fbd4be8`

A rate without watts cannot say whether the card was saturated or whether the
bench was measuring launch overhead. Both columns or neither.
