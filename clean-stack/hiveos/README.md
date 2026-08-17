# matador-miner on HiveOS

matador-miner ships a HiveOS custom miner package: `matador-miner-<version>.tar.gz`
on every [release](https://github.com/vanities/matador-miner/releases).

## Flight sheet setup

Create a flight sheet with miner **Custom**, then click **Setup Miner Config** and fill:

| Field | Value |
|---|---|
| Miner name | `matador-miner` |
| Installation URL | `https://github.com/vanities/matador-miner/releases/download/v0.8.58/matador-miner-0.8.58.tar.gz` |
| Hash algorithm | `btx` |
| Wallet and worker template | `%WAL%.%WORKER_NAME%` |
| Pool URL | `stratum+tcp://btx-us-east.lproute.com:8660` |
| Pass | `x` |
| Extra config arguments | (optional, plain miner CLI flags) |

Set the flight sheet wallet to your BTX address (`btx1...`). Ready to import flight
sheets, and a table of verified pool URLs, are in [hiveos/](../../hiveos/).

## TLS pools (ssl://)

Give the Pool URL its scheme: `ssl://`, `tls://`, `stratum+ssl://` or `stratum+tls://`
encrypt the stratum socket, and `stratum://`, `stratum+tcp://` or a bare `host:port`
stay plaintext. ninjaraider's TLS endpoint is `ssl://ninjaraider.com:44921`.

The certificate chain and hostname are verified by default; a pool with a self-signed
or expired certificate needs `--pool-tls-insecure` in Extra config arguments.

## What the package does

- Mines on **all GPUs** by default, reported to the pool as a **single worker name**
  (the rig shows as one row on the pool dashboard, like other miners). Add
  `--gpu-suffix` to Extra config arguments for one worker per card
  (`myrig-gpu0`, `myrig-gpu1`, ...), or pin specific cards with `--gpus 0,1`.
- On a multi-GPU rig the miner log shows rig-level `[stats-all]` and
  `[stats-all-avg]` roll-ups every 60s (`gpus=5/5 ep/s=6.05 ... pow=2980W
  maxtemp=54C | per-gpu ep/s: ...`) — a `gpus=4/5` means a card is
  down/respawning. Per-card console lines are suppressed by default; export
  `MATADOR_MGPU_CHILD_STATS=1` in `h-run.sh` to restore them (per-card data is
  always available via the per-card status API the dashboard uses).
- Ampere and newer only (RTX 30xx and up). A rig whose cards are all pre-Ampere
  stops with a clear message: ENC_RC needs tensor cores those cards do not have.
- Reports rate, per GPU temperature, accepted and rejected shares to the HiveOS
  dashboard.
- Works with both stratum pools and login style pools (LuckyPool). Solo mining
  through a pool that supports it: use `solo:%WAL%.%WORKER_NAME%` as the template.
- Multiple pool URLs (space separated) become primary plus fallbacks.

## Health: what a working v4 rig looks like

ENC_RC's unit of work is the **episode**: one full dependent INT8 GEMM chain for one
nonce, roughly 825 ms on a 5090. So the rate is small and fractional by design.

```
grep '\[stats\]' /var/log/miner/custom/custom.log | tail -1
```

Read three fields:

| Field | Healthy | Meaning |
|-------|---------|---------|
| `rc-active` | `1` | The miner latched the ENC_RC activation from the pool's job. **`0` means it is mining nothing** -- the pool is not announcing RC, or an operator pinned a wrong height. |
| `ep/s` | ~1.2 per 5090 | Episodes completed per second. Small is normal; zero with `rc-active=1` is a stall. |
| `rej%` | ~0 | Share reject rate. |

There are no CPU-side tuning knobs. The v3 `BTX_MATMUL_CPU_SCAN_AHEAD*` variables that
earlier packages documented are gone along with the v3 solver: the episode is entirely
tensor-core bound and spare CPU cores cannot assist it. If you set them, nothing happens.

**Pre-Ampere cards (compute < 8.0) cannot mine v4 at all** and this package ships no
binary for them; `h-run.sh` will say so and stop. Pin release v0.9.1 on those rigs.

## Updating

The self-updater is disabled under HiveOS so the package stays consistent.
To update, edit the flight sheet Installation URL to the newer release tar.gz
and reapply the flight sheet. The miner log notes when a newer release exists.

## Troubleshooting

- Miner log: `/var/log/miner/custom/custom.log` (or `miner log` in the shell).
- The status API listens on `127.0.0.1:4060` (one port per GPU counting up):
  `curl -s localhost:4060/summary | jq .`
- No hashrate on the dashboard usually means the miner failed to start; check
  the log for a config error (bad wallet address or unreachable pool).
