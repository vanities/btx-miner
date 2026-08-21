# matador-miner on HiveOS

matador-miner ships a HiveOS custom miner package: `matador-miner-<version>.tar.gz`
on every [release](https://github.com/vanities/matador-miner/releases).

## Flight sheet setup

Create a flight sheet with miner **Custom**, then click **Setup Miner Config** and fill:

| Field | Value |
|---|---|
| Miner name | `matador-miner` |
| Installation URL | `https://github.com/vanities/matador-miner/releases/download/v0.9.27/matador-miner-0.9.27.tar.gz` |
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
- Reports hashrate, per GPU temperature, accepted and rejected shares to the HiveOS
  dashboard. See [Hashrate on the dashboard](#hashrate-on-the-dashboard) for why the
  number is a handful of H/s rather than MH/s.
- Works with both stratum pools and login style pools (LuckyPool). Solo mining
  through a pool that supports it: use `solo:%WAL%.%WORKER_NAME%` as the template.
- Multiple pool URLs (space separated) become primary plus fallbacks.

## Hashrate on the dashboard

**A healthy card reads about 1.2-1.5 H/s. That is correct, not a broken counter.**

ENC_RC's unit of work is the **episode**: one episode is one nonce fully tried, so
episodes/s *is* hashrate in the classic sense. It is just that a v4 nonce costs a full
dependent INT8 GEMM chain (~825 ms on a 5090) instead of one SHA256, so the honest
number is single-digit H/s rather than tens of MH/s. HiveOS is given that same figure,
so what you see on the dashboard, in `curl localhost:4060/summary`, and on the miner
log's `[stats]` line all agree.

Use it for OC tuning: the reported figure is the **live** rate over the last heartbeat
(~30s), so a clock change shows up within a minute or two rather than being averaged
away. Watch the per-GPU column next to the card you are tuning.

> Rigs on **v0.9.24 and older reported a flat 0 H/s** here no matter how well they were
> mining: `h-stats.sh` was still reading the v3 `nonce_per_s` field, which nothing has
> emitted since v4 replaced nonces with episodes. The same release also printed the
> `[stats]` line's rates as truncated integers, so `ep/s=1.4` logged as `ep/s=1` and
> anything below 1 logged as `0`. Update the flight sheet's Installation URL to
> v0.9.26 or newer.

## Health: what a working v4 rig looks like

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
- A dashboard hashrate of a few H/s is normal — see
  [Hashrate on the dashboard](#hashrate-on-the-dashboard). A flat **0** on v0.9.24 or
  older is the fixed `nonce_per_s` bug described there; update the Installation URL.
- Otherwise no hashrate usually means the miner failed to start, or started but never
  latched RC. Check the log for a config error (bad wallet address, unreachable pool)
  and confirm `rc-active=1` on the `[stats]` line.
- To see what HiveOS is being told, run the package's own stats hook:
  `MATADOR_HIVE_DIR=/hive/miners/custom/matador-miner bash /hive/miners/custom/matador-miner/h-stats.sh`
  It prints the exact `khs:` and `stats:` the agent uploads. `khs` is in kH/s by HiveOS
  contract, so `0.00124` there is the `1.24 H/s` the dashboard shows.
