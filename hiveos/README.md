# HiveOS flight sheets

Importable flight sheets for matador-miner on HiveOS. Setup, tuning and troubleshooting
live in [clean-stack/hiveos/README.md](../clean-stack/hiveos/README.md); this folder is
just the sheets.

One sheet per live pool, re-verified 2026-08-21 (DNS, TCP, and a real stratum
handshake from a mining rig):

| Sheet | Pool | Transport |
|---|---|---|
| [flight-sheet.example.json](flight-sheet.example.json) | minebtx + byron fallback (recommended) | plaintext |
| [flight-sheet.minebtx.example.json](flight-sheet.minebtx.example.json) | minebtx | plaintext |
| [flight-sheet.byron.example.json](flight-sheet.byron.example.json) | Byron Pool | plaintext |
| [flight-sheet.btx-pool.example.json](flight-sheet.btx-pool.example.json) | btx-pool | plaintext |

Set the flight sheet wallet to your own BTX address (`btx1...`) after importing, and bump
`install_url` to the newest release.

## Pool URLs

Put the pool's URL in the flight sheet's **Pool URL** field with its scheme intact.
`ssl://`, `tls://`, `stratum+ssl://` and `stratum+tls://` select TLS; `stratum://`,
`stratum+tcp://` and a bare `host:port` are plaintext. The miner reads TLS off that scheme
and nothing else, so dropping it silently downgrades a TLS pool to a plaintext socket
against a TLS port, and the connection never completes.

Several URLs (space, comma or semicolon separated) become the primary pool plus fallbacks:

```
stratum+tcp://stratum.minebtx.com:3333 stratum+tcp://stratum.btxbyronbay.com:3335
```

TLS connections verify the pool's certificate chain and hostname by default. A pool with a
self-signed or expired certificate needs `--pool-tls-insecure` in **Extra config arguments**.

The stratum dialect (classic `mining.subscribe` vs JSON-RPC `login`) is auto-detected at
handshake, so no pool needs extra configuration.

## Verified endpoints (2026-08-21)

Probed from a mining rig: DNS, TCP, and a stratum handshake.

| Pool | URL | Result |
|---|---|---|
| minebtx | `stratum+tcp://stratum.minebtx.com:3333` | **ok** (mined against) |
| Byron | `stratum+tcp://stratum.btxbyronbay.com:3335` | **ok** (mined against) |
| btx-pool | `stratum+tcp://btx-pool.com:3334` | **ok** (handshake verified; not yet mined against) |
| LuckyPool | old `:8660` refused; new `stratum+tls://…lproute.com:8665` | **live but LOCKED to lpminer** - proprietary TLS framed transport, not stratum (see below) |
| bitminerpool | `stratum.bitminerpool.xyz:3333` | port open, **no stratum response** (dead service) |
| poolbtx | `poolbtx.com:3333` | connection refused |
| ninjaraider | `ninjaraider.com:44920/:44921` | connection refused (retired) |
| btxpool.org | `43.154.101.226:3333` | connection refused |
| Coin Miners | `eu/us.coin-miners.info:8461` | connection refused |

The July 2026 sheets for the refused pools have been removed: HiveOS retries a dead
pool forever, and "can't connect" reports kept tracing back to imported sheets pointing
at endpoints that no longer exist. If one of these pools comes back, its sheet is one
`git revert` away.

**LuckyPool is a special case.** It did not shut down - it moved its BTX pool to
`stratum+tls://btx-eu.lproute.com:8665` (and other regions) and put it behind a proprietary,
certificate-pinned **"BTX transport"** that only their own `lpminer` speaks. matador reaches
the TLS layer (with `--pool-tls-insecure`, since the cert is self-signed) but the pool drops
any client that opens with a classic `mining.subscribe`, so matador cannot mine it today.
Their own flight sheet says as much: *"lpminer is the only miner supported by this BTX-v4
transport."* Use minebtx or byron with matador; use LuckyPool only with lpminer.
