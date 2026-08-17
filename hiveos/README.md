# HiveOS flight sheets

Importable flight sheets for matador-miner on HiveOS. Setup, tuning and troubleshooting
live in [clean-stack/hiveos/README.md](../clean-stack/hiveos/README.md); this folder is
just the sheets.

One sheet per pool. Every one of them was driven through the real HiveOS chain
(`h-config.sh` -> `h-run.sh` -> miner -> `h-stats.sh`) on 2026-07-09 and reached a live job.

| Sheet | Pool | Transport |
|---|---|---|
| [flight-sheet.example.json](flight-sheet.example.json) | luckypool, us + eu fallback | plaintext |
| [flight-sheet.ninjaraider-ssl.example.json](flight-sheet.ninjaraider-ssl.example.json) | ninjaraider `:44921` | TLS |
| [flight-sheet.ninjaraider.example.json](flight-sheet.ninjaraider.example.json) | ninjaraider `:44920` | plaintext |
| [flight-sheet.minebtx.example.json](flight-sheet.minebtx.example.json) | minebtx | plaintext |
| [flight-sheet.poolbtx.example.json](flight-sheet.poolbtx.example.json) | poolbtx | plaintext |
| [flight-sheet.bitminerpool.example.json](flight-sheet.bitminerpool.example.json) | bitminerpool | plaintext |
| [flight-sheet.byron.example.json](flight-sheet.byron.example.json) | Byron Pool | plaintext |
| [flight-sheet.btx-pool.example.json](flight-sheet.btx-pool.example.json) | btx-pool | plaintext |
| [flight-sheet.btxpool-org.example.json](flight-sheet.btxpool-org.example.json) | btxpool.org | plaintext |
| [flight-sheet.coin-miners.example.json](flight-sheet.coin-miners.example.json) | Coin Miners, eu + us fallback | plaintext |

There is no sheet for minebtx's `:3334` SSL port or for DiffPool: both refuse the connection
(see the table below). minebtx's plaintext `:3333` works and has a sheet.

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
ssl://ninjaraider.com:44921,ninjaraider.com:44920
```

TLS connections verify the pool's certificate chain and hostname by default. A pool with a
self-signed or expired certificate needs `--pool-tls-insecure` in **Extra config arguments**.

The stratum dialect (classic `mining.subscribe` vs JSON-RPC `login`) is auto-detected at
handshake, so no pool needs extra configuration.

## Verified endpoints

Probed 2026-07-09 against the pool list on [btxlinks](https://btxlinks.vercel.app/): DNS,
TCP, TLS handshake where applicable, and a stratum request. "login" pools reject
`mining.subscribe` and get switched to the login dialect automatically.

| Pool | URL | Result |
|---|---|---|
| minebtx | `stratum://stratum.minebtx.com:3333` | ok, stratum |
| minebtx | `ssl://stratum.minebtx.com:3334` | **connection refused** |
| poolbtx | `stratum://poolbtx.com:3333` | ok, stratum |
| bitminerpool | `stratum://stratum.bitminerpool.xyz:3333` | ok, stratum |
| Byron | `stratum://stratum.btxbyronbay.com:3335` | ok, stratum |
| ninjaraider | `stratum://ninjaraider.com:44920` | ok, login |
| ninjaraider | `ssl://ninjaraider.com:44921` | ok, login, TLSv1.3, cert verified |
| btx-pool | `stratum://btx-pool.com:3334` | ok, stratum |
| btxpool.org | `stratum://43.154.101.226:3333` | ok, stratum |
| luckypool | `stratum://btx-us-east.lproute.com:8660` | ok, login |
| luckypool | `stratum://btx-eu.lproute.com:8660` | ok, login |
| Coin Miners | `stratum://eu.coin-miners.info:8461` | ok, stratum |
| Coin Miners | `stratum://us.coin-miners.info:8461` | ok, stratum |
| DiffPool | `stratum+tcp://btx.diffpool.xyz:3333` | **connection refused** |

The two refusals are pool-side: the port is closed, not filtered, and neither depends on
anything the miner does. minebtx serves plaintext on `:3333` regardless of what its SSL
port advertises.
