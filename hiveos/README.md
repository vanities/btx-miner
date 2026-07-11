# HiveOS flight sheets

Importable flight sheets, one per BTX pool. Setup and tuning live in the
[HiveOS section of the main README](../README.md#hiveos).

Import a sheet, set its wallet to your own BTX address (`btx1...`), and point
`install_url` at the newest release.

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

## Pool URLs

Put the pool's URL in the flight sheet's **Pool URL** field with its scheme intact.
`ssl://`, `tls://`, `stratum+ssl://` and `stratum+tls://` connect over TLS. `stratum://`,
`stratum+tcp://` and a bare `host:port` are plaintext. The scheme is the only thing that
selects TLS, so a TLS pool needs it.

Several URLs (space, comma or semicolon separated) become the primary pool plus fallbacks:

```
ssl://ninjaraider.com:44921,ninjaraider.com:44920
```

TLS connections verify the pool's certificate chain and hostname against the system trust
store. A pool with a self-signed or expired certificate needs `--pool-tls-insecure` in
**Extra config arguments**.

The stratum dialect (classic `mining.subscribe` or JSON-RPC `login`) is detected at
handshake, so no pool needs extra configuration.

Requires v0.8.53 or newer: earlier packages dropped the scheme, so an `ssl://` sheet
connected in plaintext and the pool never answered.

## Pool endpoints

Checked 2026-07-09. Every sheet above reached a live job.

| Pool | URL |
|---|---|
| minebtx | `stratum://stratum.minebtx.com:3333` |
| poolbtx | `stratum://poolbtx.com:3333` |
| bitminerpool | `stratum://stratum.bitminerpool.xyz:3333` |
| Byron | `stratum://stratum.btxbyronbay.com:3335` |
| ninjaraider | `stratum://ninjaraider.com:44920`, `ssl://ninjaraider.com:44921` |
| btx-pool | `stratum://btx-pool.com:3334` |
| btxpool.org | `stratum://43.154.101.226:3333` |
| luckypool | `stratum://btx-us-east.lproute.com:8660`, `stratum://btx-eu.lproute.com:8660` |
| Coin Miners | `stratum://eu.coin-miners.info:8461`, `stratum://us.coin-miners.info:8461` |

There is no sheet for minebtx's `:3334` SSL port or for DiffPool's
`btx.diffpool.xyz:3333`: both refuse the connection. minebtx's plaintext `:3333` works.
