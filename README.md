# matador-miner

![MATADOR - fearless BTX MatMul miner](docs/matador.png)

**A fast, headless GPU miner for [`btxchain/btx`](https://github.com/btxchain/btx) (BTX) -
the ENC_RC proof-of-work. One static binary. Solo or pool. NVIDIA. Self-updating and
fleet-ready.**

Point it at your own node and keep 100% of every block, or pool-mine against
[minebtx](https://minebtx.com/) with one flag. It is fully decoupled from the node, so the
miner can update or restart without ever touching `btxd`.

```bash
# install + start mining a pool (repeat --pool for automatic failover to a backup):
curl -fsSL https://raw.githubusercontent.com/vanities/matador-miner/main/install.sh | bash
matador-miner --mode pool \
  --pool stratum+tcp://btx-us-east.lproute.com:8660 \
  --pool ssl://ninjaraider.com:44921 \
  --worker rig1 --payoutaddress btx1zcf4z36asua8ylchysphgwfgyfr8267vvznth826epden7lar4fnqvy9gzv
```

## Why matador

- **One static binary.** No toolkit, no drivers beyond the NVIDIA one you already have, no
  Python. The CUDA backend auto-detects.
- **Solo *or* pool.** Solo mines against your own `btxd` (`getblocktemplate` -> `submitblock`)
  and keeps every block self-custodied; pool mode talks stratum to minebtx/dexbtx pools for
  steady payouts. Same solver either way.
- **Self-updating, safely.** Checks GitHub on a schedule and atomically upgrades itself - same
  PID, no node restart - so a fleet stays current with zero ops. Every update is **sha256-verified
  before it's swapped in**: a corrupt or tampered download is refused and the miner just keeps
  running the version it's on. A **bake-time** delay means it won't jump onto a brand-new release
  until it has aged, so a bad release is caught (on a canary) before the fleet adopts it. Stable
  channel only, by default; opt out any time.
- **Fleet-ready.** Run one coordinator (your node + a least-privilege work proxy + a telemetry
  dashboard) and point any number of disposable rigs at it. They hop on and off, share one
  wallet, and never collide.
- **Observable.** A read-only JSON status API (on by default, loopback) for dashboards,
  watchdogs, and the fleet hub.

## Supported hardware

BTX activated **ENC_RC (v4)** on mainnet. The proof-of-work is now a dense INT8 tensor-core
workload, which changes what can mine it:

**NVIDIA, Ampere or newer** (`sm_80` / `86` / `89` / `90` / `120`), single or multi-GPU. That
is the whole list.

Ampere is a hard floor, not a preference: the work is INT8 matrix multiplication, and a card
without the units for it is not slow, it is out of the game. Pre-Ampere NVIDIA, Apple Silicon
and AMD cannot mine BTX, and matador no longer ships builds for them.

**Rates.** The v3 rate tables that used to live here have been removed. They were measured
against a proof-of-work that no longer exists, in a unit (`nonce/s`) the miner no longer
reports, so republishing them would be misleading. v4 numbers are being collected; see
[Help wanted](#help-wanted) if you want to contribute one.

The v4 unit of work is the **episode**: one full dependent INT8 matrix chain per nonce. The
miner reports `ep/s` on its `[stats]` line, and the value is small and fractional by design.

## Quick start

**1. Install** (Linux one-liner: fetches the latest release, verifies the sha256, installs to
`/usr/local/bin` or `~/.local/bin`):

```bash
curl -fsSL https://raw.githubusercontent.com/vanities/matador-miner/main/install.sh | bash
```

This puts `matador-miner` on your `PATH`, so you run it as `matador-miner` from anywhere - no
`./`. (The `./bin/matador-miner` form further down is only for the un-extracted release bundle.)

**2. Mine.** The CUDA backend **auto-detects**, so no flag is needed.

```bash
# Pool - no node required (2nd --pool is a backup; matador fails over if the 1st is down):
matador-miner --mode pool \
  --pool stratum+tcp://btx-us-east.lproute.com:8660 \
  --pool ssl://ninjaraider.com:44921 \
  --worker rig1 --payoutaddress btx1zcf4z36asua8ylchysphgwfgyfr8267vvznth826epden7lar4fnqvy9gzv

# Solo - against your own btxd (v0.32.12+, RPC on); keep 100% of every block, no fee:
matador-miner \
  --payoutaddress btx1zcf4z36asua8ylchysphgwfgyfr8267vvznth826epden7lar4fnqvy9gzv \
  --rpccookiefile ~/.btx/.cookie          # or --rpcuser/--rpcpassword
# solo is the default mode; add --rpcconnect/--rpcport if btxd isn't on 127.0.0.1:19334
```

You should see a `[stats]` heartbeat within seconds carrying `rc-active=1` and a rising
`ep/s`. That's it. The examples
use the project's payout address so they run as-is - **set `--payoutaddress` to your own
`btx1...` to mine to yourself.**

**Prefer a config file?** Copy the template for your GPU and just run it:

```bash
cp config.example.nvidia.json matador.json
$EDITOR matador.json                          # set payout address + worker
matador-miner                                 # auto-loads ./matador.json
```

## Pools

Exact, tested URLs (verified with this miner, July 2026). Use them verbatim; repeat `--pool`
to build a failover chain, matador walks the list top to bottom if one goes down.

| pool | plain TCP | TLS (encrypted) |
|---|---|---|
| LuckyPool US East | `stratum+tcp://btx-us-east.lproute.com:8660` | - |
| LuckyPool US Central | `stratum+tcp://btx-us-central.lproute.com:8660` | - |
| LuckyPool US Chicago | `stratum+tcp://btx-us-ord.lproute.com:8660` | - |
| LuckyPool EU | `stratum+tcp://btx-eu.lproute.com:8660` | - |
| ninjaraider | `stratum+tcp://ninjaraider.com:44920` | `ssl://ninjaraider.com:44921` |
| minebtx | `stratum+tcp://stratum.minebtx.com:3333` | - |

URL schemes: `stratum+tcp://` (or a bare `host:port`) is a plain TCP connection.
`ssl://`, `tls://`, `stratum+ssl://`, and `stratum+tls://` all mean TLS-encrypted
(v0.8.34+); the port must be the pool's TLS port, they are not interchangeable with the
plain port. Pool dialect (classic stratum vs login-style, used by LuckyPool and
ninjaraider) is auto-detected, no flag needed.

## Install in detail

**Release bundle (recommended).** Each release ships a `*-bundle.tar.gz` with the binary and a
config template:

```bash
# matador-miner-<ver>-linux-x86_64-bundle.tar.gz   NVIDIA CUDA (Ampere or newer)
curl -fsSLO "<bundle-url>" && curl -fsSLO "<bundle-url>.sha256"
sha256sum -c matador-miner-*-bundle.tar.gz.sha256     # must print OK   (shasum -a 256 on macOS)
tar xzf matador-miner-*-bundle.tar.gz && cd matador-miner-*/
cp config.example.nvidia.json matador.json && $EDITOR matador.json
./bin/matador-miner
```

**Bare binary.** The one-liner above, or pin/redirect it: `VERSION=v0.4.0 PREFIX=$HOME/.local/bin`
before the pipe. To verify by hand instead of trusting the script:

```bash
api=https://api.github.com/repos/vanities/matador-miner/releases
url=$(curl -fsSL "$api" | grep -oE '"browser_download_url": *"[^"]+linux-x86_64"' | cut -d'"' -f4)
curl -fsSLO "$url" && curl -fsSLO "$url.sha256"
sha256sum -c "$(basename "$url").sha256"              # must print OK
chmod +x "$(basename "$url")" && sudo mv "$(basename "$url")" /usr/local/bin/matador-miner
matador-miner --help
```

## HiveOS

Each release ships a HiveOS custom-miner package: `matador-miner-<ver>.tar.gz`. Create a
flight sheet with miner **Custom**, click **Setup Miner Config**, and fill:

| Field | Value |
|---|---|
| Miner name | `matador-miner` |
| Installation URL | `https://github.com/vanities/matador-miner/releases/download/v0.8.58/matador-miner-0.8.58.tar.gz` |
| Hash algorithm | `btx` |
| Wallet and worker template | `%WAL%.%WORKER_NAME%` |
| Pool URL | `stratum+tcp://stratum.minebtx.com:3333` |
| Pass | `x` |
| Extra config arguments | optional matador CLI flags, e.g. `--no-gpu-suffix` |

Set the flight sheet wallet to your BTX address (`btx1...`). Ready to import flight sheets,
one per BTX pool, are in [hiveos/](hiveos/).

- Mines on **all GPUs**, one pool worker per card (`rig-gpu0`, `rig-gpu1`, ...). Add
  `--no-gpu-suffix` to report the whole rig as a single worker, or `--gpus 0,1` to pin cards.
- The package ships one binary, for Ampere and newer. A rig whose cards are all pre-Ampere
  cannot mine: `h-run.sh` says so and stops, rather than burning power for nothing.
- Rate, per-GPU temps, and accepted/rejected shares show up on the HiveOS dashboard; the JSON
  status API stays available on the rig at `127.0.0.1:4060`.
- Works with both stratum and login-style pools. Solo-through-pool: use
  `solo:%WAL%.%WORKER_NAME%` as the template where the pool supports it.
- TLS pools work from v0.8.53 on: give the Pool URL an `ssl://`, `tls://`, `stratum+ssl://`
  or `stratum+tls://` scheme, for example `ssl://ninjaraider.com:44921`. Certificates are
  verified; add `--pool-tls-insecure` for a pool with a self-signed certificate.
- Under HiveOS the self-updater is off (the flight sheet owns the install). To update, point
  the Installation URL at the newer release tar.gz and reapply the flight sheet.

## Auto-update

On by default: the miner checks GitHub releases at startup and every 30 min, and when a newer
one ships it downloads the platform binary, verifies its sha256, atomically swaps itself, and
re-exec's into it **with the same PID and no `btxd` restart**. Works the same under systemd,
`nohup`, `tmux`, `screen`, or a foreground shell.

> **Requirement:** the binary must live in a path **writable by the user running it**.
> `install.sh` uses `~/.local/bin` when you're not root, which works out of the box. If you
> `sudo`-install to `/usr/local/bin` but run as a normal user, the self-update can't replace
> the file (it logs `cannot replace binary ...` and keeps the old one) - run from a user-owned
> dir instead, e.g. `~/.local/bin` or an `/opt/matador/bin` you own.

Tune or disable: `--update-interval-s <sec>` (`0` = startup-only), `--update-channel prerelease`,
`--min-version-age-s <sec>` (bake time), or `--no-auto-update` (check + notify only). Details in
[`docs/matador-standalone-ops.md`](docs/matador-standalone-ops.md#auto-update).

## Configure

`matador-miner` picks safe defaults for the detected hardware, so there is little to turn:

| Knob (config / flag) | Default | Notes |
|---|---|---|
| `gpus` / `--gpus 0,1,2` | first GPU | multi-GPU fan-out; each card gets its own worker suffix + API port |
| `backend` / `--backend` | auto | `cuda` or `cpu`. Leave it unset. `cpu` runs a reference solver far too slow to mine with. |
| `pools` / `--pool` | - | one endpoint or an ordered failover list (pool mode) |

Full config keys and the systemd unit are in
[`docs/matador-standalone-ops.md`](docs/matador-standalone-ops.md). Example config:
[nvidia](docs/config.example.nvidia.json).

Flags retired with v3 (`--overlap`, `--no-overlap`, `--hip-solver`, and `--backend
metal|hip|rocm`) are **accepted and ignored** with a warning rather than rejected, so an old
flight sheet cannot stop a rig from starting. Clean them out when convenient.

Pool: **[minebtx](https://minebtx.com/)** is the default (the examples use
`stratum+tcp://stratum.minebtx.com:3333`) -
[live dashboard](https://pool.minebtx.com/) -
[dexbtx/minebtx source](https://github.com/dexbtx/minebtx). Any dexbtx-style pool works too,
e.g. [bitminerpool.xyz](https://bitminerpool.xyz/#miners) - just point `--pool` at its stratum
endpoint.

## Run a fleet

Mine across many machines from one node and one dashboard. A **coordinator** runs your `btxd`
+ `matador-gbt-proxy` (a least-privilege work proxy: token auth, only `getblocktemplate` /
`submitblock`) + `matador-hub` (telemetry + dashboard). **Workers** are disposable - no node,
no chain state - so they hop on and off with zero warmup. They share one payout wallet, and a
per-rig coinbase extranonce keeps their work disjoint (no duplicate effort), exactly like a
pool partitions across miners.

```bash
# on the coordinator:
FLEET_TOKEN=... NODE_COOKIE=~/.btx/.cookie \
  HUB_WORKERS="rig1=http://10.0.0.11:4060,rig2=http://10.0.0.12:4060" \
  scripts/matador-coordinator.sh --listen 10.0.0.1
# dashboard -> http://10.0.0.1:4070
# workers   -> matador-miner --mode solo --rpcconnect 10.0.0.1 --rpcport 4071 \
#                --rpcuser rig1 --rpcpassword "$FLEET_TOKEN" --worker rig1 --payoutaddress btx1...
```

Workers can also **fall back to a pool** if the coordinator drops (`--fallback-pool ...`) and
**idle-gate** the GPU so a workstation only mines when it's free (`--should-mine-command ...`).
Full copy-paste setup (VPN + laptop dashboard, fallback, idle-gate) is in
**[`docs/matador-fleet.md`](docs/matador-fleet.md)**.

## Monitor

A read-only HTTP status API runs by default on `127.0.0.1:4060` (`--no-api` to disable,
`--api-listen` to bind a LAN/VPN address). It never exposes RPC credentials or pool passwords.

```bash
curl -s http://127.0.0.1:4060/health     # {"status":"ok"}
curl -s http://127.0.0.1:4060/summary    # version, mode, backend, shares, nonces, GPU temp/power, update state
curl -s http://127.0.0.1:4060/pools      # effective failover pool list
scripts/matador-status.sh                # readable terminal dashboard over the same API
```

`/summary` is the one to scrape for a dashboard - shares, `rc_episodes` and the rolling
`episode_per_s` averages, per-GPU util/power/temp, watchdog state, and the auto-update block
(running vs latest version). For multi-GPU rigs each child increments the port: `4060`,
`4061`, ...

> **Scrapers built against v0.9.9 or earlier need updating.** The v3 throughput fields
> (`nonce_per_s`, `digest_c_per_s`, `scan_per_s`, the `batched_*` counters and the
> `validation` object) described a solver that no longer exists and have been replaced by
> `rc_episodes`, `solve_windows` and `episode_per_s`. A dashboard alerting on "nonces stopped"
> would page on a perfectly healthy v4 rig.

## Reliability

Pool mode supports ordered `pools[]` failover, a reject-streak watchdog that triggers a safe
reconnect, optional pool-fallback for solo workers, and a warning-only thermal monitor (it never
changes clocks, fans, or power limits). A 1% time-based dev fee funds development - the coinbase
pays the dev address for ~36s of each hour, logged on entry and exit.

## Trust & self-custody

- **Self-custody.** Solo submits to *your* `btxd` over localhost RPC and holds **no wallet keys**;
  rewards pay the `--payoutaddress` you provide.
- **Closed-source binary.** **Verify the sha256** of every download before running it
  (the bundle and one-liner do this for you). Use `LOG_LEVEL=debug` for troubleshooting.
- **Loopback by default.** The status API binds `127.0.0.1`; only expose it on a LAN/VPN you
  control.

## Help wanted

The RTX 5090 is mined first-hand. v4 rates for every other Ampere-or-newer card are unmeasured,
so a data point genuinely helps:

- **NVIDIA (`sm_80`-`sm_120`):** the `ep/s` value from the `[stats]` line, plus card, driver
  and whether you ran as root (which enables the built-in GPU tuning).

Easiest way to share: `scripts/matador-status.sh` (or `curl -s .../summary`) prints a clean
snapshot - open an issue with it plus your OS and driver version.

## Credits

- **[`btxchain/btx`](https://github.com/btxchain/btx)** - the BTX node, the MatMul proof-of-work,
  and the ENC_RC proof-of-work this mines.
- **[`dexbtx/minebtx`](https://github.com/dexbtx/minebtx)** (shib) - the minebtx pool and stratum
  orchestrator; the protocol reference for the v2/v3 seed + `parent_mtp` handling.

## License

Proprietary - Copyright (c) 2026 AM2 LLC. All rights reserved. See [LICENSE](LICENSE).
Third-party components (btxchain/btx and its Bitcoin Core lineage) remain under the MIT License.
matador-miner release binaries ship under their own end-user terms.
