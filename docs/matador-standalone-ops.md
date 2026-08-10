# matador-miner standalone operations

This is the first farm-operator layer for the standalone `matador-miner` binary: a stable JSON config file, a systemd service shape, and the next seams for ETH-era miner polish.

## Config file

`matador-miner` automatically loads `./matador.json` when it exists. It also accepts `--config <path>` or `MATADOR_CONFIG=<path>`. Precedence is:

```text
defaults < JSON config < environment variables < CLI flags
```

That means a bundle can be run by copying one hardware template to `matador.json`, while a rig can still keep durable settings in `/etc/matador-miner/config.json`, put secrets in a protected env file if desired, and override one setting for a one-off run.

Example pool config:

```bash
cp config.example.nvidia.json matador.json   # NVIDIA/CUDA
$EDITOR matador.json                         # set your real P2MR payout address
./bin/matador-miner                          # auto-loads ./matador.json
```

The source tree also keeps this template under `docs/` for packagers: `config.example.nvidia.json`. The AMD and Apple templates were removed when BTX activated v4 (ENC_RC), which neither backend can mine.

Supported JSON keys mirror the existing CLI/env surface:

- `mode`: `solo` or `pool`
- `pool`: `host:port`, `stratum+tcp://host:port`, or a comma-separated failover list for pool mode
- `pools`: ordered failover array. Entries may be strings (`"minebtx.com:3333"`) or objects with either `url` or `host`+`port` plus optional `label`/`name`. The first entry is primary; later entries are tried when connect/disconnect/stall recovery trips.
- `worker`, `pool_pass`
- `chain`, `backend` (`cuda`, `metal`, `cpu`, or `hip`/`rocm`). HIP/ROCm delegates solves to the external C++/HIP `btx-gbt-solve-hip` sidecar. Release bundles auto-discover it next to `bin/matador-miner`; outside a bundle use `--hip-solver <path>` or config `sidecars.hip`.
- `gpus`: optional basic multi-GPU fan-out list, e.g. `[0, 1, 2]`. This starts one normal miner process per device with `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` scoped to that child, worker suffixes like `rig1-gpu0`, and API ports incremented from the configured base (`4060`, `4061`, ...). This is not a solver optimization; it is just safe process-level fan-out.
- `api`: optional local read-only status API object: `{ "enabled": true, "listen": "127.0.0.1", "port": 4060 }`
- `api_enabled` / `api_listen` / `api_port`: flat aliases for the same status API settings
- `watchdog`: pool-mode self-supervision object: `{ "enabled": true, "check_s": 15, "reject_streak": 20, "no_share_s": 0 }`
- `watchdog_enabled`, `watchdog_check_s`, `watchdog_reject_streak`, `watchdog_no_share_s`: flat aliases. `no_share_s: 0` leaves the accepted-share timeout observe-only/off by default.
- `thermal`: warning-only GPU temp/power object: `{ "enabled": true, "warn_temp_c": 86, "critical_temp_c": 90, "warn_power_w": 0 }`. This only logs/API-reports thresholds; it does not change clocks, fans, power limits, or restart the process.
- `thermal_enabled`, `thermal_warn_temp_c`, `thermal_critical_temp_c`, `thermal_warn_power_w`: flat aliases for the same thermal settings.
- `payoutaddress` / `payout_address`
- `rpcconnect`, `rpcport`, `datadir`, `rpccookiefile` / `rpc_cookie_file`, `rpcuser`, `rpcpassword`
- `maxtries` / `max_tries`
- `devfee` / `dev_fee`, `devaddress` / `dev_address`
- `solver_threads`, `overlap`
- `update_check`, `auto_update`, `update_channel` (`stable`|`prerelease`), `update_interval_s`, `update_jitter_s`, `min_version_age_s` (see [Auto-update](#auto-update))
- `should_mine_command`, `should_mine_interval`, `gate_yield` (`abort`|`finish`) - idle-gate; see [matador-fleet.md](matador-fleet.md#idle-gate-mine-when-the-box-is-idle-yield-when-its-needed)
- `fallback_pool`, `fallback_after_s`, `solo_recheck_s` - solo->pool failover; see [matador-fleet.md](matador-fleet.md)

Do not commit a real config if it contains RPC credentials. The miner logs the config file path, byte size, and number of applied settings, but not RPC passwords or pool passwords.

## Local status API

Enable the read-only HTTP API for dashboards/watchdogs:

```bash
matador-miner --config /etc/matador-miner/config.json --api --api-port 4060
curl -s http://127.0.0.1:4060/health
curl -s http://127.0.0.1:4060/summary
curl -s http://127.0.0.1:4060/pools
```

The API binds to `127.0.0.1` by default. Keep it loopback-only unless a LAN firewall is intentionally protecting it. It exposes runtime counters, backend, worker, chain, public payout address, pool endpoints, and watchdog state; it does **not** expose RPC credentials or pool passwords.

`/summary` includes:

- `gpu_runtime`: best-effort GPU telemetry array with `gpu_uuid`, `vendor`, `util_pct`, `power_w`, and `temp_c`. NVIDIA uses `nvidia-smi`; AMD falls back to `rocm-smi` CSV when no NVIDIA rows are present (`[]` when unavailable, such as macOS/CPU-only hosts)
- `thermal`: warning-only status derived from the same GPU telemetry (`ok`, `warning`, `critical`, `disabled`, or `unavailable`)
- `watchdog.status`: `ok`, `warning`, or `disabled`
- `watchdog.reject_streak`
- `watchdog.last_notify_age_sec`, `last_share_age_sec`, `last_accept_age_sec`, `last_nonce_age_sec`
- `watchdog.last_warning`, `last_action`, and `reconnect_requested`

The first action set is deliberately safe: in pool mode the watchdog can request a reconnect/failover, which uses the same disconnect/reconnect path as normal pool loss. It does **not** restart the process, change clocks/fans, or touch power limits.

For a quick terminal dashboard once the API is enabled:

```bash
scripts/matador-status.sh
# or: MATADOR_API_URL=http://127.0.0.1:4060/summary scripts/matador-status.sh
```

CLI/env failover forms are also supported:

```bash
# repeat --pool; CLI clears lower-precedence config/env pools then appends each flag
matador-miner --mode pool --pool stratum+tcp://stratum.minebtx.com:3333 \
  --payoutaddress btx1zcf4z36asua8ylchysphgwfgyfr8267vvznth826epden7lar4fnqvy9gzv --worker rig1

# or comma-separate POOL / JSON `pool`
POOL=stratum+tcp://stratum.minebtx.com:3333 matador-miner --mode pool \
  --payoutaddress btx1zcf4z36asua8ylchysphgwfgyfr8267vvznth826epden7lar4fnqvy9gzv --worker rig1
```

## Auto-update

`matador-miner` keeps itself current. By default it checks GitHub releases at startup
**and** on an interval, and when a newer release is out it downloads the platform
binary, sha256-verifies it, atomically swaps this executable, and re-exec's into it.
The re-exec keeps the same PID and does **not** restart `btxd` - in pool mode there is
no local node, and in solo mode the separate `btxd` process is untouched.

Config keys (CLI / env / `matador.json`, same precedence as everything else):

| key | default | meaning |
|-----|---------|---------|
| `auto_update` | `true` | download+verify+swap+re-exec. `false` / `--no-auto-update` = check + notify only |
| `update_check` | `true` | `false` / `--no-update-check` disables the check entirely (startup + periodic) |
| `update_channel` | `stable` | `stable` = GitHub "Latest" (non-prerelease). `prerelease` = newest tag incl. prereleases |
| `update_interval_s` | `1800` (30min) | periodic re-check cadence. `<=0` = startup-only |
| `update_jitter_s` | `300` | random `0..N`s delay before each periodic check, to de-sync a fleet |
| `min_version_age_s` | `3600` (1h) | bake-time: only auto-adopt a release once it is this old. Stops a whole fleet jumping onto a brand-new bad release the instant it is published |

The status API exposes update state for dashboards/fleet views:

```bash
curl -s http://127.0.0.1:4060/summary | python3 -c 'import sys,json;print(json.load(sys.stdin)["update"])'
# {"current":"v0.4.1","latest_seen":"v0.4.1","last_check_age_sec":42,"channel":"stable","auto_update":true}
```

**Pinning a release** (datacenter operators who stage updates): set `auto_update: false`
(checks + logs only), or `update_check: false` (silent), and roll binaries yourself.

## systemd service template

For a standalone pool rig. Note the install path and `ReadWritePaths` - they are what
make in-place auto-update work under a hardened unit:

```ini
# /etc/systemd/system/matador-miner.service
[Unit]
Description=Matador BTX standalone miner
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=miner
Group=miner
Environment=LOG_LEVEL=info
Environment=MATADOR_CONFIG=/etc/matador-miner/config.json
# Install the binary somewhere the SERVICE USER owns, not /usr/local/bin (root-owned).
# Auto-update does rename(self+".new", self), which needs write+execute on this dir.
ExecStart=/opt/matador/bin/matador-miner
Restart=always
RestartSec=10
TimeoutStopSec=30
NoNewPrivileges=true

# Hardening that stays COMPATIBLE with self-update:
ProtectSystem=strict
ReadWritePaths=/opt/matador/bin     # REQUIRED: without this, ProtectSystem=strict
                                    # silently blocks the binary swap and auto-update
                                    # no-ops (logs "cannot replace binary ...").
ProtectHome=true

[Install]
WantedBy=multi-user.target
```

Install/start:

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin miner || true
sudo install -d -o miner -g miner /opt/matador/bin
sudo install -o miner -g miner matador-miner /opt/matador/bin/matador-miner
sudo systemctl daemon-reload
sudo systemctl enable --now matador-miner.service
journalctl -u matador-miner -f
```

Why in-process re-exec is safe under systemd: `execv` replaces the process image but
keeps the **same PID**, so for `Type=simple` the update is invisible to systemd - no
restart, no `Restart=` bump, argv preserved. The only requirements are (1) the binary
lives in a service-user-owned dir and (2) `ReadWritePaths=` covers it. Verify after a
release with `curl -s http://127.0.0.1:4060/summary | grep -o '"current":"[^"]*"'`.

### Alternative: let systemd own the update cadence (timer)

If you prefer updates in a maintenance window instead of in-process, disable the
in-process periodic check (`update_interval_s: 0`) and drive `--update-check-only` from
a timer. `--update-check-only` runs one check (which may swap+re-exec) then exits; it
needs no payout/RPC/GPU.

```ini
# /etc/systemd/system/matador-update.service
[Service]
Type=oneshot
User=miner
ExecStart=/opt/matador/bin/matador-miner --update-check-only
ProtectSystem=strict
ReadWritePaths=/opt/matador/bin

# /etc/systemd/system/matador-update.timer
[Timer]
OnCalendar=*-*-* 04:00:00
RandomizedDelaySec=1h          # fleet de-sync, like update_jitter_s
Persistent=true
[Install]
WantedBy=timers.target
```

`systemctl enable --now matador-update.timer`. The miner service itself stays running;
the timer unit swaps the binary on disk, and the running miner picks it up on its next
restart (or run the timer's `ExecStart` against the live binary path and let its re-exec
hand off in place).

For solo mode, either point `rpccookiefile` at a readable cookie or run the service as a user that can read the local `btxd` datadir cookie.

## Next operator-polish seams

Config files create a stable interface for the next ETH-era features:

1. **Temperature / backend-stall watchdogs**: expose GPU temp/power and detect nonce-counter stalls; keep the first version observe/reconnect-only.
2. **Thermal/power controls**: move the current external `gpu-tune.service` settings behind explicit per-GPU config keys.
3. **True multi-GPU**: add `devices[]` plus per-device counters; the config file gives us somewhere clean to persist those choices.
