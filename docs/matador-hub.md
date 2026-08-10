# matador-hub - fleet telemetry (phase 1)

`scripts/matador-hub.py` is the coordinator-side control-plane surface for a fleet of
`matador-miner` rigs. It scrapes each worker's existing read-only `/summary` API and
rolls them up into one aggregate view: total hashrate, per-rig status/util/power/temp,
shares, online/offline, and the auto-update state (which rigs are behind).

It needs **zero changes on the workers** - each rig just runs with `--api` enabled - and
no third-party Python deps (stdlib only; also runs under `uv run python matador-hub.py`).

## The appliance shape

This is phase 1 of the "local archive node + hop-on/hop-off fleet" idea: the stateful,
uptime-bearing parts (the `btxd` archive node, the chain) live on **one** coordinator
host; every miner is a disposable, stateless solver that exposes `/summary`. The hub
turns those into a single fleet dashboard. Later phases add a GBT proxy (workers mine
solo through the coordinator without holding RPC creds) and an idle-gate agent (rigs
join/leave on machine idle). Phase 1 ships value today with no protocol work.

## Run it

On the coordinator host (bind LAN/VPN only - it aggregates rigs, **do not expose it
publicly**):

```bash
# inline workers
python3 scripts/matador-hub.py \
  --worker rig1-5090=http://10.0.0.11:4060 \
  --worker rig2-4090=http://10.0.0.12:4060 \
  --listen 0.0.0.0 --port 4070

# or a config file (see docs/hub.example.json)
python3 scripts/matador-hub.py --config docs/hub.example.json
```

Each worker must expose its API on a reachable address. Locally that is loopback
(`--api`), but for a hub on another host bind the worker API to the LAN, e.g.
`--api-listen 0.0.0.0 --api-port 4060` behind a firewall/VPN.

## Endpoints

| path | purpose |
|------|---------|
| `/` | auto-refreshing HTML dashboard (total hashrate, per-rig table, "behind" highlight) |
| `/fleet` | aggregate JSON: `{ totals, rigs[] }` for scripting/Grafana/alerting |
| `/health` | `{"status":"ok"}` |

```bash
curl -s http://127.0.0.1:4070/fleet | python3 -m json.tool
```

`totals` includes `online`/`offline`, summed `episode_per_s` (computed by the hub from
each rig's batched-attempt counter delta - the same source the miner heartbeat uses),
`power_w`, `accepted`/`rejected`/`stale` shares, and `behind` (rigs whose `latest_seen`
tag differs from their running `current` - i.e. an auto-update is pending). Each `rigs[]`
row carries version, channel, `auto_update`, mode/backend, uptime, per-GPU telemetry,
thermal + watchdog status, and `last_seen_age_s`.

## systemd unit (coordinator host)

```ini
# /etc/systemd/system/matador-hub.service
[Unit]
Description=Matador fleet telemetry hub
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=matador
ExecStart=/usr/bin/python3 /opt/matador/scripts/matador-hub.py --config /etc/matador/hub.json
Restart=always
RestartSec=5
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now matador-hub.service
```

## Notes

- A rig that stops responding keeps its last-known row but flips to `online:false` after
  `--offline-after-s` (default 30s); its rate drops to 0.
- The hub is read-only: it never sends commands to rigs, only reads `/summary`.
- `episode_per_s` needs two polls to establish a rate, so it reads 0 for the first
  `poll_interval_s` after a rig appears.
