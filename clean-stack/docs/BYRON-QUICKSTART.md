# Mining on the Byron Pool: Quickstart

This guide gets the `matador-miner` running against the Byron pool on a fresh
Linux + NVIDIA instance.

## What you need

| Item | Notes |
|------|-------|
| Linux x86_64 box with an NVIDIA GPU | `nvidia-smi` must work (driver installed) |
| The `matador-miner` binary | Linux x86_64 build (from the release link your operator gives you) |
| A Tailscale auth key for the Byron network | Ask the pool operator. The pool is only reachable on its private network |
| Your BTX payout address | Starts with `btx1...`. Rewards go here |
| A worker name | Any label for this machine, e.g. `rig0` |

## Step 1: Join the Byron network (Tailscale)

The pool lives at `100.64.0.1:3334` on a private Tailscale network, so you join it first.

**Standard box (you have root):**
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --authkey=tskey-auth-XXXXX
```

**Cloud GPU box with no tun device (common on rented GPUs):** run Tailscale in
userspace mode, which opens a local SOCKS5 proxy the miner can use:
```bash
tailscaled --tun=userspace-networking --socks5-server=localhost:1055 --statedir=/var/lib/tailscale &
tailscale up --authkey=tskey-auth-XXXXX
```
(Any port works for the proxy, just keep it the same in Step 3.)

**Confirm you can reach the pool:**
```bash
tailscale status            # shows you connected
nc -vz 100.64.0.1 3334      # "succeeded" / "open" means the pool is reachable
```

## Step 2: Get the miner binary

```bash
# put it somewhere stable and make it runnable
mkdir -p ~/.local/bin
mv matador-miner ~/.local/bin/matador-miner
chmod +x ~/.local/bin/matador-miner
```

## Step 3: Start mining

**Standard box (on the tailnet directly):**
```bash
~/.local/bin/matador-miner --mode pool \
  --pool 100.64.0.1:3334 \
  --payoutaddress <YOUR_BTX_ADDRESS> \
  --worker <WORKER_NAME> \
  --gpus auto \
  --no-auto-update --no-update-check
```

**Cloud box (userspace Tailscale): prefix with the SOCKS5 proxy from Step 1:**
```bash
SOCKS5=127.0.0.1:1055 ~/.local/bin/matador-miner --mode pool \
  --pool 100.64.0.1:3334 \
  --payoutaddress <YOUR_BTX_ADDRESS> \
  --worker <WORKER_NAME> \
  --gpus auto \
  --no-auto-update --no-update-check
```

Placeholders:
- `<YOUR_BTX_ADDRESS>`: your `btx1...` payout address.
- `<WORKER_NAME>`: a unique label per machine (e.g. `rig0`) so you can tell
  instances apart on the dashboard.
- `--gpus auto`: uses every GPU in the box. To pin specific cards instead:
  `--gpus 0,1`.

## Step 4: Confirm it's working

A healthy miner shows:
- `[stats]` lines with a rate and a count of accepted shares.
- Rejected (`rej`) staying at or near **0**.
- Your worker appearing on the pool dashboard with a hashrate.

If you see accepted shares climbing and `rej` near 0, you're mining.

## Step 5: Keep it running (recommended)

Install it as a service so it survives reboots and restarts on crash. Create
`/etc/systemd/system/matador.service`:
```ini
[Unit]
Description=Matador BTX miner (Byron pool)
After=network-online.target

[Service]
Environment=SOCKS5=127.0.0.1:1055
ExecStart=/home/USER/.local/bin/matador-miner --mode pool --pool 100.64.0.1:3334 --payoutaddress btx1YOURADDRESS --worker rig0 --gpus auto --no-auto-update --no-update-check
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
(Drop the `Environment=SOCKS5=...` line if you're on a standard tailnet, not
userspace mode. Replace `USER`, the address, and the worker name.)
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now matador
journalctl -u matador -f      # watch the logs
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `connection refused`, can't reach `100.64.0.1` | Not on the Byron network, or auth key expired | Re-run `tailscale up --authkey=...` with a fresh key from the operator; confirm with `tailscale status` and `nc -vz 100.64.0.1 3334` |
| Connects but rate stays 0 on a cloud box | No tun device, traffic not routed | Use the userspace + `SOCKS5=127.0.0.1:1055` path in Steps 1 and 3 |
| Shares rejected (`rej` climbing) | Modified or outdated binary, or bad address | Run the official unmodified binary; double-check your `btx1...` address; contact the operator |
| `error while loading shared libraries`, exits at start | Missing GPU driver or libraries | Install the NVIDIA driver (`nvidia-smi` must work); run `ldd ~/.local/bin/matador-miner` to spot missing libs |
| Stops after reboot | Not installed as a service | Use the systemd unit in Step 5 |
