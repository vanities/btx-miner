#!/usr/bin/env python3
"""Purge the stuck-cluster nodes: probe every configured/known/connected node's height, then
(1) remove confirmed-stuck addnode= lines from btx.conf so btxd stops dialing them, and
(2) setban the stuck IPs (24h, reversible) so it refuses to reconnect -- freeing every
connection slot for the live chain. Never touches the MATADOR LIVE-TIP block or unreachable
addresses (those are unknown, could be live nodes that are momentarily down)."""
import hashlib, socket, struct, time, subprocess, json, os, concurrent.futures

MAGIC = bytes([0xb7, 0x54, 0x58, 0x01]); PROTO = 800001
C = "btx-miner-0332"
BTXCONF = "/home/vanities/_attic/matador-miner/btx-data/btx.conf"
STUCK_CEIL = 185600     # height <= this = the frozen self-qual cluster
BAN_S = 86400

def sha256d(b): return hashlib.sha256(hashlib.sha256(b).digest()).digest()
def m(cmd, p=b""): return MAGIC + cmd.ljust(12, b"\x00") + struct.pack("<I", len(p)) + sha256d(p)[:4] + p
def vi(n): return bytes([n]) if n < 0xfd else b"\xfd" + struct.pack("<H", n)
def vpay():
    ua = b"/clean:0.1/"; a = struct.pack("<Q", 0)+b"\x00"*16+struct.pack(">H", 0)
    return struct.pack("<iQq", PROTO, 0, int(time.time()))+a+a+struct.pack("<Q", 9)+vi(len(ua))+ua+struct.pack("<i", 0)+b"\x00"
def height(host, port, to=3):
    s = socket.create_connection((host, port), timeout=to); s.settimeout(to)
    s.sendall(m(b"version", vpay()))
    h = b""
    while len(h) < 24: h += s.recv(24-len(h))
    ln = struct.unpack("<I", h[16:20])[0]; pl = b""
    while len(pl) < ln: pl += s.recv(min(4096, ln-len(pl)))
    s.close()
    off = 4+8+8+26+26+8; n = pl[off]; off += 1+n
    return struct.unpack("<i", pl[off:off+4])[0]
def cli(*a):
    return subprocess.run(["docker", "exec", C, "timeout", "20", "btx-cli", "-datadir=/data", *a],
                          capture_output=True, text=True).stdout.strip()

# gather candidate addresses: btx.conf addnode lines + connected peers
conf = subprocess.run(["sudo", "-n", "cat", BTXCONF], capture_output=True, text=True).stdout
conf_addr = []
for ln in conf.splitlines():
    s = ln.strip()
    if s.startswith("addnode="):
        conf_addr.append(s.split("=", 1)[1])
peers = []
try:
    for p in json.loads(cli("getpeerinfo") or "[]"):
        peers.append(p["addr"])
except Exception:
    pass
cands = sorted(set(conf_addr) | set(peers))

def probe(a):
    ip, _, port = a.rpartition(":")
    ip = ip.strip("[]")
    try:
        return (a, ip, height(ip, int(port)))
    except Exception:
        return (a, ip, None)   # unreachable = unknown, leave alone

with concurrent.futures.ThreadPoolExecutor(max_workers=40) as ex:
    results = list(ex.map(probe, cands))

stuck = [(a, ip) for a, ip, h in results if h is not None and h <= STUCK_CEIL]
live = [(a, ip, h) for a, ip, h in results if h is not None and h > STUCK_CEIL]
unknown = [a for a, ip, h in results if h is None]
print(f"probed {len(cands)}: {len(stuck)} STUCK, {len(live)} live, {len(unknown)} unreachable")

# 1) ban stuck IPs (reversible)
banned = 0
for a, ip in stuck:
    r = subprocess.run(["docker", "exec", C, "timeout", "10", "btx-cli", "-datadir=/data",
                        "setban", ip, "add", str(BAN_S)], capture_output=True, text=True)
    if "already banned" not in r.stderr.lower():
        banned += 1
    subprocess.run(["docker", "exec", C, "timeout", "10", "btx-cli", "-datadir=/data",
                    "disconnectnode", a], capture_output=True, text=True)
print(f"banned {banned} stuck IPs for {BAN_S//3600}h + disconnected")

# 2) strip confirmed-stuck addnode= lines from btx.conf (keep unknown + live + LIVE-TIP block)
stuck_addrs = {a for a, _ in stuck}
out_lines, removed = [], 0
for ln in conf.splitlines():
    s = ln.strip()
    if s.startswith("addnode=") and s.split("=", 1)[1] in stuck_addrs:
        removed += 1
        continue
    out_lines.append(ln)
newconf = "\n".join(out_lines).rstrip("\n") + "\n"
subprocess.run(["sudo", "-n", "cp", BTXCONF, BTXCONF + ".bak-pre-clean"], capture_output=True)
p = subprocess.run(["sudo", "-n", "tee", BTXCONF], input=newconf, capture_output=True, text=True)
print(f"removed {removed} stuck addnode= lines from btx.conf (backup .bak-pre-clean)"
      if p.returncode == 0 else f"btx.conf write FAILED: {p.stderr[:80]}")
for a, ip, h in sorted(live, key=lambda x: -x[2])[:6]:
    print(f"  keeping live: {a} h={h}")
