#!/usr/bin/env python3
"""Discover nodes on the LIVE chain (not the stuck ~185570 cluster) and refresh livepeers.txt.
P2P-probes candidate addresses (the node's address book + current livepeers + seeds), reads each
one's version-handshake height, and keeps those within 20 of the highest reachable height AND
clearly above the stuck cluster. That set = the true-tip nodes we want pinned."""
import hashlib, socket, struct, time, subprocess, json, os

MAGIC = bytes([0xb7, 0x54, 0x58, 0x01]); PROTO = 800001
C = "btx-miner-0332"
LIVEFILE = os.path.expanduser("~/matador-solo/livepeers.txt")
STUCK_CEIL = 185600     # anything at/below this is the frozen self-qual cluster, not live

def sha256d(b): return hashlib.sha256(hashlib.sha256(b).digest()).digest()
def m(cmd, p=b""): return MAGIC + cmd.ljust(12, b"\x00") + struct.pack("<I", len(p)) + sha256d(p)[:4] + p
def vi(n): return bytes([n]) if n < 0xfd else b"\xfd" + struct.pack("<H", n)
def vpay():
    ua = b"/disc:0.1/"; a = struct.pack("<Q", 0)+b"\x00"*16+struct.pack(">H", 0)
    return struct.pack("<iQq", PROTO, 0, int(time.time()))+a+a+struct.pack("<Q", 7)+vi(len(ua))+ua+struct.pack("<i", 0)+b"\x00"

def height(host, port, to=6):
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

cands = set()
for ln in open(LIVEFILE) if os.path.exists(LIVEFILE) else []:
    ln = ln.strip()
    if ln: cands.add(tuple(ln.rsplit(":", 1)))
for seed in ["node.btx.dev:19335", "node.btxchain.org:19335", "146.190.179.86:19335", "89.85.40.184:19335"]:
    cands.add(tuple(seed.rsplit(":", 1)))
try:
    for x in json.loads(cli("getnodeaddresses", "80")):
        cands.add((x["address"], str(x["port"])))
except Exception:
    pass
try:
    for p in json.loads(cli("getpeerinfo") or "[]"):
        ip, _, port = p["addr"].rpartition(":")
        cands.add((ip.strip("[]"), port))
except Exception:
    pass

import concurrent.futures
cand_list = list(cands)[:60]
def _probe(cp):
    ip, port = cp
    try:
        return (height(ip, int(port), to=3), ip, int(port))
    except Exception:
        return None
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=30) as ex:
    for r in ex.map(_probe, cand_list):
        if r: results.append(r)
if not results:
    print("no reachable candidates (keeping existing livepeers)"); raise SystemExit(0)
top = max(h for h, _, _ in results)
# LIVE = clearly above the frozen stuck cluster. Not "within N of the very tip" -- a node at
# 185700 while the tip is 185811 is still on the live chain and can serve bodies for its range;
# pinning more live sources = more reliable body flow. STUCK_CEIL cleanly separates the two tiers.
newly_live = {f"{ip}:{port}" for h, ip, port in results if h > STUCK_CEIL}
confirmed_stuck = {f"{ip}:{port}" for h, ip, port in results if h <= STUCK_CEIL}
# MERGE: keep existing pins (they may be intermittently unreachable live nodes), add the newly
# discovered live ones, and drop only addresses PROVED to be stuck-cluster this probe. Never
# shrink to just what happened to answer this round.
existing = set()
if os.path.exists(LIVEFILE):
    existing = {l.strip() for l in open(LIVEFILE) if l.strip()}
merged = (existing | newly_live) - confirmed_stuck
merged = sorted(merged)[:15]
print(f"probed {len(cand_list)} cands, {len(results)} reachable, top height={top}, "
      f"pinned live nodes={len(merged)} (+{len(newly_live - existing)} new, -{len(existing & confirmed_stuck)} stuck)")
with open(LIVEFILE, "w") as f:
    for a in merged:
        f.write(a + "\n")
        print(f"  live-pin {a}")
print(f"true-tip ~= {top}")

# DURABLE PIN: maintain a managed addnode block in btx.conf so btxd connects to the live chain
# at EVERY startup (survives wedge-restarts, reboots, and this script not running). Idempotent:
# rewrite only the lines between our markers; never touch the rest of the operator's config.
BTXCONF = "/home/vanities/_attic/matador-miner/btx-data/btx.conf"
BEGIN = "# === MATADOR LIVE-TIP PINS (auto-maintained by find_live_peers.py) ==="
END = "# === END MATADOR LIVE-TIP PINS ==="
try:
    cur = subprocess.run(["sudo", "-n", "cat", BTXCONF], capture_output=True, text=True).stdout
    lines, skip = [], False
    for ln in cur.splitlines():
        if ln.strip() == BEGIN:
            skip = True; continue
        if ln.strip() == END:
            skip = False; continue
        if not skip:
            lines.append(ln)
    while lines and lines[-1].strip() == "":
        lines.pop()
    block = [BEGIN] + [f"addnode={a}" for a in merged] + [END]
    newconf = "\n".join(lines + [""] + block) + "\n"
    p = subprocess.run(["sudo", "-n", "tee", BTXCONF], input=newconf, capture_output=True, text=True)
    if p.returncode == 0:
        print(f"  btx.conf: durable pin block updated ({len(merged)} live nodes)")
    else:
        print(f"  btx.conf update FAILED: {p.stderr.strip()[:80]}")
except Exception as e:
    print(f"  btx.conf update error: {e}")
