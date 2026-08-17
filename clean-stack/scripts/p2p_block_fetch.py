#!/usr/bin/env python3
"""Fetch a block directly over BTX P2P (v1 transport) from a serving peer and
submit it via local RPC. Usage: p2p_fetch.py <blockhash> [<blockhash>...]"""
import hashlib, json, socket, struct, subprocess, sys, time, random

MAGIC = bytes([0xb7, 0x54, 0x58, 0x01])
PROTO = 800001
C = "27229a02f55e"

def sha256d(b):
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()

def msg(cmd, payload=b""):
    return MAGIC + cmd.ljust(12, b"\x00") + struct.pack("<I", len(payload)) + sha256d(payload)[:4] + payload

def varint(n):
    if n < 0xfd: return bytes([n])
    return b"\xfd" + struct.pack("<H", n)

def version_payload():
    ts = int(time.time())
    addr = struct.pack("<Q", 0) + b"\x00" * 16 + struct.pack(">H", 0)
    ua = b"/matador-sync:0.1/"
    return (struct.pack("<iQq", PROTO, 0, ts) + addr + addr +
            struct.pack("<Q", random.getrandbits(64)) + varint(len(ua)) + ua +
            struct.pack("<i", 0) + b"\x00")

def recv_msg(s):
    hdr = b""
    while len(hdr) < 24:
        c = s.recv(24 - len(hdr))
        if not c: raise ConnectionError("closed")
        hdr += c
    if hdr[:4] != MAGIC: raise ConnectionError("bad magic")
    cmd = hdr[4:16].rstrip(b"\x00").decode()
    ln = struct.unpack("<I", hdr[16:20])[0]
    pl = b""
    while len(pl) < ln:
        c = s.recv(min(65536, ln - len(pl)))
        if not c: raise ConnectionError("closed mid-payload")
        pl += c
    return cmd, pl

def fetch_from(ip, port, want_hashes, timeout=45):
    s = socket.create_connection((ip, port), timeout=10)
    s.settimeout(timeout)
    s.sendall(msg(b"version", version_payload()))
    got = {}
    veracked = False
    deadline = time.time() + timeout
    want = {bytes.fromhex(h)[::-1]: h for h in want_hashes}
    asked = False
    while time.time() < deadline and len(got) < len(want_hashes):
        try:
            cmd, pl = recv_msg(s)
        except socket.timeout:
            break
        if cmd == "version":
            s.sendall(msg(b"verack"))
        elif cmd == "verack":
            veracked = True
        elif cmd == "ping":
            s.sendall(msg(b"pong", pl))
        elif cmd == "block":
            # BTX v4 headers are 182 bytes (matmul fields), NOT 80 -- hash the full header
            bh = sha256d(pl[:182])
            if bh in want:
                got[want[bh]] = pl.hex()
                print(f"  got {want[bh][:12]} ({len(pl)} bytes) from {ip}", flush=True)
            else:
                print(f"  block from {ip} hash-mismatch (len {len(pl)})", flush=True)
        if veracked and not asked:
            # ask with BOTH inv types: witness-flagged first, plain MSG_BLOCK as fallback
            # (a chain without segwit typing ignores 0x40000002 silently)
            inv_w = varint(len(want)) + b"".join(struct.pack("<I", 0x40000002) + h for h in want)
            inv_p = varint(len(want)) + b"".join(struct.pack("<I", 2) + h for h in want)
            s.sendall(msg(b"getdata", inv_w))
            s.sendall(msg(b"getdata", inv_p))
            asked = True
    s.close()
    return got

def cli(*a):
    r = subprocess.run(["docker", "exec", C, "timeout", "30", "btx-cli", "-datadir=/data", *a],
                       capture_output=True, text=True)
    return r.stdout.strip(), r.stderr.strip()

def cli_stdin(cmd, payload):
    r = subprocess.run(["docker", "exec", "-i", C, "timeout", "120", "btx-cli", "-datadir=/data",
                        "-stdin", cmd], input=payload + "\n", capture_output=True, text=True)
    return r.stdout.strip(), r.stderr.strip()

args = sys.argv[1:]
explicit = [a for a in args if ":" in a]          # host:port peer targets
want = [a for a in args if ":" not in a]
if not want:
    print("usage: p2p_fetch.py [host:port...] <hash>..."); sys.exit(2)
if explicit:
    cands = explicit
else:
    peers = json.loads(cli("getpeerinfo")[0])
    cands = [p["addr"] for p in sorted(peers, key=lambda p: -p.get("startingheight", 0))
             if p.get("startingheight", 0) > 185100]
print("candidate peers:", cands[:6], flush=True)

remaining = list(want)
blocks = {}
for addr in cands[:8]:
    if not remaining: break
    ip, _, port = addr.rpartition(":")
    ip = ip.strip("[]")
    try:
        got = fetch_from(ip, int(port), remaining)
    except Exception as e:
        print(f"  {addr}: {e}", flush=True)
        continue
    blocks.update(got)
    remaining = [h for h in remaining if h not in blocks]

for h in want:
    if h not in blocks:
        print(f"MISSING {h}", flush=True)
        continue
    out, err = cli_stdin("submitblock", blocks[h])
    print(f"submitblock {h[:12]} -> {(out or err or 'OK')[:80]}", flush=True)
print("tip now:", cli("getblockcount")[0], flush=True)
