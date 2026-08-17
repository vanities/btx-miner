#!/usr/bin/env python3
"""Independent tip check: P2P-connect to public BTX nodes and read the block height each
reports in its version handshake. Consensus among well-connected nodes ~= the true tip."""
import hashlib, socket, struct, time, sys

MAGIC = bytes([0xb7, 0x54, 0x58, 0x01])
PROTO = 800001

NODES = [
    ("node.btx.dev", 19335), ("node.btxchain.org", 19335),
    ("146.190.179.86", 19335), ("89.85.40.184", 19335),
    ("206.189.253.106", 19335), ("178.128.156.73", 19335),
    ("46.101.240.240", 19335), ("142.189.93.78", 48537),
    ("seed.btx.dev", 19335), ("seed.btxchain.org", 19335),
]
if len(sys.argv) > 1:
    NODES = [(a.rsplit(":", 1)[0], int(a.rsplit(":", 1)[1])) for a in sys.argv[1:]]

def sha256d(b): return hashlib.sha256(hashlib.sha256(b).digest()).digest()
def msg(cmd, p=b""): return MAGIC + cmd.ljust(12, b"\x00") + struct.pack("<I", len(p)) + sha256d(p)[:4] + p
def varint(n): return bytes([n]) if n < 0xfd else b"\xfd" + struct.pack("<H", n)

def version_payload():
    ua = b"/probe:0.1/"
    addr = struct.pack("<Q", 0) + b"\x00"*16 + struct.pack(">H", 0)
    return (struct.pack("<iQq", PROTO, 0, int(time.time())) + addr + addr +
            struct.pack("<Q", 12345) + varint(len(ua)) + ua + struct.pack("<i", 0) + b"\x00")

def parse_start_height(pl):
    # version payload: ver(4) serv(8) ts(8) addr_recv(26) addr_from(26) nonce(8) ua(varstr) height(4) relay(1)
    off = 4+8+8+26+26+8
    n = pl[off]; off += 1
    if n >= 0xfd:  # varint (unlikely for UA len)
        n = struct.unpack("<H", pl[off:off+2])[0]; off += 2
    off += n
    return struct.unpack("<i", pl[off:off+4])[0]

def probe(host, port, timeout=8):
    s = socket.create_connection((host, port), timeout=timeout)
    s.settimeout(timeout)
    s.sendall(msg(b"version", version_payload()))
    hdr = b""
    while len(hdr) < 24: hdr += s.recv(24-len(hdr))
    ln = struct.unpack("<I", hdr[16:20])[0]
    pl = b""
    while len(pl) < ln: pl += s.recv(min(4096, ln-len(pl)))
    s.close()
    return parse_start_height(pl), pl[4+8+8+26+26+8+1:pl.index(b"/", pl.index(b"/")+1)+1].decode("latin1", "ignore")

print("independent node heights (their reported tip):")
heights = []
for host, port in NODES:
    try:
        h, ua = probe(host, port)
        heights.append(h)
        print(f"  {host}:{port:<6} height={h}  {ua[:20]}")
    except Exception as e:
        print(f"  {host}:{port:<6} unreachable ({type(e).__name__})")
if heights:
    heights.sort()
    print(f"\n  reachable: {len(heights)}  min={heights[0]} max={heights[-1]}  median={heights[len(heights)//2]}")
