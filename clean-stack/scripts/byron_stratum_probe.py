#!/usr/bin/env python3
"""
byron_stratum_probe.py - protocol-level test harness for the Byron BTX pool.

Exercises the Byron stratum handshake WITHOUT a GPU and WITHOUT touching the
running miner, so the capability handshake (Byron's job-dispatch gate) can be
validated and the real capability.req / mining.notify JSON captured BEFORE
rebuilding the CUDA miner.

Flow it drives (mirrors matador-miner's StratumClient::HandleCapabilityReq):
  -> mining.subscribe
  -> mining.authorize
  <- mining.set_difficulty
  <- mining.capability.req   [byron-pool/2, [matmul_nonce_seed_v2]]
  -> mining.capability.ack   [byron-pool/2, [matmul_nonce_seed_v2], <solver-sha256>]
  <- mining.notify           <== SUCCESS: handshake worked, jobs are flowing
     ...or mining.notify_incompatible  <== ack rejected; the printed reason says why

It never solves or submits. It only proves the pool will hand us work and prints
the exact wire JSON so the C++ ack format / notify parser can be finalized. If the
ack format below is wrong, tweak build_capability_ack() and re-run (seconds), then
port the confirmed format to matador-miner.cpp and rebuild once.

stdlib only (no pip). Reaches 100.64.0.1 either directly (once on Byron's tailnet)
or through the existing SOCKS5 proxy (--socks5 host:port).

Examples:
  # over the tailnet (after `tailscale up ...`):
  ./byron_stratum_probe.py --user btx1zkjl....rig0
  # over your current SOCKS5 proxy (no tailnet needed):
  ./byron_stratum_probe.py --user btx1zkjl....rig0 --socks5 127.0.0.1:1080
  # advertise the exact binary the miner runs (for the allowlist):
  ./byron_stratum_probe.py --user btx1zkjl....rig0 --solver-bin ~/.local/bin/matador-miner
"""

import argparse
import hashlib
import json
import os
import socket
import sys
import time

# Capabilities matador-miner actually implements (keep in sync with
# StratumClient::SupportedCapability in matador-miner.cpp).
SUPPORTED_CAPS = {"matmul_nonce_seed_v2"}


def ts():
    return time.strftime("%H:%M:%S")


def log(direction, msg):
    # direction: "->" sent, "<-" recv, "**" event, "!!" problem
    print(f"[{ts()}] {direction} {msg}", flush=True)


def socks5_connect(proxy_host, proxy_port, dst_host, dst_port,
                   puser=None, ppass=None, timeout=30):
    """Minimal SOCKS5 CONNECT (no-auth or user/pass). Returns a connected socket."""
    t0 = time.time()
    s = socket.create_connection((proxy_host, proxy_port), timeout=timeout)
    methods = b"\x00\x02" if puser is not None else b"\x00"
    s.sendall(b"\x05" + bytes([len(methods)]) + methods)
    resp = s.recv(2)
    if len(resp) < 2 or resp[0] != 5:
        raise RuntimeError(f"socks5 bad greeting resp: {resp!r}")
    method = resp[1]
    if method == 2:
        if puser is None:
            raise RuntimeError("socks5 proxy requires user/pass but none provided")
        u = puser.encode()
        p = (ppass or "").encode()
        s.sendall(b"\x01" + bytes([len(u)]) + u + bytes([len(p)]) + p)
        ar = s.recv(2)
        if len(ar) < 2 or ar[1] != 0:
            raise RuntimeError(f"socks5 user/pass auth failed: {ar!r}")
    elif method != 0:
        raise RuntimeError(f"socks5 no acceptable auth method (server chose {method})")
    dh = dst_host.encode()
    s.sendall(b"\x05\x01\x00\x03" + bytes([len(dh)]) + dh + dst_port.to_bytes(2, "big"))
    rep = s.recv(4)
    if len(rep) < 4 or rep[1] != 0:
        raise RuntimeError(f"socks5 CONNECT failed (rep={rep!r})")
    atyp = rep[3]                       # drain the bound address + port
    if atyp == 1:
        s.recv(4)
    elif atyp == 3:
        s.recv(s.recv(1)[0])
    elif atyp == 4:
        s.recv(16)
    s.recv(2)
    log("**", f"socks5 CONNECT ok via {proxy_host}:{proxy_port} "
              f"in {(time.time() - t0) * 1000:.0f}ms")
    return s


def resolve_solver_sha(args):
    """The solver sha advertised in capability.ack. Byron treats it as compatibility
    metadata (it re-verifies every share), so a placeholder still gets us jobs - but
    --solver-bin lets us advertise the exact binary for the allowlist."""
    if args.solver_sha:
        return args.solver_sha
    env = os.environ.get("MATADOR_SOLVER_SHA256")
    if env:
        return env
    if args.solver_bin and os.path.exists(args.solver_bin):
        h = hashlib.sha256()
        with open(args.solver_bin, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        d = h.hexdigest()
        log("**", f"hashed {args.solver_bin} -> sha256 {d[:12]}...")
        return d
    log("!!", "no --solver-sha / MATADOR_SOLVER_SHA256 / --solver-bin; advertising zeros "
              "(fine for testing - pool re-verifies shares)")
    return "0" * 64


def build_capability_ack(req_obj, sha):
    """Mirror matador-miner's HandleCapabilityReq. If the pool rejects the ack,
    THIS is the one function to tweak (id echo vs null, method-call vs result,
    param order), then re-run."""
    params = req_obj.get("params") or []
    proto = params[0] if len(params) >= 1 and isinstance(params[0], str) else "byron-pool/2"
    requested = params[1] if len(params) >= 2 and isinstance(params[1], list) else []
    caps = [c for c in requested if c in SUPPORTED_CAPS]
    ack = {
        "id": req_obj.get("id"),                 # echo the request id (None -> JSON null)
        "method": "mining.capability.ack",
        "params": [proto, caps, sha],
    }
    return ack, len(caps), len(requested)


def main():
    ap = argparse.ArgumentParser(description="Byron stratum capability-handshake probe")
    ap.add_argument("--host", default="100.64.0.1")
    ap.add_argument("--port", type=int, default=3334)
    ap.add_argument("--user", required=True, help="<BTX-address>.<worker> (payout.worker)")
    ap.add_argument("--pass", dest="passwd", default="x", help="authorize password (Byron ignores)")
    ap.add_argument("--socks5", help="SOCKS5 proxy host:port (omit to connect directly)")
    ap.add_argument("--socks5-user")
    ap.add_argument("--socks5-pass")
    ap.add_argument("--solver-sha", help="advertise this sha256 verbatim")
    ap.add_argument("--solver-bin", help="hash this binary for the advertised solver sha")
    ap.add_argument("--ua", default="byron-probe/0.1", help="mining.subscribe user-agent")
    ap.add_argument("--timeout", type=int, default=120, help="give up after N s with no job")
    args = ap.parse_args()

    sha = resolve_solver_sha(args)
    log("**", f"connecting to {args.host}:{args.port}"
              + (f" via socks5 {args.socks5}" if args.socks5 else " (direct)"))

    try:
        if args.socks5:
            ph, pp = args.socks5.rsplit(":", 1)
            s = socks5_connect(ph, int(pp), args.host, args.port,
                               args.socks5_user, args.socks5_pass)
        else:
            s = socket.create_connection((args.host, args.port), timeout=30)
    except Exception as e:
        log("!!", f"connect failed: {e}")
        log("!!", "if direct: are you on Byron's tailnet? (`tailscale status`, "
                  "`nc -vz 100.64.0.1 3334`). else pass --socks5 host:port")
        sys.exit(2)

    s.settimeout(5.0)
    log("**", f"connected; advertising solver_sha={sha[:12]}...")

    def send(obj):
        line = json.dumps(obj) + "\n"
        s.sendall(line.encode())
        log("->", line.strip())

    send({"id": 1, "method": "mining.subscribe", "params": [args.ua]})
    send({"id": 2, "method": "mining.authorize", "params": [args.user, args.passwd]})

    buf = b""
    deadline = time.time() + args.timeout
    job_linger_until = None
    got_req = got_ack = got_job = False

    while True:
        now = time.time()
        if job_linger_until is not None and now > job_linger_until:
            break
        if now > deadline:
            log("!!", f"no mining.notify within {args.timeout}s - giving up "
                      f"(capability.req seen={got_req}, ack sent={got_ack})")
            break
        try:
            data = s.recv(4096)
        except socket.timeout:
            continue
        except Exception as e:
            log("!!", f"recv error: {e}")
            break
        if not data:
            log("!!", "pool closed the connection")
            break
        buf += data
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            raw = raw.strip()
            if not raw:
                continue
            log("<-", raw.decode(errors="replace"))
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            method = obj.get("method")
            if method == "mining.capability.req":
                got_req = True
                log("**", "*** capability.req received - THIS is the job-dispatch gate ***")
                ack, nsup, nreq = build_capability_ack(obj, sha)
                send(ack)
                got_ack = True
                log("**", f"sent capability.ack ({nsup}/{nreq} requested caps supported)")
                if nsup == 0:
                    log("!!", "0 caps supported - pool will likely send notify_incompatible")
            elif method == "mining.notify":
                if not got_job:
                    log("**", "*** mining.notify received - HANDSHAKE WORKED, jobs flowing ***")
                    log("**", "    ^ capture this job object; it answers header-based vs seed-only")
                    job_linger_until = time.time() + 3   # grab a 2nd job, then exit clean
                got_job = True
            elif method == "mining.notify_incompatible":
                log("!!", "*** notify_incompatible - the ack was REJECTED ***")
                log("!!", "reason: " + json.dumps(obj.get("params")))
                log("!!", "-> tweak build_capability_ack() (id echo? result vs method? "
                          "param order?) and re-run")

    log("**", f"summary: capability.req={got_req} ack_sent={got_ack} job_received={got_job}")
    try:
        s.close()
    except Exception:
        pass
    sys.exit(0 if got_job else 1)


if __name__ == "__main__":
    main()
