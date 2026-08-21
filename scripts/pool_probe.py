#!/usr/bin/env python3
# Read-only-ish pool probe: DNS -> TCP -> stratum/login handshake; optionally one bogus
# submit WITHOUT authorize to capture the pool's unauthorized wording. Times out fast.
import json, socket, sys, time

def probe(host, port, mode):
    out = {"host": host, "port": port}
    t0 = time.time()
    try:
        addrs = sorted({ai[4][0] for ai in socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)})
        out["dns"] = addrs
    except Exception as e:
        out["dns_error"] = f"{type(e).__name__}: {e}"; return out
    try:
        s = socket.create_connection((host, port), timeout=6)
        s.settimeout(6)
        out["tcp"] = "connected in %.0fms" % ((time.time()-t0)*1000)
    except Exception as e:
        out["tcp_error"] = f"{type(e).__name__}: {e}"; return out
    def send(obj): s.sendall((json.dumps(obj)+"\n").encode())
    def lines(max_s=6):
        buf = b""; end = time.time()+max_s; got=[]
        while time.time() < end and len(got) < 8:
            try: chunk = s.recv(4096)
            except socket.timeout: break
            if not chunk: got.append("<closed>"); break
            buf += chunk
            while b"\n" in buf:
                ln, buf = buf.split(b"\n",1)
                got.append(ln.decode(errors="replace")[:300])
        return got
    try:
        if mode == "login":
            send({"id":3,"method":"login","params":{"address":"btx1qtestprobeonly","agent":"probe/0.1","worker":"probe"}})
            out["login_resp"] = lines()
        elif mode == "sub_only_submit":
            send({"id":1,"method":"mining.subscribe","params":["probe/0.1"]})
            out["subscribe_resp"] = lines(4)
            # bogus submit with NO authorize -> capture the unauthorized wording
            send({"id":11,"method":"mining.submit","params":["btx1qtestprobeonly.probe","job-0","00000000","00000000","00000000"]})
            out["unauth_submit_resp"] = lines(5)
        else:  # full classic handshake
            send({"id":1,"method":"mining.subscribe","params":["probe/0.1"]})
            send({"id":2,"method":"mining.authorize","params":["btx1qtestprobeonly.probe","x"]})
            out["handshake_resp"] = lines(5)
    except Exception as e:
        out["proto_error"] = f"{type(e).__name__}: {e}"
    try: s.close()
    except Exception: pass
    return out

for spec in sys.argv[1:]:
    host, port, mode = spec.split(",")
    print(json.dumps(probe(host, int(port), mode), indent=1))
