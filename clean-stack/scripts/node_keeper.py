#!/usr/bin/env python3
"""matador node keeper: keep the node ON the live chain tip, forever.

Steady state (every 10 s): attest any new block above our tip (GPU ExactReplay gate,
quorum-checked first so nothing is ever replayed twice) - blocks then connect on their own.

Stall state (tip unchanged while a higher header tip exists, 3+ ticks): run the full
recovery ladder proved out on 2026-08-10:
  1. attest EVERY branch block lacking quorum, INCLUDING reorg-side blocks at/below our
     tip (the attest_all blind spot that wedged the 185003 fork recovery),
  2. fetch missing bodies: getblockfrompeer round-robin over tall peers, then the direct
     P2P fetcher (182-byte-header aware) as the hammer,
  3. submitblock sweep oldest-first (bypasses the 1-per-minute trusted-mirror drip).

Container resolved by NAME (survives recreation). Cookie read fresh per submit
(survives rotation). Quiet when idle; one line per state change; ALERT line if the
gap survives the whole ladder for ~10 minutes (grep journal for KEEPER-ALERT).
"""
import json, os, statistics, subprocess, time

CONTAINER = "btx-miner-0332"
RPC = "http://127.0.0.1:19334/"
BIN = os.path.expanduser("~/git/wt-solo-stats/clean-stack/build-solo-stats/rc_attest")
P2P = os.path.expanduser("~/matador-solo/p2p_fetch3.py")
CTX = "531e9f3a57dd383ce6611502ee29d166c9fe66ced5397a88be5aeb4cee641bd0"
KEY = open(os.path.expanduser("~/matador-solo/attest.key")).read().strip()
SEEDS = ["89.85.40.184:19335", "node.btx.dev:19335", "146.190.179.86:19335"]
TICK_S = 10
STALL_TICKS = 3          # same tip this many ticks with a higher header tip -> ladder
ALERT_S = 600            # gap survives the ladder this long -> KEEPER-ALERT

def log(m):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}", flush=True)

def cli(*a, t="30"):
    r = subprocess.run(["docker", "exec", CONTAINER, "timeout", t, "btx-cli", "-datadir=/data", *a],
                       capture_output=True, text=True)
    return r.stdout.strip(), r.stderr.strip()

def cli_stdin(cmd, payload):
    r = subprocess.run(["docker", "exec", "-i", CONTAINER, "timeout", "120", "btx-cli",
                        "-datadir=/data", "-stdin", cmd], input=payload + "\n",
                       capture_output=True, text=True)
    return r.stdout.strip(), r.stderr.strip()

def cookie():
    return subprocess.run(["sudo", "-n", "cat", "/home/vanities/_attic/matador-miner/btx-data/.cookie"],
                          capture_output=True, text=True).stdout.strip()

def prev_of(raw):
    return bytes.fromhex(raw)[4:36][::-1].hex()

def ntime(raw):
    return int.from_bytes(bytes.fromhex(raw)[68:72], "little")

def nbits_of(raw):
    # v4 182-byte header: version(4) prev(32) merkle(32) time@68 nBits@72 nonce64@76 ...
    return int.from_bytes(bytes.fromhex(raw)[72:76], "little")

def compact_work(nbits):
    """Chainwork contribution of one header: 2^256 // (target+1), target from compact nBits."""
    exp, mant = nbits >> 24, nbits & 0x007FFFFF
    target = mant << (8 * (exp - 3)) if exp > 3 else mant >> (8 * (3 - exp))
    if target <= 0:
        return 0
    return (1 << 256) // (target + 1)

def connected_chainwork(h):
    """Chainwork of a CONNECTED block (verbose getblockheader works for connected)."""
    info, _ = cli("getblockheader", h)
    try:
        return int(json.loads(info)["chainwork"], 16)
    except Exception:
        return None

def branch_work(chain, fork_parent):
    """Total chainwork of a headers-only branch = fork parent's chainwork + header works."""
    base = connected_chainwork(fork_parent)
    if base is None:
        return None
    return base + sum(compact_work(nbits_of(raw)) for _, raw in chain)

def peer_claim_max():
    """Highest tip any peer has claimed (handshake height + best_peer_tip if exposed)."""
    best = 0
    out, _ = cli("getpeerinfo")
    try:
        for p in json.loads(out):
            best = max(best, int(p.get("startingheight", 0) or 0))
    except Exception:
        pass
    out, _ = cli("getmatmulchallengeprofile")
    if out:
        import re
        m = re.search(r'"best_peer_tip":\s*(\d+)', out)
        if m:
            best = max(best, int(m.group(1)))
    return best

def tip_height():
    out, _ = cli("getblockcount")
    return int(out) if out else -1

_workcache = {}          # candidate tip hash -> (branch_work, chain, fork_parent); tips are immutable

def best_work_candidate(tip):
    """The candidate branch with the MOST WORK (not height). Returns
    (work, tips_entry, chain, fork_parent) or None when no candidate is worth chasing."""
    out, _ = cli("getchaintips")
    if not out:
        return None
    cands = [x for x in json.loads(out)
             if x["status"] in ("headers-only", "valid-headers") and x["height"] > tip - 200]
    cands.sort(key=lambda c: -c["height"])
    best, walked = None, False
    for c in cands:
        cached = _workcache.get(c["hash"])
        if cached is None:
            # walk budget: at most ONE uncached branch per tick (highest first), so a cold
            # start amortizes the big stale-fork walks across ticks instead of stalling.
            if walked:
                continue
            walked = True
            chain, fork_parent = walk_branch(c)
            if not chain:
                continue
            w = branch_work(chain, fork_parent)
            if w is None:
                continue
            _workcache[c["hash"]] = (w, chain, fork_parent)
            cached = _workcache[c["hash"]]
        w, chain, fork_parent = cached
        if best is None or w > best[0]:
            best = (w, c, chain, fork_parent)
    while len(_workcache) > 64:
        _workcache.pop(next(iter(_workcache)))
    return best

def has_quorum(h):
    out, _ = cli("getmatmulattestations", h)
    try:
        return bool(json.loads(out).get("quorum"))
    except Exception:
        return False

def has_body(h):
    out, err = cli("getblock", h, "0")
    return bool(out and not err)

def walk_branch(best):
    """Oldest-first [(hash, raw_header)] for the whole branch (raw headers only)."""
    chain, h = [], best["hash"]
    for _ in range(best["branchlen"]):
        raw, _ = cli("getblockheader", h, "false")
        if not raw:
            break
        chain.append((h, raw))
        h = prev_of(raw)
    chain.reverse()
    return chain, h                      # h = connected fork parent

def attest(raw, height, mtp):
    r = subprocess.run(["sudo", "-n", BIN, "--header", raw, "--height", str(height),
                        "--parent-mtp", str(mtp), "--chain-id", GENESIS, "--context", CTX,
                        "--privkey", KEY], capture_output=True, text=True)
    hexs = [l for l in r.stdout.split() if len(l) > 200 and all(c in "0123456789abcdef" for c in l)]
    if not hexs:
        return False, (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "?")
    body = json.dumps({"jsonrpc": "1.0", "id": "k", "method": "submitmatmulattestations",
                       "params": [[hexs[-1]]]})
    out = subprocess.run(["curl", "-s", "--user", cookie(), "--data-binary", body,
                          "-H", "content-type: text/plain;", RPC],
                         capture_output=True, text=True).stdout
    return "accepted" in out or "duplicate" in out, out[:80]

def attest_branch(chain, fork_h):
    """Attest every branch block lacking quorum (INCLUDING below-tip reorg blocks)."""
    times = {}
    for hh in range(fork_h - 11, fork_h + 1):
        if hh < 0:
            continue
        bh, _ = cli("getblockhash", str(hh))
        if bh:
            raw, _ = cli("getblockheader", bh, "false")
            if raw:
                times[hh] = ntime(raw)
    for i, (_, raw) in enumerate(chain):
        times[fork_h + 1 + i] = ntime(raw)
    done = 0
    for i, (bh, raw) in enumerate(chain):
        H = fork_h + 1 + i
        if has_quorum(bh):
            continue
        prior = [times[x] for x in range(H - 11, H) if x in times]
        if len(prior) < 11:
            log(f"h={H} only {len(prior)} ancestor times; cannot attest")
            break
        ok, msg = attest(raw, H, int(statistics.median(sorted(prior))))
        if ok:
            done += 1
        else:
            log(f"h={H} {bh[:12]} attest failed: {msg}")
    return done

def tall_peer_addrs():
    out, _ = cli("getpeerinfo")
    try:
        ps = json.loads(out)
    except Exception:
        return []
    ids = [(p["id"], p["addr"]) for p in ps if p.get("startingheight", 0) > 185100]
    return ids[:10]

def fetch_bodies(missing, use_p2p):
    peers = tall_peer_addrs()
    for i, h in enumerate(missing):
        if peers:
            cli("getblockfrompeer", h, str(peers[i % len(peers)][0]))
        time.sleep(0.05)
    if use_p2p and missing:
        targets = list(dict.fromkeys([a for _, a in peers] + SEEDS))
        subprocess.run(["timeout", "240", "python3", P2P, *targets, *missing[:60]],
                       capture_output=True, text=True)

def sweep(chain):
    ok = 0
    for hh, _ in chain:
        out, err = cli("getblock", hh, "0")
        if not out or err:
            break
        cli_stdin("submitblock", out)
        ok += 1
    return ok

GENESIS = ""
def main():
    global GENESIS
    while not GENESIS:
        GENESIS, _ = cli("getblockhash", "0")
        if not GENESIS:
            log("node RPC not ready; waiting")
            time.sleep(15)
    log(f"keeper up (container={CONTAINER}, work-based chain selection)")
    behind_since, hdr_lag_since, tick = None, None, 0
    while True:
        try:
            tick += 1
            tip = tip_height()
            if tip < 0:
                log("RPC error; waiting")
                time.sleep(30)
                continue
            active_hash, _ = cli("getbestblockhash")
            active_work = connected_chainwork(active_hash) or 0
            best = best_work_candidate(tip)

            # Eclipse / header-lag tripwire: peers claim tips well beyond every header we know.
            if tick % 6 == 0:
                claim = peer_claim_max()
                known = max([tip] + [c["height"] for c in
                            (json.loads(cli("getchaintips")[0] or "[]"))])
                if claim > known + 3:
                    if hdr_lag_since is None:
                        hdr_lag_since = time.time()
                    elif time.time() - hdr_lag_since > ALERT_S:
                        log(f"KEEPER-ALERT: peers claim tip {claim} but our best-known header "
                            f"is {known} -- possible eclipse or header-sync failure")
                        hdr_lag_since = time.time()
                else:
                    hdr_lag_since = None

            if best is None or best[0] <= active_work:
                # POSITIVE confirmation: our active chain carries at least as much work as
                # every known competing branch. This is the "on the right chain" check.
                if behind_since:
                    log(f"recovered: on best-work chain at tip={tip}")
                behind_since = None
                if tick % 60 == 1:
                    alt = f"alt_work_deficit={(active_work - best[0]) >> 200}" if best else "no_alt"
                    log(f"audit: ON BEST-WORK CHAIN tip={tip} work~2^{active_work.bit_length()-1} {alt}")
                time.sleep(TICK_S)
                continue

            # A competing branch carries MORE work than our active chain (behind on our own
            # chain extension, or on a secondary chain after a fork) -- same remedy either way.
            w, c, chain, fork_parent = best
            if behind_since is None:
                behind_since = time.time()
            elif time.time() - behind_since > 25:
                # persisted past the routine fresh-block window -> worth a line
                log(f"BEHIND best-work branch: tip={tip} branch_tip={c['height']} "
                    f"({c['hash'][:12]}); remedying")
            info, _ = cli("getblockheader", fork_parent)
            try:
                fh = json.loads(info)["height"]
            except Exception:
                log("cannot resolve fork parent height; retrying")
                time.sleep(TICK_S)
                continue
            n = attest_branch(chain, fh)
            missing = [hh for hh, _ in chain if not has_body(hh)]
            persist = time.time() - behind_since
            if missing:
                log(f"remedy: attested {n}, missing bodies {len(missing)} (p2p={persist > 60})")
                fetch_bodies(missing, use_p2p=persist > 60)
                time.sleep(8)
            swept = sweep(chain)
            t1 = tip_height()
            if n or missing or t1 != tip:
                log(f"remedy: swept {swept}, tip {tip} -> {t1}")
            if _workcache.get(c["hash"]) and t1 > tip:
                _workcache.pop(c["hash"], None)   # branch state changed; recompute next tick
            if persist > ALERT_S:
                log(f"KEEPER-ALERT: heavier branch {c['hash'][:12]} (h={c['height']}) "
                    f"unresolved for {int(persist)}s despite ladder")
                behind_since = time.time()        # re-arm so the alert repeats
            time.sleep(TICK_S)
        except Exception as e:
            log(f"tick error: {type(e).__name__}: {e}")
            time.sleep(20)

main()
