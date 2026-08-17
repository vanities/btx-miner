#!/usr/bin/env python3
# dexbtx_poolcore_bridge.py -- adapter between the dexbtx pool test endpoint (newline-JSON
# over TCP, schema v1: job / set_target / submit / result) and matador-miner --poolcore
# (poolcore-v0 JSON-lines on stdio). Written for the 2026-08-04 dexbtx integration test
# (tcp://165.22.247.143:3355); the mapping is the general dexbtx<->poolcore contract:
#
#   dexbtx job (182-byte header hex, spec s2)  ->  poolcore enc-rc-v47 job:
#       header_hex  = first 76 bytes of the 182-byte template + 4 zero bytes (classic
#                     80-byte layout; poolcore derives nonce64/dim/seeds itself)
#       target_hex  = share_target verbatim (both sides: 256-bit big-endian display hex)
#       height / parent_mtp / matmul_n = passthrough
#   dexbtx set_target  ->  replacement poolcore job (same header, new target; a new job
#       always supersedes -- poolcore preempts in-flight work)
#   poolcore solution  ->  dexbtx submit:
#       nonce64       = int(nonce_hex, 16)                     (decimal on their wire)
#       matmul_digest = byte-REVERSED digest_hex               (poolcore emits GetHex
#                       display order; dexbtx wire wants raw sha256 order, spec s4.1)
#       seed_a/seed_b = derived here (single-SHA256 V3 preimage, wire order) -- poolcore
#                       solutions do not carry seeds; the pool wants them for fast-check
#
# BYTE-ORDER RULE (spec s4.1): every 32-byte value on the dexbtx wire is raw sha256 /
# header-serialization order EXCEPT share_target, which is big-endian display -- the same
# convention poolcore's target_hex already uses, so targets pass through untouched.
#
# Usage:
#   python3 dexbtx_poolcore_bridge.py --endpoint 165.22.247.143:3355 \
#       --miner ~/.local/bin/matador-miner [--max-seconds 900] [--max-accepted 15]
#
# The miner runs with the v0.9.2 default RC latch (no env needed); this script is also a
# live demonstration that an enc-rc job dispatches out of the box.

import argparse, hashlib, json, socket, struct, subprocess, sys, threading, time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [bridge] {msg}", file=sys.stderr, flush=True)

def seed_v3(prev_raw, merkle_raw, version, ntime, nbits, nonce64, dim, height, parent_mtp, which):
    tag = b"BTX_MATMUL_SEED_V3"
    pre = (bytes([len(tag)]) + tag + prev_raw + struct.pack('<q', parent_mtp)
           + struct.pack('<I', height) + struct.pack('<i', version) + merkle_raw
           + struct.pack('<I', ntime) + struct.pack('<I', nbits)
           + struct.pack('<Q', nonce64) + struct.pack('<H', dim) + bytes([which]))
    return hashlib.sha256(pre).digest()

class Bridge:
    def __init__(self, host, port, miner_path):
        self.sock = socket.create_connection((host, port), timeout=30)
        self.sock.settimeout(None)   # 30s applies to CONNECT only; job stream is quiet for long stretches
        self.sock_file = self.sock.makefile('rw', encoding='utf-8', newline='\n')
        self.core = subprocess.Popen([miner_path, '--poolcore'], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1)
        self.job = None            # parsed current pool job (dict + header fields)
        self.core_jid = 0          # poolcore numeric job id
        self.jid_to_pool = {}      # core_jid -> pool job_id string
        self.accepted = 0
        self.rejected = 0
        self.submitted = 0
        self.lock = threading.Lock()
        self.done = threading.Event()

    def to_core(self, obj):
        line = json.dumps(obj, separators=(',', ':'))
        self.core.stdin.write(line + '\n')
        self.core.stdin.flush()

    def to_pool(self, obj):
        self.sock_file.write(json.dumps(obj, separators=(',', ':')) + '\n')
        self.sock_file.flush()

    def parse_header(self, header_hex):
        # accepts BOTH forms: the original 182-byte wire template (spec s2) and the
        # 2026-08-04 "poolcore-native" 80-byte classic template
        raw = bytes.fromhex(header_hex)
        if len(raw) not in (80, 182):
            raise ValueError(f"header is {len(raw)} bytes, want 80 or 182")
        return {
            'version':  struct.unpack_from('<i', raw, 0)[0],
            'prev':     raw[4:36],
            'merkle':   raw[36:68],
            'ntime':    struct.unpack_from('<I', raw, 68)[0],
            'nbits':    struct.unpack_from('<I', raw, 72)[0],
            'template80': raw[:76].hex() + '00000000',
        }

    def start_job(self, pool_job, target_hex):
        h = self.parse_header(pool_job.get('header') or pool_job['template'])
        self.core_jid += 1
        self.jid_to_pool[self.core_jid] = pool_job['job_id']
        self.job = {**pool_job, **h, 'target': target_hex}
        t0 = time.monotonic()
        self.to_core({'type': 'job', 'job_id': self.core_jid, 'profile': 'enc-rc-v47',
                      'header_hex': h['template80'], 'target_hex': target_hex,
                      'height': pool_job['height'], 'parent_mtp': pool_job['parent_mtp'],
                      'matmul_n': pool_job.get('matmul_dim') or pool_job.get('matmul_n') or 0})
        log(f"job -> core: pool_id={pool_job['job_id']} core_jid={self.core_jid} "
            f"height={pool_job['height']} dim={pool_job.get('matmul_dim')} "
            f"target={target_hex[:16]}.. in {1000*(time.monotonic()-t0):.1f}ms")

    def pool_reader(self):
        try:
            for line in self.sock_file:
                line = line.strip()
                if not line:
                    continue
                m = json.loads(line)
                t = m.get('type')
                if t == 'job':
                    # native form: 'template' + 'target_hex', submits in display order;
                    # wire form: 'header' (182B) + 'share_target', submits in raw order
                    self.native = 'template' in m
                    tgt = m.get('target_hex') or m['share_target']
                    log(f"pool job {m.get('job_id')} clean={m.get('clean')} "
                        f"native={self.native} target={tgt[:16]}..")
                    self.start_job(m, tgt)
                elif t == 'set_target':
                    tgt = m.get('target_hex') or m['share_target']
                    log(f"pool set_target {tgt[:16]}.. (vardiff) -> re-job")
                    if self.job:
                        self.start_job(self.job, tgt)
                elif t == 'result':
                    with self.lock:
                        if m.get('status') == 'accepted':
                            self.accepted += 1
                        else:
                            self.rejected += 1
                        a, r = self.accepted, self.rejected
                    log(f"result nonce64={m.get('nonce64')} status={m.get('status')} "
                        f"reason={m.get('reason','-')} is_block={m.get('is_block')} "
                        f"[acc={a} rej={r}]")
                else:
                    log(f"pool msg: {line[:160]}")
        except Exception as e:
            log(f"pool_reader ended: {e}")
        finally:
            self.done.set()

    def core_reader(self):
        try:
            for line in self.core.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                except ValueError:
                    # poolcore stdout also carries the version banner + human log lines;
                    # only JSON objects are protocol traffic
                    continue
                t = m.get('type')
                if t == 'solution':
                    jid = m.get('job_id')
                    pool_id = self.jid_to_pool.get(jid)
                    if pool_id is None or jid != self.core_jid:
                        log(f"stale solution for core_jid={jid} (current {self.core_jid}) -- dropped")
                        continue
                    j = self.job
                    nonce = int(m['nonce_hex'], 16)
                    dim = j.get('matmul_dim') or j.get('matmul_n') or 4096
                    sa = seed_v3(j['prev'], j['merkle'], j['version'], j['ntime'], j['nbits'],
                                 nonce, dim, j['height'], j['parent_mtp'], 0)
                    sb = seed_v3(j['prev'], j['merkle'], j['version'], j['ntime'], j['nbits'],
                                 nonce, dim, j['height'], j['parent_mtp'], 1)
                    if getattr(self, 'native', False):
                        # display/GetHex order end-to-end: digest passes through untouched,
                        # python's raw seed bytes get reversed
                        digest_out, sa_out, sb_out = m['digest_hex'], sa[::-1].hex(), sb[::-1].hex()
                    else:
                        digest_out, sa_out, sb_out = (bytes.fromhex(m['digest_hex'])[::-1].hex(),
                                                      sa.hex(), sb.hex())
                    self.to_pool({'v': 1, 'type': 'submit', 'job_id': pool_id, 'nonce64': nonce,
                                  'matmul_digest': digest_out,
                                  'seed_a': sa_out, 'seed_b': sb_out})
                    with self.lock:
                        self.submitted += 1
                        n = self.submitted
                    log(f"submit #{n}: nonce64={nonce} kind={m.get('kind')} "
                        f"digest={digest_out[:16]}.. order={'display' if getattr(self, 'native', False) else 'wire'}")
                elif t in ('hello', 'job_ack'):
                    log(f"core: {line[:200]}")
                elif t == 'error':
                    log(f"core ERROR: {line[:200]}")
        except Exception as e:
            log(f"core_reader ended: {e}")
        finally:
            self.done.set()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoint', default='165.22.247.143:3355')
    ap.add_argument('--miner', required=True)
    ap.add_argument('--max-seconds', type=int, default=900)
    ap.add_argument('--max-accepted', type=int, default=15)
    args = ap.parse_args()

    host, port = args.endpoint.rsplit(':', 1)
    t0 = time.monotonic()
    log(f"connecting {host}:{port}, miner={args.miner}")
    b = Bridge(host, int(port), args.miner)
    b.to_core({'type': 'init'})
    threading.Thread(target=b.pool_reader, daemon=True).start()
    threading.Thread(target=b.core_reader, daemon=True).start()

    while not b.done.is_set():
        if time.monotonic() - t0 > args.max_seconds:
            log("max-seconds reached")
            break
        with b.lock:
            if b.accepted >= args.max_accepted:
                log("max-accepted reached")
                break
        time.sleep(1)

    try:
        b.to_core({'type': 'shutdown'})
        b.core.wait(timeout=10)
    except Exception:
        b.core.kill()
    elapsed = time.monotonic() - t0
    log(f"DONE in {elapsed:.0f}s: submitted={b.submitted} accepted={b.accepted} "
        f"rejected={b.rejected}")
    return 0 if b.accepted > 0 else 1

if __name__ == '__main__':
    sys.exit(main())
