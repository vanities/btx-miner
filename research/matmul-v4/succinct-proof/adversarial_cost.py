# Copyright (c) 2026 The BTX developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or http://www.opensource.org/licenses/mit-license.php.
"""
Adversarial cost model for the succinct sketch proof: can a miner pass
verification while doing LESS compute than the honest matmul? Every known
strategy is priced here, and the tests assert none beats honest mining.

Attack surface (see README.md for the protocol):
  A. PARTIAL COMPUTE  - compute most of the sketch honestly, garbage-fill a
     fraction eps of entries, commit, hope the ~Q sampled openings miss the
     garbage. Acceptance ~ (1-eps)^Q; the work saved is linear in eps while
     the acceptance loss is exponential, so the expected cost per accepted
     block RISES for every eps > 0 (d/deps at 0: Q - save_share > 0).
  B. SKIPPED PROJECTION ROWS - same, but skip rows of P = U*A (bigger saving
     per corrupted entry: each skipped row corrupts a whole sketch row).
     Same exponential loss; still never profitable.
  C. GARBAGE COMMIT + SCAN - commit to a cheap/fake polynomial per nonce and
     scan at hash speed instead of matmul speed. Each hash-winning candidate
     survives verification only with Schwartz-Zippel probability
     sz = 2(m-1)/q per round. The nonce-throughput speedup is bounded by
     (matmul ops)/(header-hash ops) ~ 2^30 at the theoretical limit; net
     efficiency = speedup * sz^rounds << 1.
  D. CHALLENGE GRINDING - re-randomize the commitment and test whether the
     new Fiat-Shamir point collides. Each test costs >= the O(n^2)
     verifier-side scalar; expected grinds = 1/sz. Expected work per success
     dwarfs one honest block.

The acceptance model for A/B is also MEASURED, not just derived: a toy
Merkle commitment with root-derived query sampling runs thousands of
deterministic trials and the observed acceptance matches (1-eps)^Q.

Run:  python3 adversarial_cost.py     # prints the cost table
"""

from __future__ import annotations

import hashlib

Q_QUERIES = 34            # FRI-style sampled openings (~100-bit proximity)
FIELD_Q = (1 << 61) - 1


# ---------------------------------------------------------------------------
# Work accounting (MAC counts for the honest ENC-BMX4C pipeline).
# ---------------------------------------------------------------------------
def work_shares(n: int, m: int) -> dict:
    proj_p = m * n * n          # P = U * A      (m x n)
    proj_q = m * n * n          # Q = B * V      (n x m)
    combine = m * m * n         # Chat = P * Q   (m x m)
    total = proj_p + proj_q + combine
    return {
        "proj_p": proj_p, "proj_q": proj_q, "combine": combine, "total": total,
        "combine_share": combine / total, "proj_p_share": proj_p / total,
    }


def sz_bound(m: int) -> float:
    """Schwartz-Zippel: wrong committed polynomial agrees at a random point
    with probability <= 2(m-1)/q (total degree < 2m-1 over F_q)."""
    return 2.0 * (m - 1) / FIELD_Q


# ---------------------------------------------------------------------------
# A/B: partial compute. Expected cost per ACCEPTED block, relative to honest.
# ---------------------------------------------------------------------------
def partial_compute_cost_ratio(eps: float, save_share: float,
                               queries: int = Q_QUERIES) -> float:
    """(work per attempt) / (acceptance probability), normalized so honest = 1.
    save_share = fraction of total work the skipped entries actually save
    (combine-only: ~0.2 at m=n/2; P-rows: ~0.4)."""
    if eps <= 0.0:
        return 1.0
    work = 1.0 - save_share * eps
    accept = (1.0 - eps) ** queries
    return work / accept


# ---------------------------------------------------------------------------
# C: garbage commit + hash-speed scan.
# ---------------------------------------------------------------------------
def garbage_commit_efficiency(speedup: float, m: int, rounds: int = 1) -> float:
    """Cheater efficiency relative to honest mining: (nonce-throughput
    multiplier from skipping the matmul) x (probability an eligible candidate
    survives verification). > 1 would mean cheating pays; real values are
    astronomically below 1."""
    return speedup * (sz_bound(m) ** rounds)


# ---------------------------------------------------------------------------
# D: challenge grinding.
# ---------------------------------------------------------------------------
def grinding_cost_ratio(n: int, m: int) -> float:
    """Expected work to land one accepted cheat via commitment re-randomization,
    relative to one honest block. Each grind must evaluate the true scalar
    (>= n^2 MACs) to know whether the collision landed."""
    w = work_shares(n, m)
    per_try = (n * n) / w["total"]
    return per_try / sz_bound(m)


# ---------------------------------------------------------------------------
# Empirical acceptance: toy Merkle commitment + root-derived query sampling.
# Deterministic (hash-derived trials), no RNG.
# ---------------------------------------------------------------------------
def _h(*parts: bytes) -> bytes:
    hh = hashlib.sha256()
    for p in parts:
        hh.update(p)
    return hh.digest()


def merkle_root(leaves: list[bytes]) -> bytes:
    level = [_h(b"leaf", x) for x in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [_h(b"node", level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def derive_queries(root: bytes, salt: int, count: int, domain: int) -> list[int]:
    out = []
    ctr = 0
    while len(out) < count:
        d = _h(b"query", root, salt.to_bytes(8, "big"), ctr.to_bytes(4, "big"))
        for i in range(0, 32, 4):
            if len(out) >= count:
                break
            out.append(int.from_bytes(d[i:i + 4], "big") % domain)
        ctr += 1
    return out


def empirical_acceptance(eps: float, domain: int, trials: int,
                         queries: int = Q_QUERIES) -> tuple[float, float]:
    """Measure how often `queries` root-derived samples all miss a garbage set
    covering ~eps of the domain. Returns (measured, predicted (1-e)^Q with the
    EXACT realized garbage fraction)."""
    garbage = {i for i in range(domain)
               if int.from_bytes(_h(b"garb", i.to_bytes(4, "big"))[:4], "big")
               % 10_000 < int(eps * 10_000)}
    eps_actual = len(garbage) / domain
    # One committed vector; per-trial salt models the per-block challenge.
    leaves = [b"honest-%d" % i if i not in garbage else b"garbage-%d" % i
              for i in range(domain)]
    root = merkle_root(leaves)
    accepted = 0
    for t in range(trials):
        idxs = derive_queries(root, t, queries, domain)
        if all(i not in garbage for i in idxs):
            accepted += 1
    return accepted / trials, (1.0 - eps_actual) ** queries


# ---------------------------------------------------------------------------
# The table.
# ---------------------------------------------------------------------------
def cost_table(n: int = 4096, m: int = 2048):
    w = work_shares(n, m)
    rows = []
    for eps in (0.01, 0.05, 0.10, 0.25, 0.50):
        rows.append(("A skip combine entries", f"eps={eps:.2f}",
                     partial_compute_cost_ratio(eps, w["combine_share"])))
    for eps in (0.01, 0.05, 0.10, 0.25):
        # skipping eps of P's rows saves eps * proj_p_share of total work and
        # corrupts the same eps fraction of sketch rows (hence entries)
        rows.append(("B skip P rows", f"eps={eps:.2f}",
                     partial_compute_cost_ratio(eps, w["proj_p_share"])))
    return rows


if __name__ == "__main__":
    n, m = 4096, 2048   # profile D
    w = work_shares(n, m)
    print(f"=== adversarial cost table | n={n} m={m} (profile D) | Q={Q_QUERIES} queries ===")
    print(f" honest work: proj 2x{w['proj_p']:.3g} + combine {w['combine']:.3g} "
          f"= {w['total']:.3g} MACs (combine share {w['combine_share']:.2f})")
    print(f" Schwartz-Zippel per round: {sz_bound(m):.3e} (~2^{__import__('math').log2(sz_bound(m)):.1f})")
    print("\n expected cost per ACCEPTED block, relative to honest = 1.0:")
    for name, param, ratio in cost_table(n, m):
        print(f"   {name:26s} {param:10s} -> {ratio:12.2f}x")
    eff1 = garbage_commit_efficiency(2 ** 30, m, rounds=1)
    eff2 = garbage_commit_efficiency(2 ** 30, m, rounds=2)
    print(f"   C garbage commit + scan   speedup 2^30, 1 round -> efficiency {eff1:.3e} (loses)")
    print(f"   C garbage commit + scan   speedup 2^30, 2 rounds -> efficiency {eff2:.3e}")
    print(f"   D challenge grinding      -> {grinding_cost_ratio(n, m):.3e}x honest")
    print("\n empirical acceptance vs (1-e)^Q (toy Merkle, root-derived queries, 1500 trials):")
    for eps in (0.05, 0.10, 0.20):
        meas, pred = empirical_acceptance(eps, domain=1024, trials=1500)
        print(f"   eps={eps:.2f}: measured {meas:.4f}  predicted {pred:.4f}")
    print("\n => every strategy costs MORE per accepted block than honest mining;")
    print("    partial compute loses exponentially (Q=34 queries vs linear savings).")
