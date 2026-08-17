# Copyright (c) 2026 The BTX developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or http://www.opensource.org/licenses/mit-license.php.
#
# Succinct sketch-proof reference tests (MatMul v4 chain-size decoupling;
# contrib/matmul-v4/succinct-proof/README.md). This suite pins the protocol
# the reference implements:
#
#   (a) COMPLETENESS: an honest prover's (commit, value, opening) verifies,
#       across shapes and both PCS size backends.
#   (b) SOUNDNESS: a perturbed sketch (single entry) and a garbage sketch
#       both fail verification, though the verifier never sees the sketch.
#   (c) BINDING: the correct scalar cannot be claimed against a commitment
#       to a wrong sketch; an unregistered commitment is rejected; the
#       Fiat-Shamir challenge binds to the commitment.
#   (d) DETERMINISM: run-to-run byte-identity of commit, challenge, value.
#   (e) DECOUPLING: on-chain bytes are near-flat in m (the naive payload is
#       8*m^2); profile D shrinks >100x under the conservative FRI estimate.
#   (f) VERIFIER COST: the O(n^3) matmul path is never executed on verify
#       (enforced by making it explode).
#   (g) ADVERSARIAL COST: no cheating strategy beats honest mining. Partial
#       compute loses exponentially (Q sampled queries vs linear savings,
#       measured against a real Merkle sampler, not just the formula);
#       garbage-commit scanning and challenge grinding are priced out by
#       Schwartz-Zippel even at theoretical-limit speedups.
#   (h) LATTICE BACKEND: the exact-binding front-runner's commitment is REAL
#       (ring-Ajtai over Dilithium's ring): NTT validated against schoolbook
#       negacyclic multiplication, gadget roundtrip exact, commitment
#       deterministic and sensitive to any single coefficient change,
#       Z-linear over digit vectors, end-to-end prove/verify honest and
#       tampered, and the per-nonce commit op-count sits in the same class
#       as today's digest hashing.
#   + GOLDEN vectors: pinned commitment/challenge/value hex at fixed
#     headers, so any silent change to operand gen, Fiat-Shamir, or the
#     evaluation fails loudly.
#
# Run:  python3 test_succinct.py        (no dependencies)
#   or: pytest test_succinct.py

from __future__ import annotations

import succinct_matmul_pow as sp
from succinct_matmul_pow import (
    IdealPC, Q, SuccinctProof, compute_sketch, eval_sketch_poly,
    fiat_shamir_ab, naive_onchain_bytes, prove, regen_operands, verify,
    verifier_side,
)

HDR = b"blockheader-height-777"
SEED = b"seed-from-header-777"


def make_header(nonce: int) -> tuple[bytes, bytes]:
    """Fixed test header/seed pair, keyed by nonce (mirrors MakeV4Header)."""
    return b"golden-header-nonce-%d" % nonce, b"golden-seed-%d" % nonce


def _bump_one(C):
    C2 = [row[:] for row in C]
    C2[0][0] = (C2[0][0] + 1) % Q
    return C2


def _garbage(C):
    m = len(C)
    return [[(7 * i + 13 * j + 1) % Q for j in range(m)] for i in range(m)]


# ---------------------------------------------------------------------------
# (a) COMPLETENESS
# ---------------------------------------------------------------------------
def test_verifier_accepts_honest_proof():
    p = prove(HDR, SEED, n=24, m=8)
    assert verify(HDR, SEED, 24, 8, p)


def test_verifier_accepts_honest_proof_across_shapes():
    for n, m in ((16, 4), (32, 8), (48, 16), (40, 20)):
        p = prove(HDR, SEED, n=n, m=m)
        assert verify(HDR, SEED, n, m, p), f"honest proof failed at n={n} m={m}"


def test_verifier_accepts_honest_proof_kzg_backend():
    p = prove(HDR, SEED, n=24, m=8, backend="kzg")
    assert verify(HDR, SEED, 24, 8, p)


# ---------------------------------------------------------------------------
# (b) SOUNDNESS
# ---------------------------------------------------------------------------
def test_verifier_rejects_single_entry_perturbation():
    p = prove(HDR, SEED, n=24, m=8, _tamper=_bump_one)
    assert not verify(HDR, SEED, 24, 8, p)


def test_verifier_rejects_garbage_sketch():
    p = prove(HDR, SEED, n=24, m=8, _tamper=_garbage)
    assert not verify(HDR, SEED, 24, 8, p)


def test_verifier_rejects_perturbation_across_shapes():
    for n, m in ((16, 4), (32, 8), (48, 16)):
        p = prove(HDR, SEED, n=n, m=m, _tamper=_bump_one)
        assert not verify(HDR, SEED, n, m, p), f"tamper missed at n={n} m={m}"


def test_verifier_rejects_wrong_seed_or_header():
    p = prove(HDR, SEED, n=24, m=8)
    assert not verify(HDR, b"other-seed", 24, 8, p)
    assert not verify(b"other-header", SEED, 24, 8, p)


# ---------------------------------------------------------------------------
# (c) BINDING
# ---------------------------------------------------------------------------
def test_verifier_rejects_right_value_on_wrong_commitment():
    # Adversary commits to garbage but CLAIMS the correct verifier-side value.
    # PCS evaluation binding must reject: the commitment does not open to that
    # value at the challenge point.
    n, m = 24, 8
    A, B, U, V = regen_operands(SEED, n, m)
    pc = IdealPC("fri")
    commit = pc.commit(_garbage(compute_sketch(A, B, U, V)))
    a, b = fiat_shamir_ab(HDR, commit)
    honest_value = verifier_side(A, B, U, V, a, b)
    _, opening = pc.open(a, b)   # a REAL opening (of the garbage poly)
    forged = SuccinctProof(commit=commit, value=honest_value, opening=opening)
    assert not verify(HDR, SEED, n, m, forged)


def test_verifier_rejects_unknown_commitment():
    p = prove(HDR, SEED, n=24, m=8)
    fake = SuccinctProof(commit=b"\x00" * 32, value=p.value, opening=p.opening)
    assert not verify(HDR, SEED, 24, 8, fake)


def test_challenge_binds_to_commitment():
    # Fiat-Shamir: a different commitment must yield a different challenge,
    # so a prover cannot pick the challenge first and fit a sketch to it.
    p1 = prove(HDR, SEED, n=24, m=8)
    p2 = prove(HDR, SEED, n=24, m=8, _tamper=_bump_one)
    assert p1.commit != p2.commit
    assert fiat_shamir_ab(HDR, p1.commit) != fiat_shamir_ab(HDR, p2.commit)


def test_opening_value_matches_direct_poly_eval():
    # Internal consistency: the PCS opens to exactly f(a,b) of the committed poly.
    n, m = 24, 8
    A, B, U, V = regen_operands(SEED, n, m)
    Chat = compute_sketch(A, B, U, V)
    pc = IdealPC("fri")
    commit = pc.commit(Chat)
    a, b = fiat_shamir_ab(HDR, commit)
    val, _ = pc.open(a, b)
    assert val == eval_sketch_poly(Chat, a, b)


# ---------------------------------------------------------------------------
# (d) DETERMINISM
# ---------------------------------------------------------------------------
def test_proof_determinism_run_to_run():
    p1 = prove(HDR, SEED, n=32, m=8)
    p2 = prove(HDR, SEED, n=32, m=8)
    assert p1.commit == p2.commit
    assert p1.value == p2.value
    assert fiat_shamir_ab(HDR, p1.commit) == fiat_shamir_ab(HDR, p2.commit)


# ---------------------------------------------------------------------------
# (e) DECOUPLING
# ---------------------------------------------------------------------------
def test_pinned_constants():
    assert Q == (1 << 61) - 1                       # the sketch field
    assert naive_onchain_bytes(1024) == 8 << 20     # profile C: 8 MiB
    assert naive_onchain_bytes(2048) == 32 << 20    # profile D: 32 MiB
    assert sp.kzg_proof_bytes(2048 * 2048) == 128   # constant opening
    assert sp.FELT == 8 and sp.HASH == 32


def test_onchain_size_decoupling_at_profile_d():
    # Profile D: n=4096, m=2048 -> naive 32 MiB. Succinct must be >100x smaller
    # even under the conservative FRI upper estimate, and ~constant for KZG.
    m = 2048
    naive = naive_onchain_bytes(m)
    fri = 32 + sp.FELT + sp.fri_proof_bytes(m * m)
    kzg = 32 + sp.FELT + sp.kzg_proof_bytes(m * m)
    assert fri * 100 < naive, f"FRI {fri} not >100x under naive {naive}"
    assert kzg < 256, f"KZG proof should be constant ~168 B, got {kzg}"


def test_onchain_size_near_flat_in_m():
    # m doubling quadruples the naive payload; the FRI proof grows only ~log^2.
    fri_1024 = sp.fri_proof_bytes(1024 * 1024)
    fri_2048 = sp.fri_proof_bytes(2048 * 2048)
    assert naive_onchain_bytes(2048) == 4 * naive_onchain_bytes(1024)
    assert fri_2048 < 1.6 * fri_1024, "FRI growth should be ~log^2, not 4x"
    # KZG literally does not move.
    assert sp.kzg_proof_bytes(1024 * 1024) == sp.kzg_proof_bytes(2048 * 2048)


def test_proof_object_reports_onchain_bytes():
    p = prove(HDR, SEED, n=24, m=8)
    assert p.onchain_bytes() == 32 + sp.FELT + p.opening.nbytes


# ---------------------------------------------------------------------------
# (f) VERIFIER COST
# ---------------------------------------------------------------------------
def test_verifier_never_runs_cubic_path():
    p = prove(HDR, SEED, n=24, m=8)      # prover NEEDS matmat (that is the PoW)
    real = sp.matmat

    def boom(*_a, **_k):
        raise AssertionError("verifier executed the O(n^3) matmul path")

    sp.matmat = boom
    try:
        assert verify(HDR, SEED, 24, 8, p)   # must still verify, O(n^2) only
    finally:
        sp.matmat = real


# ---------------------------------------------------------------------------
# Soundness margin arithmetic (Schwartz-Zippel)
# ---------------------------------------------------------------------------
def test_soundness_margin_schwartz_zippel():
    # One round at m=2048: cheat survives with prob <= 2(m-1)/q ~ 2^-49.
    m = 2048
    per_round = 2 * (m - 1) / Q
    assert per_round < 2 ** -49
    # Two rounds (independent challenges) or one extension-field round clears
    # any realistic PoW grinding budget: 2^-98 vs a hash-of-sketch cost per try.
    assert per_round ** 2 < 2 ** -98


# ---------------------------------------------------------------------------
# (g) ADVERSARIAL COST
# ---------------------------------------------------------------------------
def test_partial_compute_never_profitable():
    import adversarial_cost as ac
    w = ac.work_shares(4096, 2048)   # profile D
    for save_share in (w["combine_share"], w["proj_p_share"]):
        prev = 1.0
        for eps in (0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75):
            ratio = ac.partial_compute_cost_ratio(eps, save_share)
            assert ratio > 1.0, f"profitable skip at eps={eps} share={save_share}"
            assert ratio > prev, "cost must worsen monotonically with eps"
            prev = ratio
    # marginal condition at eps->0: d/deps = Q - save_share > 0 for any share <= 1
    assert ac.Q_QUERIES > 1.0


def test_partial_compute_acceptance_matches_sampling():
    # The (1-e)^Q model is verified against an actual Merkle commitment with
    # root-derived query sampling (deterministic trials), not assumed.
    import adversarial_cost as ac
    for eps in (0.05, 0.15):
        measured, predicted = ac.empirical_acceptance(eps, domain=1024, trials=1200)
        assert abs(measured - predicted) < 0.03, \
            f"eps={eps}: measured {measured:.4f} vs predicted {predicted:.4f}"


def test_garbage_commit_unprofitable_at_limit_speedup():
    import adversarial_cost as ac
    # Even granting the theoretical-limit nonce-throughput speedup (2^30) from
    # skipping the matmul entirely, one SZ round leaves the cheater >30,000x
    # less efficient than honest; a second round is astronomical.
    assert ac.garbage_commit_efficiency(2 ** 30, 2048, rounds=1) < 2 ** -15
    assert ac.garbage_commit_efficiency(2 ** 30, 2048, rounds=2) < 2 ** -60


def test_challenge_grinding_unprofitable():
    import adversarial_cost as ac
    # Each grind costs at least the O(n^2) scalar; expected grinds 1/sz.
    assert ac.grinding_cost_ratio(4096, 2048) > 2 ** 30


# ---------------------------------------------------------------------------
# (h) LATTICE BACKEND
# ---------------------------------------------------------------------------
def test_lattice_ntt_matches_schoolbook_ring_mult():
    # Proves the transform IS the negacyclic ring isomorphism (zeta table and
    # both directions), not merely self-consistent.
    import lattice_pc as lp
    x = [(13 * i + 7) % lp.QP for i in range(lp.D)]
    y = [(5 * i * i + 3) % lp.QP for i in range(lp.D)]
    via_ntt = lp.intt([a * b % lp.QP for a, b in zip(lp.ntt(x), lp.ntt(y))])
    assert via_ntt == lp.ring_mul_schoolbook(x, y)
    assert lp.intt(lp.ntt(x)) == [v % lp.QP for v in x]


def test_lattice_gadget_roundtrip_exact():
    import lattice_pc as lp
    coeffs = [(1 << 61) - 2, 0, 1, (1 << 60) + 12345, Q - 1, 987654321]
    digits = lp.gadget_decompose(coeffs)
    assert all(0 <= d < (1 << lp.GBITS) for d in digits)   # SIS-small
    assert lp.gadget_recompose(digits) == coeffs


def test_lattice_commit_deterministic_and_twiddle_sensitive():
    import lattice_pc as lp
    coeffs = [(3 * i + 1) % Q for i in range(128)]
    t1 = lp.ajtai_commit(b"seed", coeffs)
    assert lp.ajtai_commit(b"seed", coeffs) == t1          # deterministic
    for pos in (0, 63, 127):
        mod = list(coeffs)
        mod[pos] = (mod[pos] + 1) % Q
        assert lp.ajtai_commit(b"seed", mod) != t1, \
            f"single-coefficient change at {pos} must change the commitment"


def test_lattice_commit_linear_in_digit_vectors():
    # The Z-linearity Greyhound-style evaluation proofs exploit:
    # A*(g1 + g2) == A*g1 + A*g2 over the ring.
    import lattice_pc as lp
    g1 = [(7 * i) % 16 for i in range(1024)]
    g2 = [(3 * i + 1) % 16 for i in range(1024)]
    g_sum = [a + b for a, b in zip(g1, g2)]
    t1 = lp.commit_t(b"seed", g1)
    t2 = lp.commit_t(b"seed", g2)
    ts = lp.commit_t(b"seed", g_sum)
    for r in range(lp.NOUT):
        for i in range(lp.D):
            assert ts[r][i] == (t1[r][i] + t2[r][i]) % lp.QP


def test_lattice_backend_end_to_end():
    p = prove(HDR, SEED, n=24, m=8, backend="lattice")
    assert len(p.commit) == 4096                            # real 4 KiB Ajtai t
    assert verify(HDR, SEED, 24, 8, p)
    bad = prove(HDR, SEED, n=24, m=8, backend="lattice", _tamper=_bump_one)
    assert not verify(HDR, SEED, 24, 8, bad)


def test_lattice_per_nonce_commit_in_digest_class():
    import lattice_pc as lp
    for m in (1024, 2048):
        c = lp.per_nonce_cost_analysis(m)
        assert c["ratio_vs_digest"] < 2.0, \
            f"m={m}: NTT commit {c['ratio_vs_digest']:.2f}x digest hashing"


# ---------------------------------------------------------------------------
# GOLDEN vectors: pinned at fixed headers. Any change to operand generation,
# the commitment serialization, Fiat-Shamir, or the evaluation must trip these.
# ---------------------------------------------------------------------------
GOLDEN = [
    # (nonce, n, m, commit_hex, challenge_a, challenge_b, value)
    (1, 32, 8,
     "504775270bb056f75315e121f1d4728d77bd4cf56c0926d0c8c754802ac467f5",
     1549702366784100016, 678731951599216324, 521372265953688527),
    (2, 48, 16,
     "9e4c29602e00fd8789c2f9078e5e92a350525cc54327d2b376c2b3c949b5ea4b",
     904962308943062721, 392921846655836019, 1884555757301130349),
]


def test_golden_proof_vectors():
    for nonce, n, m, commit_hex, ga, gb, gval in GOLDEN:
        hdr, seed = make_header(nonce)
        p = prove(hdr, seed, n=n, m=m)
        assert p.commit.hex() == commit_hex, f"commit drift at nonce={nonce}"
        a, b = fiat_shamir_ab(hdr, p.commit)
        assert (a, b) == (ga, gb), f"challenge drift at nonce={nonce}"
        assert p.value == gval, f"value drift at nonce={nonce}"
        assert verify(hdr, seed, n, m, p)


def _main():
    import sys
    fails = 0
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    _main()
