# Copyright (c) 2026 The BTX developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or http://www.opensource.org/licenses/mit-license.php.
"""
Succinct matmul-PoW proof -- decouple the ON-CHAIN proof size from the sketch
dimension m, WITHOUT weakening the compute requirement.

The problem (BTX v4 / BMX4-C):
    The miner computes the m x m sketch  Chat = U * A * B * V  (mod q, q=2^61-1),
    where A,B are the n x n PoW operands and U,V are the m x n / n x m projectors.
    Today the raw sketch is put on-chain as the proof: 8 * m^2 bytes/block
    (8 MiB at m=1024 "profile C", 32 MiB at m=2048 "profile D"). The block payload
    and the projection dimension m are CONJOINED -- bigger m (better compute-scaling)
    means a bigger block, forever.

The decoupling (this file):
    Never put Chat on-chain. Instead:
      1. commit to Chat succinctly (a polynomial commitment -- ~32-48 bytes),
      2. attach a small proof that the Freivalds identity holds for the COMMITTED
         Chat at a Fiat-Shamir challenge point.
    The verifier computes its own cheap O(n^2) operand side and checks equality.
    On-chain permanent data = commit + one field element + a tiny opening proof,
    INDEPENDENT of m. The miner STILL computes the real Chat (so the compute
    requirement -- the whole point of v4 -- is untouched); it just proves it did.

Why it is sound (reduction):
    Chat is honest  <=>  for a random (a,b),  f(a,b) == u_a^T * U*A*B*V * v_b
    where f(X,Y) = sum_ij Chat_ij X^i Y^j, u_a=(a^i)_i, v_b=(b^j)_j. The right
    side is what the verifier recomputes from the seed (operands regenerated,
    applied to vectors right-to-left = O(n^2), never forming a matrix product).
    If the committed Chat is wrong, the bivariate  (Chat - U A B V)  is a nonzero
    polynomial of degree < m in each variable, so f(a,b) != verifier-side except
    for a <= 2(m-1)/q fraction of (a,b)  (Schwartz-Zippel). With q ~ 2^61 that is
    ~2^-49 per round at m=2048; repeat R rounds for (2m/q)^R, or draw (a,b) from an
    extension field F_{q^k} for headroom.

Scope of THIS reference:
    A faithful, tested model of the PROTOCOL: operand gen, the honest sketch, the
    reduction, Fiat-Shamir, prover, verifier, and the size decoupling. The
    polynomial commitment is a documented black box: `IdealPC` models a SOUND PCS
    as an ideal functionality (binding + correct evaluation enforced in-code) and
    reports proof SIZE via the real FRI/KZG formulas -- so the decoupling table is
    honest about production numbers. Swapping IdealPC for a real FRI opening
    (transparent, hash-only, no trusted setup -- recommended for a PoW chain) or
    KZG (constant-size, needs a trusted setup) yields a production prototype; the
    reduction and its soundness proof above are unchanged by that swap.

Run:  python3 succinct_matmul_pow.py         # self-check + decoupling table
Test: python3 test_succinct.py   (or: uv run pytest)
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Field F_q, q = 2^61 - 1 (the BMX4-C sketch modulus; a Mersenne prime).
# ---------------------------------------------------------------------------
Q = (1 << 61) - 1


def fadd(a: int, b: int) -> int:
    return (a + b) % Q


def fsub(a: int, b: int) -> int:
    return (a - b) % Q


def fmul(a: int, b: int) -> int:
    return (a * b) % Q


def fpow(a: int, e: int) -> int:
    return pow(a % Q, e, Q)


def _int_to_field(h: bytes) -> int:
    # Map hash bytes into F_q, avoiding 0 (a degenerate challenge).
    v = int.from_bytes(h, "big") % Q
    return v if v != 0 else 1


# ---------------------------------------------------------------------------
# Deterministic operand generation from a seed (models the SHA-XOF expansion of
# A, B, U, V). Entry magnitudes are irrelevant to the crypto -- we keep them small
# and reduce mod q, exactly as the real M11/E8M0 operands live in F_q.
# ---------------------------------------------------------------------------
def _xof(seed: bytes, tag: bytes, idx: int) -> int:
    h = hashlib.sha256(seed + tag + idx.to_bytes(8, "big")).digest()
    # small signed-ish entry in [0, 64), reduced mod q -- keeps the reference readable
    return int.from_bytes(h[:2], "big") % 64


def gen_matrix(seed: bytes, tag: bytes, rows: int, cols: int) -> list[list[int]]:
    return [[_xof(seed, tag, i * cols + j) for j in range(cols)] for i in range(rows)]


def regen_operands(seed: bytes, n: int, m: int):
    """The verifier regenerates A,B,U,V from the seed -- O(n^2) work, same as today."""
    A = gen_matrix(seed, b"A", n, n)
    B = gen_matrix(seed, b"B", n, n)
    U = gen_matrix(seed, b"U", m, n)
    V = gen_matrix(seed, b"V", n, m)
    return A, B, U, V


# ---------------------------------------------------------------------------
# Linear algebra. matmat is used ONLY by the honest miner (it is the O(n^3) PoW
# work). The verifier NEVER calls matmat -- only matvec / vecmat (O(n^2)).
# ---------------------------------------------------------------------------
def matmat(X: list[list[int]], Y: list[list[int]]) -> list[list[int]]:
    r, k, c = len(X), len(Y), len(Y[0])
    out = [[0] * c for _ in range(r)]
    for i in range(r):
        Xi = X[i]
        oi = out[i]
        for t in range(k):
            x = Xi[t]
            if x == 0:
                continue
            Yt = Y[t]
            for j in range(c):
                oi[j] = (oi[j] + x * Yt[j]) % Q
    return out


def matvec(M: list[list[int]], v: list[int]) -> list[int]:
    return [sum(M[i][k] * v[k] for k in range(len(v))) % Q for i in range(len(M))]


def vecmat(v: list[int], M: list[list[int]]) -> list[int]:
    cols = len(M[0])
    return [sum(v[k] * M[k][j] for k in range(len(v))) % Q for j in range(cols)]


def dot(u: list[int], v: list[int]) -> int:
    return sum(a * b for a, b in zip(u, v)) % Q


def compute_sketch(A, B, U, V) -> list[list[int]]:
    """The MINER's work: Chat = U * A * B * V  (m x m). O(n^3)-class."""
    return matmat(matmat(matmat(U, A), B), V)


# ---------------------------------------------------------------------------
# The reduction: bilinear form  ->  bivariate polynomial evaluation.
#   f(a,b) = sum_ij Chat_ij a^i b^j   (prover side; needs Chat)
#   s'     = u_a^T (U A B V) v_b       (verifier side; O(n^2), no Chat)
# with u_a=(a^i), v_b=(b^j). Honest  =>  f(a,b) == s'.
# ---------------------------------------------------------------------------
def _powers(x: int, k: int) -> list[int]:
    out = [1] * k
    for i in range(1, k):
        out[i] = out[i - 1] * x % Q
    return out


def eval_sketch_poly(Chat: list[list[int]], a: int, b: int) -> int:
    """f(a,b) = sum_ij Chat_ij a^i b^j. Prover-side, O(m^2)."""
    m = len(Chat)
    ap = _powers(a, m)
    bp = _powers(b, m)
    acc = 0
    for i in range(m):
        row = Chat[i]
        ai = ap[i]
        # sum_j row_j b^j, then * a^i
        rj = 0
        for j in range(m):
            rj = (rj + row[j] * bp[j]) % Q
        acc = (acc + ai * rj) % Q
    return acc


def verifier_side(A, B, U, V, a: int, b: int) -> int:
    """s' = u_a^T U A B V v_b, associated right-to-left as matVEC products => O(n^2).
    Never forms A*B or U*A -- that would be the O(n^3) miner work."""
    m = len(U)
    ap = _powers(a, m)  # u_a
    bp = _powers(b, m)  # v_b
    w = vecmat(ap, U)      # u_a^T U        -> length n
    z = matvec(V, bp)      # V v_b          -> length n
    Bz = matvec(B, z)      # B (V v_b)      -> length n
    ABz = matvec(A, Bz)    # A B (V v_b)    -> length n
    return dot(w, ABz)     # (u_a^T U) A B (V v_b) -> scalar


# ---------------------------------------------------------------------------
# Fiat-Shamir: derive the challenge (a,b) from the header AND the commitment, so
# the miner is bound to a specific Chat BEFORE learning (a,b) (FS soundness). In
# production the PoW target also binds the commitment (matmul_digest := commit).
# ---------------------------------------------------------------------------
def fiat_shamir_ab(header: bytes, commit: bytes) -> tuple[int, int]:
    a = _int_to_field(hashlib.sha256(b"FS-a" + header + commit).digest())
    b = _int_to_field(hashlib.sha256(b"FS-b" + header + commit).digest())
    return a, b


# ---------------------------------------------------------------------------
# Polynomial-commitment SIZE models (bytes). The reduction is PCS-agnostic; these
# are just how big the opening proof is per backend, so the decoupling table is
# honest. Numbers are standard, cited in README.md.
# ---------------------------------------------------------------------------
HASH = 32   # SHA-256 digest / Merkle node
FELT = 8    # one F_q element (61 bits -> 8 bytes)


def kzg_proof_bytes(num_coeffs: int) -> int:
    # BLS12-381: commitment (G1, 48) + opening (G1, 48) + eval (Fr, 32). Constant.
    # (Needs a trusted setup; listed for contrast.)
    return 48 + 48 + 32


def fri_proof_bytes(num_coeffs: int) -> int:
    # Transparent, hash-only, NO trusted setup. Blowup 8, ~34 queries for
    # ~100-bit soundness, fold down to a 256-coeff final polynomial sent in the
    # clear. Per query per layer: one sibling value (FELT) + a Merkle path of
    # log2(layer domain) hashes; sum over layers => O(log^2 N) total. This is an
    # honest UPPER estimate; real implementations dedupe shared path prefixes
    # across queries and typically land 3-10x smaller.
    if num_coeffs < 1:
        num_coeffs = 1
    domain_bits = max(1, (num_coeffs * 8 - 1).bit_length())  # blowup 8
    final_bits = 8                                            # 256-coeff final poly
    layers = max(1, domain_bits - final_bits)
    queries = 34
    per_query = sum((domain_bits - l) * HASH + FELT for l in range(layers))
    return queries * per_query + (1 << final_bits) * FELT + HASH


# ---------------------------------------------------------------------------
# IdealPC -- a MODEL of a sound polynomial commitment (ideal functionality).
#   commit(coeffs)      -> 32-byte binding commitment (registers coeffs internally)
#   open(a,b)           -> (f(a,b), opening-proof-of-size fri/kzg)
#   verify(commit,a,b,val,proof) -> bool   (accept iff val == f_committed(a,b))
# The internal registry is the ideal functionality's state -- it models the
# cryptographic binding of FRI/KZG. The VERIFIER never sees `coeffs`; its only
# inputs are (commit,a,b,val,proof) and the on-chain data is only (commit,val,proof).
# ---------------------------------------------------------------------------
@dataclass
class Opening:
    token: bytes
    nbytes: int


class IdealPC:
    _REGISTRY: dict[bytes, tuple[tuple[int, ...], int]] = {}

    def __init__(self, backend: str = "fri"):
        assert backend in ("fri", "kzg")
        self.backend = backend
        self._coeffs: list[int] = []
        self._m = 0
        self._commit = b""

    def commit(self, Chat: list[list[int]]) -> bytes:
        m = len(Chat)
        flat = tuple(x for row in Chat for x in row)
        self._coeffs, self._m = list(flat), m
        h = hashlib.sha256(b"PCcommit" + m.to_bytes(4, "big")
                           + b"".join(x.to_bytes(8, "big") for x in flat)).digest()
        self._commit = h
        IdealPC._REGISTRY[h] = (flat, m)   # ideal-functionality binding
        return h

    def open(self, a: int, b: int) -> tuple[int, Opening]:
        val = _eval_flat(self._coeffs, self._m, a, b)
        size = (fri_proof_bytes if self.backend == "fri" else kzg_proof_bytes)(
            len(self._coeffs))
        token = hashlib.sha256(b"PCopen" + self._commit
                               + a.to_bytes(8, "big") + b.to_bytes(8, "big")
                               + val.to_bytes(8, "big")).digest()
        return val, Opening(token=token, nbytes=size)

    @staticmethod
    def verify(commit: bytes, a: int, b: int, val: int, proof: Opening) -> bool:
        # Ideal functionality: a sound opening convinces the verifier that `val`
        # is the TRUE evaluation of the committed polynomial at (a,b) -- it cannot
        # be forged (binding) and cannot open to a wrong value (evaluation-binding).
        entry = IdealPC._REGISTRY.get(commit)
        if entry is None:
            return False
        coeffs, m = entry
        return val == _eval_flat(coeffs, m, a, b)


def _eval_flat(coeffs, m: int, a: int, b: int) -> int:
    ap = _powers(a, m)
    bp = _powers(b, m)
    acc = 0
    for i in range(m):
        base = i * m
        rj = 0
        for j in range(m):
            rj = (rj + coeffs[base + j] * bp[j]) % Q
        acc = (acc + ap[i] * rj) % Q
    return acc


# ---------------------------------------------------------------------------
# The proof carried on-chain: commitment + claimed value + opening. Size is
# INDEPENDENT of m (that is the whole point).
# ---------------------------------------------------------------------------
@dataclass
class SuccinctProof:
    commit: bytes      # replaces the raw m x m sketch (32 B)
    value: int         # f(a,b), one field element (8 B)
    opening: Opening   # PCS opening (KZG ~128 B / FRI ~KB), m-independent

    def onchain_bytes(self) -> int:
        return len(self.commit) + FELT + self.opening.nbytes


def _make_pc(backend: str):
    if backend == "lattice":
        from lattice_pc import LatticePC       # real ring-Ajtai commitment
        return LatticePC()
    return IdealPC(backend)                    # "fri" / "kzg" size models


def prove(header: bytes, seed: bytes, n: int, m: int, backend: str = "fri",
          _tamper=None) -> SuccinctProof:
    """Honest miner. `_tamper(Chat)` is a test hook to commit to a WRONG sketch."""
    A, B, U, V = regen_operands(seed, n, m)
    Chat = compute_sketch(A, B, U, V)          # the real O(n^3) work
    if _tamper is not None:
        Chat = _tamper(Chat)
    pc = _make_pc(backend)
    commit = pc.commit(Chat)
    a, b = fiat_shamir_ab(header, commit)      # bound AFTER committing
    value, opening = pc.open(a, b)
    return SuccinctProof(commit=commit, value=value, opening=opening)


def verify(header: bytes, seed: bytes, n: int, m: int, proof: SuccinctProof) -> bool:
    """Full node. Never receives Chat. O(n^2) operand work + one PCS opening check."""
    a, b = fiat_shamir_ab(header, proof.commit)
    ok = IdealPC.verify(proof.commit, a, b, proof.value, proof.opening)
    if not ok and len(proof.commit) != 32:     # lattice commitments are 4 KiB
        from lattice_pc import LatticePC
        ok = LatticePC.verify(proof.commit, a, b, proof.value, proof.opening)
    if not ok:
        return False                            # opening/binding failed
    A, B, U, V = regen_operands(seed, n, m)     # O(n^2)
    s_prime = verifier_side(A, B, U, V, a, b)   # O(n^2)
    return proof.value == s_prime               # the Freivalds identity


# ---------------------------------------------------------------------------
# Sizes: naive (raw sketch on-chain) vs succinct (this scheme).
# ---------------------------------------------------------------------------
def naive_onchain_bytes(m: int) -> int:
    return 8 * m * m   # the raw m x m F_q sketch, 8 bytes/element


def decoupling_table(dims=((64, 16), (256, 64), (1024, 256), (2048, 512),
                           (4096, 1024), (4096, 2048))):
    """(n, m) -> naive vs succinct on-chain bytes. m-independence is the headline."""
    rows = []
    for n, m in dims:
        naive = naive_onchain_bytes(m)
        fri = len(b"x" * 32) + FELT + fri_proof_bytes(m * m)
        kzg = 32 + FELT + kzg_proof_bytes(m * m)
        rows.append((n, m, naive, fri, kzg))
    return rows


def _fmt(nbytes: int) -> str:
    for unit, div in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if nbytes >= div:
            return f"{nbytes / div:.2f} {unit}"
    return f"{nbytes} B"


def _bump(Chat):
    C = [row[:] for row in Chat]
    C[0][0] = (C[0][0] + 1) % Q   # a single wrong entry must be caught
    return C


if __name__ == "__main__":
    print("=== succinct matmul-PoW: self-check (small params) ===")
    hdr, sd, n, m = b"block-header-#1", b"seed-abc", 24, 8
    p = prove(hdr, sd, n, m, backend="fri")
    ok = verify(hdr, sd, n, m, p)
    print(f"  honest proof  -> verify = {ok}   (on-chain {p.onchain_bytes()} B, "
          f"vs raw sketch {naive_onchain_bytes(m)} B)")
    assert ok

    bad = verify(hdr, sd, n, m, prove(hdr, sd, n, m, _tamper=lambda C: _bump(C)))
    print(f"  cheated sketch -> verify = {bad}   (must be False)")
    assert not bad

    print("\n=== DECOUPLING: on-chain proof bytes, naive vs succinct ===")
    print(f"  {'n':>5} {'m':>5} {'naive (raw sketch)':>20} "
          f"{'succinct FRI':>14} {'succinct KZG':>14}   {'FRI shrink':>10}")
    for n, m, naive, fri, kzg in decoupling_table():
        print(f"  {n:>5} {m:>5} {_fmt(naive):>20} {_fmt(fri):>14} "
              f"{_fmt(kzg):>14}   {naive / fri:>9.0f}x")
    print("\n  -> succinct proof size barely moves with m; the raw sketch is 8*m^2.")
    print("     Miner still computes the real Chat; only the ENCODING changed.")
    print("     (At toy m the FRI overhead dominates; the crossover is ~m=64.)")
