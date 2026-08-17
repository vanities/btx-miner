# Copyright (c) 2026 The BTX developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or http://www.opensource.org/licenses/mit-license.php.
"""
Ring-Ajtai (SIS-based) commitment over the sketch coefficients: the
exact-binding commitment candidate (Greyhound-class) from README.md, in code.

Why this candidate: digest uniqueness (README) requires that a modified
commitment have NO producible opening. Hash-based PCS commit to redundant
encodings, so twiddled commitment strings still open as the same polynomial;
an Ajtai commitment t = A*g is a deterministic algebraic function of the
exact digit vector g, and producing any opening for a modified t is a SIS
collision. Uniqueness is structural, not statistical.

What is REAL here:
  - gadget decomposition of the F_q sketch coefficients into small digits,
  - the negacyclic NTT over Z_q'[X]/(X^d + 1) with Dilithium's parameters
    (d = 256, q' = 8380417, zeta = 1753), validated against schoolbook
    multiplication in the ring,
  - the seed-derived public matrix A (NTT domain, nothing-up-my-sleeve),
  - the commitment, its determinism, its sensitivity to any single
    coefficient change, and its Z-linearity over digit vectors (the
    property Greyhound-style evaluation proofs exploit),
  - the per-nonce operation-count analysis against today's digest hashing.

What is MODELED: the succinct evaluation opening (Greyhound / LaBRADOR
recursion) uses the same ideal-functionality pattern as IdealPC, with the
proof size taken from the literature (~50 KB class). Swapping the model for
the real recursion changes neither the commitment nor the uniqueness
argument, which live above it.

Parameters are ILLUSTRATIVE: chosen for clarity and comfortable SIS margins
(digit base 2^4 keeps norms tiny), not a production sizing. Production
needs a concrete SIS analysis (norm bound vs q' and module rank).
"""

from __future__ import annotations

import hashlib
import math

# ---------------------------------------------------------------------------
# Ring and commitment parameters.
# ---------------------------------------------------------------------------
D = 256                      # ring degree: R = Z_QP[X]/(X^D + 1)
QP = 8380417                 # Dilithium prime, 2^23 - 2^13 + 1; 512 | QP-1
ZETA = 1753                  # primitive 512th root of unity mod QP
GBITS = 4                    # digit base 2^GBITS
GDIGITS = 16                 # ceil(61 / GBITS) digits per F_q coefficient
NOUT = 4                     # module rank (rows of A) -> t is NOUT ring elems
COMMIT_BYTES = NOUT * D * 4  # 4 KiB serialized commitment
GREYHOUND_OPEN_BYTES = 50_000  # literature-class evaluation-proof size


def _brv8(x: int) -> int:
    return int("{:08b}".format(x)[::-1], 2)


_ZETA_TABLE: list[int] | None = None


def _zetas() -> list[int]:
    global _ZETA_TABLE
    if _ZETA_TABLE is None:
        _ZETA_TABLE = [pow(ZETA, _brv8(i), QP) for i in range(D)]
    return _ZETA_TABLE


# ---------------------------------------------------------------------------
# Negacyclic NTT (Dilithium layout): X^D + 1 splits fully mod QP, so ring
# multiplication is elementwise in the NTT domain.
# ---------------------------------------------------------------------------
def ntt(poly: list[int]) -> list[int]:
    a = [x % QP for x in poly]
    z = _zetas()
    k = 0
    length = D // 2
    while length >= 1:
        start = 0
        while start < D:
            k += 1
            zeta = z[k]
            for j in range(start, start + length):
                t = zeta * a[j + length] % QP
                a[j + length] = (a[j] - t) % QP
                a[j] = (a[j] + t) % QP
            start += 2 * length
        length //= 2
    return a


def intt(a_in: list[int]) -> list[int]:
    a = list(a_in)
    z = _zetas()
    k = D
    length = 1
    while length < D:
        start = 0
        while start < D:
            k -= 1
            zeta = QP - z[k]
            for j in range(start, start + length):
                t = a[j]
                a[j] = (t + a[j + length]) % QP
                a[j + length] = (t - a[j + length]) % QP
                a[j + length] = zeta * a[j + length] % QP
            start += 2 * length
        length *= 2
    f = pow(D, QP - 2, QP)
    return [x * f % QP for x in a]


def ring_mul_schoolbook(x: list[int], y: list[int]) -> list[int]:
    """Reference negacyclic multiplication (X^D = -1), for validating the NTT."""
    out = [0] * D
    for i in range(D):
        xi = x[i] % QP
        if xi == 0:
            continue
        for j in range(D):
            k = i + j
            v = xi * (y[j] % QP) % QP
            if k < D:
                out[k] = (out[k] + v) % QP
            else:
                out[k - D] = (out[k - D] - v) % QP
    return out


# ---------------------------------------------------------------------------
# Gadget decomposition: each F_q coefficient (61 bits) -> GDIGITS small
# digits in [0, 2^GBITS). Z-linear reconstruction; norms stay tiny for SIS.
# ---------------------------------------------------------------------------
def gadget_decompose(coeffs: list[int]) -> list[int]:
    mask = (1 << GBITS) - 1
    out = []
    for c in coeffs:
        for i in range(GDIGITS):
            out.append((c >> (GBITS * i)) & mask)
    return out


def gadget_recompose(digits: list[int]) -> list[int]:
    n = len(digits) // GDIGITS
    out = []
    for i in range(n):
        c = 0
        for j in range(GDIGITS):
            c |= digits[i * GDIGITS + j] << (GBITS * j)
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Seed-derived public matrix A (NTT domain). Uniform mod QP from a SHA XOF;
# the ~2^-40 sampling bias from 4-byte reduction is irrelevant at reference
# level (production uses rejection sampling as in Dilithium).
# ---------------------------------------------------------------------------
_A_CACHE: dict[tuple[bytes, int, int], list[int]] = {}


def _a_poly(seed: bytes, row: int, col: int) -> list[int]:
    key = (seed, row, col)
    got = _A_CACHE.get(key)
    if got is not None:
        return got
    out = []
    ctr = 0
    while len(out) < D:
        d = hashlib.sha256(seed + b"A" + row.to_bytes(2, "big")
                           + col.to_bytes(4, "big") + ctr.to_bytes(4, "big")).digest()
        for i in range(0, 32, 4):
            if len(out) >= D:
                break
            out.append(int.from_bytes(d[i:i + 4], "big") % QP)
        ctr += 1
    _A_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# The commitment: t = A * g over R_QP, g = NTT-packed digit chunks.
# ---------------------------------------------------------------------------
def commit_t(seed: bytes, digits: list[int]) -> list[list[int]]:
    """Raw commitment as NOUT ring elements (NTT domain)."""
    k_chunks = math.ceil(len(digits) / D)
    t = [[0] * D for _ in range(NOUT)]
    for j in range(k_chunks):
        chunk = digits[j * D:(j + 1) * D]
        if len(chunk) < D:
            chunk = chunk + [0] * (D - len(chunk))
        gh = ntt(chunk)
        for r in range(NOUT):
            arj = _a_poly(seed, r, j)
            tr = t[r]
            for i in range(D):
                tr[i] = (tr[i] + arj[i] * gh[i]) % QP
    return t


def serialize_t(t: list[list[int]]) -> bytes:
    return b"".join(x.to_bytes(4, "little") for row in t for x in row)


def ajtai_commit(seed: bytes, coeffs: list[int]) -> bytes:
    return serialize_t(commit_t(seed, gadget_decompose(coeffs)))


# ---------------------------------------------------------------------------
# Per-nonce cost analysis: commitment op-count vs today's digest hashing.
# Rough model, clearly labeled: NTT butterfly ~3 ops, pointwise MAC ~2 ops,
# SHA256 compression ~4000 ops over 64-byte blocks. The point is the CLASS.
# ---------------------------------------------------------------------------
def per_nonce_cost_analysis(m: int) -> dict:
    n_coeffs = m * m
    n_digits = n_coeffs * GDIGITS
    k_chunks = n_digits // D
    butterflies = k_chunks * (D // 2) * 8          # log2(256) stages
    pointwise = k_chunks * D * NOUT
    ntt_ops = butterflies * 3 + pointwise * 2
    sha_compressions = (n_coeffs * 8) // 64        # digest over 8*m^2 bytes
    sha_ops = sha_compressions * 4000
    return {
        "k_chunks": k_chunks,
        "ntt_ops": ntt_ops,
        "sha_ops": sha_ops,
        "ratio_vs_digest": ntt_ops / sha_ops,
    }


# ---------------------------------------------------------------------------
# LatticePC: same API as IdealPC. The COMMITMENT is real (above); the
# evaluation opening is modeled with the same ideal-functionality pattern
# and a literature-class proof size.
# ---------------------------------------------------------------------------
class LatticePC:
    _REGISTRY: dict[bytes, tuple[tuple[int, ...], int]] = {}
    _SEED = b"btx-succinct-lattice-A"

    def __init__(self):
        self._coeffs: list[int] = []
        self._m = 0
        self._commit = b""

    def commit(self, Chat: list[list[int]]) -> bytes:
        m = len(Chat)
        flat = tuple(x for row in Chat for x in row)
        self._coeffs, self._m = list(flat), m
        self._commit = ajtai_commit(self._SEED, list(flat))
        LatticePC._REGISTRY[self._commit] = (flat, m)
        return self._commit

    def open(self, a: int, b: int):
        from succinct_matmul_pow import Opening, _eval_flat
        val = _eval_flat(self._coeffs, self._m, a, b)
        token = hashlib.sha256(b"LPCopen" + self._commit
                               + a.to_bytes(8, "big") + b.to_bytes(8, "big")
                               + val.to_bytes(8, "big")).digest()
        return val, Opening(token=token, nbytes=GREYHOUND_OPEN_BYTES)

    @staticmethod
    def verify(commit: bytes, a: int, b: int, val: int, proof) -> bool:
        from succinct_matmul_pow import _eval_flat
        entry = LatticePC._REGISTRY.get(commit)
        if entry is None:
            return False
        coeffs, m = entry
        return val == _eval_flat(coeffs, m, a, b)


if __name__ == "__main__":
    print(f"=== ring-Ajtai commitment | R = Z_{QP}[X]/(X^{D}+1) | "
          f"digits base 2^{GBITS} x {GDIGITS} | rank {NOUT} ===")
    x = [(7 * i + 3) % QP for i in range(D)]
    y = [(11 * i + 5) % QP for i in range(D)]
    via_ntt = intt([a * b % QP for a, b in zip(ntt(x), ntt(y))])
    print(f" NTT vs schoolbook negacyclic mult: "
          f"{'MATCH' if via_ntt == ring_mul_schoolbook(x, y) else 'MISMATCH'}")
    coeffs = [(1 << 60) - 3 * i for i in range(64)]
    t1 = ajtai_commit(b"seed", coeffs)
    t2 = ajtai_commit(b"seed", coeffs)
    coeffs[0] ^= 1
    t3 = ajtai_commit(b"seed", coeffs)
    print(f" deterministic: {t1 == t2}   twiddle changes commitment: {t1 != t3}")
    print(f" commitment size: {len(t1)} bytes (header carries H(sigma||t), 32 B)")
    for m in (1024, 2048):
        c = per_nonce_cost_analysis(m)
        print(f" per-nonce op-count m={m}: NTT-commit {c['ntt_ops']:.3g} ops vs "
              f"digest-hash {c['sha_ops']:.3g} ops -> {c['ratio_vs_digest']:.2f}x")
    print(" => commit cost is in the same class as today's per-nonce digest")
    print("    hashing (and embarrassingly parallel: independent 256-pt NTTs).")
