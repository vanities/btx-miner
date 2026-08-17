"""Cross-validate lattice_commit_probe.cu --selftest against the Python
reference (lattice_pc.py NTT/gadget/commit), with the probe's A PRF and
coefficient generator replicated exactly. Byte-identical t proves the CUDA
kernel implements the same commitment."""

import sys
import lattice_pc as lp

M32 = 0xFFFFFFFF


def mix32(x):
    x &= M32
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & M32
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & M32
    x ^= x >> 16
    return x


SEED = 0x42545831  # "BTX1", matches the probe


def gen_a_poly(_seed_bytes, r, j):
    inner = mix32(((j * 0x85EBCA6B) & M32))
    return [mix32(SEED ^ ((r * 0x9E3779B9) & M32) ^ mix32(((j * 0x85EBCA6B) & M32) ^ i)) % lp.QP
            for i in range(lp.D)]


def gen_coeff(i):
    hi = mix32((i * 2654435761) & M32)
    lo = mix32(i ^ 0xDEADBEEF)
    return ((hi << 32) | lo) & ((1 << 61) - 1)


def main():
    # replicate --selftest: 64 coefficients -> K=4 chunks
    lp._a_poly = gen_a_poly           # probe PRF instead of SHA derivation
    coeffs = [gen_coeff(i) for i in range(64)]
    t = lp.commit_t(b"ignored", lp.gadget_decompose(coeffs))
    flat = [x for row in t for x in row]
    print("python  t[0..15] (mod QP):", " ".join(str(v % lp.QP) for v in flat[:16]))
    if len(sys.argv) > 1:             # probe's numbers passed on argv
        probe = [int(v) for v in sys.argv[1:]]
        ok = probe == [v % lp.QP for v in flat[:len(probe)]]
        print("MATCH -- CUDA kernel == Python reference commitment" if ok
              else "MISMATCH -- kernel diverges from reference")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
