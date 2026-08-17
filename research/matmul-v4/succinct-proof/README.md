# Succinct sketch proof: decoupling the v4 proof size from m

Reference implementation + tests for replacing the raw on-chain sketch payload
with a constant-or-near-constant-size proof. Target problem: in the current v4
design the block carries the full m x m sketch (8 * m^2 bytes: 8 MiB at profile
C, 32 MiB at profile D), so the proof size and the projection dimension m are
conjoined. This scheme cuts that link: m can grow (better compute-scaling) while
the on-chain proof stays small and (for KZG) literally constant.

## The idea in one paragraph

The verifier never needs the sketch matrix itself. It needs to be convinced that
the committed sketch equals U*A*B*V. Freivalds already reduces that to comparing
two scalars. So: commit to Chat as the coefficients of a bivariate polynomial
f(X,Y) = sum_ij Chat_ij X^i Y^j (32-byte commitment), derive a challenge point
(a,b) by Fiat-Shamir from the header and the commitment, and have the miner
prove the single evaluation f(a,b) with a polynomial-commitment opening. The
verifier recomputes the same scalar from the seed on its own:
u_a^T (U (A (B (V v_b)))) with u_a = (a^i), v_b = (b^j), pure matrix-vector
work, O(n^2), never forming any matrix product. Equality at a random point
implies (whp) equality of the polynomials, i.e. the committed sketch is the
real one. On-chain data: commitment + one field element + one opening proof.
Independent of m.

## Soundness

If the committed Chat is wrong, (f - true)(X,Y) is a nonzero polynomial of
degree < m in each variable. By Schwartz-Zippel a random (a,b) in F_q^2
satisfies it with probability <= 2(m-1)/q. With q = 2^61 - 1 and m = 2048 that
is about 2^-49 per round. Fiat-Shamir binds (a,b) to the commitment, so the
prover commits first and learns the challenge second. Grinding interplay: each
grind attempt costs a full commit over the sketch, so a 2^-49-per-try cheat is
already worse than honest mining; two independent rounds (or one round with
(a,b) drawn from a degree-2 extension field) push it to ~2^-98, out of reach of
any physical grinding budget. The compute requirement of the PoW is untouched:
the miner must still produce the real Chat to commit to it.

## What the miner pays

- Per candidate nonce: one commitment over the sketch. The current design
  already hashes the full serialized sketch per candidate (the digest
  H(sigma || Chat)), and a Merkle-style commitment over the coefficients is
  about 2x that hashing. So the per-nonce cost class does not change.
- One flavor detail to settle in production: a textbook FRI opening commits to
  the low-degree-extended codeword (blowup 8), which per nonce would add an NTT
  plus 8x the hashing. Still small next to the matmul itself, but the cleaner
  design is a coefficient commitment per nonce with the LDE built only for the
  won block, which requires the PCS flavor that supports opening against a
  coefficient root. This is the main open engineering question.
- Per WON block only: one opening-proof generation (FRI: seconds-class on the
  same hardware; amortized once per block, not per nonce). Note this adds proof
  latency at block-publish time, which trades against orphan risk; pipelining
  or header-first relay with the proof following are the standard mitigations.
- This argues for a transparent hash-based commitment (FRI/Merkle) over KZG for
  the mining side: KZG commitment per nonce would be a multi-million-point MSM,
  which is not viable per candidate. KZG numbers are included below only to
  show the constant-size endpoint.

## Sizes (measured by this reference's size models)

| n | m | naive raw sketch (8m^2) | succinct FRI (upper est.) | succinct KZG |
|------|------|------------------------|---------------------------|--------------|
| 1024 | 256 | 512 KiB | ~169 KiB | 168 B |
| 2048 | 512 | 2 MiB | ~213 KiB | 168 B |
| 4096 | 1024 (profile C) | 8 MiB | ~261 KiB | 168 B |
| 4096 | 2048 (profile D) | 32 MiB | ~314 KiB | 168 B |

The FRI column is a conservative upper estimate (34 queries, blowup 8, no path
deduplication); production implementations typically land 3-10x smaller. The
headline: naive grows 4x per m-doubling, FRI grows ~log^2, KZG does not move.
At profile D the shrink is >100x even under the conservative estimate, and the
permanent chain no longer contains any m-sized object at all.

## What this reference is, and is not

- IS: a faithful, tested model of the protocol. Operand generation from seed,
  the honest O(n^3) sketch, the bilinear-to-polynomial reduction, Fiat-Shamir,
  prover, verifier, tamper detection, and the size accounting.
- IS NOT: production cryptography. The polynomial commitment is modeled as an
  ideal functionality (`IdealPC`): binding and evaluation-correctness are
  enforced in code, and proof sizes are reported via standard FRI/KZG formulas.
  Swapping `IdealPC` for a real FRI opening (transparent, hash-only, no trusted
  setup, the recommended flavor for a PoW chain) yields a production prototype.
  The reduction and the soundness argument are unchanged by that swap.

## Relation to the segregated-proof design

Complementary, not competing. Segregation + pruning (the existing design doc)
relocates the raw sketch off the permanent chain and is the right near-term
ship: standard pattern, no new cryptography. This scheme is the endgame: the
sketch never exists on-chain in any form, archival nodes carry nothing m-sized,
block relay carries kilobytes instead of megabytes, and the C-vs-D choice stops
having any chain-size consequence at all. A natural path is to ship segregation
first and introduce the succinct proof as a later upgrade that reuses the same
header commitment slot (matmul_digest becomes the polynomial commitment).

## Adversarial cost

Can a miner pass while doing less compute? `adversarial_cost.py` prices every
known strategy and the tests assert none beats honest mining. Partial compute
(garbage-fill a fraction e of entries) loses exponentially: Q=34 sampled
openings accept with (1-e)^Q while the savings are linear, so skipping 1% of
the sketch already costs 1.4x honest per accepted block, 10% costs ~35x, and
the curve never turns profitable. The acceptance model is verified against an
actual Merkle commitment with root-derived query sampling, not just the
formula. Garbage-commit scanning is Schwartz-Zippel-priced at ~2^-49 per
round against a nonce-throughput speedup bounded by ~2^30, and challenge
grinding costs ~10^11 honest blocks per success. Three implementation MUSTs
fall out: the commitment stays in the header hash as the per-nonce
eligibility gate; the PCS admits no free re-randomization; and the
commitment must bind EXACTLY to the coefficients (below).

## Digest uniqueness: the exact-binding requirement

Today's design has an unstated load-bearing property: per nonce there is
exactly ONE valid digest value (the hash of the exact true sketch, enforced
by exact Freivalds over the full payload). That uniqueness is what makes
eligibility tries cost matmuls. A succinct scheme must preserve it, and the
choice of commitment basis decides whether it does:

- Evaluation-domain commitments (the textbook LDE-FRI flavor) are
  proximity-decoded: a single twiddled leaf stays within decoding radius of
  the true polynomial, the opening decodes the twiddle away, the identity
  check passes on the decoded value, and ~34 queries miss one position in
  millions. A miner could then compute ONE sketch and grind eligibility by
  twiddling leaves at Merkle-update speed. That is a PoW break, not a
  discount.
- Coefficient-basis commitments close it: every change to committed data IS
  a different polynomial, so the Fiat-Shamir identity catches any twiddle at
  the Schwartz-Zippel bound and ground twiddles only produce invalid blocks
  (the single-entry-perturbation test in this reference is exactly that
  catch). Exact-binding algebraic PCS constructions (lattice-based
  Greyhound-class) have the same property.

So the coefficient-root flavor is not just the cheaper per-nonce shape; it
is the SOUND one. The production requirement: evaluation claims must bind to
the committed leaves themselves, never to a decoded nearby codeword.

One sharpening: exactness is a property of the commitment's BINDING, not the
basis alone (this reference is exact because IdealPC is exact by
construction). Any hash-based PCS commits to an encoding with redundancy, so
many commitment strings open as the same polynomial. Fiat-Shamir folding
does force a full proof regeneration per grind try, capping the twiddle
speedup near 10-20x instead of 2^31, but a 10x grind is still a dead PoW.
Candidate ranking that follows: exact-binding lattice PCS (Ajtai/Greyhound
class: a twiddled commitment has no known opening at all; transparent,
post-quantum, linear prover) is the structurally clean candidate; the FRI
family needs a bespoke leaf-exactness construction to survive this role; KZG
and Pedersen-class are exact but per-nonce MSM kills them for mining; plain
commit-plus-spot-check (open k sampled entries, recompute from seed) has no
global identity, so single-entry twiddles pass sampling with probability ~1
and it cannot stand alone. A sumcheck layer (Thaler's matmul protocol)
composes with any commitment and shrinks what the PCS must open to a single
point.

The front-runner is now IN CODE here: `lattice_pc.py` implements the
ring-Ajtai commitment for real (Dilithium's ring Z_8380417[X]/(X^256+1),
NTT validated against schoolbook negacyclic multiplication, gadget
decomposition to base-2^4 digits, seed-derived A, Z-linearity over digit
vectors), exposed as `backend="lattice"` with the evaluation opening modeled
at the literature's ~50 KB class. It answers the per-nonce budget question
at op-count level: the commitment costs ~0.64x today's full-sketch digest
hashing (independent 256-point NTTs, embarrassingly parallel), so the
eligibility-gate role fits the nonce loop with margin. Commitment is 4 KiB
(the header carries H(sigma || t), 32 bytes; t rides in the block body), so
the on-chain footprint at profile D is ~54 KiB: commitment + opening +
value, against 32 MiB raw.

MEASURED on real silicon (RTX 5090, boost clocks; CUDA port of this exact
commitment, cross-validated byte-identical against lattice_pc.py on a
shared test vector). Per commit, two arms bracketing the production cost:

| profile | A regenerated on the fly | A read from a precomputed table |
|---------|--------------------------|---------------------------------|
| C (m=1024) | 0.12 ms | 0.22 ms (256 MB table) |
| D (m=2048) | 0.46 ms | 0.84 ms (1 GB table, 1.3 TB/s effective) |

Both arms sit comfortably inside a realistic per-nonce budget at these
dimensions. The memory arm runs at ~ the card's bandwidth ceiling, so the
on-the-fly arm is the right production shape: ~1.8x faster and it removes
the resident gigabyte; a hardened PRF for A lands between the two arms.
The op-count estimate held up on silicon.

## Run

```
python3 succinct_matmul_pow.py   # self-check + decoupling table
python3 adversarial_cost.py      # adversarial cost table (modeled + sampled)
python3 lattice_pc.py            # ring-Ajtai commitment self-check + op-count
python3 test_succinct.py         # 29 tests: completeness, soundness, binding,
                                 # decoupling, verifier-cost, adversarial cost,
                                 # lattice backend
```

No dependencies beyond the Python standard library.
