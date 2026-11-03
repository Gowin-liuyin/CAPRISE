"""M7 gate: Eq. (2) -- distance-comparison preservation for query-vs-database.

    ||e_q - e_1|| < ||e_q - e_2|| - beta  =>  ||e'_q - e'_1|| < ||e'_q - e'_2||

Important construction note
---------------------------
Uniform random unit vectors in d=768 have pairwise distances concentrated at
sqrt(2) with std ~ 1/sqrt(2d) ~ 0.025, so the premise (a distance gap > beta=0.2)
is essentially never satisfied by random triples.  The premise is about real
retrieval geometry: the gap between the k-th and (k+1)-th neighbour.  We
therefore construct triples *exactly on the premise boundary*: place e_1 at
distance d1 from e_q and e_2 at distance d1 + beta + margin (margin in
[1e-6, 0.05]) in independent random directions.  This is the strongest possible
test of the claim (gaps just above the guaranteed threshold).
"""

from __future__ import annotations

import numpy as np

from gates.common import Gate, add_src_to_path

add_src_to_path()
from caprise.caprise import DecQ, EncDB, EncQ, KeyGen  # noqa: E402

g = Gate("M7", "eq2", owner="lead", paper_ref="paper Eq. (2) / SPEC 5.2")

key = KeyGen(dim=768, beta=0.2, s=3)
rng = np.random.default_rng(11)
beta = key.beta
D = key.dim


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _perp(base, v):
    v = v - np.einsum("ij,ij->i", v, base)[:, None] * base
    return _unit(v)


def _rot(base, dist, v):
    """Unit vector at exact L2 distance ``dist`` from each row of ``base``."""
    theta = 2.0 * np.arcsin(dist / 2.0)
    return np.cos(theta)[:, None] * base + np.sin(theta)[:, None] * _perp(base, v)


def make_triples(n: int, margin_lo: float, margin_hi: float, rng):
    eq = _unit(rng.standard_normal((n, D)))
    d1 = rng.uniform(0.3, 1.0, size=n)
    margin = rng.uniform(margin_lo, margin_hi, size=n)
    d2 = d1 + beta + margin
    e1 = _rot(eq, d1, rng.standard_normal((n, D)))
    e2 = _rot(eq, d2, rng.standard_normal((n, D)))
    return eq, e1, e2, d1, d2


# --- 50k triples with margin in [0.001, 0.05] ------------------------------
n = 50_000
EQ, E1, E2, d1, d2 = make_triples(n, 1e-3, 0.05, rng)
premise = np.linalg.norm(EQ - E1, axis=1) < np.linalg.norm(EQ - E2, axis=1) - beta
g.check_true("premise_holds_by_construction", bool(premise.all()), f"{n} triples")

q_primes = np.asarray([EncQ(key, q)[0] for q in EQ])
E1p, _ = EncDB(key, E1)
E2p, _ = EncDB(key, E2)
d_q1 = np.linalg.norm(q_primes - E1p, axis=1)
d_q2 = np.linalg.norm(q_primes - E2p, axis=1)
violations = int((d_q1 >= d_q2).sum())
g.check("preservation_violations", 0, violations, f"{violations}/{n} violated Eq.(2)")
margin = d_q2 - d_q1
g.note(f"ciphertext margin: min={margin.min():.4e}, median={np.median(margin):.4f} "
       f"(theoretical worst case s*beta/2 = {key.s*beta/2:.4f})")

# --- extreme boundary: margin = 1e-6 exactly -------------------------------
n_b = 2000
EQb, E1b, E2b, _, _ = make_triples(n_b, 1e-6, 1e-6, rng)
qb = np.asarray([EncQ(key, q)[0] for q in EQb])
E1bp, _ = EncDB(key, E1b)
E2bp, _ = EncDB(key, E2b)
viol_b = int((np.linalg.norm(qb - E1bp, axis=1) >= np.linalg.norm(qb - E2bp, axis=1)).sum())
g.check("boundary_violations", 0, viol_b, "gap = beta + 1e-6 still preserved")

# --- NTK-style: plaintext ordering is preserved exactly in ciphertext? -----
g.note("Random-vector isotropy note: d=768 random unit vectors give pair-distance std ~0.025, "
       "so premise gaps must be constructed; real retrieval gaps are measured in exp2.")
g.finish()
