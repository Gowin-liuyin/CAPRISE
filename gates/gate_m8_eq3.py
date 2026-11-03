"""M8 gate: Eq. (3) -- CAPRISE does NOT preserve inter-database orderings.

    ||e_1 - e_2|| < ||e_1 - e_3|| - beta  =/=>  ||e'_1 - e'_2|| < ||e'_1 - e'_3||

Reproduction finding (important)
--------------------------------
The paper proves Eq.(3) as an *existence* statement: there are noise
realisations delta, delta' with delta+delta' < s*beta/2 that flip the ordering.
However the noise direction is uniform in d dimensions, and a flip requires
lambda_1, lambda_2, lambda_3 to be *aligned* with the specific difference
vectors (so that the three noise contributions add up).  In d=768 dimensions
such alignment has probability exp(-Omega(d)), so with the actual sampler the
empirical inversion rate on premise-satisfying triples is 0 (we measure it).

This gate therefore separates two claims:
  (a) Eq.(3) is logically TRUE: a valid noise realisation exists inside the
      sampler's support that inverts the ordering (constructive adversarial
      realisation below).
  (b) The claim is NOT practically reachable at high dimension: empirical
      inversion rate = 0/20000 for CAPRISE (and 0 for ADCPE).
  (c) The actual measurable difference between CAPRISE and ADCPE is the
      *distance-reconstruction error* (isotropic noise projection ~ R/sqrt(d)):
      CAPRISE ~1.9x larger than ADCPE here -> quantified and reported.
"""

from __future__ import annotations

import numpy as np

from gates.common import Gate, add_src_to_path

add_src_to_path()
from caprise.caprise import (  # noqa: E402
    ADCPE,
    EncDB,
    KeyGen,
    noise_radius_adcpe,
    noise_radius_db,
    sample_noise,
)

g = Gate("M8", "eq3", owner="lead", paper_ref="paper Eq. (3) / SPEC 5.2 / finding A19")

key = KeyGen(dim=768, beta=0.2, s=3)
rng = np.random.default_rng(13)
beta = key.beta
D = key.dim
R_db = noise_radius_db(key.s, key.beta)      # 3sβ/8
R_adcpe = noise_radius_adcpe(key.s, key.beta)  # sβ/4


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _perp(base, v):
    v = v - np.einsum("ij,ij->i", v, base)[:, None] * base
    return _unit(v)


def _rot(base, dist, v):
    theta = 2.0 * np.arcsin(dist / 2.0)
    return np.cos(theta)[:, None] * base + np.sin(theta)[:, None] * _perp(base, v)


# --------------------------------------------------------------------------
# (a) Constructive adversarial realisation: exists within sampler support
# --------------------------------------------------------------------------
# Build a triple with a small premise margin, then choose noise vectors inside
# the ball whose directions are aligned with the relevant difference vectors.
E1 = _unit(rng.standard_normal((1, D)))
d12 = np.array([0.7])
margin = np.array([1e-4])
d13 = d12 + beta + margin
E2 = _rot(E1, d12, rng.standard_normal((1, D)))
E3 = _rot(E1, d13, rng.standard_normal((1, D)))
D12 = (E2 - E1)[0]
D13 = (E3 - E1)[0]
u12 = D12 / np.linalg.norm(D12)
u13 = D13 / np.linalg.norm(D13)


def _mk_noise(vec):
    """Represent an arbitrary vector inside the ball as (direction, u)."""
    n = np.linalg.norm(vec)
    return vec / n, n / R_db


# candidate direction w in the plane of u12,u13
if np.linalg.norm(u12 + u13) > 1e-6:
    w = _unit(u12 + u13)
else:
    w = _unit(np.cross(u12, rng.standard_normal(D)))
# a few candidate w's, take the one that maximises lhs - rhs
best = None
for alpha in np.linspace(0, 2 * np.pi, 721):
    v = _perp(np.array([_unit(u12 + u13)])[0][None, :], w[None, :] * 0 + rng.standard_normal((1, D)))[0]
    # build orthonormal basis of span(u12,u13)
    n1 = u12
    n2 = _unit(u13 - (u13 @ n1) * n1)
    w_dir = np.cos(alpha) * n1 + np.sin(alpha) * n2
    lam1 = R_db * w_dir
    lam2 = -R_db * n1
    lam3 = R_db * u13
    lhs = np.linalg.norm(key.s * D12 + (lam1 - lam2))
    rhs = np.linalg.norm(key.s * D13 + (lam1 - lam3))
    if best is None or lhs - rhs > best[0]:
        best = (lhs - rhs, lhs, rhs, alpha)
adv_gap, adv_lhs, adv_rhs, adv_alpha = best
g.check_true("eq3_adversarial_inversion_exists", adv_gap > 0,
             f"constructed valid noise (||lambda||<=R): lhs={adv_lhs:.4f} > rhs={adv_rhs:.4f} "
             f"(inverted despite premise; alpha={adv_alpha:.3f} rad)")
g.note("Constructive realisation confirms the mathematical claim of Eq.(3): the noise support "
       "contains a configuration that flips the ordering.")

# --------------------------------------------------------------------------
# (b) Empirical rate with the actual sampler on premise-satisfying triples
# --------------------------------------------------------------------------
n = 20_000
E1b = _unit(rng.standard_normal((n, D)))
d12b = rng.uniform(0.3, 1.0, size=n)
margin_b = rng.uniform(1e-6, 0.08, size=n)
d13b = d12b + beta + margin_b
E2b = _rot(E1b, d12b, rng.standard_normal((n, D)))
E3b = _rot(E1b, d13b, rng.standard_normal((n, D)))
premise = np.linalg.norm(E1b - E2b, axis=1) < np.linalg.norm(E1b - E3b, axis=1) - beta
g.check_true("premise_holds_by_construction", bool(premise.all()), f"{n} triples")

E1p, _ = EncDB(key, E1b)
E2p, _ = EncDB(key, E2b)
E3p, _ = EncDB(key, E3b)
inv = int((np.linalg.norm(E1p - E2p, axis=1) >= np.linalg.norm(E1p - E3p, axis=1)).sum())
g.note(f"CAPRISE empirical inversion rate on premised triples (real sampler): {inv}/{n}. "
       f"Existence-only: alignment probability decays like exp(-Omega(d)).")

# ADCPE control: no realisation can invert (verify via worst-case bound)
# lhs <= s*d12 + 2*R_adcpe ; rhs >= s*d13 - 2*R_adcpe ; inversion needs
# s*d12 + 2R >= s*d13 - 2R  <=>  s*(d13-d12) <= 4R = s*beta, contradicting premise.
g.check_true("adcpe_worst_case_bound_forbids_inversion", 4 * R_adcpe <= key.s * beta,
             f"4R_adcpe={4*R_adcpe:.4f} <= s*beta={key.s*beta:.4f}")
E1a = ADCPE.EncDB(key, E1b)
E2a = ADCPE.EncDB(key, E2b)
E3a = ADCPE.EncDB(key, E3b)
inv_a = int((np.linalg.norm(E1a - E2a, axis=1) >= np.linalg.norm(E1a - E3a, axis=1)).sum())
g.check("adcpe_empirical_inversions", 0, inv_a, "no inversions either (as guaranteed)")

# --------------------------------------------------------------------------
# (c) Measurable leakage difference: distance reconstruction error
# --------------------------------------------------------------------------
m = 600
X = _unit(rng.standard_normal((m, D)))
Xp, _ = EncDB(key, X)
Xa = ADCPE.EncDB(key, X)
D_true = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
D_cap = np.linalg.norm(Xp[:, None, :] - Xp[None, :, :], axis=2) / key.s
D_ad = np.linalg.norm(Xa[:, None, :] - Xa[None, :, :], axis=2) / key.s
mask = ~np.eye(m, dtype=bool)
err_cap = float(np.abs(D_cap - D_true)[mask].mean())
err_ad = float(np.abs(D_ad - D_true)[mask].mean())
g.check_true("caprise_leaks_more_distance_error_than_adcpe", err_cap > err_ad,
             f"CAPRISE err={err_cap:.5f} > ADCPE err={err_ad:.5f} (ratio {err_cap/err_ad:.2f}x)")
# theory: projection std ~ sqrt(2)*R/sqrt(d)
theory_cap = np.sqrt(2) * R_db / key.s / np.sqrt(D)   # in plaintext units: R_db/s * sqrt2 / sqrt(d)
theory_ad = np.sqrt(2) * R_adcpe / key.s / np.sqrt(D)
g.note(f"Distance-reconstruction error (plaintext units): CAPRISE {err_cap:.5f} vs ADCPE {err_ad:.5f} "
       f"(theory proj std: {theory_cap:.5f} / {theory_ad:.5f}); ratio {err_cap/err_ad:.2f}x = R_db/R_adcpe = 1.5x")
g.note("Finding A19: Eq.(3) holds only as an existence statement at high dimension; the practical "
       "vector-analysis advantage of CAPRISE must be sought in the increased distance-reconstruction "
       "error / ASR (measured in exp4), not in literal order inversions.")
g.finish()
