"""M3 gate: noise sampler (Algorithm 1 line 11 / Eq. 7).

Protocol under test: lambda is sampled uniformly inside the d-dimensional ball
of radius R = 3*s*beta/8, via

    lambda = R * u^(1/d) * n/||n||,   n ~ N(0, I_d),  u ~ U(0,1)

Gate criteria
-------------
1. ||lambda|| < R strictly for every sample (required by Eq. 7 and the proof).
2. ||lambda||/R is close to 1 (u^(1/d) concentration in high dimension).
3. Uniform-in-ball: (||lambda||/R)^d ~ U(0,1) (KS test, p > 0.01).
4. Direction n/||n|| is uniform on the sphere: mean ~ 0, unit norm.
5. Determinism given (K, r).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kstest

from gates.common import Gate, add_src_to_path

add_src_to_path()
from caprise.caprise import (  # noqa: E402
    KeyGen,
    noise_radius_db,
    noise_radius_q,
    sample_noise,
)
from caprise.prf import new_nonce  # noqa: E402

g = Gate("M3", "noise", owner="lead", paper_ref="Algorithm 1 line 11 / Eq. 7 / SPEC 2.2")

key = KeyGen(dim=768, beta=0.2, s=3)
R_db = noise_radius_db(3, 0.2)
R_q = noise_radius_q(3, 0.2)
g.check_true("radius_formulas", abs(R_db - 3 * 3 * 0.2 / 8) < 1e-15 and abs(R_q - 3 * 0.2 / 8) < 1e-15,
             f"R_db={R_db:.4f} R_q={R_q:.4f}")

rng = np.random.default_rng(7)
lam = np.asarray([sample_noise(key, new_nonce(rng), R_db) for _ in range(4000)])
norms = np.linalg.norm(lam, axis=1)

g.check_true("norm_strictly_below_radius", bool((norms < R_db).all()), f"max||lambda||={norms.max():.6f} < R={R_db}")
g.check_true("norm_close_to_radius", float(norms.max()) > 0.99 * R_db, f"max/R={norms.max()/R_db:.5f}")
g.check_true("norm_min_above_zero", float(norms.min()) > 0.0)

# uniform-in-ball test: (||lambda||/R)^d ~ U(0,1)
x = (norms / R_db) ** 768
p_ball = kstest(x, "uniform").pvalue
g.check_true("uniform_in_ball_ks", p_ball > 0.01, f"KS p={p_ball:.4f}")

# direction uniformity: standardised coordinate means z_i ~ N(0,1)
dirs = lam / norms[:, None]
z = dirs.mean(axis=0) * np.sqrt(len(lam) * 768)
max_z = float(np.abs(z).max())
frac3 = float((np.abs(z) < 3).mean())
g.check_true("direction_mean_near_zero", max_z < 5.0 and frac3 > 0.99,
             f"max|z|={max_z:.2f} (<5), frac|z|<3={frac3:.4f} (>0.99)")
g.check_true("direction_unit_norm", bool(np.allclose(np.linalg.norm(dirs, axis=1), 1.0)))

# determinism
nonce = new_nonce(rng)
g.check_true("deterministic_given_nonce", np.array_equal(sample_noise(key, nonce, R_db), sample_noise(key, nonce, R_db)))

# query radius is exactly one third of the DB radius (1/8 vs 3/8 of s*beta)
g.check_true("ratio_db_over_q", abs(R_db / R_q - 3.0) < 1e-12, f"R_db/R_q={R_db/R_q:.6f}")

g.note("Noise is bounded (ball), not Gaussian: n supplies only the direction.")
g.note(f"d=768: E[u^(1/d)] = {1/(1+1/768):.5f} -> norms concentrate just below R.")
g.finish()
