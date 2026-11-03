"""M12 gate -- DistanceDP protocol (paper Sec. III-B2, adopted from RemoteRAG).

Acceptance criteria (docs/MODULE_PLAN.md M12; SPEC.md Sec. 4):

1. gamma mode: r ~ Gamma(shape=n, scale=1/eps), E[r] = n/eps.
   With n=768, eps=7680 -> mean radius 0.1, relative error < 2% over 5000 draws,
   and a KS test against Gamma(768, 1/7680) with p > 0.01.
2. direction is uniform on the unit (n-1)-sphere: standardised coordinates
   z = sqrt(n) * x have mean 0 / variance 1; coordinate-mean z-scores
   max|z| < 5, and fraction of |z| < 3 is > 0.99 (Gaussian tail reference).
3. ||e~ - e|| == realised r (fixed and gamma mode), floating error <= 1e-12.
4. fixed-radius and gamma modes are reproducible under a seeded RNG and random
   across seeds; fixed mode returns exactly the requested radius.
5. eps <-> mean radius conversion: epsilon_for_radius(n, r_bar) == n / r_bar and
   RemoteRAG's range eps in [10n, 50n] maps to r_bar in [0.02, 0.1].

Run:  .venv/bin/python -m gates.gate_m12_distance_dp
"""

from __future__ import annotations

import math
import sys

import numpy as np
from scipy import stats

from gates.common import Gate, add_src_to_path

add_src_to_path()

from caprise.dp import distance_dp, epsilon_for_radius, sample_radius  # noqa: E402

N_DIM = 768
EPS = 7680.0          # = 10 * n  ->  mean radius n / eps = 0.1
MEAN_R = N_DIM / EPS  # 0.1


def main() -> int:
    g = Gate(
        "M12",
        "distance_dp",
        owner="theory-efficiency",
        paper_ref="arXiv:2601.12331 Sec. III-B2 / RemoteRAG arXiv:2412.12775 Sec. 3.2.1",
    )

    # ------------------------------------------------------------------ #
    # 1. Gamma(shape=n, scale=1/eps) radius law and mean n/eps
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(20260118)
    radii = np.array([sample_radius(N_DIM, EPS, rng) for _ in range(5000)])
    mean_r = float(radii.mean())
    rel_err = abs(mean_r - MEAN_R) / MEAN_R
    g.check_true(
        "gamma_mean_equals_n_over_eps",
        rel_err < 0.02,
        detail=f"n={N_DIM}, eps={EPS:g}: sample mean={mean_r:.6f}, expected={MEAN_R:g}, rel_err={rel_err:.5f} (<0.02), N=5000",
    )
    ks = stats.kstest(radii, "gamma", args=(float(N_DIM), 0.0, 1.0 / EPS))
    g.check_true(
        "gamma_ks_test_p_gt_0.01",
        ks.pvalue > 0.01,
        detail=f"KS D={ks.statistic:.5f}, p={ks.pvalue:.5f} vs Gamma(shape={N_DIM}, scale=1/{EPS:g})",
    )
    std_ratio = float(radii.std(ddof=1) / mean_r)
    expected_std_ratio = 1.0 / math.sqrt(N_DIM)
    g.check_true(
        "gamma_std_over_mean_equals_1_over_sqrt_n",
        abs(std_ratio - expected_std_ratio) / expected_std_ratio < 0.05,
        detail=f"empirical std/mean={std_ratio:.6f}, expected 1/sqrt(n)={expected_std_ratio:.6f} (rel err <5%)",
    )

    # ------------------------------------------------------------------ #
    # 2. Direction uniformity on S^{n-1}
    # ------------------------------------------------------------------ #
    n_dirs = 5000
    e0 = np.zeros(N_DIM)
    dirs = np.empty((n_dirs, N_DIM))
    for i in range(n_dirs):
        pert, r_used = distance_dp(e0, radius=0.1, rng=rng)
        assert abs(r_used - 0.1) < 1e-15
        dirs[i] = pert / r_used

    norms = np.linalg.norm(dirs, axis=1)
    max_norm_dev = float(np.max(np.abs(norms - 1.0)))
    g.check_true(
        "directions_are_unit_norm",
        max_norm_dev < 1e-12,
        detail=f"max | ||direction|| - 1 | = {max_norm_dev:.3e} over {n_dirs} draws (<=1e-12)",
    )

    Z = math.sqrt(N_DIM) * dirs  # standardised coordinates, target ~ N(0,1)
    coord_mean = Z.mean(axis=0)                       # (n,)
    coord_z = coord_mean * math.sqrt(n_dirs)          # ~N(0,1) under uniformity
    max_abs_z = float(np.max(np.abs(coord_z)))
    g.check_true(
        "uniformity_coord_mean_zscores_max_lt_5",
        max_abs_z < 5.0,
        detail=f"max |z_j| = {max_abs_z:.3f} over {N_DIM} coordinates (each z_j = sqrt(N)*mean_j ~ N(0,1)); <5 required",
    )
    frac_within_3 = float(np.mean(np.abs(Z) < 3.0))
    g.check_true(
        "uniformity_frac_abs_z_lt_3_gt_0.99",
        frac_within_3 > 0.99,
        detail=f"fraction of standardised coordinates with |z|<3: {frac_within_3:.5f} (Gaussian reference 0.9973); >0.99 required",
    )
    var_z = float(np.mean(Z**2))
    g.check_true(
        "uniformity_unit_variance",
        abs(var_z - 1.0) < 0.02,
        detail=f"E[z^2] = {var_z:.5f} (target 1.0 +/- 0.02; for uniform S^(n-1), Var(x_j)=1/n)",
    )

    # ------------------------------------------------------------------ #
    # 3. ||e~ - e|| == realised radius r
    # ------------------------------------------------------------------ #
    e = rng.standard_normal(N_DIM)
    e /= np.linalg.norm(e)
    fixed_errs = []
    for _ in range(200):
        pert, r_used = distance_dp(e, radius=0.033, rng=rng)
        fixed_errs.append(abs(float(np.linalg.norm(pert - e)) - r_used))
    gamma_errs = []
    for _ in range(200):
        pert, r_used = distance_dp(e, radius="gamma", eps=N_DIM / 0.05, rng=rng)
        gamma_errs.append(abs(float(np.linalg.norm(pert - e)) - r_used))
    max_fixed = float(max(fixed_errs))
    max_gamma = float(max(gamma_errs))
    g.check_true(
        "norm_shift_equals_realised_radius_fixed",
        max_fixed <= 1e-12,
        detail=f"max | ||e~-e|| - r | = {max_fixed:.3e} over 200 draws (radius=0.033, tol 1e-12)",
    )
    g.check_true(
        "norm_shift_equals_realised_radius_gamma",
        max_gamma <= 1e-12,
        detail=f"max | ||e~-e|| - r | = {max_gamma:.3e} over 200 gamma draws (mean radius 0.05, tol 1e-12)",
    )
    # perturbed query must not be re-normalised (it is e + r*d, not e~ / ||e~||)
    pert, r_used = distance_dp(e, radius=0.05, rng=np.random.default_rng(7))
    expected_norm = math.sqrt(1.0 + 0.05**2 + 2 * 0.05 * float(e @ (pert - e) / r_used))
    g.check_true(
        "perturbation_not_renormalised",
        abs(float(np.linalg.norm(pert)) - expected_norm) < 1e-12,
        detail=f"||e~||={float(np.linalg.norm(pert)):.12f}, ||e + r d||={expected_norm:.12f} (DistanceDP adds, never normalises)",
    )

    # ------------------------------------------------------------------ #
    # 4. Reproducibility / randomness of both modes
    # ------------------------------------------------------------------ #
    p1, r1 = distance_dp(e, radius="gamma", eps=EPS, rng=np.random.default_rng(1234))
    p2, r2 = distance_dp(e, radius="gamma", eps=EPS, rng=np.random.default_rng(1234))
    g.check_true(
        "gamma_seeded_reproducible",
        np.array_equal(p1, p2) and r1 == r2,
        detail=f"seed=1234 twice: vectors identical={np.array_equal(p1, p2)}, radii equal={r1 == r2}",
    )
    p3, r3 = distance_dp(e, radius="gamma", eps=EPS, rng=np.random.default_rng(1235))
    g.check_true(
        "gamma_different_seed_differs",
        not np.array_equal(p1, p3) and r1 != r3,
        detail="seed=1234 vs 1235: vectors/radii differ",
    )
    f1, fr1 = distance_dp(e, radius=0.02, rng=np.random.default_rng(99))
    f2, fr2 = distance_dp(e, radius=0.02, rng=np.random.default_rng(99))
    g.check_true(
        "fixed_radius_seeded_reproducible_and_exact",
        np.array_equal(f1, f2) and fr1 == fr2 == 0.02,
        detail=f"seed=99 twice with radius=0.02: identical={np.array_equal(f1, f2)}, returned radius={fr1}",
    )
    g.check_true(
        "fixed_radius_returned_exactly",
        all(
            distance_dp(e, radius=0.033, rng=np.random.default_rng(i))[1] == 0.033
            for i in range(20)
        ),
        detail="20 draws with radius=0.033 all return r==0.033 exactly (no sampling of the radius)",
    )

    # invalid arguments must be rejected
    rejected = []
    for args, kwargs in [
        ((e,), {"radius": "gamma"}),                 # eps missing
        ((e,), {"radius": -0.1}),                    # negative radius
        ((e,), {"radius": 0.0}),                     # zero radius
        ((e,), {"radius": "uniform"}),               # unknown mode
        ((e,), {"radius": "gamma", "eps": -1.0}),    # negative eps
    ]:
        try:
            distance_dp(*args, **kwargs)
        except (ValueError, TypeError):
            rejected.append(True)
        else:
            rejected.append(False)
    g.check_true(
        "invalid_arguments_rejected",
        all(rejected),
        detail=f"rejected {sum(rejected)}/5 invalid configurations (missing eps, r<=0, unknown mode, eps<=0)",
    )

    # ------------------------------------------------------------------ #
    # 5. eps <-> mean radius conversion
    # ------------------------------------------------------------------ #
    conv_ok = abs(epsilon_for_radius(N_DIM, MEAN_R) - EPS) < 1e-6
    g.check_true(
        "epsilon_for_radius",
        conv_ok,
        detail=f"epsilon_for_radius({N_DIM}, {MEAN_R:g}) = {epsilon_for_radius(N_DIM, MEAN_R):.6f} (expected {EPS:g})",
    )
    # RemoteRAG: eps in [10n, 50n]  <->  mean radius in [0.02, 0.1]
    eps_lo = epsilon_for_radius(N_DIM, 0.1)
    eps_hi = epsilon_for_radius(N_DIM, 0.02)
    g.check_true(
        "ridge_range_eps_10n_50n_maps_to_rbar_0.02_0.1",
        abs(eps_lo - 10 * N_DIM) < 1e-6 and abs(eps_hi - 50 * N_DIM) < 1e-6,
        detail=f"r_bar=0.1 -> eps={eps_lo:g} (10n={10*N_DIM}), r_bar=0.02 -> eps={eps_hi:g} (50n={50*N_DIM})",
    )
    # realised mean radius for eps=50n should be 0.02
    rng2 = np.random.default_rng(11)
    radii2 = np.array([sample_radius(N_DIM, 50 * N_DIM, rng2) for _ in range(3000)])
    rel2 = abs(float(radii2.mean()) - 0.02) / 0.02
    g.check_true(
        "gamma_mean_at_eps_50n_is_0.02",
        rel2 < 0.02,
        detail=f"eps=50n=38400: sample mean radius={radii2.mean():.6f}, expected 0.02, rel_err={rel2:.5f} (<0.02)",
    )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    g.note("DistanceDP mechanism: e~ = e_q + r*d, r ~ Gamma(shape=n, scale=1/eps), d uniform on S^{n-1}.")
    g.note("Mean radius r_bar = n/eps; RemoteRAG's typical range r_bar in [0.02,0.1] corresponds to eps in [10n,50n].")
    g.note("Two modes implemented: radius='gamma' + eps (actual DP mechanism) and fixed radius (Table III / ablations).")
    g.note("Perturbed query is NOT re-normalised, matching RemoteRAG and Eq. 21's e~_q = e_q + noise.")
    res = g.finish()
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
