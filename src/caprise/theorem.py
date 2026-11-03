"""Theorem 1 of arXiv:2601.12331: retrieval-set expansion under query DP noise.

The theorem (Eq. 21) states

    k' = k + m * (Omega_{n-1}(pi) / Omega_n(pi)) * int_{alpha_k}^{alpha_k'}
             sin^{n-2}(theta) dtheta

with alpha_k' = alpha_k + Delta alpha_k, Omega_n(pi) = 2 pi^{n/2} / Gamma(n/2).

Implementation notes
--------------------
* For a uniform random direction on the unit sphere in R^n, the angle theta to a
  fixed query has CDF

      F(alpha) = P(theta <= alpha)
               = 1 - I_{(1+cos alpha)/2}((n-1)/2, (n-1)/2)

  where I_x(a,b) is the regularised incomplete Beta function.  The paper writes
  the same quantity as (Omega_{n-1}/Omega_n) * int_0^alpha sin^{n-2}.  We use
  the Beta form because the direct integral under-flows to zero for n ~ 1e3 in
  the tails (sin(theta)^{n-2} ~ 1e-3000).
* alpha_k is the angle of the k-th nearest neighbour: F(alpha_k) = k / m.
* The paper/RemoteRAG approximate the angular perturbation by the radial one:
  Delta alpha_k ~ r.  The reproduction also exposes ``delta_alpha_ratio`` so the
  paper's Table III convention can be probed (see SPEC.md section 5).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import betainc, betaincinv

__all__ = [
    "cap_fraction",
    "alpha_for_fraction",
    "k_prime",
    "k_prime_table",
    "monte_carlo_k_prime",
    "omega_ratio",
]


def omega_ratio(n: int) -> float:
    """Omega_{n-1}(pi) / Omega_n(pi) = Gamma(n/2) / (sqrt(pi) Gamma((n-1)/2))."""
    from scipy.special import gammaln

    return math.exp(gammaln(n / 2.0) - 0.5 * math.log(math.pi) - gammaln((n - 1) / 2.0))


def cap_fraction(alpha: float, n: int) -> float:
    """P(theta <= alpha) for a uniform random direction on S^{n-1}."""
    if alpha <= 0.0:
        return 0.0
    if alpha >= math.pi:
        return 1.0
    a = (n - 1) / 2.0
    t = math.cos(alpha)
    x = (1.0 + t) / 2.0
    return float(1.0 - betainc(a, a, x))


def alpha_for_fraction(f: float, n: int) -> float:
    """Inverse of :func:`cap_fraction` (F(alpha) = f)."""
    if not (0.0 < f < 1.0):
        raise ValueError("fraction must be in (0,1)")
    a = (n - 1) / 2.0
    x = float(betaincinv(a, a, 1.0 - f))
    t = 2.0 * x - 1.0
    t = min(max(t, -1.0), 1.0)
    return math.acos(t)


def k_prime(
    k: int,
    m: int,
    n: int,
    r: float,
    delta_alpha_ratio: float = 1.0,
) -> float:
    """top-k' needed so that the perturbed query's top-k' contains the original
    top-k (Theorem 1).

    Parameters
    ----------
    k : int      original top-k
    m : int      database size
    n : int      embedding dimension
    r : float    query perturbation radius (e.g. 0.02, 0.033)
    delta_alpha_ratio : float
        Delta alpha_k = delta_alpha_ratio * r (paper/RemoteRAG use ~1.0;
        Table III of the paper is only reproduced with ~1.2, see SPEC.md).
    """
    if not (0 < k < m):
        raise ValueError("need 0 < k < m")
    alpha_k = alpha_for_fraction(k / m, n)
    alpha_kp = alpha_k + delta_alpha_ratio * r
    return k + m * (cap_fraction(alpha_kp, n) - cap_fraction(alpha_k, n))


def k_prime_table(
    m: int = 100_000,
    dims: tuple[int, ...] = (768, 1536),
    radii: tuple[float, ...] = (0.033, 0.02),
    ks: tuple[int, ...] = (5, 10, 15, 20),
    delta_alpha_ratio: float = 1.0,
) -> list[dict]:
    """Recompute the paper's Table III layout."""
    rows = []
    for n in dims:
        for k in ks:
            row = {"n": n, "k": k}
            for r in radii:
                kp = k_prime(k, m, n, r, delta_alpha_ratio=delta_alpha_ratio)
                row[f"kprime@{r}"] = kp
                row[f"ratio@{r}"] = kp / k
            rows.append(row)
    return rows


def monte_carlo_k_prime(
    n: int,
    m: int,
    k: int,
    r: float,
    n_queries: int = 20,
    seed: int = 0,
    chunk: int = 8192,
) -> dict:
    """Empirically measure the top-k' expansion by simulating real vectors.

    Random unit database vectors and random unit queries are drawn; the query is
    perturbed by a random direction of norm r (``e~ = e + r * direction``).  For
    each query we find the smallest k' such that the top-k' under the perturbed
    query contains the original top-k set.  This validates Theorem 1 without
    relying on the angular approximation.

    Returns dict with mean/median/std of the measured k' and the theoretical one.
    """
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n_queries, n))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    direction = rng.standard_normal((n_queries, n))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    q_tilde = q + r * direction  # not re-normalised (matches DistanceDP)

    measured = np.empty(n_queries, dtype=np.int64)
    for j in range(n_queries):
        # database distances from the plaintext query
        d_true = np.empty(m)
        d_pert = np.empty(m)
        start = 0
        while start < m:
            stop = min(start + chunk, m)
            E = rng.standard_normal((stop - start, n))
            E /= np.linalg.norm(E, axis=1, keepdims=True)
            d_true[start:stop] = np.linalg.norm(E - q[j], axis=1)
            d_pert[start:stop] = np.linalg.norm(E - q_tilde[j], axis=1)
            start = stop
        topk = np.argpartition(d_true, k)[:k]
        threshold = d_pert[topk].max()
        # k' = rank of the worst true-positive under the perturbed query
        measured[j] = int((d_pert < threshold).sum() + 1)

    theoretical = k_prime(k, m, n, r, delta_alpha_ratio=1.0)
    return {
        "n": n,
        "m": m,
        "k": k,
        "r": r,
        "n_queries": n_queries,
        "mean": float(measured.mean()),
        "median": float(np.median(measured)),
        "std": float(measured.std()),
        "theoretical_dalpha_eq_r": float(theoretical),
        "measured": measured.tolist(),
    }
