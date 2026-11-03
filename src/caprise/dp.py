"""DistanceDP -- the query-side differential-privacy mechanism (RemoteRAG).

The paper (Sec. III-B2) adopts DistanceDP from
"Cheng et al., RemoteRAG: A Privacy-Preserving LLM Cloud RAG Service,
arXiv:2412.12775", which defines (n, eps)-DistanceDP with the noise mechanism:

    radial component    r ~ Gamma(shape=n, scale=1/eps),   E[r] = n/eps
    direction           uniform on the unit (n-1)-sphere
    perturbed embedding e~_q = e_q + r * direction

For normalised embeddings typical r is 0.02-0.1 (RemoteRAG Sec. 3.2.1), which is
what the paper's Table III uses as the "radius" parameter.

Two modes are provided:
  * ``radius="gamma"`` + ``eps``: the actual DP mechanism (random radius).
  * ``radius=r`` (float): fixed-radius perturbation, used by Table III and the
    privacy/utility ablation experiments.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["sample_radius", "distance_dp", "epsilon_for_radius"]


def sample_radius(dim: int, eps: float, rng: np.random.Generator | None = None) -> float:
    """r ~ Gamma(shape=dim, scale=1/eps); mean = dim/eps."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    rng = np.random.default_rng() if rng is None else rng
    return float(rng.gamma(shape=dim, scale=1.0 / eps))


def epsilon_for_radius(dim: int, r: float) -> float:
    """Inverse of the mean-radius relation: eps = dim / r_bar."""
    if r <= 0:
        raise ValueError("r must be positive")
    return dim / r


def distance_dp(
    e_q: np.ndarray,
    radius: float | str = "gamma",
    eps: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Return (perturbed query embedding, realised radius r).

    Parameters
    ----------
    e_q : (d,) query embedding (assumed L2-normalised in RAG settings)
    radius : "gamma" | float
        "gamma" samples r ~ Gamma(d, 1/eps) (requires eps); a float fixes r.
    eps : float, optional
        privacy budget for the Gamma mode (RemoteRAG suggests eps in [10n, 50n]).
    rng : numpy Generator, optional
    """
    e_q = np.asarray(e_q, dtype=np.float64)
    if e_q.ndim != 1:
        raise ValueError("e_q must be 1-D")
    d = e_q.shape[0]
    rng = np.random.default_rng() if rng is None else rng

    if isinstance(radius, str):
        if radius != "gamma":
            raise ValueError("radius must be 'gamma' or a positive float")
        if eps is None:
            raise ValueError("eps is required for radius='gamma'")
        r = sample_radius(d, eps, rng)
    else:
        r = float(radius)
        if r <= 0:
            raise ValueError("radius must be positive")

    direction = rng.standard_normal(d)
    norm = np.linalg.norm(direction)
    while norm == 0.0:
        direction = rng.standard_normal(d)
        norm = np.linalg.norm(direction)
    direction /= norm
    return e_q + r * direction, r
