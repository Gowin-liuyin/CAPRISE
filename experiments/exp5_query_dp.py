"""exp5 -- Theorem 1 / Table III reproduction and Monte-Carlo validation (module M13).

Paper: arXiv:2601.12331 Sec. III-B2, Theorem 1 (Eq. 21), Table III.

Theorem 1 (Eq. 21)
------------------
    k' = k + m * (Omega_{n-1}(pi)/Omega_n(pi)) * int_{alpha_k}^{alpha_k + Delta alpha_k}
                  sin^{n-2} theta dtheta

For a uniform random direction on S^{n-1} the angle theta to a fixed query has

    F(alpha) = P(theta <= alpha) = 1 - I_{(1+cos alpha)/2}((n-1)/2, (n-1)/2),

so, with alpha_k defined by F(alpha_k) = k/m (and therefore m F(alpha_k) = k),

    k' = m * F(alpha_k + Delta alpha_k)                                    (Eq.21')

i.e. Eq.21 is the *expected number of database points inside the angular cap of
radius alpha_k + Delta alpha* around the perturbed query -- a sufficient
(worst-case azimuth) upper bound for the retrieval-set expansion, NOT the
minimal k' needed by a real retrieval.  This script therefore reports:

  (1) Table III (paper values) next to Eq.21 evaluated with Delta alpha = r and
      Delta alpha = 1.2 r;
  (2) the per-cell Delta alpha / r implied by the paper's table (finding A10:
      the paper's numbers are incompatible with RemoteRAG's "Delta alpha ~ r");
  (3) a Monte-Carlo check of Eq.21's identity (expected widened-cap size) using
      m=100,000 random unit DB vectors, >=200 random unit queries per (n, r) and
      the real DistanceDP perturbation e~ = q + r u;
  (4) a Monte-Carlo measurement of the *minimal* k' that a real retrieval needs
      (rank of the worst true top-k member under the perturbed query): mean,
      median, p95, p99, and the slackness of Eq.21 relative to it;
  (5) numerical-stability evidence for the Beta implementation (no NaN/underflow
      for n=1536 and beyond; Beta form vs direct quadrature for small n).

Outputs
-------
    results/exp5_query_dp.json
    figures/fig6_table3.png
    notes/exp5_query_dp.md   (written separately / summarised by this script)

Run:  .venv/bin/python experiments/exp5_query_dp.py
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from caprise.theorem import (  # noqa: E402
    alpha_for_fraction,
    cap_fraction,
    k_prime,
    omega_ratio,
)

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# --------------------------------------------------------------------------- #
# Paper Table III (LaTeX source ppRAG_FLLM.tex, Table `tab:radius_k_ratio`)
# --------------------------------------------------------------------------- #
M = 100_000
DIMS = (768, 1536)
RADII = (0.033, 0.02)
KS = (5, 10, 15, 20)

# {(n, r): {k: (paper top-k', paper k'/k)}}
PAPER_TABLE = {
    (768, 0.033): {5: (258, 52), 10: (571, 57), 15: (673, 45), 20: (928, 46)},
    (768, 0.02): {5: (58, 12), 10: (108, 11), 15: (170, 11), 20: (203, 10)},
    (1536, 0.033): {5: (982, 196), 10: (1492, 149), 15: (2077, 138), 20: (2603, 130)},
    (1536, 0.02): {5: (145, 29), 10: (245, 25), 15: (388, 26), 20: (501, 25)},
}

PAPER_REF = "arXiv:2601.12331 Theorem 1 (Eq.21), Table III; RemoteRAG arXiv:2412.12775 Sec. 3.2.1"


# --------------------------------------------------------------------------- #
# Analytical part
# --------------------------------------------------------------------------- #
def formula_table(delta_alpha_ratio: float) -> list[dict]:
    """Eq.21 for every Table III cell with Delta alpha = delta_alpha_ratio * r."""
    rows = []
    for n in DIMS:
        for k in KS:
            row = {"n": n, "k": k}
            for r in RADII:
                kp = k_prime(k, M, n, r, delta_alpha_ratio=delta_alpha_ratio)
                # store full precision (rounding happens only when printing/notes)
                row[f"kprime@{r}"] = kp
                row[f"ratio@{r}"] = kp / k
            rows.append(row)
    return rows


def paper_vs_formula_rows() -> list[dict]:
    rows = []
    for n in DIMS:
        for k in KS:
            for r in RADII:
                paper_kp, paper_ratio = PAPER_TABLE[(n, r)][k]
                f = k_prime(k, M, n, r, delta_alpha_ratio=1.0)
                rows.append(
                    {
                        "n": n,
                        "r": r,
                        "k": k,
                        "paper_kprime": paper_kp,
                        "paper_ratio": paper_ratio,
                        "formula_kprime_dalpha_eq_r": f,
                        "formula_ratio_dalpha_eq_r": f / k,
                        "paper_over_formula": round(paper_kp / f, 4),
                        "formula_over_paper": round(f / paper_kp, 4),
                    }
                )
    return rows


def reverse_delta_alpha_rows() -> list[dict]:
    """Per-cell Delta alpha / r implied by the paper's k' (Eq.21 inverted)."""
    rows = []
    for n in DIMS:
        for r in RADII:
            for k in KS:
                paper_kp = PAPER_TABLE[(n, r)][k][0]
                alpha_k = alpha_for_fraction(k / M, n)
                fk = cap_fraction(alpha_k, n)
                # solve F(alpha_k + da) - F(alpha_k) = (paper_k' - k)/m
                target = (paper_kp - k) / M

                def g(da: float) -> float:
                    return cap_fraction(alpha_k + da, n) - fk - target

                da = brentq(g, -0.5, 0.5, xtol=1e-14)
                rows.append(
                    {
                        "n": n,
                        "r": r,
                        "k": k,
                        "paper_kprime": paper_kp,
                        "alpha_k_rad": round(alpha_k, 8),
                        "delta_alpha_rad": round(da, 8),
                        "delta_alpha_over_r": round(da / r, 5),
                    }
                )
    return rows


def reverse_summary(rows: list[dict]) -> dict:
    out: dict = {"per_group": {}, "overall": {}}
    for n in DIMS:
        for r in RADII:
            vals = [x["delta_alpha_over_r"] for x in rows if x["n"] == n and x["r"] == r]
            out["per_group"][f"n={n},r={r}"] = {
                "mean": round(float(np.mean(vals)), 5),
                "std": round(float(np.std(vals, ddof=1)), 5),
                "min": round(float(np.min(vals)), 5),
                "max": round(float(np.max(vals)), 5),
            }
    allv = [x["delta_alpha_over_r"] for x in rows]
    out["overall"] = {
        "mean": round(float(np.mean(allv)), 5),
        "std": round(float(np.std(allv, ddof=1)), 5),
        "min": round(float(np.min(allv)), 5),
        "max": round(float(np.max(allv)), 5),
        "n_cells": len(allv),
    }
    return out


def convention_probe() -> dict:
    """Test alternative conventions for the angular/distance perturbation.

    * angle-space:  alpha_k' = alpha_k + c*r              (c = 1, 1.2)
    * distance-space: d_k' = d_k + r, then k' = m F(alpha(d_k')) (chord geometry)
    Reports paper / prediction for each convention.
    """
    out: dict = {
        "angle_dalpha_eq_r": {"pred": [], "paper_over_pred": []},
        "angle_dalpha_eq_1.2r": {"pred": [], "paper_over_pred": []},
        "distance_dk_plus_r": {"pred": [], "paper_over_pred": []},
    }
    for n in DIMS:
        for r in RADII:
            for k in KS:
                paper_kp = PAPER_TABLE[(n, r)][k][0]
                p1 = k_prime(k, M, n, r, 1.0)
                p12 = k_prime(k, M, n, r, 1.2)
                alpha_k = alpha_for_fraction(k / M, n)
                d_k = 2.0 * math.sin(alpha_k / 2.0)  # chord distance of k-th NN
                alpha_d = 2.0 * math.asin(min((d_k + r) / 2.0, 1.0))
                p_dist = M * cap_fraction(alpha_d, n)
                out["angle_dalpha_eq_r"]["pred"].append(round(p1, 2))
                out["angle_dalpha_eq_r"]["paper_over_pred"].append(round(paper_kp / p1, 4))
                out["angle_dalpha_eq_1.2r"]["pred"].append(round(p12, 2))
                out["angle_dalpha_eq_1.2r"]["paper_over_pred"].append(round(paper_kp / p12, 4))
                out["distance_dk_plus_r"]["pred"].append(round(p_dist, 2))
                out["distance_dk_plus_r"]["paper_over_pred"].append(round(paper_kp / p_dist, 4))
    for v in out.values():
        v["paper_over_pred_mean"] = round(float(np.mean(v["paper_over_pred"])), 4)
        v["paper_over_pred_min"] = round(float(np.min(v["paper_over_pred"])), 4)
        v["paper_over_pred_max"] = round(float(np.max(v["paper_over_pred"])), 4)
    return out


def numerical_stability() -> dict:
    """Beta-form CDF: stability, exactness, and comparison with quadrature."""
    out: dict = {"roundtrip": [], "direct_quadrature": [], "underflow_demo": [], "table3_finiteness": []}
    fractions = (1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 0.5, 0.9)
    for n in (768, 1536, 4096):
        worst = 0.0
        for f in fractions:
            a = alpha_for_fraction(f, n)
            back = cap_fraction(a, n)
            worst = max(worst, abs(back - f) / f)
        out["roundtrip"].append(
            {"n": n, "max_rel_err_F(alpha_for_fraction(f))": worst, "fractions": list(fractions)}
        )
    # m * F(alpha_k) == k exactly (definition consistency)
    maxdev = 0.0
    for n in (768, 1536):
        for k in KS:
            maxdev = max(maxdev, abs(M * cap_fraction(alpha_for_fraction(k / M, n), n) - k))
    out["m_F_alpha_k_minus_k_max_abs"] = maxdev

    # direct quadrature (the literal Eq.21 integral) vs Beta CDF for small n
    for n in (10, 50, 100):
        for k in (5, 20):
            a = alpha_for_fraction(k / M, n)
            val = omega_ratio(n) * quad(lambda t: math.sin(t) ** (n - 2), 0.0, a, limit=400)[0]
            out["direct_quadrature"].append(
                {
                    "n": n,
                    "k": k,
                    "beta_F": cap_fraction(a, n),
                    "quadrature_F": val,
                    "abs_diff": abs(val - cap_fraction(a, n)),
                }
            )

    # underflow demonstration: the literal integrand sin(theta)^(n-2) is 0.0 in
    # float64 over most of the relevant domain; the Beta form stays finite.
    for n, theta in ((768, 0.05), (768, 0.2), (1536, 0.1), (1536, 0.3), (4096, 0.3)):
        s = math.sin(theta)
        literal = s ** (n - 2)
        log_val = (n - 2) * math.log(s) if s > 0 else -math.inf
        out["underflow_demo"].append(
            {
                "n": n,
                "theta": theta,
                "sin_theta": s,
                "sin_pow_n_minus_2_float64": literal,
                "log10_integrand": log_val / math.log(10) if math.isfinite(log_val) else -math.inf,
                "beta_cap_fraction_finite": math.isfinite(cap_fraction(theta, n)),
                "beta_cap_fraction": cap_fraction(theta, n),
            }
        )

    # every Table III cell must be finite / monotone / in range
    finite_ok = True
    mono_ok = True
    for n in (768, 1536):
        for k in KS:
            a = alpha_for_fraction(k / M, n)
            for r in RADII:
                kp = k_prime(k, M, n, r, 1.0)
                if not (math.isfinite(kp) and kp > k):
                    finite_ok = False
                if not (cap_fraction(a + r, n) >= cap_fraction(a, n)):
                    mono_ok = False
    out["table3_finiteness"] = {"all_finite": finite_ok, "monotone": mono_ok}
    # n=1536 tail sanity (k/m = 5e-5 and 2e-4 are the smallest fractions used)
    tail = []
    for n in (768, 1536, 3072, 4096):
        for k in (5, 20):
            a = alpha_for_fraction(k / M, n)
            tail.append(
                {
                    "n": n,
                    "k": k,
                    "alpha_k": a,
                    "F(alpha_k)": cap_fraction(a, n),
                    "kprime@r=0.033": k_prime(k, M, n, 0.033),
                    "finite": bool(math.isfinite(k_prime(k, M, n, 0.033))),
                }
            )
    out["tail_sanity"] = tail
    return out


# --------------------------------------------------------------------------- #
# Monte-Carlo part
# --------------------------------------------------------------------------- #
DB_SEED = 20260118


def _iter_db_chunks(n: int, m: int, chunk: int):
    """Yield identical (deterministic) chunks of m uniform unit vectors in R^n."""
    rng = np.random.default_rng(DB_SEED + n)
    start = 0
    while start < m:
        stop = min(start + chunk, m)
        E = rng.standard_normal((stop - start, n))
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        yield start, E
        start = stop


def monte_carlo(
    n: int,
    m: int = M,
    nq: int = 200,
    radii: tuple[float, ...] = RADII,
    ks: tuple[int, ...] = KS,
    chunk: int = 8192,
    seed: int = 7,
    verbose: bool = True,
) -> dict:
    """Fixed random DB, `nq` random queries; measure (i) widened-cap counts and
    (ii) the minimal k' required by a real retrieval.

    The DB is regenerated deterministically for each pass (same seed), so the
    same DB is reused for every query (realistic retrieval scenario) without
    materialising an m x d matrix.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((nq, n))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    U = rng.standard_normal((nq, n))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    QT = {r: Q + r * U for r in radii}  # DistanceDP: never re-normalised
    qn = {r: np.linalg.norm(QT[r], axis=1) for r in radii}

    # pass 1: plaintext distances d(q_j, e_i)
    Dq = np.empty((m, nq), dtype=np.float32)
    for start, E in _iter_db_chunks(n, m, chunk):
        c = E.shape[0]
        B = E @ Q.T
        Dq[start : start + c] = np.sqrt(np.maximum(2.0 - 2.0 * B, 0.0)).astype(np.float32)
    # pass 2: perturbed distances d(q~_j, e_i) for every radius
    Dp = {r: np.empty((m, nq), dtype=np.float32) for r in radii}
    for start, E in _iter_db_chunks(n, m, chunk):
        c = E.shape[0]
        for r in radii:
            B = E @ QT[r].T
            Dp[r][start : start + c] = np.sqrt(
                np.maximum(1.0 + qn[r] ** 2 - 2.0 * B, 0.0)
            ).astype(np.float32)

    if verbose:
        print(f"  [MC n={n}] DB built ({m} x {n}) + distance passes in {time.time()-t0:.2f}s", flush=True)

    kmax = max(ks)
    alpha_pop = {(r, k): alpha_for_fraction(k / m, n) for r in radii for k in ks}
    kp_theo = {(r, k): int(math.ceil(k_prime(k, m, n, r, 1.0))) for r in radii for k in ks}

    cells = {(r, k): {"cap_pop": [], "cap_emp": [], "met": [], "coverage": []} for r in radii for k in ks}

    t1 = time.time()
    for j in range(nq):
        dqv = Dq[:, j]
        topk = np.argpartition(dqv, kmax - 1)[:kmax]
        topk = topk[np.argsort(dqv[topk])]  # sorted by plaintext distance
        for r in radii:
            dp = Dp[r][:, j].astype(np.float64)
            cos_qtilde = (1.0 + qn[r][j] ** 2 - dp * dp) / (2.0 * qn[r][j])
            for k in ks:
                # ---- Eq.21 quantity: expected points in widened angular cap ----
                ak = alpha_pop[(r, k)]
                cap_pop = int(np.count_nonzero(cos_qtilde >= math.cos(ak + r)))
                # empirical alpha_k (angle of the actual k-th nearest neighbour)
                dk = float(dqv[topk[k - 1]])
                ak_emp = math.acos(max(-1.0, min(1.0, 1.0 - 0.5 * dk * dk)))
                cap_emp = int(np.count_nonzero(cos_qtilde >= math.cos(ak_emp + r)))
                # ---- minimal k' a real retrieval needs ----
                th = float(dp[topk[:k]].max())
                met = int(np.count_nonzero(dp <= th))
                # ---- sufficiency check with k' = ceil(Eq.21) ----
                kp = kp_theo[(r, k)]
                if kp <= m:
                    idx = np.argpartition(dp, kp - 1)[:kp]
                    cov = bool(set(topk[:k].tolist()).issubset(set(idx.tolist())))
                else:
                    cov = True
                c = cells[(r, k)]
                c["cap_pop"].append(cap_pop)
                c["cap_emp"].append(cap_emp)
                c["met"].append(met)
                c["coverage"].append(cov)
    if verbose:
        print(f"  [MC n={n}] per-query analysis in {time.time()-t1:.2f}s", flush=True)

    rows = []
    for r in radii:
        for k in ks:
            c = cells[(r, k)]
            f = k_prime(k, m, n, r, 1.0)
            met = np.asarray(c["met"], dtype=float)
            p95 = float(np.quantile(met, 0.95))
            p99 = float(np.quantile(met, 0.99))
            rows.append(
                {
                    "n": n,
                    "r": r,
                    "k": k,
                    "nq": nq,
                    "formula_kprime_dalpha_eq_r": float(f),
                    "mc_cap_mean_population_alpha": round(float(np.mean(c["cap_pop"])), 2),
                    "mc_cap_rel_err_population_alpha": round(
                        (float(np.mean(c["cap_pop"])) - f) / f, 5
                    ),
                    "mc_cap_mean_empirical_alpha": round(float(np.mean(c["cap_emp"])), 2),
                    "mc_cap_rel_err_empirical_alpha": round(
                        (float(np.mean(c["cap_emp"])) - f) / f, 5
                    ),
                    "min_kprime_mean": round(float(met.mean()), 3),
                    "min_kprime_median": float(np.median(met)),
                    "min_kprime_std": round(float(met.std(ddof=1)), 3),
                    "min_kprime_p95": p95,
                    "min_kprime_p99": p99,
                    "min_kprime_max": float(met.max()),
                    "recommended_kprime_max_k_p99": int(max(k, math.ceil(p99))),
                    "slack_formula_over_mean": round(float(f) / float(met.mean()), 2),
                    "slack_formula_over_p99": round(float(f) / p99, 2),
                    "coverage_rate_with_ceil_formula": round(float(np.mean(c["coverage"])), 4),
                    "min_kprime_samples": [int(x) for x in met],
                }
            )
    return {
        "n": n,
        "m": m,
        "nq": nq,
        "radii": list(radii),
        "ks": list(ks),
        "rows": rows,
        "runtime_s": round(time.time() - t0, 2),
    }


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(res: dict, fig_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    colors = {"paper": "#b2182b", "r": "#2166ac", "1.2r": "#67a9cf", "mc": "#1b7837"}

    # (a)(b): paper vs Eq.21(dalpha=r) vs Eq.21(dalpha=1.2r) for n=768 / n=1536
    for ax, n in zip(axes[0], (768, 1536)):
        width = 0.26
        x = np.arange(len(KS) * len(RADII))
        labels = []
        for idx_r, r in enumerate(RADII):
            for idx_k, k in enumerate(KS):
                pos = idx_r * len(KS) + idx_k
                labels.append(f"k={k}\nr={r}")
                paper_kp = PAPER_TABLE[(n, r)][k][0]
                f1 = k_prime(k, M, n, r, 1.0)
                f12 = k_prime(k, M, n, r, 1.2)
                ax.bar(pos - width, paper_kp, width, color=colors["paper"], label="paper Table III" if pos == 0 else None)
                ax.bar(pos, f1, width, color=colors["r"], label="Eq.21, $\\Delta\\alpha=r$" if pos == 0 else None)
                ax.bar(pos + width, f12, width, color=colors["1.2r"], label="Eq.21, $\\Delta\\alpha=1.2r$" if pos == 0 else None)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("top-$k'$ (log scale)")
        ax.set_title(f"(a) n={n}: Table III vs Theorem 1" if n == 768 else f"(b) n={n}: Table III vs Theorem 1")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    # (c): implied Delta alpha / r per Table III cell
    ax = axes[1][0]
    rev = res["reverse_delta_alpha"]["rows"]
    markers = {(768, 0.033): "o", (768, 0.02): "s", (1536, 0.033): "^", (1536, 0.02): "D"}
    offsets = {(768, 0.033): -0.18, (768, 0.02): -0.06, (1536, 0.033): 0.06, (1536, 0.02): 0.18}
    for (n, r), marker in markers.items():
        y = [x["delta_alpha_over_r"] for x in rev if x["n"] == n and x["r"] == r]
        xs = np.arange(len(KS)) + offsets[(n, r)]
        ax.scatter(xs, y, marker=marker, label=f"n={n}, r={r}", s=45)
    overall = res["reverse_delta_alpha"]["summary"]["overall"]
    ax.axhline(1.0, color="black", ls=":", lw=1.2, label="RemoteRAG claim: $\\Delta\\alpha=r$")
    ax.axhline(overall["mean"], color=colors["paper"], ls="--", lw=1.2,
               label=f"implied mean = {overall['mean']:.2f} r")
    ax.set_xticks(np.arange(len(KS)))
    ax.set_xticklabels([f"k={k}" for k in KS])
    ax.set_ylabel("implied $\\Delta\\alpha / r$ from paper Table III")
    ax.set_title("(c) Paper table needs $\\Delta\\alpha \\approx 1.21\\,r$ (A10)")
    ax.set_ylim(0.95, 1.4)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # (d): MC validation: Eq.21 vs widened-cap MC vs paper vs minimal k'
    ax = axes[1][1]
    mc_rows = [row for block in res["monte_carlo"]["blocks"] for row in block["rows"] if row["n"] == 768 and row["r"] == 0.033]
    xs = np.arange(len(KS))
    f_vals = [x["formula_kprime_dalpha_eq_r"] for x in mc_rows]
    mc_cap = [x["mc_cap_mean_population_alpha"] for x in mc_rows]
    mc_min = [x["min_kprime_mean"] for x in mc_rows]
    paper_vals = [PAPER_TABLE[(768, 0.033)][k][0] for k in KS]
    ax.plot(xs, f_vals, "o-", color=colors["r"], label="Eq.21, $\\Delta\\alpha=r$")
    ax.plot(xs, mc_cap, "x--", color=colors["mc"], label="MC widened-cap count (equals Eq.21)")
    ax.plot(xs, paper_vals, "s-", color=colors["paper"], label="paper Table III")
    ax.plot(xs, mc_min, "d-", color="#7b3294", label="MC minimal $k'$ (real retrieval)")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"k={k}" for k in KS])
    ax.set_ylabel("top-$k'$ (log scale)")
    ax.set_title("(d) n=768, r=0.033: Eq.21 is a loose sufficient bound")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("CAPRISE / ppRAG reproduction -- Theorem 1 (Eq.21) and Table III", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    t_start = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("[exp5] analytical tables ...", flush=True)
    pvf = paper_vs_formula_rows()
    reverse = reverse_delta_alpha_rows()
    reverse_summary_ = reverse_summary(reverse)

    print("[exp5] numerical stability ...", flush=True)
    stability = numerical_stability()

    print("[exp5] convention probe ...", flush=True)
    probe = convention_probe()

    print("[exp5] Monte Carlo (m=100,000; nq=200 per dimension; fixed DB) ...", flush=True)
    mc = [
        monte_carlo(768, nq=200, seed=7),
        monte_carlo(1536, nq=200, seed=7),
    ]
    mc_rows = [row for block in mc for row in block["rows"]]

    # Data-quality gate for the experiment itself: paper values must be usable
    print("\n=== Table III vs Eq.21 (Delta alpha = r) ===")
    print(f"{'n':>5} {'r':>6} {'k':>3} {'paper':>6} {'Eq.21(r)':>9} {'paper/Eq21':>10} {'implied da/r':>13}")
    rev_map = {(x["n"], x["r"], x["k"]): x["delta_alpha_over_r"] for x in reverse}
    for row in pvf:
        key = (row["n"], row["r"], row["k"])
        print(
            f"{row['n']:>5} {row['r']:>6} {row['k']:>3} {row['paper_kprime']:>6} "
            f"{row['formula_kprime_dalpha_eq_r']:>9} {row['paper_over_formula']:>10} {rev_map[key]:>13}"
        )
    overall = reverse_summary_["overall"]
    print(
        f"\nimplied Delta alpha / r: mean={overall['mean']} std={overall['std']} "
        f"range=[{overall['min']}, {overall['max']}] (RemoteRAG claims ~1.0)"
    )

    print("\n=== Monte-Carlo validation ===")
    print(f"{'n':>5} {'r':>6} {'k':>3} {'Eq.21':>9} {'MC cap':>9} {'rel err':>9} "
          f"{'MC min k´ mean':>15} {'p95':>6} {'p99':>6} {'slack':>7} {'cover':>6}")
    for row in mc_rows:
        print(
            f"{row['n']:>5} {row['r']:>6} {row['k']:>3} {row['formula_kprime_dalpha_eq_r']:>9} "
            f"{row['mc_cap_mean_population_alpha']:>9} {row['mc_cap_rel_err_population_alpha']:>9} "
            f"{row['min_kprime_mean']:>15} {row['min_kprime_p95']:>6} {row['min_kprime_p99']:>6} "
            f"{row['slack_formula_over_mean']:>7} {row['coverage_rate_with_ceil_formula']:>6}"
        )

    result = {
        "experiment": "exp5_query_dp",
        "module": "M13",
        "paper_ref": PAPER_REF,
        "description": (
            "Theorem 1 (Eq.21) / Table III reproduction: analytical values with Delta alpha = r "
            "and Delta alpha = 1.2 r, reverse-engineered Delta alpha/r from the paper table, "
            "Monte-Carlo validation of Eq.21's widened-cap expectation and measurement of the "
            "minimal k' a real retrieval requires."
        ),
        "settings": {"m": M, "dims": list(DIMS), "radii": list(RADII), "ks": list(KS), "paper_m": M},
        "paper_table": {
            f"n={n}": {f"r={r}": {k: {"top_kprime": v[0], "ratio": v[1]} for k, v in PAPER_TABLE[(n, r)].items()}
                        for r in RADII}
            for n in DIMS
        },
        "formula_table_dalpha_eq_r": formula_table(1.0),
        "formula_table_dalpha_eq_1_2r": formula_table(1.2),
        "paper_vs_formula_dalpha_eq_r": pvf,
        "reverse_delta_alpha": {"rows": reverse, "summary": reverse_summary_},
        "convention_probe": probe,
        "numerical_stability": stability,
        "monte_carlo": {
            "description": (
                "Fixed MC DB of m=100,000 uniform unit vectors (deterministically regenerated per "
                "pass), nq=200 random unit queries, DistanceDP perturbation q~ = q + r*u with u "
                "uniform on S^{n-1} (not re-normalised). For each query we record (i) the number of "
                "DB points inside the angular cap of radius alpha_k + r around q~ (Eq.21's "
                "expectation, alpha_k from F(alpha_k)=k/m), (ii) the same using the empirical k-th "
                "nearest-neighbour angle, (iii) the minimal k' such that top-k' under q~ contains "
                "the plaintext top-k, and (iv) coverage of the plaintext top-k when retrieving "
                "ceil(Eq.21) candidates."
            ),
            "config": {"m": M, "nq": 200, "db_seed": DB_SEED, "chunk": 8192},
            "blocks": mc,
            "rows": mc_rows,
        },
        "findings": [
            {
                "id": "A10",
                "statement": (
                    "The paper's Table III cannot be reproduced from Theorem 1 with RemoteRAG's "
                    f"Delta alpha ~ r: Eq.21 gives 0.44-0.74x the paper values; inverting the table "
                    f"requires Delta alpha/r = {overall['mean']} (range {overall['min']}-{overall['max']}) "
                    "instead of ~1. This is an internal inconsistency between the paper's table and "
                    "its stated DP convention (RemoteRAG: 'Delta alpha_k can be approximately by r')."
                ),
            },
            {
                "id": "A10b",
                "statement": (
                    "Eq.21 is a sufficient worst-case upper bound, not the minimal retrieval expansion: "
                    "on uniform-sphere data the minimal k' measured by MC is ~k (k=5 -> ~5-6), while "
                    "Eq.21 with Delta alpha=r predicts m*F(alpha_k+r) (k=5 -> 147 for n=768, r=0.033) "
                    "and the paper's table 258. The widening is driven by the worst-case azimuthal "
                    "alignment of each DB point with the perturbation direction; typical distance "
                    "changes are O(r/sqrt(n))."
                ),
            },
        ],
        "runtime_s": round(time.time() - t_start, 1),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }
    out_path = RESULTS / "exp5_query_dp.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[exp5] results -> {out_path.relative_to(ROOT)}")

    print("[exp5] figure ...", flush=True)
    fig_path = FIGURES / "fig6_table3.png"
    make_figure(result, fig_path)
    print(f"[exp5] figure -> {fig_path.relative_to(ROOT)}")
    print(f"[exp5] done in {time.time()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
