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
      for n=1536 and beyond; Beta form vs direct quadrature for small n);
  (6) an enumeration of four Delta-alpha conventions for all 16 Table III cells
      -- (a) Delta alpha = r, (b) Delta alpha = 2 asin(r/2), (c) Euclidean query
      shift alpha' = 2 asin((2 sin(alpha_k/2)+r)/2), (d) per-cell inversion of the
      paper's k' -- showing that only the undeclared convention (d), Delta alpha ~
      1.2 r, reproduces the paper's numbers;
  (7) a cross-check of the same formula against the highlighted example of the
      cited implementation (RemoteRAG arXiv:2412.12775 Sec. 5.4 / Fig. 5(c):
      k=5, r=0.03, n=768 -> k'=160), which is consistent with Delta alpha = r.

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


# --------------------------------------------------------------------------- #
# Delta-alpha conventions: enumerate every reading of Delta alpha and test which
# one can reproduce the paper's Table III.
# --------------------------------------------------------------------------- #
DELTA_ALPHA_CONVENTIONS = {
    "a_dalpha_eq_r": (
        "Delta alpha_k = r (RemoteRAG arXiv:2412.12775 Sec. 3.2.2: 'the perturbed angle "
        "Delta alpha_k between e_k and e_k' can be approximately by r')"
    ),
    "b_dalpha_chord": (
        "Delta alpha_k = 2*asin(r/2): a Euclidean displacement of norm r on the unit sphere "
        "maps to this angular deviation (equals r + r^3/24 + ...)"
    ),
    "c_alpha_euclid_expansion": (
        "alpha_k' = 2*asin((2*sin(alpha_k/2)+r)/2): the chord distance of the k-th nearest "
        "neighbour (2 sin(alpha_k/2)) grows by the Euclidean radius r; Delta alpha_k = alpha_k' - alpha_k"
    ),
    "d_inverse_paper": (
        "Delta alpha_k solved per cell from Eq.21 so that the formula equals the paper's top-k' "
        "(the convention the paper's table would require; it is NOT stated in the paper or in RemoteRAG)"
    ),
}


def delta_alpha_conventions() -> dict:
    """Enumerate four Delta-alpha conventions for all 16 Table III cells."""
    rows: list[dict] = []
    for n in DIMS:
        for r in RADII:
            for k in KS:
                paper_kp = PAPER_TABLE[(n, r)][k][0]
                alpha_k = alpha_for_fraction(k / M, n)
                fk = cap_fraction(alpha_k, n)

                # (a) Delta alpha = r
                da_a = r
                kp_a = k + M * (cap_fraction(alpha_k + da_a, n) - fk)

                # (b) Delta alpha = 2 asin(r/2)
                da_b = 2.0 * math.asin(r / 2.0)
                kp_b = k + M * (cap_fraction(alpha_k + da_b, n) - fk)

                # (c) Euclidean expansion: chord of the k-th NN grows by r
                chord_k = 2.0 * math.sin(alpha_k / 2.0)
                alpha_c = 2.0 * math.asin(min((chord_k + r) / 2.0, 1.0))
                da_c = alpha_c - alpha_k
                kp_c = k + M * (cap_fraction(alpha_c, n) - fk)

                # (d) per-cell Delta alpha required by the paper's k'
                target = (paper_kp - k) / M
                da_d = brentq(
                    lambda x: cap_fraction(alpha_k + x, n) - fk - target,
                    -0.5,
                    0.5,
                    xtol=1e-14,
                )
                kp_d = k + M * (cap_fraction(alpha_k + da_d, n) - fk)

                rows.append(
                    {
                        "n": n,
                        "r": r,
                        "k": k,
                        "paper_kprime": paper_kp,
                        "alpha_k_rad": alpha_k,
                        "chord_k": chord_k,
                        "delta_alpha_a_rad": da_a,
                        "kprime_a_dalpha_eq_r": kp_a,
                        "delta_alpha_b_rad": da_b,
                        "kprime_b_dalpha_chord": kp_b,
                        "alpha_c_rad": alpha_c,
                        "delta_alpha_c_rad": da_c,
                        "kprime_c_euclid_expansion": kp_c,
                        "delta_alpha_d_rad": da_d,
                        "kprime_d_inverse_paper": kp_d,
                        "delta_alpha_over_r_c": da_c / r,
                        "delta_alpha_over_r_d": da_d / r,
                        "rel_err_a": kp_a / paper_kp - 1.0,
                        "rel_err_b": kp_b / paper_kp - 1.0,
                        "rel_err_c": kp_c / paper_kp - 1.0,
                        "rel_err_d": kp_d / paper_kp - 1.0,
                    }
                )

    def _stats(vals: list[float]) -> dict:
        return {
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "n_cells": len(vals),
        }

    summary: dict = {"rel_err": {}, "inverse": {}, "inverse_per_group": {}, "euclid_effective": {}}
    for tag in ("a", "b", "c", "d"):
        errs = [row[f"rel_err_{tag}"] for row in rows]
        summary["rel_err"][tag] = {
            "mean": float(np.mean(errs)),
            "mean_abs": float(np.mean(np.abs(errs))),
            "max_abs": float(np.max(np.abs(errs))),
            "min": float(np.min(errs)),
            "max": float(np.max(errs)),
        }
    summary["inverse"] = _stats([row["delta_alpha_over_r_d"] for row in rows])
    for n in DIMS:
        for r in RADII:
            vals = [row["delta_alpha_over_r_d"] for row in rows if row["n"] == n and row["r"] == r]
            summary["inverse_per_group"][f"n={n},r={r}"] = _stats(vals)
    summary["euclid_effective"] = _stats([row["delta_alpha_over_r_c"] for row in rows])
    best = [min(abs(row["rel_err_a"]), abs(row["rel_err_b"]), abs(row["rel_err_c"])) for row in rows]
    summary["best_derivable_convention_abs_rel_err"] = {
        "min": float(np.min(best)),
        "mean": float(np.mean(best)),
        "max": float(np.max(best)),
        "description": (
            "per cell: min |rel err| over the three conventions derivable from the stated DP "
            "mechanism (a, b, c); even the best one never reproduces the paper's table"
        ),
    }
    summary["only_inverse_reproduces_paper"] = (
        "the paper's Table III is reproduced exactly (|rel err| < 1e-9) only by convention (d), "
        "Delta alpha = 1.15-1.30 r, which is derivable neither from Eq.21 nor from RemoteRAG's "
        "'Delta alpha ~ r'; 1.2 r is a fitted convention implicit in the table, not a correct value"
    )
    return {"definitions": DELTA_ALPHA_CONVENTIONS, "rows": rows, "summary": summary}


# --------------------------------------------------------------------------- #
# Cross-check against the cited implementation (RemoteRAG)
# --------------------------------------------------------------------------- #
REMOTERAG_REF = {
    "paper": "RemoteRAG: A Privacy-Preserving LLM Cloud RAG Service (arXiv:2412.12775v1)",
    "formula": (
        "Same Lemma 1 / Theorem 1 as Eq.21 (Sec. 3.2.2): Delta k = N * (Omega_{n-1}(pi)/Omega_n(pi)) "
        "* int_{alpha_k}^{alpha_k'} sin^{n-2} theta dtheta, alpha_k' = alpha_k + Delta alpha_k, "
        "with 'the perturbed angle Delta alpha_k ... can be approximately by r'."
    ),
    "highlight": (
        "Sec. 5.4 Efficiency Study: 'We highlight k'=160, r=0.03 in Fig. 5(c)'; "
        "Fig. 5(c) caption 'Cost breakdown (k'=160)'."
    ),
    "default_setting": (
        "Appendix B.4: 'Unless otherwise specified, our experiments used the following hyperparameter "
        "settings: N=10^5, k=5, k'=160, and T5 as the embedding model.'"
    ),
    "reference_point": {"k": 5, "r": 0.03, "n": 768, "kprime": 160},
    "source_url": "https://arxiv.org/abs/2412.12775 (TeX source e-print, files 5_experiments.tex, "
    "12_experiments_appendix.tex)",
    "caveats": [
        "Appendix B.4 says 'Unless otherwise specified, our experiments used ... N=10^5, k=5, k'=160, "
        "and T5'; Sec. 5.4 does not specify otherwise, so the documented default is N=10^5.",
        "r is printed to two decimals (0.03 -> [0.025,0.035)); at N=10^5 the Delta-alpha=r formula "
        "reaches k'=160 exactly at r=0.033921, which rounds to 0.03 -- i.e. the reference point is "
        "consistent with Delta alpha=r once the printed precision of r is honoured.",
        "The earlier N=10^6 reading (r=0.02941 -> 170.34, +6.46%) would also round to r=0.03, but it is "
        "not the documented default and is therefore only a secondary cross-check.",
    ],
}


def paper_table_rounding_consistency() -> dict:
    """Is Table III consistent with Delta alpha = r once r's printed precision is honoured?

    The paper prints r=0.033 (3 decimals -> true r in [0.0325, 0.0335)) and r=0.02
    (2 decimals -> true r in [0.015, 0.025)).  For every cell we compute the k'
    range that Delta alpha = r_true would produce for r_true inside the printed
    interval and check whether the paper's k' lies inside that range.  This
    separates a genuine convention mismatch from a rounding artefact.
    """
    half_width = {0.033: 0.0005, 0.02: 0.005}
    rows = []
    for n in DIMS:
        for r in RADII:
            half = half_width[r]
            for k in KS:
                paper_kp = PAPER_TABLE[(n, r)][k][0]
                kp_at_r = k_prime(k, M, n, r)
                lo = k_prime(k, M, n, r - half)
                hi = k_prime(k, M, n, r + half)
                rows.append(
                    {
                        "n": n,
                        "r": r,
                        "k": k,
                        "paper_kprime": paper_kp,
                        "r_printed_half_width": half,
                        "r_true_interval": [r - half, r + half],
                        "kprime_at_printed_r": kp_at_r,
                        "kprime_over_interval": [lo, hi],
                        "paper_inside_interval": bool(lo <= paper_kp <= hi),
                        "paper_over_kprime_at_printed_r": paper_kp / kp_at_r,
                        "implied_extra_r_if_inside": "consistent" if lo <= paper_kp <= hi else "inconsistent",
                    }
                )
    by_column = {}
    for r in RADII:
        col = [x for x in rows if x["r"] == r]
        by_column[f"r={r}"] = {
            "n_cells": len(col),
            "n_inside_interval": int(sum(x["paper_inside_interval"] for x in col)),
            "r_printed_half_width": half_width[r],
            "paper_over_kprime_at_printed_r_min": min(x["paper_over_kprime_at_printed_r"] for x in col),
            "paper_over_kprime_at_printed_r_max": max(x["paper_over_kprime_at_printed_r"] for x in col),
        }
    inside_total = int(sum(x["paper_inside_interval"] for x in rows))
    col33 = by_column["r=0.033"]
    col02 = by_column["r=0.02"]
    conclusion = (
        f"At Delta alpha=r, {inside_total}/16 Table III cells lie inside the prediction interval implied "
        f"by the printed precision of r. The split is systematic: the r=0.02 column (2 decimals, true r in "
        f"[0.015,0.025)) is compatible in {col02['n_inside_interval']}/8 cells, while the r=0.033 column "
        f"(3 decimals, true r in [0.0325,0.0335)) is compatible in {col33['n_inside_interval']}/8 cells. "
        "The r=0.033 column therefore cannot be explained by rounding: its paper values are "
        f"{col33['paper_over_kprime_at_printed_r_min']:.2f}-{col33['paper_over_kprime_at_printed_r_max']:.2f}x "
        "the Delta-alpha=r formula and would require a true r of roughly 0.039-0.043, i.e. 19-30% above the "
        "printed 0.033 (far outside 3-decimal rounding). The inconsistency of Table III is thus localised "
        "in the r=0.033 half; the r=0.02 half is consistent with Delta alpha=r."
    )
    return {
        "description": (
            "Rounding-consistency check of paper Table III under Delta alpha=r: for each cell, does the "
            "paper's k' fall inside the k' range produced by the true r inside the printed interval?"
        ),
        "rows": rows,
        "by_column": by_column,
        "n_inside_interval_total": inside_total,
        "conclusion": conclusion,
    }


def remoterag_reference_check(implied_ratio: float) -> dict:
    """Recompute RemoteRAG's highlighted point (k=5, r=0.03, n=768, k'=160).

    Resolution of the N ambiguity (2026-11-04, arXiv:2412.12775 TeX source):
    Sec. 5.4 does not restate N, but Appendix B.4 states "Unless otherwise
    specified, our experiments used the following hyperparameter settings:
    N=10^5, k=5, k'=160, and T5 as the embedding model."  Sec. 5.4 does not
    specify otherwise, so the documented default is N=10^5.  At N=10^5 the
    Delta-alpha=r formula reaches k'=160 exactly at r*=0.033921, which is what
    "r=0.03" (two decimals) quotes; the prediction over the quoted interval
    r in [0.025, 0.035) is [69.77, 176.19] and contains 160.  The reference
    point is therefore consistent with Delta alpha=r once the printed precision
    of r is honoured.  The N=10^6 reading (r=0.02941 -> 170.34, +6.46%) is
    retained only as a secondary cross-check -- it is not the documented N.
    """
    k, r_quoted, n, kp_ref = 5, 0.03, 768, 160
    r_lo, r_hi = 0.025, 0.035

    def _kp(N: int, da: float) -> float:
        a = alpha_for_fraction(k / N, n)
        return k + N * (cap_fraction(a + da, n) - cap_fraction(a, n))

    evals = []
    for N in (10**4, 10**5, 10**6):
        alpha_k = alpha_for_fraction(k / N, n)
        chord_k = 2.0 * math.sin(alpha_k / 2.0)
        alpha_c = 2.0 * math.asin(min((chord_k + r_quoted) / 2.0, 1.0))
        row = {
            "N": N,
            "alpha_k_rad": alpha_k,
            "kprime_a_dalpha_eq_r": _kp(N, r_quoted),
            "kprime_b_dalpha_chord": _kp(N, 2.0 * math.asin(r_quoted / 2.0)),
            "kprime_c_euclid_expansion": k + N * (cap_fraction(alpha_c, n) - cap_fraction(alpha_k, n)),
            "kprime_dalpha_at_implied_ratio": _kp(N, implied_ratio * r_quoted),
        }
        for tag, key in (
            ("a", "kprime_a_dalpha_eq_r"),
            ("b", "kprime_b_dalpha_chord"),
            ("c", "kprime_c_euclid_expansion"),
            ("dalpha_at_implied_ratio", "kprime_dalpha_at_implied_ratio"),
        ):
            row[f"rel_err_{tag}_vs_reference"] = row[key] / kp_ref - 1.0
        evals.append(row)

    primary_N = 10**5  # Appendix B.4 documented default
    primary = next(x for x in evals if x["N"] == primary_N)
    kp_lo, kp_hi = _kp(primary_N, r_lo), _kp(primary_N, r_hi)
    r_star = brentq(lambda x: _kp(primary_N, x) - kp_ref, 1e-6, 0.2, xtol=1e-15)
    r_star_1e6 = brentq(lambda x: _kp(10**6, x) - kp_ref, 1e-6, 0.2, xtol=1e-15)
    inside = bool(kp_lo <= kp_ref <= kp_hi)
    rounds_to_quoted = bool(round(r_star, 2) == r_quoted)
    abs_rel = abs(primary["rel_err_a_vs_reference"])
    conclusion = (
        "With RemoteRAG's documented default N=10^5 (Appendix B.4) and r printed to two decimals, "
        f"Delta alpha=r reaches k'=160 at r*={r_star:.5f} (rounds to the quoted 0.03); the prediction "
        f"over r in [0.025,0.035) is [{kp_lo:.2f},{kp_hi:.2f}], which contains 160. The reference point "
        "is therefore consistent with Delta alpha=r once the printed precision of r is honoured "
        f"(evaluating the formula at the printed r=0.03 alone gives {primary['kprime_a_dalpha_eq_r']:.2f}, "
        f"{100 * primary['rel_err_a_vs_reference']:+.2f}%, which is a rounding artefact, not a convention "
        "difference). The paper-implied ~1.2 r convention and the Euclidean-expansion convention remain "
        f"excluded at the reference point ({primary['kprime_dalpha_at_implied_ratio']:.1f} = "
        f"{primary['kprime_dalpha_at_implied_ratio'] / kp_ref:.2f}x and "
        f"{primary['kprime_c_euclid_expansion']:.1f} = {primary['kprime_c_euclid_expansion'] / kp_ref:.2f}x)."
    )
    return {
        "reference": REMOTERAG_REF,
        "implied_ratio_used": implied_ratio,
        "evaluations": evals,
        "primary": {
            "N": primary_N,
            "documented_default": True,
            "note": "N=10^5 is the default stated in RemoteRAG Appendix B.4; Sec. 5.4 does not override it.",
            "kprime_a_dalpha_eq_r": primary["kprime_a_dalpha_eq_r"],
            "kprime_c_euclid_expansion": primary["kprime_c_euclid_expansion"],
            "kprime_at_implied_ratio": primary["kprime_dalpha_at_implied_ratio"],
            "rel_err_vs_reference": primary["rel_err_a_vs_reference"],
            "abs_rel_err_vs_reference": abs_rel,
            "r_printed_interval": [r_lo, r_hi],
            "kprime_dalpha_eq_r_interval": [kp_lo, kp_hi],
            "reference_inside_interval": inside,
        },
        "pointwise_inversion": {
            "N": primary_N,
            "r_star": r_star,
            "r_star_rounds_to_quoted_r": rounds_to_quoted,
            "delta_alpha_over_quoted_r": r_star / r_quoted,
            "note": (
                "r*=0.033921 reproduces k'=160 exactly and rounds to the printed 0.03; the ratio to the "
                "printed r is 1.1307 but the ratio to the true r* is 1.0 by construction."
            ),
        },
        "secondary_N1e6": {
            "kprime_a_dalpha_eq_r": next(x for x in evals if x["N"] == 10**6)["kprime_a_dalpha_eq_r"],
            "r_star_at_N1e6": r_star_1e6,
            "note": (
                "N=10^6 would also give k'=160 with r=0.02941 (which also rounds to 0.03), but it is not "
                "the documented default; kept only to document the degeneracy of the printed point."
            ),
        },
        "conclusion": conclusion,
        "verdict": (
            "the cited implementation is consistent with Delta alpha = r at its documented default N=10^5 "
            "once the 2-decimal printed r is honoured (r*=0.033921 -> 0.03); it is NOT consistent with the "
            "paper-implied ~1.2 r nor with the Euclidean-expansion convention"
        ),
    }


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

    colors = {
        "paper": "#b2182b",
        "r": "#2166ac",
        "euclid": "#f4a582",
        "inverse": "#1b7837",
        "mc": "#1b7837",
    }
    conv_rows = res["delta_alpha_conventions"]["rows"]
    conv_index = {(row["n"], row["r"], row["k"]): row for row in conv_rows}

    # (a)(b): paper vs Eq.21(Delta alpha=r) vs the Euclidean expansion (c) vs the
    # per-cell inverted Delta alpha (d, which reproduces the paper exactly)
    for ax, n, panel in zip(axes[0], (768, 1536), ("a", "b")):
        width = 0.2
        x = np.arange(len(KS) * len(RADII))
        labels = []
        for idx_r, r in enumerate(RADII):
            for idx_k, k in enumerate(KS):
                pos = idx_r * len(KS) + idx_k
                labels.append(f"k={k}\nr={r}")
                paper_kp = PAPER_TABLE[(n, r)][k][0]
                row = conv_index[(n, r, k)]
                ax.bar(pos - 1.5 * width, paper_kp, width, color=colors["paper"],
                       label="paper Table III" if pos == 0 else None)
                ax.bar(pos - 0.5 * width, row["kprime_a_dalpha_eq_r"], width, color=colors["r"],
                       label="Eq.21, $\\Delta\\alpha=r$" if pos == 0 else None)
                ax.bar(pos + 0.5 * width, row["kprime_c_euclid_expansion"], width, color=colors["euclid"],
                       label="Eq.21, Euclidean expansion (c)" if pos == 0 else None)
                ax.bar(pos + 1.5 * width, row["kprime_d_inverse_paper"], width, color=colors["inverse"],
                       label="Eq.21, inverted $\\Delta\\alpha$ (d) = paper" if pos == 0 else None)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("top-$k'$ (log scale)")
        ax.set_title(f"({panel}) n={n}: Table III, $\\Delta\\alpha=r$, Euclidean (c), inverted (d)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        if panel == "a":
            ax.text(
                0.02, 0.02,
                "$\\Delta\\alpha=2\\,\\mathrm{asin}(r/2)$ differs from $r$ by <0.02% (not shown)",
                transform=ax.transAxes, fontsize=7, color="0.35",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
            )

    # (c): paper-implied Delta alpha / r per Table III cell (filled) vs the
    # effective Delta alpha / r of the Euclidean convention (c) (hollow).
    ax = axes[1][0]
    rev = res["reverse_delta_alpha"]["rows"]
    markers = {(768, 0.033): "o", (768, 0.02): "s", (1536, 0.033): "^", (1536, 0.02): "D"}
    offsets = {(768, 0.033): -0.18, (768, 0.02): -0.06, (1536, 0.033): 0.06, (1536, 0.02): 0.18}
    for (n, r), marker in markers.items():
        y = [x["delta_alpha_over_r"] for x in rev if x["n"] == n and x["r"] == r]
        xs = np.arange(len(KS)) + offsets[(n, r)]
        ax.scatter(xs, y, marker=marker, label=f"paper-implied (n={n}, r={r})", s=45)
    for (n, r), marker in markers.items():
        y = [conv_index[(n, r, k)]["delta_alpha_over_r_c"] for k in KS]
        xs = np.arange(len(KS)) + offsets[(n, r)]
        ax.scatter(xs, y, marker=marker, facecolors="none", edgecolors=colors["euclid"], s=40,
                   label="Euclidean expansion (c): effective $\\Delta\\alpha/r$"
                   if (n, r) == (768, 0.033) else None)
    overall = res["reverse_delta_alpha"]["summary"]["overall"]
    ax.axhline(1.0, color="black", ls=":", lw=1.2, label="RemoteRAG claim: $\\Delta\\alpha=r$")
    ax.axhline(overall["mean"], color=colors["paper"], ls="--", lw=1.2,
               label=f"paper-implied mean = {overall['mean']:.2f} r")
    ax.set_xticks(np.arange(len(KS)))
    ax.set_xticklabels([f"k={k}" for k in KS])
    ax.set_ylabel("$\\Delta\\alpha / r$")
    ax.set_title("(c) Only the inverted $\\Delta\\alpha \\approx 1.2\\,r$ reproduces Table III (A10)")
    ax.set_ylim(0.95, 1.45)
    ax.legend(fontsize=7, loc="upper left")
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

    print("[exp5] Delta-alpha conventions (four readings) ...", flush=True)
    conventions = delta_alpha_conventions()
    conv_summary = conventions["summary"]

    print("[exp5] Table III rounding consistency under Delta alpha=r ...", flush=True)
    rounding = paper_table_rounding_consistency()

    print("[exp5] RemoteRAG reference-implementation cross-check ...", flush=True)
    remoterag = remoterag_reference_check(conv_summary["inverse"]["mean"])

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

    print("\n=== Four Delta-alpha conventions (paper Table III, m=100,000) ===")
    print(
        f"{'n':>5} {'r':>6} {'k':>3} {'paper':>7} {'(a) da=r':>10} {'(b) chord':>10} "
        f"{'(c) euclid':>11} {'(d) da/r':>9} {'err_a':>8} {'err_c':>8} {'err_d':>8}"
    )
    for row in conventions["rows"]:
        print(
            f"{row['n']:>5} {row['r']:>6} {row['k']:>3} {row['paper_kprime']:>7} "
            f"{row['kprime_a_dalpha_eq_r']:>10.2f} {row['kprime_b_dalpha_chord']:>10.2f} "
            f"{row['kprime_c_euclid_expansion']:>11.2f} {row['delta_alpha_over_r_d']:>9.4f} "
            f"{row['rel_err_a']:>+8.3f} {row['rel_err_c']:>+8.3f} {row['rel_err_d']:>+8.1e}"
        )
    inv = conv_summary["inverse"]
    best = conv_summary["best_derivable_convention_abs_rel_err"]
    print(
        f"\n(d) inverted Delta alpha/r: min={inv['min']:.4f} mean={inv['mean']:.4f} "
        f"max={inv['max']:.4f} std={inv['std']:.4f} (16 cells)"
    )
    print(
        f"best derivable convention (a/b/c) is still off by >= {best['min'] * 100:.1f}% per cell "
        f"(mean {best['mean'] * 100:.1f}%, max {best['max'] * 100:.1f}%)"
    )

    print("\n=== Table III rounding consistency under Delta alpha=r (printed r precision) ===")
    print(rounding["conclusion"])
    for col, info in rounding["by_column"].items():
        print(f"  {col}: {info['n_inside_interval']}/{info['n_cells']} cells inside the rounding interval "
              f"(paper/Eq21 = {info['paper_over_kprime_at_printed_r_min']:.2f}-{info['paper_over_kprime_at_printed_r_max']:.2f}x)")

    print("\n=== RemoteRAG reference cross-check (k=5, r=0.03, n=768, k'=160) ===")
    for ev in remoterag["evaluations"]:
        print(
            f"N={ev['N']:>9.0e}: (a) da=r {ev['kprime_a_dalpha_eq_r']:8.2f} "
            f"({ev['rel_err_a_vs_reference']:+7.2%})  (b) chord {ev['kprime_b_dalpha_chord']:8.2f}  "
            f"(c) euclid {ev['kprime_c_euclid_expansion']:8.2f}  "
            f"da={remoterag['implied_ratio_used']:.3f}r {ev['kprime_dalpha_at_implied_ratio']:8.2f}"
        )
    print(remoterag["conclusion"])

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
            "four Delta-alpha conventions enumerated per cell (r / chord / Euclidean expansion / "
            "per-cell inversion), a cross-check against the cited RemoteRAG implementation, "
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
        "delta_alpha_conventions": conventions,
        "paper_table_rounding": rounding,
        "remoterag_reference": remoterag,
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
                    "Delta alpha ~ r: Eq.21 gives 0.44-0.74x the paper values "
                    f"(paper/formula = {min(x['paper_over_formula'] for x in pvf):.3f}-"
                    f"{max(x['paper_over_formula'] for x in pvf):.3f}); inverting the table requires "
                    f"Delta alpha/r = {inv['mean']:.3f} (range {inv['min']:.3f}-{inv['max']:.3f}) instead "
                    "of ~1. The paper's Table III can therefore only be reproduced with an UNDECLARED "
                    "Delta alpha ~ 1.2 r; neither the paper's text nor Eq.21 yields that convention. "
                    "1.2 r is NOT a 'correct' value -- it is the fitted convention implicit in the table. "
                    f"Rounding-aware split: {rounding['by_column']['r=0.02']['n_inside_interval']}/8 cells in the "
                    f"r=0.02 column are consistent with Delta alpha=r given the printed 2-decimal r, while "
                    f"{rounding['by_column']['r=0.033']['n_inside_interval']}/8 cells in the r=0.033 column are "
                    "not (3-decimal r cannot absorb a 19-30% shift)."
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
            {
                "id": "A10c",
                "statement": (
                    "Four Delta-alpha conventions were enumerated on all 16 Table III cells: "
                    "(a) Delta alpha = r; (b) Delta alpha = 2 asin(r/2); (c) Euclidean query shift "
                    "alpha' = 2 asin((2 sin(alpha_k/2)+r)/2); (d) per-cell inversion of the paper's k'. "
                    "(a) and (b) agree to <0.02%. None of the three derivable conventions reproduces the "
                    f"table: the best of (a,b,c) is still off by >= {best['min'] * 100:.1f}% in every cell "
                    f"((a) {conv_summary['rel_err']['a']['min'] * 100:+.0f}%..{conv_summary['rel_err']['a']['max'] * 100:+.0f}%, "
                    f"(c) {conv_summary['rel_err']['c']['min'] * 100:+.0f}%..{conv_summary['rel_err']['c']['max'] * 100:+.0f}%), "
                    f"while (d) requires Delta alpha/r = {inv['min']:.3f}-{inv['max']:.3f} "
                    f"(mean {inv['mean']:.3f}, std {inv['std']:.3f}). The paper's Table III is reproduced "
                    "exactly only by the undeclared (d) ~1.2 r convention. A rounding-aware check localises "
                    f"the mismatch: r=0.02 column {rounding['by_column']['r=0.02']['n_inside_interval']}/8 inside the "
                    f"Delta-alpha=r prediction interval, r=0.033 column "
                    f"{rounding['by_column']['r=0.033']['n_inside_interval']}/8 inside."
                ),
            },
            {
                "id": "A10d",
                "statement": (
                    "Cross-check against the cited implementation (RemoteRAG arXiv:2412.12775 Sec. 5.4 / "
                    "Fig. 5(c): highlighted k'=160, r=0.03, n=768) using its DOCUMENTED default N=10^5 "
                    "(Appendix B.4: 'Unless otherwise specified ... N=10^5, k=5, k'=160, and T5'). At N=10^5 "
                    f"the Delta-alpha=r formula reaches k'=160 exactly at r*={remoterag['pointwise_inversion']['r_star']:.5f}, "
                    "which rounds to the printed 0.03; the prediction over the printed interval "
                    f"r in [{remoterag['primary']['r_printed_interval'][0]},{remoterag['primary']['r_printed_interval'][1]}) "
                    f"is [{remoterag['primary']['kprime_dalpha_eq_r_interval'][0]:.2f},"
                    f"{remoterag['primary']['kprime_dalpha_eq_r_interval'][1]:.2f}], which contains 160. "
                    f"Evaluating at the printed r=0.03 alone gives {remoterag['primary']['kprime_a_dalpha_eq_r']:.2f} "
                    f"({100 * remoterag['primary']['rel_err_vs_reference']:+.2f}%), a rounding artefact rather than a "
                    "convention difference. The reference point excludes the paper-implied convention "
                    f"(Delta alpha={remoterag['implied_ratio_used']:.3f}r -> "
                    f"{remoterag['primary']['kprime_at_implied_ratio']:.1f} = "
                    f"{remoterag['primary']['kprime_at_implied_ratio'] / 160:.2f}x) and the Euclidean-expansion "
                    f"convention ({remoterag['primary']['kprime_c_euclid_expansion']:.1f} = "
                    f"{remoterag['primary']['kprime_c_euclid_expansion'] / 160:.2f}x). Conclusion: the cited "
                    "implementation supports Delta alpha=r at its documented N=10^5; the paper's Table III "
                    "remains inconsistent with Delta alpha=r only in its r=0.033 column."
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
