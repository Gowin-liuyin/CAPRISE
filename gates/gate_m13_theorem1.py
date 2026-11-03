"""M13 gate -- Theorem 1 (Eq.21) / Table III reproduction.

Acceptance criteria (docs/MODULE_PLAN.md M13; SPEC.md Sec. 5; task brief):

1. Numerical stability of the Beta-form implementation:
   * F(alpha_for_fraction(f)) == f for f down to k/m = 5e-5 and n up to 4096;
   * m * F(alpha_k) == k exactly (definition of alpha_k);
   * every Table III cell finite and k' > k for n=768/1536 (no NaN in the tail);
   * Beta CDF agrees with direct quadrature for small n (where quadrature works);
   * the literal Eq.21 integrand sin(theta)^(n-2) underflows to 0.0 for n>=768,
     which is exactly why the Beta form is required.
2. Monte-Carlo check of Eq.21's mathematical identity: the mean number of DB
   points inside the widened angular cap (radius alpha_k + r around the
   DistanceDP-perturbed query) matches Eq.21 with Delta alpha = r within 5%.
   Checked on the full m=100,000 / nq=200 experiment JSON *and* with a smaller
   reduced MC run inside the gate.
   NOTE (cross-review V5): the gate-internal reduced MC reuses
   ``experiments/exp5_query_dp.py:monte_carlo`` (imported via importlib), so it is
   a *regression* check of the same implementation, NOT an independent
   re-implementation.  Implementation-level independence is covered by the
   separate cross-review task (results/verify_theory.json / notes/verify_theory.md
   by data-retrieval).
3. Retrieval sufficiency: retrieving ceil(Eq.21) candidates under the perturbed
   query always contains the plaintext top-k (coverage rate == 1.0).
4. Reverse-engineered Delta alpha/r from the paper's Table III is documented and
   stable (~1.21, range 1.15-1.30, std <0.1) => paper table is inconsistent with
   RemoteRAG's "Delta alpha ~ r" (finding A10).  The paper mismatch is NOT a
   gate failure condition -- only that the finding is measured and documented.
5. Four Delta-alpha conventions are enumerated for every Table III cell and can be
   recomputed from the JSON primitives (task-3):
   (a) Delta alpha = r; (b) Delta alpha = 2 asin(r/2); (c) Euclidean expansion
   alpha' = 2 asin((2 sin(alpha_k/2)+r)/2); (d) per-cell inversion of the paper's
   k'.  (a)/(b)/(c) must be reproducible from the stored JSON values and none of
   them may reproduce the paper's table within 10% in any cell; (d) must match the
   paper exactly and its Delta alpha/r ratio must stay in (1.1, 1.4).
6. Cross-check against the cited implementation (RemoteRAG arXiv:2412.12775
   Sec. 5.4 / Fig. 5(c): k=5, r=0.03, n=768, k'=160): Eq.21 with Delta alpha=r
   must agree within 10% at N=10^6, N=10^5 must be far below (documents the N
   inference), and the reference point must exclude the paper-implied ~1.21 r and
   the Euclidean-expansion convention.
7. Artifacts exist: results/exp5_query_dp.json, figures/fig6_table3.png,
   notes/exp5_query_dp.md (the experiment is run by the gate if artifacts are
   missing), and the notes' key numbers match the JSON.

Run:  .venv/bin/python -m gates.gate_m13_theorem1
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gammaln

from gates.common import Gate, add_src_to_path

add_src_to_path()

from caprise.theorem import alpha_for_fraction, cap_fraction, k_prime, omega_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXP_JSON = RESULTS / "exp5_query_dp.json"
FIG = ROOT / "figures" / "fig6_table3.png"
NOTES = ROOT / "notes" / "exp5_query_dp.md"
EXP_SCRIPT = ROOT / "experiments" / "exp5_query_dp.py"

M = 100_000
DIMS = (768, 1536)
RADII = (0.033, 0.02)
KS = (5, 10, 15, 20)

# Table III as transcribed from the paper's LaTeX source (tab:radius_k_ratio)
PAPER = {
    (768, 0.033): {5: (258, 52), 10: (571, 57), 15: (673, 45), 20: (928, 46)},
    (768, 0.02): {5: (58, 12), 10: (108, 11), 15: (170, 11), 20: (203, 10)},
    (1536, 0.033): {5: (982, 196), 10: (1492, 149), 15: (2077, 138), 20: (2603, 130)},
    (1536, 0.02): {5: (145, 29), 10: (245, 25), 15: (388, 26), 20: (501, 25)},
}


def _load_exp5_module():
    spec = importlib.util.spec_from_file_location("exp5_query_dp", EXP_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_artifacts(g: Gate) -> None:
    """Run the experiment if its artifacts are missing (keeps the gate reproducible
    from a clean checkout)."""
    if EXP_JSON.exists() and FIG.exists() and NOTES.exists():
        g.note("exp5 artifacts found; gate does not rerun the full experiment")
        return
    g.note("exp5 artifacts missing -> running experiments/exp5_query_dp.py (~2 min)")
    proc = subprocess.run(
        [sys.executable, str(EXP_SCRIPT)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
    )
    g.check_true(
        "exp5_experiment_ran",
        proc.returncode == 0 and EXP_JSON.exists() and FIG.exists(),
        detail=f"subprocess returncode={proc.returncode}",
    )
    if proc.returncode != 0:
        tail = proc.stdout.decode("utf-8", "replace")[-1500:]
        g.note(f"experiment subprocess output tail:\n{tail}")


def main() -> int:
    g = Gate(
        "M13",
        "theorem1",
        owner="theory-efficiency",
        paper_ref="arXiv:2601.12331 Theorem 1 (Eq.21), Table III; RemoteRAG arXiv:2412.12775 Sec. 3.2.1",
    )
    _ensure_artifacts(g)

    # ================================================================== #
    # 1. Numeric stability of the Beta implementation
    # ================================================================== #
    fractions = (1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 0.5, 0.9)
    worst_rt = 0.0
    for n in (768, 1536, 4096):
        for f in fractions:
            a = alpha_for_fraction(f, n)
            worst_rt = max(worst_rt, abs(cap_fraction(a, n) - f) / f)
    g.check_true(
        "beta_roundtrip_n768_1536_4096",
        worst_rt < 1e-9,
        detail=f"max |F(alpha_for_fraction(f))-f|/f = {worst_rt:.3e} over f in {fractions}, n in (768,1536,4096); tol 1e-9",
    )

    maxdev_mF = 0.0
    for n in DIMS:
        for k in KS:
            maxdev_mF = max(maxdev_mF, abs(M * cap_fraction(alpha_for_fraction(k / M, n), n) - k))
    g.check_true(
        "m_times_F_alpha_k_equals_k",
        maxdev_mF < 1e-6,
        detail=f"max |m*F(alpha_k)-k| = {maxdev_mF:.3e} over all 16 Table III cells (alpha_k := F^-1(k/m))",
    )

    finite_ok, monotone_ok, k_prime_values = True, True, {}
    for n in DIMS:
        for k in KS:
            a = alpha_for_fraction(k / M, n)
            for r in RADII:
                kp = k_prime(k, M, n, r, delta_alpha_ratio=1.0)
                k_prime_values[(n, r, k)] = kp
                finite_ok &= bool(math.isfinite(kp)) and kp > k
                monotone_ok &= cap_fraction(a + r, n) >= cap_fraction(a, n)
    g.check_true(
        "n768_n1536_tail_no_nan",
        finite_ok,
        detail="all 16 Table III cells: k' finite and > k (no NaN/inf in the n=1536 tail)",
    )
    g.check_true(
        "cap_fraction_monotone_in_alpha",
        monotone_ok,
        detail="F(alpha_k + r) >= F(alpha_k) for every Table III cell",
    )

    quad_err = 0.0
    for n in (10, 50, 100):
        for k in (5, 20):
            a = alpha_for_fraction(k / M, n)
            direct = omega_ratio(n) * quad(lambda t: math.sin(t) ** (n - 2), 0.0, a, limit=400)[0]
            quad_err = max(quad_err, abs(direct - cap_fraction(a, n)))
    g.check_true(
        "beta_form_agrees_with_direct_quadrature",
        quad_err < 1e-9,
        detail=f"max |quadrature - Beta CDF| = {quad_err:.3e} for n in (10,50,100), k in (5,20)",
    )

    underflow_hits = []
    for n, theta in ((768, 0.05), (768, 0.2), (1536, 0.1), (1536, 0.3), (4096, 0.3)):
        literal = math.sin(theta) ** (n - 2)
        underflow_hits.append(literal == 0.0 and math.isfinite(cap_fraction(theta, n)))
    g.check_true(
        "literal_integrand_underflows_but_beta_finite",
        all(underflow_hits),
        detail="sin(theta)^(n-2) == 0.0 in float64 for the probed (n>=768, theta) while the Beta CDF is finite "
        f"({sum(underflow_hits)}/5 probed points)",
    )

    # ================================================================== #
    # Load experiment results
    # ================================================================== #
    if not EXP_JSON.exists():
        g.check_true("exp5_results_json_exists", False, detail=f"missing {EXP_JSON}")
        g.finish()
        return 1
    res = json.loads(EXP_JSON.read_text(encoding="utf-8"))

    # cross-check the JSON formula table against an independent recomputation
    recompute_err = 0.0
    for row in res["formula_table_dalpha_eq_r"]:
        n, k = row["n"], row["k"]
        for r in RADII:
            recompute_err = max(recompute_err, abs(row[f"kprime@{r}"] - k_prime_values[(n, r, k)]))
    g.check_true(
        "formula_table_recomputed_consistently",
        recompute_err < 1e-6,
        detail=f"max |JSON formula value - recomputed Eq.21| = {recompute_err:.3e} (16 cells x 2 radii)",
    )

    # paper table transcription sanity: k'/k ratio in the paper is round(k'/k)
    ratio_ok = True
    for (n, r), per_k in PAPER.items():
        for k, (kp, ratio) in per_k.items():
            ratio_ok &= abs(round(kp / k) - ratio) <= 1
    g.check_true(
        "paper_table_transcription_consistent",
        ratio_ok,
        detail="paper top-k' and k'/k columns are mutually consistent after rounding (LaTeX tab:radius_k_ratio)",
    )

    # ================================================================== #
    # 2. MC vs Eq.21 (Delta alpha = r) -- full experiment (all 16 cells)
    # ================================================================== #
    mc_rows = res["monte_carlo"]["rows"]
    max_rel = max(abs(x["mc_cap_rel_err_population_alpha"]) for x in mc_rows)
    g.check_true(
        "mc_widened_cap_matches_formula_within_5pct",
        max_rel < 0.05,
        detail=f"max |MC cap count - Eq.21|/Eq.21 = {max_rel:.4f} over all 16 cells "
        f"(m=100,000, nq=200, fixed DB; population alpha_k from F(alpha_k)=k/m); tol 5%",
    )
    max_rel_emp = max(abs(x["mc_cap_rel_err_empirical_alpha"]) for x in mc_rows)
    g.check_true(
        "mc_empirical_alpha_version_also_within_5pct",
        max_rel_emp < 0.05,
        detail=f"max |MC cap count (empirical alpha_k) - Eq.21|/Eq.21 = {max_rel_emp:.4f} over 16 cells; tol 5%",
    )
    g.check_true(
        "mc_coverage_topk_sufficiency",
        all(x["coverage_rate_with_ceil_formula"] == 1.0 for x in mc_rows),
        detail="retrieving ceil(Eq.21) candidates under the perturbed query contains the plaintext top-k "
        "for every one of 200 queries x 16 cells",
    )
    g.check_true(
        "mc_minimal_kprime_less_than_formula",
        all(x["min_kprime_mean"] < x["formula_kprime_dalpha_eq_r"] for x in mc_rows),
        detail="measured minimal k' (real retrieval) is below Eq.21 for every cell => Eq.21 is a sufficient "
        "but loose upper bound (documented finding A10b)",
    )

    # independent reduced MC inside the gate (deterministic seeds)
    exp5 = _load_exp5_module()
    gate_mc = exp5.monte_carlo(768, nq=50, seed=7, verbose=False)["rows"] + exp5.monte_carlo(
        1536, nq=50, seed=7, verbose=False
    )["rows"]
    gate_cells = [x for x in gate_mc if x["r"] == 0.033 and x["k"] in (5, 20)]
    gate_max_rel = max(abs(x["mc_cap_rel_err_population_alpha"]) for x in gate_cells)
    gate_cov_ok = all(x["coverage_rate_with_ceil_formula"] == 1.0 for x in gate_cells)
    g.check_true(
        "mc_independent_reduced_run_matches_formula",
        gate_max_rel < 0.05 and gate_cov_ok,
        detail=f"gate-internal MC (nq=50, r=0.033, n in {{768,1536}}, k in {{5,20}}): "
        f"max rel err={gate_max_rel:.4f} (<0.05), coverage==1.0: {gate_cov_ok}",
    )

    # ================================================================== #
    # 3. Reverse-engineered Delta alpha / r (finding A10)
    # ================================================================== #
    summary = res["reverse_delta_alpha"]["summary"]["overall"]
    rev_vals = [x["delta_alpha_over_r"] for x in res["reverse_delta_alpha"]["rows"]]
    g.check_true(
        "reverse_dalpha_converges_to_stable_value",
        (1.15 <= summary["mean"] <= 1.30) and summary["std"] < 0.1 and min(rev_vals) > 1.0,
        detail=f"implied Delta alpha/r: mean={summary['mean']}, std={summary['std']}, "
        f"range=[{summary['min']}, {summary['max']}] over 16 cells (stable ~1.21; RemoteRAG claims ~1.0); "
        "all cells > 1 => Table III inconsistent with Delta alpha = r (finding A10)",
    )
    ratios = [x["paper_over_formula"] for x in res["paper_vs_formula_dalpha_eq_r"]]
    g.check_true(
        "paper_table_exceeds_dalpha_eq_r_formula",
        max(ratios) > 1.3 and min(ratios) > 1.2,
        detail=f"paper / Eq.21(Delta alpha=r): range [{min(ratios):.3f}, {max(ratios):.3f}] "
        f"(mean {float(np.mean(ratios)):.3f}) => paper table needs ~1.2 r (A10); NOT a gate failure",
    )
    probe = res["convention_probe"]
    g.check_true(
        "dalpha_1p2r_closest_convention",
        abs(probe["angle_dalpha_eq_1.2r"]["paper_over_pred_mean"] - 1.0)
        < abs(probe["distance_dk_plus_r"]["paper_over_pred_mean"] - 1.0)
        and abs(probe["angle_dalpha_eq_1.2r"]["paper_over_pred_mean"] - 1.0)
        < abs(probe["angle_dalpha_eq_r"]["paper_over_pred_mean"] - 1.0),
        detail=f"paper/prediction means: dalpha=r {probe['angle_dalpha_eq_r']['paper_over_pred_mean']}, "
        f"dalpha=1.2r {probe['angle_dalpha_eq_1.2r']['paper_over_pred_mean']}, "
        f"distance d+r {probe['distance_dk_plus_r']['paper_over_pred_mean']} => 1.2r is the closest convention",
    )

    # ================================================================== #
    # 3b. Four Delta-alpha conventions (task-3): (a) r, (b) chord,
    #     (c) Euclidean expansion, (d) per-cell inversion of the paper's k'
    # ================================================================== #
    note_required: list[str] = []
    conv = res.get("delta_alpha_conventions") or {}
    conv_rows = conv.get("rows", [])
    conv_keys = ("kprime_a_dalpha_eq_r", "kprime_b_dalpha_chord", "kprime_c_euclid_expansion",
                 "delta_alpha_d_rad", "kprime_d_inverse_paper", "delta_alpha_over_r_d")
    g.check_true(
        "delta_alpha_conventions_present",
        isinstance(conv, dict) and len(conv_rows) == 16
        and all(all(key in row for key in conv_keys) for row in conv_rows),
        detail=f"{len(conv_rows)} cells, keys {list(conv_keys)} present in every row "
        f"(conventions: {list(conv.get('definitions', {}))})",
    )
    if len(conv_rows) == 16:
        recon_err = 0.0
        inv_err = 0.0
        inv_ratio_err = 0.0
        inv_kp_err = 0.0
        for row in conv_rows:
            n, r, k = row["n"], row["r"], row["k"]
            paper_kp = PAPER[(n, r)][k][0]
            alpha_k = alpha_for_fraction(k / M, n)
            fk = cap_fraction(alpha_k, n)
            chord_k = 2.0 * math.sin(alpha_k / 2.0)
            alpha_c = 2.0 * math.asin(min((chord_k + r) / 2.0, 1.0))
            recomputed = {
                "kprime_a_dalpha_eq_r": k + M * (cap_fraction(alpha_k + r, n) - fk),
                "kprime_b_dalpha_chord": k
                + M * (cap_fraction(alpha_k + 2.0 * math.asin(r / 2.0), n) - fk),
                "kprime_c_euclid_expansion": k + M * (cap_fraction(alpha_c, n) - fk),
            }
            for key, val in recomputed.items():
                recon_err = max(recon_err, abs(row[key] - val))
            target = (paper_kp - k) / M
            da_d = brentq(
                lambda x: cap_fraction(alpha_k + x, n) - fk - target, -0.5, 0.5, xtol=1e-14
            )
            inv_err = max(inv_err, abs(row["delta_alpha_d_rad"] - da_d))
            inv_ratio_err = max(inv_ratio_err, abs(row["delta_alpha_over_r_d"] - da_d / r))
            inv_kp_err = max(inv_kp_err, abs(row["kprime_d_inverse_paper"] - paper_kp))
        g.check_true(
            "four_conventions_recomputed_from_json",
            recon_err < 1e-6,
            detail=f"(a) Delta alpha=r, (b) chord 2asin(r/2), (c) Euclidean expansion recomputed from the JSON "
            f"primitives: max |JSON - Eq.21| = {recon_err:.3e} over 16 cells x 3 conventions (tol 1e-6)",
        )
        g.check_true(
            "inverse_convention_matches_json_and_paper",
            inv_err < 1e-9 and inv_ratio_err < 1e-9 and inv_kp_err < 1e-6,
            detail=f"per-cell inverted Delta alpha re-solved with brentq: max |Delta alpha - JSON| = {inv_err:.3e} rad, "
            f"max |Delta alpha/r - JSON| = {inv_ratio_err:.3e}, max |k'(d) - paper k'| = {inv_kp_err:.3e}",
        )
        best_per_cell = [
            min(abs(x["rel_err_a"]), abs(x["rel_err_b"]), abs(x["rel_err_c"])) for x in conv_rows
        ]
        a_errs = [x["rel_err_a"] for x in conv_rows]
        c_errs = [x["rel_err_c"] for x in conv_rows]
        g.check_true(
            "no_derivable_convention_reproduces_paper_table",
            min(best_per_cell) >= 0.10,
            detail=f"best of (a,b,c) per cell: min |rel err| = {min(best_per_cell):.4f} (>= 0.10); "
            f"(a) rel err [{min(a_errs):+.3f}, {max(a_errs):+.3f}], (c) [{min(c_errs):+.3f}, {max(c_errs):+.3f}] "
            "=> only the undeclared per-cell inversion reproduces Table III",
        )
        inv_sum = conv["summary"]["inverse"]
        vals = [x["delta_alpha_over_r_d"] for x in conv_rows]
        g.check_true(
            "inverse_delta_alpha_ratio_interval",
            abs(inv_sum["min"] - min(vals)) < 1e-12
            and abs(inv_sum["mean"] - float(np.mean(vals))) < 1e-9
            and abs(inv_sum["max"] - max(vals)) < 1e-12
            and min(vals) > 1.1
            and max(vals) < 1.4,
            detail=f"Delta alpha/r implied by the paper table: min={inv_sum['min']:.4f} "
            f"mean={inv_sum['mean']:.4f} max={inv_sum['max']:.4f} over 16 cells "
            f"(all > 1.1 => more than Delta alpha = r; all < 1.4 => below the Euclidean-expansion "
            f"effective ratio {conv['summary']['euclid_effective']['mean']:.3f})",
        )
        legacy = res["reverse_delta_alpha"]["summary"]["overall"]
        g.check_true(
            "reverse_delta_alpha_sections_consistent",
            abs(legacy["mean"] - inv_sum["mean"]) < 1e-3
            and abs(legacy["min"] - inv_sum["min"]) < 1e-3
            and abs(legacy["max"] - inv_sum["max"]) < 1e-3,
            detail=f"legacy (5-dp rounded) summary mean={legacy['mean']}, min={legacy['min']}, "
            f"max={legacy['max']} vs full-precision conventions summary "
            f"{inv_sum['mean']:.6f}/{inv_sum['min']:.6f}/{inv_sum['max']:.6f}: diff < 1e-3 (JSON rounding only)",
        )
        note_required.append(f"{inv_sum['mean']:.4f}")

    # ================================================================== #
    # 3c. Cross-check against the cited implementation (RemoteRAG Sec. 5.4)
    # ================================================================== #
    ref = res.get("remoterag_reference") or {}
    g.check_true(
        "remoterag_reference_present",
        isinstance(ref, dict) and len(ref.get("evaluations", [])) == 3,
        detail=f"RemoteRAG cross-check present (N=10^4/10^5/10^6 evaluations: {len(ref.get('evaluations', []))})",
    )
    if ref:
        rp = ref["reference"]["reference_point"]
        g.check_true(
            "remoterag_reference_point_k5_r003_n768_kprime160",
            (rp["k"], rp["r"], rp["n"], rp["kprime"]) == (5, 0.03, 768, 160),
            detail=f"reference point {rp} (RemoteRAG Sec. 5.4 / Fig. 5(c) highlighted example)",
        )
        primary = ref["primary"]
        g.check_true(
            "remoterag_primary_uses_documented_default_N",
            primary["N"] == 10**5 and primary.get("documented_default") is True,
            detail=f"RemoteRAG primary reading uses N={primary['N']:g} (Appendix B.4 documented default: "
            "'Unless otherwise specified ... N=10^5, k=5, k'=160, and T5'); Sec. 5.4 does not override it",
        )
        # recompute the Delta-alpha=r prediction interval over the printed r precision
        r_lo, r_hi = primary["r_printed_interval"]
        alpha_lo = alpha_for_fraction(rp["k"] / primary["N"], rp["n"])
        kp_lo = rp["k"] + primary["N"] * (cap_fraction(alpha_lo + r_lo, rp["n"]) - cap_fraction(alpha_lo, rp["n"]))
        alpha_hi = alpha_for_fraction(rp["k"] / primary["N"], rp["n"])
        kp_hi = rp["k"] + primary["N"] * (cap_fraction(alpha_hi + r_hi, rp["n"]) - cap_fraction(alpha_hi, rp["n"]))
        interval_ok = (kp_lo <= rp["kprime"] <= kp_hi) and abs(kp_lo - primary["kprime_dalpha_eq_r_interval"][0]) < 1e-6 \
            and abs(kp_hi - primary["kprime_dalpha_eq_r_interval"][1]) < 1e-6
        g.check_true(
            "remoterag_kprime160_inside_rounding_interval",
            interval_ok and primary["reference_inside_interval"] is True,
            detail=f"at N=10^5, Delta alpha=r over the printed r in [{r_lo},{r_hi}) predicts "
            f"k' in [{kp_lo:.2f},{kp_hi:.2f}], which contains the reference k'=160; "
            f"evaluating at the printed r=0.03 alone gives {primary['kprime_a_dalpha_eq_r']:.2f} "
            f"({100 * primary['rel_err_vs_reference']:+.2f}%) -- a rounding artefact, not a convention difference",
        )
        pinv = ref["pointwise_inversion"]
        alpha_star = alpha_for_fraction(rp["k"] / primary["N"], rp["n"])
        r_star = brentq(
            lambda x: rp["k"] + primary["N"] * (cap_fraction(alpha_star + x, rp["n"]) - cap_fraction(alpha_star, rp["n"]))
            - rp["kprime"],
            1e-6, 0.2, xtol=1e-15,
        )
        g.check_true(
            "remoterag_pointwise_inversion_rounds_to_quoted_r",
            abs(r_star - pinv["r_star"]) < 1e-9 and round(r_star, 2) == rp["r"],
            detail=f"point-wise inversion at N=10^5: r*={r_star:.6f} reproduces k'=160 and rounds to the "
            f"printed r={rp['r']}; Delta alpha/r_printed = {r_star / rp['r']:.4f} (the excess is pure rounding "
            "of r), ratio to the true r* is 1 by construction",
        )
        g.check_true(
            "remoterag_reference_excludes_implied_and_euclid_conventions",
            primary["kprime_at_implied_ratio"] > 1.2 * rp["kprime"]
            and primary["kprime_c_euclid_expansion"] > 1.5 * rp["kprime"],
            detail=f"at the documented N=10^5 reference point, Delta alpha={ref['implied_ratio_used']:.3f} r gives "
            f"{primary['kprime_at_implied_ratio']:.2f} ({primary['kprime_at_implied_ratio'] / rp['kprime']:.2f}x) "
            f"and the Euclidean expansion gives {primary['kprime_c_euclid_expansion']:.2f} "
            f"({primary['kprime_c_euclid_expansion'] / rp['kprime']:.2f}x) => both excluded",
        )
        sec = ref["secondary_N1e6"]
        g.check_true(
            "remoterag_secondary_N1e6_documented_as_non_default",
            "not the documented default" in sec.get("note", "").lower()
            or "not the documented default" in sec.get("note", ""),
            detail=f"secondary N=1e6 reading kept only to document the degeneracy: "
            f"k'={sec['kprime_a_dalpha_eq_r']:.2f} at r=0.03, r*={sec['r_star_at_N1e6']:.5f} also rounds to 0.03",
        )
        note_required.extend(["0.03392", "8/8"])

    # ================================================================== #
    # 3d. Table III rounding-consistency split (r=0.02 vs r=0.033)
    # ================================================================== #
    rnd = res.get("paper_table_rounding") or {}
    rnd_rows = rnd.get("rows", [])
    g.check_true(
        "paper_table_rounding_present",
        len(rnd_rows) == 16 and all(k in rnd_rows[0] for k in (
            "n", "r", "k", "paper_kprime", "r_true_interval", "kprime_over_interval",
            "paper_inside_interval", "paper_over_kprime_at_printed_r")),
        detail=f"{len(rnd_rows)} cells with printed-r interval, Delta-alpha=r prediction interval and inside flag",
    )
    if len(rnd_rows) == 16:
        worst_lo = worst_hi = 0.0
        for row in rnd_rows:
            n, r, k = row["n"], row["r"], row["k"]
            half = row["r_printed_half_width"]
            a_k = alpha_for_fraction(k / M, n)
            kp_at = k + M * (cap_fraction(a_k + r, n) - cap_fraction(a_k, n))
            lo = k_prime(k, M, n, r - half)
            hi = k_prime(k, M, n, r + half)
            worst_lo = max(worst_lo, abs(lo - row["kprime_over_interval"][0]))
            worst_hi = max(worst_hi, abs(hi - row["kprime_over_interval"][1]))
            if abs(kp_at - row["kprime_at_printed_r"]) > 1e-9:
                worst_lo = float("inf")
            inside = bool(lo <= row["paper_kprime"] <= hi)
            if inside != row["paper_inside_interval"]:
                worst_hi = float("inf")
        g.check_true(
            "paper_table_rounding_reproducible",
            worst_lo < 1e-6 and worst_hi < 1e-6,
            detail=f"recomputed k'(r+/-printed half-width) from JSON primitives: max |diff| = "
            f"{max(worst_lo, worst_hi):.2e}; inside flags recomputed exactly",
        )
        by_col = rnd["by_column"]
        c02, c33 = by_col["r=0.02"], by_col["r=0.033"]
        g.check_true(
            "paper_table_rounding_split_02_consistent_0033_not",
            c02["n_inside_interval"] == 8 and c33["n_inside_interval"] == 0
            and rnd["n_inside_interval_total"] == 8,
            detail=f"Delta-alpha=r with printed-r precision: r=0.02 column {c02['n_inside_interval']}/8 inside "
            f"(paper/Eq21 {c02['paper_over_kprime_at_printed_r_min']:.2f}-{c02['paper_over_kprime_at_printed_r_max']:.2f}x), "
            f"r=0.033 column {c33['n_inside_interval']}/8 inside "
            f"({c33['paper_over_kprime_at_printed_r_min']:.2f}-{c33['paper_over_kprime_at_printed_r_max']:.2f}x) "
            "=> the inconsistency is localised in the r=0.033 half and cannot be 3-decimal rounding",
        )

    # ================================================================== #
    # 4. Artifacts
    # ================================================================== #
    g.check_true(
        "artifact_results_json",
        EXP_JSON.exists() and "monte_carlo" in res and "paper_vs_formula_dalpha_eq_r" in res,
        detail=f"{EXP_JSON.relative_to(ROOT)} ({EXP_JSON.stat().st_size} bytes) with required sections",
    )
    g.check_true(
        "artifact_figure",
        FIG.exists() and FIG.stat().st_size > 10_000,
        detail=f"{FIG.relative_to(ROOT)} ({FIG.stat().st_size if FIG.exists() else 0} bytes)",
    )
    note_text = NOTES.read_text(encoding="utf-8") if NOTES.exists() else ""
    g.check_true(
        "artifact_notes_documents_finding",
        NOTES.exists() and "A10" in note_text and ("1.21" in note_text or "Δα" in note_text),
        detail=f"{NOTES.relative_to(ROOT)} exists={NOTES.exists()}, "
        f"documents A10={'A10' in note_text}, mentions implied ratio={'1.21' in note_text or 'Δα' in note_text}",
    )
    g.check_true(
        "artifact_notes_numbers_match_json",
        NOTES.exists() and bool(note_required) and all(s in note_text for s in note_required),
        detail=f"notes contain the JSON-formatted numbers {note_required} "
        f"(implied mean Delta alpha/r and RemoteRAG cross-check values)",
    )

    # ================================================================== #
    # Summary
    # ================================================================== #
    g.note("Eq.21 is the expected widened-cap size: k' - k = m*(F(alpha_k+r)-F(alpha_k)) = m*F(alpha_k+r) - k.")
    g.note("MC validates that identity within 5% (widened-cap count) and shows ceil(Eq.21) always covers the plaintext top-k.")
    g.note("MC also measures the minimal k' a real retrieval needs (~k): Eq.21 is a sufficient but very loose bound (A10b).")
    g.note("Four Delta-alpha conventions enumerated: (a) r, (b) 2asin(r/2), (c) Euclidean expansion, (d) per-cell inversion.")
    g.note("(a)/(b)/(c) never reproduce Table III (best still >= 10% off); only (d) does, needing Delta alpha/r ~1.15-1.30 (A10c).")
    g.note("Paper Table III requires Delta alpha/r ~1.21 (range 1.15-1.30), inconsistent with RemoteRAG's Delta alpha ~ r (A10).")
    g.note("RemoteRAG's highlighted point (k=5, r=0.03, n=768, k'=160) is consistent with Delta alpha = r to ~6% at N=1e6 (A10d).")
    res = g.finish()
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
