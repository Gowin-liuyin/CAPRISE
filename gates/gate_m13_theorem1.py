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
5. Artifacts exist: results/exp5_query_dp.json, figures/fig6_table3.png,
   notes/exp5_query_dp.md (the experiment is run by the gate if artifacts are
   missing).

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

    # ================================================================== #
    # Summary
    # ================================================================== #
    g.note("Eq.21 is the expected widened-cap size: k' - k = m*(F(alpha_k+r)-F(alpha_k)) = m*F(alpha_k+r) - k.")
    g.note("MC validates that identity within 5% (widened-cap count) and shows ceil(Eq.21) always covers the plaintext top-k.")
    g.note("MC also measures the minimal k' a real retrieval needs (~k): Eq.21 is a sufficient but very loose bound (A10b).")
    g.note("Paper Table III requires Delta alpha/r ~1.21 (range 1.15-1.30), inconsistent with RemoteRAG's Delta alpha ~ r (A10).")
    res = g.finish()
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
