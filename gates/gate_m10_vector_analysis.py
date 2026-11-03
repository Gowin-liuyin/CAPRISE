"""M10 gate: vector-analysis attack ASR -- CAPRISE vs ADCPE (paper Fig. 5).

Module M10 of docs/MODULE_PLAN.md; paper Section V-C + Fig. 5
(`figures/exp_asr.png` in the paper LaTeX source).

What this gate verifies
-----------------------
1. Pipeline is executable and *deterministic*: re-running the reduced exp4
   pipeline with the same seed reproduces the ASR values bit-for-bit (the
   experiment feeds seed-derived deterministic nonces to EncDB / ADCPE.Enc).
2. Pipeline self-check: a noise-free ciphertext (E' = s*E) must yield ASR@t = 1
   and distance-reconstruction error ~ 0, proving the ASR computation itself
   is correct (and that the NaN bug of the first draft is gone).
3. The full experiment artifact results/exp4_vector_analysis.json exists and
   contains all required cells and metrics:
     - real GTR d=768 (m >= 5000, |Q| >= 100, from data/passage_emb.npy),
     - synthetic d=1536/3072,
     - the paper's 12-size grid {64,...,2048},
     - m sweep, beta sweep, normalisation control,
     - ASR@1/5/10, Kendall tau, Spearman rho, distance-reconstruction error.
4. Metric ranges are sane (ASR in [0,1], Kendall/Spearman in [-1,1], dist_err>=0,
   all finite) and the attack is actually *successful* (ASR@10 not degenerate),
   yet not perfect for both methods (no accidental plaintext leak).
5. Paper's qualitative claim (CAPRISE consistently lower ASR than ADCPE) holds
   in every measured cell; a reversal is reported explicitly as a failure with
   the offending cells (the module plan requires "CAPRISE < ADCPE (trend)").
6. Finding A19 is quantified: CAPRISE's distance-reconstruction error is larger
   than ADCPE's, ratio in the theory-consistent band [1.3, 2.5] (radii ratio
   1.5; measured 1.65-2.16 because of the second-order perpendicular term).
7. Paper Fig. 5 anchor comparison: at d=768 our controlled synthetic ASR@10 is
   within 3 pp of the digitised paper values (83.1 / 87.5); the paper figure
   values are stored in the artifact for auditability.
8. Figure and human-readable notes exist (and the figure is regenerated from
   the artifact if missing).

If the full artifact is missing, the gate runs the full exp4 experiment once
(~30 s) and writes it; existing up-to-date artifacts are reused.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

from gates.common import Gate, ROOT, add_src_to_path

add_src_to_path()

ARTIFACT = ROOT / "results" / "exp4_vector_analysis.json"
CSV_PATH = ROOT / "results" / "exp4_vector_analysis.csv"
FIG_PATH = ROOT / "figures" / "fig5_vector_analysis.png"
NOTES_PATH = ROOT / "notes" / "exp4_vector_analysis.md"

g = Gate(
    "M10",
    "vector_analysis",
    owner="privacy-attacks",
    paper_ref="paper Section V-C / Fig. 5 (figures/exp_asr.png); A12, A13, A19",
)


# --------------------------------------------------------------------------- #
# Load the experiment module (experiments/ is not a package)
# --------------------------------------------------------------------------- #
def _load_exp4():
    path = ROOT / "experiments" / "exp4_vector_analysis.py"
    spec = importlib.util.spec_from_file_location("exp4_vector_analysis", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp4_vector_analysis"] = mod
    spec.loader.exec_module(mod)
    return mod


exp4 = _load_exp4()


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _iter_cells(artifact: dict):
    """Yield (section, dim, cell) for every cell that stores ADCPE/CAPRISE metrics."""
    for sec in ("main_synthetic", "main_real_gtr", "paper_dims", "m_sweep", "real_m_sweep", "beta_sweep"):
        if sec == "main_synthetic":
            mapping = artifact["main"]["synthetic"]
        elif sec == "main_real_gtr":
            mapping = artifact["main"]["real_gtr"]
        elif sec == "paper_dims":
            mapping = artifact.get("paper_dims_synthetic", {})
        elif sec == "m_sweep":
            mapping = artifact.get("m_sweep", {})
        elif sec == "real_m_sweep":
            mapping = artifact.get("real_m_sweep", {})
        else:
            mapping = artifact.get("beta_sweep", {})
        for dim, cell in mapping.items():
            if isinstance(cell, dict) and "ADCPE" in cell and "CAPRISE" in cell:
                yield sec, dim, cell


# --------------------------------------------------------------------------- #
# 1) reduced pipeline: executable + deterministic
# --------------------------------------------------------------------------- #
cfg = exp4.quick_config()
run_a = exp4.run_experiment(cfg)
run_b = exp4.run_experiment(cfg)
g.check_true("reduced_pipeline_executes", len(run_a["main"]["synthetic"]) >= 2,
             f"quick config cells: {sorted(run_a['main']['synthetic'])}")
det = True
for d in run_a["main"]["synthetic"]:
    for method in ("ADCPE", "CAPRISE"):
        for key in ("asr@1", "asr@5", "asr@10", "kendall", "spearman", "dist_err"):
            if run_a["main"]["synthetic"][d][method][key] != run_b["main"]["synthetic"][d][method][key]:
                det = False
g.check_true("pipeline_deterministic_same_seed", det,
             "two runs with identical seed and seed-derived nonces give identical metrics")

# --------------------------------------------------------------------------- #
# 2) noiseless self-check: E' = s*E -> ASR = 1, dist_err = 0
# --------------------------------------------------------------------------- #
sc = run_a["selfcheck"]
g.check_true("selfcheck_noiseless_asr1", abs(sc["asr@1"] - 1.0) < 1e-12 and abs(sc["asr@10"] - 1.0) < 1e-12,
             f"ASR@1={sc['asr@1']}, ASR@10={sc['asr@10']}")
g.check_true("selfcheck_noiseless_dist_err_zero", sc["dist_err"] < 1e-12,
             f"dist_err={sc['dist_err']:.3e} (no NaN/inf)")
g.check_true("selfcheck_kendall_one", abs(sc["kendall"] - 1.0) < 1e-12, f"tau={sc['kendall']}")

# --------------------------------------------------------------------------- #
# 3) full artifact: load, or generate it once if missing / stale
# --------------------------------------------------------------------------- #
need_full = True
artifact = None
if ARTIFACT.exists():
    try:
        artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        required = {"paper_dims_synthetic", "real_m_sweep", "normalization_control", "beta_sweep", "main", "findings"}
        need_full = not required.issubset(artifact.keys())
    except Exception:
        need_full = True
if need_full:
    print("  [gate] full exp4 artifact missing/stale -> running full experiment (~30 s)", flush=True)
    full_cfg = exp4.default_config()
    artifact = exp4.run_experiment(full_cfg)
    artifact["config"] = full_cfg
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    exp4.write_csv(artifact, CSV_PATH)
    exp4.make_figure(artifact, FIG_PATH)
g.check_true("full_artifact_present", ARTIFACT.exists() and artifact is not None,
             str(ARTIFACT.relative_to(ROOT)))

# --------------------------------------------------------------------------- #
# 4) artifact completeness: real GTR + synthetic dims + paper grid + sweeps
# --------------------------------------------------------------------------- #
real = artifact["main"]["real_gtr"]
real_cell = real.get("768")
real_ok = isinstance(real_cell, dict) and "ADCPE" in real_cell
g.check_true("real_gtr_d768_cell", real_ok, f"real_gtr keys={list(real)}")
if real_ok:
    prov = real_cell.get("provenance", {})
    g.check_true("real_gtr_uses_passage_emb", prov.get("source") == "data/passage_emb.npy",
                 f"source={prov.get('source')}")
    g.check("real_gtr_m_at_least_5000", 5000, int(real_cell["ADCPE"]["m"]),
            f"m={real_cell['ADCPE']['m']}", passed=int(real_cell["ADCPE"]["m"]) >= 5000)
    g.check_true("real_gtr_queries_at_least_100", int(real_cell["ADCPE"]["n_queries"]) >= 100,
                 f"|Q|={real_cell['ADCPE']['n_queries']}")
synth = artifact["main"]["synthetic"]
g.check_true("synthetic_dims_1536_3072", {"1536", "3072"}.issubset(synth.keys()),
             f"synthetic dims={sorted(synth)}")
paper_dims = sorted(int(d) for d in artifact.get("paper_dims_synthetic", {}))
g.check("paper_fig5_dim_grid", exp4.PAPER_FIG5["dims"], paper_dims,
        "paper Fig. 5 embedding sizes {64,...,2048}",
        passed=paper_dims == exp4.PAPER_FIG5["dims"])
g.check_true("m_sweeps_and_beta_sweep_present",
             len(artifact.get("m_sweep", {})) >= 2 and len(artifact.get("real_m_sweep", {})) >= 2
             and len(artifact.get("beta_sweep", {})) >= 2,
             f"m_sweep={sorted(artifact.get('m_sweep', {}))}, real_m_sweep={sorted(artifact.get('real_m_sweep', {}))}, "
             f"beta_sweep={sorted(artifact.get('beta_sweep', {}))}")

# --------------------------------------------------------------------------- #
# 5) metric ranges / sanity over every cell of the artifact
# --------------------------------------------------------------------------- #
cells = list(_iter_cells(artifact))
g.check_true("artifact_has_cells", len(cells) >= 20, f"{len(cells)} metric cells")
range_ok = True
bad = []
for sec, dim, cell in cells:
    for method in ("ADCPE", "CAPRISE"):
        c = cell[method]
        for t in (1, 5, 10):
            v = c.get(f"asr@{t}")
            if not _finite(v) or not (0.0 <= float(v) <= 1.0):
                range_ok = False
                bad.append(f"{sec}/{dim}/{method}/asr@{t}={v}")
        for key, lo, hi in (("kendall", -1.0, 1.0), ("spearman", -1.0, 1.0)):
            v = c.get(key)
            if not _finite(v) or not (lo <= float(v) <= hi):
                range_ok = False
                bad.append(f"{sec}/{dim}/{method}/{key}={v}")
        v = c.get("dist_err")
        if not _finite(v) or float(v) < 0.0:
            range_ok = False
            bad.append(f"{sec}/{dim}/{method}/dist_err={v}")
g.check_true("metrics_in_range_and_finite", range_ok, f"{len(cells)} cells; bad={bad[:5]}")

asr10 = [c[m]["asr@10"] for _, _, c in cells for m in ("ADCPE", "CAPRISE")]
g.check_true("attack_is_successful_not_degenerate", float(np.median(asr10)) > 0.6,
             f"median ASR@10 over all cells={np.median(asr10):.3f}")
g.check_true("not_perfect_no_plaintext_leak", float(np.mean(asr10)) < 0.999,
             f"mean ASR@10={np.mean(asr10):.4f} (<1: encryption noise does flip ranks)")

# --------------------------------------------------------------------------- #
# 6) paper's qualitative claim: CAPRISE ASR@10 lower than ADCPE in every cell
# --------------------------------------------------------------------------- #
reversals = []
for sec, dim, cell in cells:
    if cell["CAPRISE"]["asr@10"] >= cell["ADCPE"]["asr@10"]:
        reversals.append(
            f"{sec}/d={dim}: CAPRISE {cell['CAPRISE']['asr@10']:.3f} >= ADCPE {cell['ADCPE']['asr@10']:.3f}"
        )
g.check_true("caprise_asr10_lower_in_all_cells", len(reversals) == 0,
             f"{len(cells)} cells checked; reversals={reversals[:5]}")
main_cells = [(sec, dim, c) for sec, dim, c in cells if sec in ("main_synthetic", "main_real_gtr")]
deltas = [c["CAPRISE"]["asr@10"] - c["ADCPE"]["asr@10"] for _, _, c in main_cells]
g.check_true("main_delta_direction_consistent", all(d < 0 for d in deltas),
             f"main cells CAPRISE-ADCPE ASR@10 deltas (pt)=" 
             + ", ".join(f"{d*100:+.1f}" for d in deltas))

# --------------------------------------------------------------------------- #
# 7) finding A19: CAPRISE leaks *more* distance-reconstruction error
# --------------------------------------------------------------------------- #
ratios = []
for sec, dim, cell in cells:
    e_ad, e_cap = cell["ADCPE"]["dist_err"], cell["CAPRISE"]["dist_err"]
    ratios.append(e_cap / e_ad if e_ad else float("nan"))
ratios = np.asarray(ratios)
g.check_true("caprise_more_dist_err", bool((ratios > 1.0).all()),
             f"CAPRISE/ADCPE dist_err ratio min={ratios.min():.2f} max={ratios.max():.2f}")
g.check_true("dist_err_ratio_theory_consistent", bool(((ratios >= 1.3) & (ratios <= 2.5)).all()),
             f"all ratios in [1.3,2.5] (radii ratio 1.5 + second-order term); "
             f"median={np.median(ratios):.2f}")
if real_ok:
    rr = real_cell["CAPRISE"]["dist_err"] / real_cell["ADCPE"]["dist_err"]
    g.check_true("real_gtr_dist_err_ratio_gt_1_5", rr > 1.5,
                 f"real GTR d=768 ratio={rr:.2f} (M8/A19: CAPRISE ~1.8x)")

# --------------------------------------------------------------------------- #
# 8) paper Fig. 5 anchor (d=768) + digitised reference stored
# --------------------------------------------------------------------------- #
pf = artifact.get("paper_fig5_reference", {})
g.check("paper_fig5_reference_stored", True,
        pf.get("dims") == exp4.PAPER_FIG5["dims"] and len(pf.get("CAPRISE", [])) == 12,
        "digitised paper Fig. 5 values present in artifact",
        passed=bool(pf.get("dims") == exp4.PAPER_FIG5["dims"] and len(pf.get("CAPRISE", [])) == 12))
anchor = artifact.get("paper_dims_synthetic", {}).get("768", {})
i768 = exp4.PAPER_FIG5["dims"].index(768)
anchor_ok = True
anchor_detail = []
for method in ("ADCPE", "CAPRISE"):
    ours = anchor[method]["asr@10"]
    paper = exp4.PAPER_FIG5[method][i768] / 100.0
    anchor_detail.append(f"{method} ours={ours:.3f} paper={paper:.3f}")
    if abs(ours - paper) > 0.03:
        anchor_ok = False
g.check_true("paper_anchor_d768_within_3pp", anchor_ok, "; ".join(anchor_detail))

# findings recorded for the report
kinds = {f.get("kind") for f in artifact.get("findings", [])}
g.check_true("findings_recorded",
             {"main_cell", "dist_err_ratio", "paper_fig5_comparison", "real_m_density", "normalization_control"}.issubset(kinds),
             f"finding kinds={sorted(k for k in kinds if k)}")

# --------------------------------------------------------------------------- #
# 9) figure + CSV + notes artifacts
# --------------------------------------------------------------------------- #
if not FIG_PATH.exists() or FIG_PATH.stat().st_size < 20_000:
    exp4.make_figure(artifact, FIG_PATH)
g.check_true("figure_generated", FIG_PATH.exists() and FIG_PATH.stat().st_size > 20_000,
             f"{FIG_PATH.relative_to(ROOT)} ({FIG_PATH.stat().st_size if FIG_PATH.exists() else 0} bytes)")
if not CSV_PATH.exists():
    exp4.write_csv(artifact, CSV_PATH)
g.check_true("csv_generated", CSV_PATH.exists() and CSV_PATH.stat().st_size > 200,
             f"{CSV_PATH.relative_to(ROOT)}")
notes_text = NOTES_PATH.read_text(encoding="utf-8") if NOTES_PATH.exists() else ""
g.check_true("notes_written", len(notes_text) > 3000 and "ASR" in notes_text and "A19" in notes_text
             and "Fig. 5" in notes_text,
             f"{NOTES_PATH.relative_to(ROOT)} ({len(notes_text)} chars)")

# --------------------------------------------------------------------------- #
# 10) runtime sanity: main table must be cheap enough to be reproducible
# --------------------------------------------------------------------------- #
rt = artifact.get("runtime", {}).get("seconds", 9999)
g.check_true("full_run_time_reasonable", float(rt) < 600.0, f"exp4 full run {rt} s")

g.note(
    "ASR definition (A12, paper gives none): ASR@t = |top_t(cipher) & top_t(plain)|/t averaged over "
    "100 stored-ciphertext queries; auxiliary Kendall/Spearman and distance-reconstruction error. "
    "Full definition in experiments/exp4_vector_analysis.py + notes/exp4_vector_analysis.md."
)
g.note(
    "Finding A19/F2: CAPRISE's vector-analysis advantage is real but modest "
    f"({', '.join(f'{d*100:+.1f} pt' for d in deltas)} ASR@10 on the main cells); the stable, large "
    f"difference is the distance-reconstruction error ({np.median(ratios):.2f}x)."
)
g.note(
    "Paper Fig. 5 values were digitised from figures/exp_asr.png (+-1 pp) because the paper only "
    "publishes them as a bar chart; at d=768 our controlled synthetic point matches within 1.3 pp. "
    "The paper's ASR-vs-dimension *trend* (53.5%->88.8%) is NOT reproduced by a controlled "
    "dimension sweep (flat 81.7-84.9%) - recorded as paper ambiguity A13 (dimensions 64..2048 cannot "
    "all come from the 768-dim gtr-t5-base model)."
)

g.finish()
