"""M20 gate -- ASR protocol study (paper Fig.5 trend reproduction), supplementary.

Acceptance criteria (docs/MODULE_PLAN.md M20; task C1/C2 follow-up):

1. Artifact results/exp6_asr_protocol.json exists and was produced on real MS
   MARCO GTR embeddings (data/passage_emb.npy or its 50k backup), with the
   attack protocol from the paper (database vector as test query, CAPRISE vs
   ADCPE ciphertext-distance ranking, ASR@t = top-t overlap) and m=5000 /
   nq>=100 for the main grid.
2. Three data protocols are enumerated on the paper's 12 Fig.5 sizes:
   H1 Gaussian projection + L2 renormalisation, H2 coordinate truncation with
   NO renormalisation (zero-padded above 768), H3 truncation + L2
   renormalisation.  Every cell stores ASR@1/5/10 and Kendall tau for both
   methods; all values finite and in [0,1].
3. The comparison to the digitised paper Fig.5 is stored per protocol with
   Spearman correlation and mean absolute deviation.  The trend test is:
     * H2 (truncation, no renormalisation) reproduces the paper's monotone
       trend: Spearman(CAPRISE) >= 0.9 and its ASR@10 range >= 25 pp;
     * H1 and H3 (unit-sphere protocols) are flat: ASR@10 range <= 20 pp;
     * H2's mean absolute deviation from the paper is <= 15 pp (residual level
       offset documented, not fitted away).
4. A beta sweep (0.2/0.3/0.4) and an m sweep are present; for each beta the
   truncation protocol keeps Spearman >= 0.9 against the paper trend.
5. Human-readable notes/exp6_asr_protocol.md exist and document the protocol,
   the result and the residual offset (finding F11 update).

The gate is deliberately a *trend/protocol* gate: the paper does not state the
data protocol, beta or ASR definition for Fig.5 (F11/A13), so no gate requires
an exact numerical match; it requires that the question is answered with
evidence and that the flat alternative protocols are excluded.

Run:  .venv/bin/python -m gates.gate_m20_asr_protocol
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from gates.common import Gate, add_src_to_path

add_src_to_path()
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ART = RESULTS / "exp6_asr_protocol.json"
NOTES = ROOT / "notes" / "exp6_asr_protocol.md"
PAPER_DIMS = (64, 128, 256, 312, 384, 512, 768, 896, 1024, 1536, 1792, 2048)

g = Gate("M20", "asr_protocol", owner="lead",
         paper_ref="arXiv:2601.12331 Fig.5 / Sec. V-C (ASR vs embedding size)")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


if not ART.exists():
    g.check_true("artifact_present", False, detail=f"missing {ART}")
    g.finish()
    raise SystemExit(1)

art = json.loads(ART.read_text(encoding="utf-8"))
paper = art.get("paper_fig5") or {}
paper_c = np.asarray(paper.get("CAPRISE", []), dtype=float)
paper_a = np.asarray(paper.get("ADCPE", []), dtype=float)

g.check_true("artifact_present", True, detail=f"{ART.relative_to(ROOT)}")
g.check("paper_reference_points", 12, len(paper_c),
        "digitised paper Fig.5 values for the 12 sizes",
        passed=len(paper_c) == 12 and len(paper_a) == 12)

setup = art.get("setup", {})
cells = art.get("cells", {})
cmp = art.get("comparison_to_paper", {})
g.check_true(
    "protocols_complete",
    all(name in cells and all(str(d) in cells[name] for d in PAPER_DIMS)
        for name in ("H1_proj_renorm", "H2_trunc_raw", "H3_trunc_renorm")),
    detail=f"protocols={sorted(cells)}; 12 paper dims each: "
           f"{[len(cells.get(n, {})) for n in ('H1_proj_renorm', 'H2_trunc_raw', 'H3_trunc_renorm')]}",
)
g.check_true(
    "attack_setup_recorded",
    setup.get("m_main") == 5000 and int(setup.get("n_queries", 0)) >= 100
    and setup.get("s") == 3 and setup.get("beta") == 0.2,
    detail=f"m_main={setup.get('m_main')}, nq={setup.get('n_queries')}, "
           f"s={setup.get('s')}, beta={setup.get('beta')}, source={setup.get('source')}",
)

bad = []
for name, per_dim in cells.items():
    for d, cell in per_dim.items():
        for method in ("CAPRISE", "ADCPE"):
            for key in ("asr@1", "asr@5", "asr@10", "kendall"):
                v = cell[method][key]
                if not _finite(v) or not (-1e-9 <= v <= 1 + 1e-9 and -1 <= v <= 1 + 1e-9):
                    bad.append((name, d, method, key, v))
g.check_true("asr_values_finite_and_in_range", not bad, detail=f"bad={bad[:5]}")

# trend / level checks
h2 = cmp.get("H2_trunc_raw", {})
h1 = cmp.get("H1_proj_renorm", {})
h3 = cmp.get("H3_trunc_renorm", {})
g.check_true(
    "truncation_protocol_reproduces_paper_trend",
    _finite(h2.get("spearman_caprise_vs_paper")) and h2["spearman_caprise_vs_paper"] >= 0.9
    and h2.get("our_caprise_range_pp", [0, 0])[1] - h2.get("our_caprise_range_pp", [0, 0])[0] >= 25.0,
    detail=f"H2 truncation: Spearman(CAPRISE)={h2.get('spearman_caprise_vs_paper'):.3f}, "
           f"range={h2.get('our_caprise_range_pp')} pp vs paper {h2.get('paper_caprise_range_pp')} pp",
)
g.check_true(
    "unit_sphere_protocols_are_flat",
    h1.get("our_caprise_range_pp", [0, 0])[1] - h1.get("our_caprise_range_pp", [0, 0])[0] <= 20.0
    and h3.get("our_caprise_range_pp", [0, 0])[1] - h3.get("our_caprise_range_pp", [0, 0])[0] <= 20.0,
    detail=f"H1 range={h1.get('our_caprise_range_pp')} pp, H3 range={h3.get('our_caprise_range_pp')} pp "
           f"(both flat => the paper's trend is not a unit-sphere dimension effect)",
)
g.check_true(
    "truncation_level_within_15pp",
    _finite(h2.get("mean_abs_delta_caprise_pp")) and h2["mean_abs_delta_caprise_pp"] <= 15.0,
    detail=f"H2 mean|Δ| vs paper = {h2.get('mean_abs_delta_caprise_pp'):.1f} pp (CAPRISE), "
           f"{h2.get('mean_abs_delta_adcpe_pp'):.1f} pp (ADCPE); residual offset documented, not fitted",
)

beta_sweep = art.get("beta_sweep_truncation", {})
beta_ok = len(beta_sweep) == 3
for beta, payload in beta_sweep.items():
    c = payload.get("comparison_to_paper", {})
    beta_ok = beta_ok and _finite(c.get("spearman_caprise_vs_paper")) and c["spearman_caprise_vs_paper"] >= 0.9
g.check_true(
    "beta_sweep_keeps_trend",
    beta_ok,
    detail=f"betas={sorted(beta_sweep)}; Spearman(CAPRISE)="
           f"{[round(beta_sweep[b]['comparison_to_paper']['spearman_caprise_vs_paper'], 3) for b in sorted(beta_sweep)]}",
)
g.check_true(
    "m_sweep_recorded",
    bool(art.get("m_sweep")) and all(len(v) >= 2 for v in art["m_sweep"].values()),
    detail=f"m sweep dims={sorted(art.get('m_sweep', {}))}; m values="
           f"{[sorted(v) for v in art.get('m_sweep', {}).values()]}",
)
g.check_true(
    "conclusion_recorded",
    bool(art.get("summary", {}).get("per_protocol")) and bool(art.get("summary", {}).get("note")),
    detail=f"best protocol by mean|Δ| = {art.get('summary', {}).get('best_protocol_by_mean_abs_delta')}",
)
g.check_true(
    "notes_written",
    NOTES.exists() and len(NOTES.read_text(encoding="utf-8")) > 1500
    and "F11" in NOTES.read_text(encoding="utf-8"),
    detail=f"{NOTES.relative_to(ROOT)} exists={NOTES.exists()}",
)

res = g.finish()
raise SystemExit(0 if res["pass"] else 1)
