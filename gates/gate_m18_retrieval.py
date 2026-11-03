"""M18 gate: exp2 retrieval-accuracy experiment (supplementary accuracy study).

Protocol under test (docs/MODULE_PLAN.md M18 + Lead's exp2 design)
------------------------------------------------------------------
Real-data retrieval pipeline from M14/M15 (CAPRISE Phase-1 upload + Phase-2
retrieval with local DecDB + exact plaintext re-rank) compared against the
plaintext baseline under the five required axes:

  (a) plaintext baseline
  (b) CAPRISE k' = k            (strict Algorithm 2, no DistanceDP)
  (c) CAPRISE oversampling      (k' = 15 and 2k)
  (d) CAPRISE + DistanceDP      (r = 0.02 / 0.05, k' from Theorem 1)
  (e) beta sensitivity          (beta in {0.05, 0.1, 0.2, 0.4})
  plus ADCPE (SAP) baseline.

Gate criteria
-------------
1. artifacts exist and are non-empty: results/exp2_retrieval.json/.csv,
   figures/fig_retrieval.png, notes/exp2_retrieval.md;
2. setup recorded: m>=20000, n_queries>=200, k=10, dim=768, s=3;
3. all required configurations present; metrics finite and in [0,1];
4. plaintext baseline sanity (recall vs itself = 1, qrels MRR plausible);
5. CAPRISE strict k'=k keeps >= 90% of the plaintext top-10; oversampling
   recovers it (k'=2k gives >= 0.99); DP k' equals Theorem 1 and recall stays
   >= 0.99; ADCPE baseline present;
6. beta sensitivity is present for 4 values and is monotone (smaller beta ->
   less noise -> higher fidelity);
7. end-to-end qrels MRR@10 of every encrypted config stays within 0.01 of the
   plaintext MRR (re-rank restores ordering);
8. timings recorded and consistent: encrypted k'=k latency <= 2x plaintext,
   DP(r=0.05) latency > k'=k latency (k'=785 decryptions); Phase-1 upload time
   and throughput recorded;
9. CSV rows == number of configs, figure non-trivial.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from gates.common import Gate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
NOTES = ROOT / "notes"

g = Gate("M18", "retrieval", owner="data-retrieval",
         paper_ref="§V accuracy claim + MODULE_PLAN M18 / exp2 design")

json_path = RESULTS / "exp2_retrieval.json"
csv_path = RESULTS / "exp2_retrieval.csv"
png_path = FIGURES / "fig_retrieval.png"
md_path = NOTES / "exp2_retrieval.md"

files_ok = all(p.exists() and p.stat().st_size > 0 for p in (json_path, csv_path, png_path, md_path))
g.check_true("01_artifacts_present", files_ok,
             f"json={json_path.stat().st_size if json_path.exists() else 0}B, "
             f"csv={csv_path.stat().st_size if csv_path.exists() else 0}B, "
             f"png={png_path.stat().st_size if png_path.exists() else 0}B, "
             f"md={md_path.stat().st_size if md_path.exists() else 0}B")
if not files_ok:
    g.finish()
    sys.exit(1)

data = json.loads(json_path.read_text())
meta = data["meta"]
configs = {c["config"]: c for c in data["configs"]}
plain = data["plaintext"]

# --------------------------------------------------------------------------- #
# 2. experiment setup
# --------------------------------------------------------------------------- #
g.check_true("02_setup_scale", meta.get("m", 0) >= 20_000 and meta.get("n_queries", 0) >= 200,
             f"m={meta.get('m')}, n_queries={meta.get('n_queries')}")
g.check_true("03_setup_params", meta.get("k") == 10 and meta.get("dim") == 768 and meta.get("s") == 3,
             f"k={meta.get('k')}, dim={meta.get('dim')}, s={meta.get('s')}, beta={meta.get('beta_default')}")

# --------------------------------------------------------------------------- #
# 3. required configurations + metric sanity
# --------------------------------------------------------------------------- #
REQUIRED = [
    "caprise_kp=k",
    "caprise_kp=15",
    "caprise_kp=2k",
    "caprise_dp_r0.02_theorem-kp",
    "caprise_dp_r0.05_theorem-kp",
    "adcpe_kp=k",
    "adcpe_kp=2k",
    "caprise_beta0.05_kp=k",
    "caprise_beta0.1_kp=k",
    "caprise_beta0.4_kp=k",
]
missing = [c for c in REQUIRED if c not in configs]
g.check_true("04_required_configs_present", not missing,
             f"{len(configs)} configs (+ plaintext); missing={missing}; names={sorted(configs)}")

bad = []
for name, c in configs.items():
    r = c.get("recall@10_vs_plaintext")
    q = c.get("qrels", {})
    if not (isinstance(r, (int, float)) and 0.0 <= r <= 1.0):
        bad.append(f"{name}:recall")
    for key in ("MRR@10", "Recall@10", "nDCG@10"):
        if not (isinstance(q.get(key), (int, float)) and 0.0 <= q[key] <= 1.0):
            bad.append(f"{name}:{key}")
    if not (isinstance(c.get("k_prime_mean"), (int, float)) and c["k_prime_mean"] >= 1):
        bad.append(f"{name}:k'")
    if not (isinstance(c.get("latency_ms", {}).get("mean"), (int, float)) and c["latency_ms"]["mean"] >= 0):
        bad.append(f"{name}:latency")
g.check_true("05_metrics_finite_in_range", not bad, f"invalid entries={bad}")

# --------------------------------------------------------------------------- #
# 4. plaintext baseline sanity
# --------------------------------------------------------------------------- #
g.check_true("06_plaintext_recall_is_one", plain["recall@10_vs_plaintext"] == 1.0,
             f"plaintext recall vs itself = {plain['recall@10_vs_plaintext']}")
pmrr = plain["qrels"]["MRR@10"]
g.check_true("07_plaintext_qrels_reasonable", 0.2 <= pmrr <= 0.9 and plain["qrels"]["n_queries_with_qrels"] == meta["n_queries"],
             f"plaintext qrels MRR@10={pmrr:.4f} for {plain['qrels']['n_queries_with_qrels']} queries "
             f"(full-corpus reference ~0.39, 50k-subset corpus higher)")

# --------------------------------------------------------------------------- #
# 5. CAPRISE / DP / oversampling / ADCPE behaviour
# --------------------------------------------------------------------------- #
rec_k = configs["caprise_kp=k"]["recall@10_vs_plaintext"]
g.check_true("08_caprise_strict_kp_eq_k_high_fidelity", rec_k >= 0.90,
             f"CAPRISE k'=k recall@10 vs plaintext = {rec_k:.4f} (>=0.90; encryption-noise boundary flips)")

rec_2k = configs["caprise_kp=2k"]["recall@10_vs_plaintext"]
rec_15 = configs["caprise_kp=15"]["recall@10_vs_plaintext"]
g.check_true("09_oversampling_recovers", rec_15 >= rec_k and rec_2k >= rec_k and rec_2k >= 0.99,
             f"recall: k'=k {rec_k:.4f}, k'=15 {rec_15:.4f}, k'=2k {rec_2k:.4f}")

theorem = meta.get("theorem_k_prime", {})
ok_dp = True
detail_dp = []
for r, name in ((0.02, "caprise_dp_r0.02_theorem-kp"), (0.05, "caprise_dp_r0.05_theorem-kp")):
    c = configs[name]
    expected = theorem.get(f"r={r}")
    rec = c["recall@10_vs_plaintext"]
    detail_dp.append(f"r={r}: k'={c['k_prime_mean']:.0f} (expected {expected}), recall={rec:.4f}")
    ok_dp &= (expected is not None and abs(c["k_prime_mean"] - expected) < 1e-9 and rec >= 0.99)
g.check_true("10_dp_theorem_kprime_and_recall", bool(ok_dp), "; ".join(detail_dp))

adc = configs["adcpe_kp=k"]
g.check_true("11_adcpe_baseline_present", 0.0 <= adc["recall@10_vs_plaintext"] <= 1.0,
             f"ADCPE k'=k recall@10 vs plaintext = {adc['recall@10_vs_plaintext']:.4f}")

# --------------------------------------------------------------------------- #
# 6. beta sensitivity (monotone: smaller beta -> less noise -> higher fidelity)
# --------------------------------------------------------------------------- #
betas = {float(b): v for b, v in data["beta_sensitivity"]}
g.check_true("12_beta_sweep_4_values", set(betas) == {0.05, 0.1, 0.2, 0.4},
             f"beta values={sorted(betas)}")
rec_b = {b: betas[b]["recall@10_vs_plaintext"] for b in sorted(betas)}
monotone = all(rec_b[a] >= rec_b[b] - 1e-9 for a, b in zip(sorted(betas), sorted(betas)[1:]))
g.check_true("13_beta_monotone_fidelity", monotone,
             "recall@10 vs plaintext: " + ", ".join(f"beta={b}:{rec_b[b]:.4f}" for b in sorted(betas)))

# --------------------------------------------------------------------------- #
# 7. qrels MRR of every encrypted config within 0.01 of plaintext
# --------------------------------------------------------------------------- #
deltas = {name: abs(c["qrels"]["MRR@10"] - pmrr) for name, c in configs.items()}
worst = max(deltas, key=deltas.get)
g.check_true("14_qrels_mrr_preserved", deltas[worst] <= 0.01,
             f"max |qrels MRR@10 - plaintext| = {deltas[worst]:.5f} ({worst}); "
             f"local decryption + exact re-rank restores ordering")

# --------------------------------------------------------------------------- #
# 8. timings / Phase-1 throughput
# --------------------------------------------------------------------------- #
enc_lat = configs["caprise_kp=k"]["latency_ms"]["mean"]
plain_lat = plain["latency_ms"]["mean"]
g.check_true("15_encrypted_latency_reasonable", enc_lat <= 2.0 * plain_lat,
             f"CAPRISE k'=k {enc_lat:.2f} ms/query vs plaintext {plain_lat:.2f} ms/query "
             f"(server search ~{configs['caprise_kp=k']['server_search_ms']['mean']:.2f} ms)")
dp5 = configs["caprise_dp_r0.05_theorem-kp"]["latency_ms"]["mean"]
g.check_true("16_dp_cost_grows_with_kprime", dp5 > enc_lat,
             f"DP(r=0.05,k'=785) {dp5:.2f} ms/query > CAPRISE k'=k {enc_lat:.2f} ms/query "
             f"(cost dominated by decrypting k' candidates)")

phase1 = [c for c in data["configs"] if c.get("upload_s")]
ok_upload = bool(phase1) and all(0 < c["upload_s"] < 120 for c in phase1)
max_thr = max((c["enc_throughput_vec_s"] or 0) for c in phase1) if phase1 else 0
g.check_true("17_phase1_upload_measured", ok_upload and max_thr >= 1000,
             f"{len(phase1)} uploads recorded; max EncDB+AES throughput = {max_thr:.0f} vec/s")

# --------------------------------------------------------------------------- #
# 9. CSV / figure consistency
# --------------------------------------------------------------------------- #
with open(csv_path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
g.check_true("18_csv_matches_json", len(rows) == len(data["configs"]),
             f"CSV rows={len(rows)} == configs={len(data['configs'])}")
g.check_true("19_figure_nontrivial", png_path.stat().st_size > 10_000,
             f"figures/fig_retrieval.png = {png_path.stat().st_size} B")

g.note(f"exp2 on real data: m={meta['m']}, n_queries={meta['n_queries']}, beta_default={meta['beta_default']}, "
       f"total run {meta['total_wall_s']}s")
g.note(f"plaintext qrels MRR@10={pmrr:.4f} | CAPRISE k'=k recall={rec_k:.4f} -> k'=2k recall={rec_2k:.4f}")
g.note("paper gives no retrieval table; this supports the qualitative 'high retrieval accuracy' claim. "
       "Paper Table III uses m=100,000; here m=50,000, and Delta-alpha = r (SPEC §5 caveat noted).")
g.note(f"beta sensitivity: {', '.join(f'{b}:{rec_b[b]:.4f}' for b in sorted(betas))}")

res = g.finish()
print(json.dumps({k: v for k, v in res.items() if k in ("module", "name", "pass")}, indent=2))
sys.exit(0 if res["pass"] else 1)
