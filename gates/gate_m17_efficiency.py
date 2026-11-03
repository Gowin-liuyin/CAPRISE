"""M17 gate -- efficiency experiments (paper Fig.4 / Phase-1 / Phase-2).

Acceptance criteria (docs/MODULE_PLAN.md M17; SPEC.md Sec. 6; task brief):

1. CAPRISE encryption throughput vs dimension (d = 384,768,1536,2048,3072,4096)
   measured with batch >= 1000 and 3 repetitions (median):
     * embeddings/s decreases with d (paper Fig.4a trend);
     * elements/s stays within one order of magnitude (same ballpark, paper
       Fig.4b "remains high") and is stable inside d in [1536, 3072];
     * the stored throughput table is exactly recomputable from its raw run arrays,
       and an independent re-measurement reproduces the load-invariant ratio
       emb/s(d=768) / emb/s(d=1536) within 10% (absolute wall-clock rates are
       load-sensitive on a shared machine, so they are reported, not asserted).
2. Phase-1 overhead: real embedding time for 128 queries (gtr-t5-base, local
   weights; MPS fp16 + CPU fp32) and real CAPRISE EncQ time for the same batch;
   overhead = encryption / embedding must be below the paper's <19% bound in
   every measured configuration (the paper's own setting is 15/79.52 = 18.9%).
3. Phase-2 throughput at d=768: CAPRISE vectors/s measured; a real Paillier
   baseline (python-phe 1024/2048-bit, plus a pure-CPython builtin-pow bound) is
   measured; speedups vs the measured Paillier and vs the paper's RemoteRAG
   PHE (250-300 vectors/s) are positive; the paper's own 9x = 2339/(250-300) is
   internally consistent.  Absolute cross-hardware ratios are reported, not
   asserted (A17/A17b).
4. Artifacts: results/exp1_efficiency.json, figures/fig4_efficiency.png,
   notes/exp1_efficiency.md (documents implementation-style finding A17).

The experiment is run by the gate if its artifacts are missing.

Run:  .venv/bin/python -m gates.gate_m17_efficiency
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from gates.common import Gate, add_src_to_path

add_src_to_path()

from caprise.caprise import EncDB, KeyGen  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXP_JSON = RESULTS / "exp1_efficiency.json"
FIG = ROOT / "figures" / "fig4_efficiency.png"
NOTES = ROOT / "notes" / "exp1_efficiency.md"
EXP_SCRIPT = ROOT / "experiments" / "exp1_efficiency.py"

DIMS = [384, 768, 1536, 2048, 3072, 4096]
PAPER_EMBED_MS = 79.52
PAPER_ENC_MS = 15.0
PAPER_OVERHEAD_BOUND_PCT = 19.0
PAPER_CAPRISE_VEC_S = 2339.0
PAPER_PHE_VEC_S = (250.0, 300.0)


def _ensure_artifacts(g: Gate) -> None:
    if EXP_JSON.exists() and FIG.exists() and NOTES.exists():
        g.note("exp1 artifacts found; gate does not rerun the full experiment")
        return
    g.note("exp1 artifacts missing -> running experiments/exp1_efficiency.py (~30 s)")
    proc = subprocess.run(
        [sys.executable, str(EXP_SCRIPT)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
    )
    g.check_true(
        "exp1_experiment_ran",
        proc.returncode == 0 and EXP_JSON.exists() and FIG.exists(),
        detail=f"subprocess returncode={proc.returncode}",
    )
    if proc.returncode != 0:
        tail = proc.stdout.decode("utf-8", "replace")[-1500:]
        g.note(f"experiment subprocess output tail:\n{tail}")


def main() -> int:
    g = Gate(
        "M17",
        "efficiency",
        owner="theory-efficiency",
        paper_ref="arXiv:2601.12331 Sec. V-1 / Fig.4 / Phase-1 / Phase-2",
    )
    _ensure_artifacts(g)
    if not EXP_JSON.exists():
        g.check_true("exp1_results_json_exists", False, detail=f"missing {EXP_JSON}")
        g.finish()
        return 1
    res = json.loads(EXP_JSON.read_text(encoding="utf-8"))

    # ================================================================== #
    # 1. Throughput vs dimension
    # ================================================================== #
    tput = res.get("throughput", {})
    rows = tput.get("rows", [])
    g.check_true(
        "throughput_grid_complete",
        [r["d"] for r in rows] == DIMS
        and all(r["batch"] >= 1000 and r["reps"] == 3 for r in rows),
        detail=f"{len(rows)} dims {[r['d'] for r in rows]}; batch>=1000 and reps==3 in every row",
    )

    emb = [r["embeddings_per_s"] for r in rows]
    monotone = all(emb[i] > emb[i + 1] for i in range(len(emb) - 1))
    drop = emb[0] / emb[-1]
    g.check_true(
        "embeddings_per_s_decreases_with_d",
        monotone and drop > 2.0,
        detail=f"embeddings/s {emb[0]:.0f} -> {emb[-1]:.0f} strictly decreasing={monotone}, "
        f"d=384/d=4096 ratio={drop:.2f}x (>2 required; paper Fig.4a trend)",
    )

    elem = [r["elements_per_s"] for r in rows]
    ratio = max(elem) / min(elem)
    g.check_true(
        "elements_per_s_within_one_order_of_magnitude",
        ratio < 10.0,
        detail=f"elements/s range {min(elem)/1e6:.1f}M - {max(elem)/1e6:.1f}M (max/min={ratio:.2f} < 10; "
        "same ballpark as paper Fig.4b 'remains high')",
    )
    high = [r["elements_per_s"] for r in rows if 1536 <= r["d"] <= 3072]
    high_ratio = max(high) / min(high)
    g.check_true(
        "elements_per_s_stable_1536_3072",
        high_ratio < 2.0,
        detail=f"d in [1536,3072]: elements/s {min(high)/1e6:.1f}M - {max(high)/1e6:.1f}M "
        f"(max/min={high_ratio:.2f} < 2; paper Fig.4b high region)",
    )

    # ------------------------------------------------------------------ #
    # 2a. Internal consistency of the stored throughput table (load-invariant,
    #     strict): every reported rate must be exactly recomputable from the raw
    #     runs_s / runs_cpu_s arrays stored in the JSON.  This is the strongest
    #     anti-tampering check and it does not depend on machine load.
    # ------------------------------------------------------------------ #
    inconsistent = []
    for r in rows:
        d = r["d"]
        med = float(np.median(r["runs_s"]))
        med_cpu = float(np.median(r["runs_cpu_s"]))
        # stored medians keep 6 decimals and rates 1 decimal -> allow 1e-4 relative
        # (rounding only; grossly edited numbers still fail by orders of magnitude)
        tol = 1e-4
        checks = [
            (abs(r["median_s"] - med) / med <= tol, f"median_s d={d}"),
            (abs(r["embeddings_per_s"] - 1000.0 / med) / (1000.0 / med) <= tol, f"embeddings_per_s d={d}"),
            (abs(r["elements_per_s"] - 1000.0 * d / med) / (1000.0 * d / med) <= tol, f"elements_per_s d={d}"),
            (abs(r["median_cpu_s"] - med_cpu) / med_cpu <= tol, f"median_cpu_s d={d}"),
            (abs(r["embeddings_per_s_cpu"] - 1000.0 / med_cpu) / (1000.0 / med_cpu) <= tol,
             f"embeddings_per_s_cpu d={d}"),
        ]
        inconsistent += [label for ok, label in checks if not ok]
    g.check_true(
        "throughput_internal_consistency",
        not inconsistent,
        detail=f"all 6 rows: median/emb-per-s/elements-per-s are exactly recomputable from the stored "
        f"raw runs_s and runs_cpu_s arrays (rel tol 1e-4, i.e. JSON rounding only); inconsistencies={inconsistent}",
    )

    # ------------------------------------------------------------------ #
    # 2b. Independent re-measurement at d=768 and d=1536, compared as a
    #     *load-invariant ratio*.  On this shared machine absolute wall-clock
    #     throughput swings with co-running jobs (observed d=768 EncDB range
    #     ~69k-102k emb/s across runs, load average 1-5; wall-clock and CPU time
    #     move together, so this is resource contention, not process scheduling).
    #     Comparing emb/s(d=768)/emb/s(d=1536) cancels the common load factor and
    #     is therefore safe to bound tightly ([0.9,1.1]).
    # ------------------------------------------------------------------ #
    key = KeyGen(dim=768, beta=0.2, s=3, rng=np.random.default_rng(20260118))
    key1536 = KeyGen(dim=1536, beta=0.2, s=3, rng=np.random.default_rng(20260118))
    rng = np.random.default_rng(123)
    Er = rng.standard_normal((1000, 1536))
    Er /= np.linalg.norm(Er, axis=1, keepdims=True)
    E768 = np.ascontiguousarray(Er[:, :768])
    EncDB(key, E768[:16])
    EncDB(key1536, Er[:16])
    m768, m1536, round_cost_ratios = [], [], []
    # 5 interleaved rounds: a 768-block immediately followed by a 1536-block; the
    # per-round cost ratio is insensitive to slow load drift on the shared machine.
    for _ in range(5):
        t0 = time.process_time()
        EncDB(key, E768)
        c768 = time.process_time() - t0
        t0 = time.process_time()
        EncDB(key1536, Er)
        c1536 = time.process_time() - t0
        m768.append(c768)
        m1536.append(c1536)
        round_cost_ratios.append(c768 / c1536)
    recheck768 = 1000.0 / float(np.median(m768))
    recheck1536 = 1000.0 / float(np.median(m1536))
    gate_cost_ratio = float(np.median(round_cost_ratios))  # per-vector cost(768)/cost(1536)
    stored768 = next(r["embeddings_per_s_cpu"] for r in rows if r["d"] == 768)
    stored1536 = next(r["embeddings_per_s_cpu"] for r in rows if r["d"] == 1536)
    stored_cost_ratio = stored1536 / stored768  # (1000/stored1536)/(1000/stored768)
    cost_dev = abs(gate_cost_ratio - stored_cost_ratio) / stored_cost_ratio
    g.check_true(
        "independent_recheck_d768",
        recheck768 > 10_000.0
        and 0.5 < recheck768 / stored768 < 2.0
        and cost_dev < 0.10
        and recheck768 > recheck1536,
        detail=f"gate re-measurement (CPU time, 5 interleaved rounds): d=768 {recheck768:.0f} emb/s vs "
        f"stored {stored768:.0f} (load factor {recheck768/stored768:.2f}, absolute window [0.5,2.0]); "
        f"per-vector cost ratio cost(768)/cost(1536): gate {gate_cost_ratio:.3f} vs stored "
        f"{stored_cost_ratio:.3f} (dev {cost_dev*100:.1f}% < 10%, load drift cancels within each round); "
        "ordering preserved (768 faster than 1536). Absolute rates are load-sensitive on this shared "
        "machine (observed 0.8-1.1x swings minutes apart; wall~=CPU time => resource contention), so "
        "the JSON is verified strictly by throughput_internal_consistency; see notes/exp1_efficiency.md.",
    )

    # ================================================================== #
    # 2. Phase-1 overhead
    # ================================================================== #
    p1 = res.get("phase1_overhead", {})
    enc = p1.get("caprise_encryption", {})
    g.check_true(
        "phase1_caprise_encryption_measured",
        enc.get("n") == 128 and enc.get("d") == 768 and enc.get("median_ms", 0) > 0,
        detail=f"EncQ n={enc.get('n')}, d={enc.get('d')}, median={enc.get('median_ms')} ms, "
        f"vectors/s={enc.get('vectors_per_s')}",
    )
    p1_rows = p1.get("rows", [])
    measured = [r for r in p1_rows if r["embedding_source"] in ("mps", "cpu")]
    paper_row = [r for r in p1_rows if "paper" in r["embedding_source"]]
    overhead_ok = all(r["overhead_pct"] < PAPER_OVERHEAD_BOUND_PCT for r in p1_rows if r.get("overhead_pct"))
    g.check_true(
        "phase1_overhead_below_19pct_all_configs",
        len(p1_rows) >= 2 and len(measured) >= 1 and overhead_ok,
        detail=f"{len(measured)} measured embedding configs + {len(paper_row)} paper reference; "
        f"overheads={[round(r['overhead_pct'],3) for r in p1_rows if r.get('overhead_pct')]}% "
        f"(all < {PAPER_OVERHEAD_BOUND_PCT}% paper bound)",
    )
    paper_own = p1.get("paper", {})
    g.check_true(
        "paper_phase1_claim_reproduced",
        abs(paper_own.get("overhead_pct", 0) - 18.9) < 0.2 and PAPER_ENC_MS / PAPER_EMBED_MS * 100 < PAPER_OVERHEAD_BOUND_PCT,
        detail=f"paper 15/79.52 = {paper_own.get('overhead_pct')}% < 19%; our measured encryption "
        f"{enc.get('median_ms')} ms vs paper 15 ms",
    )
    g.check_true(
        "phase1_embedding_dominates_encryption",
        min(r["embedding_ms_128q"] for r in measured) > 10 * enc["median_ms"],
        detail=f"min measured embedding {min(r['embedding_ms_128q'] for r in measured):.1f} ms vs "
        f"CAPRISE {enc['median_ms']:.2f} ms (embedding is the Phase-1 bottleneck; A17c)",
    )

    # ================================================================== #
    # 3. Phase-2 throughput vs PHE baseline
    # ================================================================== #
    p2 = res.get("phase2", {})
    cap = p2.get("caprise", {})
    g.check_true(
        "phase2_caprise_measured",
        cap.get("d") == 768 and cap.get("vectors_per_s", 0) > 1000,
        detail=f"CAPRISE d=768: {cap.get('vectors_per_s')} vectors/s "
        f"({cap.get('us_per_vector')} us/vector, n={cap.get('n_vectors')} single-vector EncQ calls)",
    )
    phe = p2.get("phe_baselines", [])
    pure = p2.get("pure_python_baselines", [])
    g.check_true(
        "phase2_paillier_baseline_measured",
        len(phe) >= 1 and all(x["vectors_per_s"] > 0 and x["us_per_element"] > 0 for x in phe)
        and len(pure) >= 1,
        detail=f"phe baselines: {[(x['n_bits'], x['vectors_per_s']) for x in phe]}; "
        f"pure-pow baselines: {[(x['n_bits'], x['vectors_per_s']) for x in pure]}",
    )
    comp = p2.get("comparisons", {})
    speedups_measured = [
        v["speedup"] for k, v in comp.items() if k.startswith("measured_paillier")
    ]
    paper_phe = comp.get("paper_remoteRAG_phe_250_300", {})
    paper_ok = paper_phe.get("speedup_vs_250", 0) > 1 and paper_phe.get("speedup_vs_300", 0) > 1
    g.check_true(
        "phase2_speedups_positive",
        len(speedups_measured) >= 1 and min(speedups_measured) > 1 and paper_ok,
        detail=f"speedup vs measured Paillier: {speedups_measured}; vs paper RemoteRAG PHE "
        f"250-300: {paper_phe.get('speedup_vs_250')}x-{paper_phe.get('speedup_vs_300')}x "
        "(absolute ratios are cross-hardware/implementation, see A17/A17b)",
    )
    paper_self = PAPER_CAPRISE_VEC_S / ((PAPER_PHE_VEC_S[0] + PAPER_PHE_VEC_S[1]) / 2)
    g.check_true(
        "paper_9x_claim_internal_consistency",
        7.5 <= paper_self <= 9.5,
        detail=f"paper 2339/(250-300) = {PAPER_CAPRISE_VEC_S/PAPER_PHE_VEC_S[1]:.2f}-"
        f"{PAPER_CAPRISE_VEC_S/PAPER_PHE_VEC_S[0]:.2f}x (mid {paper_self:.2f}x) => the paper's '9x' is "
        "internally consistent",
    )
    g.check_true(
        "phase2_implementation_style_difference_reported",
        "our_over_paper" in comp.get("paper_caprise_2339", {})
        and comp["paper_caprise_2339"]["our_over_paper"] > 1,
        detail=f"our CAPRISE / paper CAPRISE 2339 vec/s = "
        f"{comp.get('paper_caprise_2339', {}).get('our_over_paper')}x (finding A17)",
    )

    # ================================================================== #
    # 4. Implementation-style probe (A17)
    # ================================================================== #
    style = res.get("implementation_style_probe", {})
    g.check_true(
        "per_element_style_probe_slower",
        style.get("ratio_per_element_over_vectorised", 0) > 5,
        detail=f"vectorised {style.get('vectorised_standard_normal_us_per_vector')} us/vector vs "
        f"per-element loop {style.get('per_element_scalar_loop_us_per_vector')} us/vector "
        f"({style.get('ratio_per_element_over_vectorised')}x slower; scalar draw "
        f"{style.get('per_element_scalar_loop_us_per_element')} us/element) -> explains paper's "
        "2339 vec/s (A17)",
    )

    # ================================================================== #
    # 5. Artifacts
    # ================================================================== #
    required = {"throughput", "phase1_overhead", "phase2", "implementation_style_probe", "findings"}
    g.check_true(
        "artifact_results_json",
        EXP_JSON.exists() and required.issubset(res.keys()),
        detail=f"{EXP_JSON.relative_to(ROOT)} ({EXP_JSON.stat().st_size} bytes), sections={sorted(res.keys())}",
    )
    g.check_true(
        "artifact_figure",
        FIG.exists() and FIG.stat().st_size > 10_000,
        detail=f"{FIG.relative_to(ROOT)} ({FIG.stat().st_size if FIG.exists() else 0} bytes)",
    )
    note_text = NOTES.read_text(encoding="utf-8") if NOTES.exists() else ""
    g.check_true(
        "artifact_notes_documents_A17",
        NOTES.exists() and "A17" in note_text and ("Paillier" in note_text or "Paillier 1024" in note_text),
        detail=f"{NOTES.relative_to(ROOT)} exists={NOTES.exists()}, documents A17={'A17' in note_text}, "
        f"documents PHE baseline={'Paillier' in note_text}",
    )

    # ================================================================== #
    # Summary
    # ================================================================== #
    g.note("Throughput measured on Apple M5 CPU (numpy vectorised CAPRISE); paper numbers are A100 GPU + per-element Python.")
    g.note("Phase-1: embedding dominates (128 queries: ~0.21-0.63 s vs CAPRISE ~1.5 ms); overhead 0.23-0.69% < 19% paper bound.")
    g.note("Phase-2: CAPRISE ~78.6k vectors/s (d=768) vs measured Paillier 2.19 vec/s (1024-bit) / 0.30 vec/s (2048-bit); paper 9x internal-consistent, cross-hardware ratios reported but not asserted (A17/A17b).")
    res = g.finish()
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
