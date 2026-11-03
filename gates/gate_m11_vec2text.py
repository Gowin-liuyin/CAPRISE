"""M11 gate: Vec2Text reconstruction attack -- paper Table II.

Module M11 of docs/MODULE_PLAN.md; paper Section V-B / Table II.

What this gate verifies
-----------------------
1. The full experiment artifact results/exp3_vec2text.json exists and was
   produced under the recorded setting: s=3, beta=0.2, d=768, 10-30 real
   MS MARCO passages, num_steps >= 10, the vec2text 0.0.13 GTR corrector,
   transformers 4.x environment.  References are constrained to the corrector's
   32-token generation budget (finding F6/A11: the paper does not state its
   text-length protocol; longer references can never be reconstructed in full).
2. Attack-embedding convention (finding F14): the attack operates on
   vec2text-convention embeddings (T5 mean-pool, no sentence-transformers Dense,
   no L2 normalisation -> empirical norms vary in ~[0.5, 1.0]), NOT on the
   unit-norm data/passage_emb.npy embeddings (which vec2text cannot invert).
3. The saved attack arrays are audited: E (attack embeddings), E' (CAPRISE
   ciphertext) and E_rescaled must satisfy
       max_i || E'_i - s*E_i || < 3*s*beta/8 = 0.225
       E_rescaled_i == E'_i * ||E_i|| / ||E'_i||   (attacker rescaling)
   and the references stored in the npz equal the JSON references.
4. Both attacker adaptations required by ambiguity A11 are present and scored:
   "caprise_raw" (invert e' directly, naive attacker) and "caprise_rescaled"
   (restore the original per-vector norm, strongest attack -> main Table II row).
5. All five Table II metrics (BLEU 0-100, token P/R/F1, ROUGE-L) are finite and
   in range.  The plaintext baseline must be strong: paper-style BLEU > 40, or
   -- if the CPU/MPS budget (20 steps, beam 4, N=20, full-length passages)
   yields 25-40 -- the deviation must be explicitly recorded in the artifact
   (finding "paper_baseline") and explained in the notes.  A weak baseline with
   no such record FAILS the gate (no silently accepting a broken pipeline).
6. Paper's privacy claim holds quantitatively in the paper-closest setting
   ("caprise_raw", i.e. the ciphertext fed to vec2text without adaptation):
   BLEU must drop by more than 50% relative to plaintext and token F1 by at
   least 10%.  The stronger scale-normalising attacker ("caprise_rescaled",
   e'*||E||/||E'||) is required to recover substantially more than raw but to
   stay below the plaintext baseline -- it is reported as the privacy upper
   bound (finding F3/A11).  Absolute paper Table II values are recorded for the
   report but not fitted.
7. Metrics are reproducible: recomputing text_metrics from the stored
   hypotheses + references in this (main) environment reproduces the stored
   numbers to 1e-6 -- the artefact is auditable without the vec2text env.
8. Figure and human-readable notes exist (a minimal figure is regenerated from
   the artefact if missing).

The gate deliberately does NOT re-run Vec2Text (needs .venv-vec2text and
~18 min); it validates the committed artefact and recomputes the cheap,
deterministic metric/encryption-audit layer.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from gates.common import Gate, ROOT, add_src_to_path

add_src_to_path()

from caprise import metrics as M  # noqa: E402

ARTIFACT = ROOT / "results" / "exp3_vec2text.json"
ARRAYS = ROOT / "results" / "exp3_vec2text_arrays.npz"
CSV_PATH = ROOT / "results" / "exp3_vec2text.csv"
FIG_PATH = ROOT / "figures" / "fig_vec2text.png"
NOTES_PATH = ROOT / "notes" / "exp3_vec2text.md"

PAPER_TABLE_II = {
    "w/o Enc": {"bleu": 83.021, "precision": 0.947, "recall": 0.950, "f1": 0.948, "rouge": 0.950},
    "CAPRISE": {"bleu": 12.425, "precision": 0.482, "recall": 0.498, "f1": 0.487, "rouge": 0.492},
}
METRIC_KEYS = ("bleu", "precision", "recall", "f1", "rouge")
REQUIRED_MODES = ("plaintext", "caprise_raw", "caprise_rescaled")
CONVENTION_TAG = "vec2text_t5_meanpool_v1"

g = Gate(
    "M11",
    "vec2text",
    owner="privacy-attacks",
    paper_ref="paper Section V-B / Table II (Vec2Text reconstruction); ambiguities A11, finding F14",
)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# 1) artifact presence / experimental setting
# --------------------------------------------------------------------------- #
exists = ARTIFACT.exists()
g.check_true("artifact_present", exists, str(ARTIFACT.relative_to(ROOT)))
artifact: dict = {}
if exists:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

setup = artifact.get("setup", {})
g.check("n_passages_10_to_30", "10..30", setup.get("n_passages"),
        "paper does not state N; this reproduction uses the Lead-specified 20 MS MARCO passages",
        passed=isinstance(setup.get("n_passages"), int) and 10 <= int(setup["n_passages"]) <= 30)
g.check_true("num_steps_at_least_10", int(setup.get("num_steps", 0)) >= 10,
             f"num_steps={setup.get('num_steps')} (paper does not state it; CPU/MPS budget documented in notes)")
g.check("setting_s3_beta02_d768",
        {"s": 3, "beta": 0.2, "d": 768},
        {"s": setup.get("s"), "beta": setup.get("beta"), "d": setup.get("d")},
        "paper Table II setting: s=3, beta=0.2 (gtr-t5-base -> d=768)")
env = setup.get("environments", {})
g.check_true("vec2text_env_recorded",
             "0.0.13" in str(setup.get("vec2text", "")) and "transformers 4" in str(env.get("run_with", "")),
             f"vec2text={setup.get('vec2text')}; run_with={env.get('run_with')}")
g.check_true("device_and_load_time_recorded",
             setup.get("device") not in (None, "", "unknown") and _finite(setup.get("corrector_load_seconds")),
             f"device={setup.get('device')}, corrector_load={setup.get('corrector_load_seconds')}s")
g.check_true("beam_width_recorded", isinstance(setup.get("sequence_beam_width"), int),
             f"sequence_beam_width={setup.get('sequence_beam_width')}")

# --------------------------------------------------------------------------- #
# 2) real MS MARCO data + F14 attack-embedding convention
# --------------------------------------------------------------------------- #
passages = artifact.get("passages", {})
texts = passages.get("texts", [])
g.check_true("real_m_marco_passages_used",
             len(texts) == setup.get("n_passages")
             and "passages.jsonl" in str(passages.get("source", "")),
             f"source={passages.get('source')}; n_texts={len(texts)}")
g.check("one_reference_per_passage", int(setup.get("n_passages", 0)), len(texts),
        "original full passage texts stored as BLEU/P/R/F1/ROUGE references")
max_ref_tokens = passages.get("max_ref_tokens")
ref_lens = passages.get("ref_token_lengths") or []
g.check_true("refs_within_corrector_generation_budget",
             isinstance(max_ref_tokens, int) and max_ref_tokens == 32
             and len(ref_lens) == len(texts) and max(ref_lens) <= max_ref_tokens,
             f"max reference tokens={max_ref_tokens}; tokenised lengths={ref_lens} (vec2text gtr__nq__32 "
             f"generates <= 32 tokens -> longer references are not a valid BLEU ceiling; F6/A11)")
g.check_true("sample_protocol_recorded",
             "selection" in passages and "generation budget" in str(passages.get("selection", "")),
             f"selection={passages.get('selection')}")

emb = artifact.get("embeddings", {})
g.check("attack_convention_tag", CONVENTION_TAG, emb.get("convention"),
        "attack embeddings = vec2text T5 mean-pool convention (F14)")
norms = [emb.get("norm_min"), emb.get("norm_max"), emb.get("norm_mean")]
norms_ok = all(_finite(v) for v in norms)
g.check_true("attack_embeddings_not_unit_norm",
             norms_ok
             and (float(emb["norm_max"]) - float(emb["norm_min"])) > 0.02
             and (float(emb["norm_min"]) < 0.995 or float(emb["norm_max"]) > 1.005),
             f"||E|| in [{emb.get('norm_min'):.4f}, {emb.get('norm_max'):.4f}] mean {emb.get('norm_mean'):.4f} "
             f"(sentence-transformers GTR embeddings are ~exactly 1.0 with <1e-3 spread -> F14 check)")
g.check_true("attack_embedding_sha_recorded",
             isinstance(emb.get("sha256"), str) and len(emb.get("sha256", "")) == 64,
             f"sha256={str(emb.get('sha256'))[:16]}...; arrays_file={emb.get('arrays_file')}")

# --------------------------------------------------------------------------- #
# 3) saved-array audit: CAPRISE encryption + rescaling attack inputs
# --------------------------------------------------------------------------- #
arrays_ok = ARRAYS.exists()
arrays = {}
if arrays_ok:
    arrays = dict(np.load(ARRAYS, allow_pickle=True))
    required = {"E_attack", "E_prime", "E_rescaled", "references"}
    arrays_ok = required.issubset(arrays.keys())
g.check_true("arrays_file_present", arrays_ok, str(ARRAYS.relative_to(ROOT)))

if arrays_ok:
    E = np.asarray(arrays["E_attack"], dtype=np.float64)
    Ep = np.asarray(arrays["E_prime"], dtype=np.float64)
    Er = np.asarray(arrays["E_rescaled"], dtype=np.float64)
    refs = [str(x) for x in arrays["references"]]
    g.check("arrays_shapes", True,
            E.shape == (len(texts), 768) and Ep.shape == E.shape and Er.shape == E.shape,
            f"E{E.shape}, E'{Ep.shape}, E_rescaled{Er.shape}",
            passed=E.shape == (len(texts), 768) and Ep.shape == E.shape and Er.shape == E.shape)
    g.check_true("arrays_sha256_matches_json",
                 hashlib.sha256(ARRAYS.read_bytes()).hexdigest() == emb.get("arrays_sha256"),
                 "artifact records the npz checksum (tamper audit)")
    g.check_true("npz_references_match_artifact", refs == texts,
                 "references in npz identical to JSON references")
    g.check_true("npz_E_sha256_matches_json",
                 hashlib.sha256(np.asarray(arrays["E_attack"], dtype=np.float32).tobytes()).hexdigest()
                 == emb.get("sha256"),
                 "attack-convention embeddings checksum matches JSON")
    enc = artifact.get("encryption", {})
    s = float(setup.get("s", 3))
    lam = Ep - s * E
    lam_norms = np.linalg.norm(lam, axis=1)
    bound = 3.0 * s * float(setup.get("beta", 0.2)) / 8.0
    g.check_true("noise_norm_within_paper_bound",
                 float(lam_norms.max()) < bound + 1e-6,
                 f"max||lambda||={lam_norms.max():.5f} < 3s*beta/8={bound} (from saved arrays)")
    g.check_true("encryption_stats_match_arrays",
                 abs(float(lam_norms.min()) - float(enc.get("noise_norm_min", -1))) < 1e-4
                 and abs(float(lam_norms.max()) - float(enc.get("noise_norm_max", -1))) < 1e-4
                 and abs(float(lam_norms.mean()) - float(enc.get("noise_norm_mean", -1))) < 1e-4,
                 f"JSON lam norms [{enc.get('noise_norm_min')}, {enc.get('noise_norm_max')}] vs arrays "
                 f"[{lam_norms.min():.6f}, {lam_norms.max():.6f}]")
    e_norms = np.linalg.norm(E, axis=1, keepdims=True)
    ep_norms = np.linalg.norm(Ep, axis=1, keepdims=True)
    expected_r = Ep * (e_norms / ep_norms)
    max_rescale_dev = float(np.abs(Er - expected_r).max())
    g.check_true("rescaled_attack_is_norm_restored_ciphertext", max_rescale_dev < 1e-5,
                 f"max|E_rescaled - E'*||E||/||E'|||={max_rescale_dev:.2e} (File: main attack definition)")
    g.check_true("raw_and_rescaled_inputs_differ",
                 float(np.abs(Er - Ep).max()) > 0.1,
                 f"max|E_rescaled - E'|={float(np.abs(Er - Ep).max()):.3f} "
                 f"(raw norm ~{np.linalg.norm(Ep, axis=1).mean():.2f} vs rescaled ~{np.linalg.norm(Er, axis=1).mean():.2f})")
    g.check_true("deterministic_nonces_recorded",
                 bool(enc.get("deterministic_nonces")) and isinstance(enc.get("nonces_sha256"), str)
                 and len(enc.get("nonces_sha256", "")) == 64,
                 f"nonces_sha256={str(enc.get('nonces_sha256'))[:16]}...")

# --------------------------------------------------------------------------- #
# 4) both attack adaptations present and scored
# --------------------------------------------------------------------------- #
modes = artifact.get("modes", {})
for mode in REQUIRED_MODES:
    payload = modes.get(mode, {})
    hyps = payload.get("hypotheses", [])
    mets = payload.get("metrics", {})
    g.check_true(f"mode_{mode}_present_and_scored",
                 len(hyps) == len(texts) and all(k in mets for k in METRIC_KEYS),
                 f"n_hypotheses={len(hyps)}, metrics={sorted(mets)}")
g.check_true("attack_adaptation_recorded_A11",
             "raw" in json.dumps(artifact.get("definition", {}))
             and "rescaled" in json.dumps(artifact.get("definition", {})),
             "raw-ciphertext and norm-restored attacks both defined (A11)")

# --------------------------------------------------------------------------- #
# 5) metric ranges, baseline strength, CAPRISE collapse
# --------------------------------------------------------------------------- #
ranges_ok, bad = True, []
for mode in REQUIRED_MODES:
    mets = modes.get(mode, {}).get("metrics", {})
    for k in METRIC_KEYS:
        v = mets.get(k)
        hi = 100.0 if k == "bleu" else 1.0
        if not _finite(v) or not (0.0 <= float(v) <= hi):
            ranges_ok = False
            bad.append(f"{mode}.{k}={v}")
g.check_true("metrics_finite_in_range", ranges_ok, f"bad={bad[:5]}")

plain = modes.get("plaintext", {}).get("metrics", {})
resc = modes.get("caprise_rescaled", {}).get("metrics", {})
raw = modes.get("caprise_raw", {}).get("metrics", {})
kinds = {f.get("kind") for f in artifact.get("findings", [])}

plain_bleu = float(plain.get("bleu", float("nan")))
baseline_deviation_recorded = (
    "paper_baseline" in kinds
    and isinstance(artifact.get("comparison", {}).get("w/o Enc"), dict)
    and float(artifact["comparison"]["w/o Enc"].get("paper", {}).get("bleu", 0.0)) == PAPER_TABLE_II["w/o Enc"]["bleu"]
)
g.check("baseline_bleu_strong",
        ">40 (paper-scale) or >=25 with recorded deviation",
        round(plain_bleu, 3),
        "capability floor: Vec2Text must reconstruct the plaintext attack-convention embeddings; "
        "the CPU/MPS budget (20 steps, beam 4, N=10, references calibrated to the corrector's "
        "32-token generation budget) keeps BLEU below the "
        "paper's 83.021 -> the deviation is recorded in findings/comparison and explained in notes",
        passed=plain_bleu > 40.0 or (plain_bleu >= 25.0 and baseline_deviation_recorded))

g.check_true("caprise_raw_bleu_collapse",
             _finite(raw.get("bleu")) and float(raw["bleu"]) <= 0.5 * plain_bleu,
             f"paper-closest (raw ciphertext) BLEU={raw.get('bleu')} <= 0.5*{plain_bleu} "
             f"({100 * (1 - float(raw.get('bleu', 0)) / plain_bleu):.1f}% relative drop)")
g.check_true("caprise_raw_f1_significant_drop",
             float(raw.get("f1", 1.0)) <= 0.90 * float(plain.get("f1", 0.0)),
             f"F1 {plain.get('f1')} -> {raw.get('f1')} (raw ciphertext)")
g.check_true("caprise_raw_rouge_drop",
             float(raw.get("rouge", 1.0)) <= 0.90 * float(plain.get("rouge", 0.0)),
             f"ROUGE-L {plain.get('rouge')} -> {raw.get('rouge')} (raw ciphertext)")
g.check_true("rescaled_attack_stronger_than_raw",
             float(resc.get("bleu", 0.0)) >= float(raw.get("bleu", 1.0)),
             f"rescaled {resc.get('bleu')} >= raw {raw.get('bleu')} "
             f"(scale-normalising adversary recovers reconstruction ability -> strongest-attack upper bound)")
g.check_true("rescaled_attack_still_below_plaintext",
             float(resc.get("bleu", 1e9)) <= 0.85 * plain_bleu,
             f"rescaled BLEU={resc.get('bleu')} <= 0.85*{plain_bleu} "
             f"(CAPRISE still degrades even the strongest attacker)")

# --------------------------------------------------------------------------- #
# 6) metrics reproducibility: recompute text_metrics from stored hypotheses
# --------------------------------------------------------------------------- #
repro_ok = True
worst = 0.0
detail = []
for mode in REQUIRED_MODES:
    payload = modes.get(mode, {})
    hyps = payload.get("hypotheses", [])
    stored = payload.get("metrics", {})
    if len(hyps) != len(texts):
        repro_ok = False
        detail.append(f"{mode}: hypothesis count mismatch")
        continue
    recomputed = M.text_metrics(hyps, texts)
    for k in METRIC_KEYS:
        d = abs(float(recomputed[k]) - float(stored.get(k, float("nan"))))
        worst = max(worst, d)
        if d > 1e-6:
            repro_ok = False
            detail.append(f"{mode}.{k}: recomputed {recomputed[k]:.6f} vs stored {stored.get(k)}")
g.check_true("metrics_reproducible_from_artifact", repro_ok,
             f"max |recomputed-stored|={worst:.2e} over 3 modes x 5 metrics; {detail[:3]}")
g.check_true("per_passage_bleu_recorded",
             all(len(modes.get(m, {}).get("per_passage_bleu", [])) == len(texts) for m in REQUIRED_MODES),
             "per-passage BLEU stored for all modes (figure data)")

# --------------------------------------------------------------------------- #
# 7) paper Table II reference + comparison recorded
# --------------------------------------------------------------------------- #
paper = artifact.get("paper_table_ii", {})
paper_ok = True
for name, vals in PAPER_TABLE_II.items():
    for k, v in vals.items():
        if abs(float(paper.get(name, {}).get(k, float("nan"))) - v) > 1e-9:
            paper_ok = False
g.check_true("paper_table_ii_values_recorded", paper_ok,
             "paper Table II reference values stored verbatim "
             "(83.021/0.947/0.950/0.948/0.950 vs 12.425/0.482/0.498/0.487/0.492)")
comp = artifact.get("comparison", {})
g.check_true("comparison_vs_paper_recorded",
             {"w/o Enc", "CAPRISE", "CAPRISE_rescaled_attack"}.issubset(comp.keys())
             and all("delta_ours_minus_paper" in comp[k]
                     for k in ("w/o Enc", "CAPRISE", "CAPRISE_rescaled_attack")),
             "ours-vs-paper deltas stored for the baseline, the paper-closest CAPRISE (raw) row and "
             "the stronger rescaled-attack row (F3/A11)")
g.check_true("findings_recorded",
             {"main_effect", "scale_adaptation_recovery", "paper_baseline", "paper_caprise_row",
              "attack_adaptation", "embedding_convention_F14"}.issubset(kinds),
             f"finding kinds={sorted(k for k in kinds if k)}")

# --------------------------------------------------------------------------- #
# 8) figure + csv + notes
# --------------------------------------------------------------------------- #
if not FIG_PATH.exists() or FIG_PATH.stat().st_size < 20_000:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = [m for m in REQUIRED_MODES if m in modes]
    xs = np.arange(len(order))
    ax.bar(xs, [modes[m]["metrics"]["bleu"] for m in order],
           color=["#2ca02c", "#ff9896", "#d62728"][: len(order)], edgecolor="black")
    ax.axhline(paper.get("w/o Enc", {}).get("bleu", 83.021), ls="--", color="gray", label="paper w/o Enc")
    ax.axhline(paper.get("CAPRISE", {}).get("bleu", 12.425), ls=":", color="firebrick", label="paper CAPRISE")
    ax.set_xticks(xs)
    ax.set_xticklabels(order, fontsize=8)
    ax.set_ylabel("BLEU (0-100)")
    ax.set_title("Vec2Text attack (paper Table II) -- BLEU by attack mode")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
g.check_true("figure_generated", FIG_PATH.exists() and FIG_PATH.stat().st_size > 20_000,
             f"{FIG_PATH.relative_to(ROOT)} ({FIG_PATH.stat().st_size if FIG_PATH.exists() else 0} bytes)")
g.check_true("csv_generated", CSV_PATH.exists() and CSV_PATH.stat().st_size > 200,
             f"{CSV_PATH.relative_to(ROOT)}")
notes_text = NOTES_PATH.read_text(encoding="utf-8") if NOTES_PATH.exists() else ""
g.check_true("notes_written",
             len(notes_text) > 3000 and "Table II" in notes_text and "Vec2Text" in notes_text
             and "A11" in notes_text and ("F14" in notes_text or "attack-convention" in notes_text),
             f"{NOTES_PATH.relative_to(ROOT)} ({len(notes_text)} chars)")

# --------------------------------------------------------------------------- #
# 9) timing / compute budget documented
# --------------------------------------------------------------------------- #
pt = artifact.get("summary", {}).get("per_text_seconds_mean")
g.check_true("per_text_time_recorded", _finite(pt) and 0.0 < float(pt) < 600.0,
             f"mean {pt} s/text on {setup.get('device')}")
rt = artifact.get("runtime", {}).get("seconds_total")
g.check_true("total_runtime_reasonable", _finite(rt) and float(rt) < 7200.0,
             f"total {rt} s (CPU/MPS scale difference vs paper A100 documented in notes)")

g.note(
    f"Main result (paper-closest row = raw ciphertext): plaintext BLEU={round(plain_bleu, 2)} -> "
    f"CAPRISE raw BLEU={round(float(raw.get('bleu', 0)), 2)} "
    f"(-{100 * (1 - float(raw.get('bleu', 0)) / plain_bleu):.1f}%); the stronger scale-normalising "
    f"attacker reaches BLEU={round(float(resc.get('bleu', 0)), 2)} "
    f"(still below plaintext; privacy upper bound, F3/A11). Paper Table II: 83.021 -> 12.425."
)
g.note(
    "F14 (critical): Vec2Text gtr-base inverts vec2text-convention embeddings (T5 mean-pool, no Dense, "
    "no L2 normalisation, norms ~0.5-1.0); the M0 sentence-transformers GTR embeddings give garbage even "
    "without encryption. The gate audits that convention via saved arrays + recorded norms."
)
g.note(
    "The absolute difference from paper Table II is reported, not fitted: the paper does not state "
    "num_steps / beam width / N / per-passage length (A11) and uses an A100; our CPU/MPS-feasible "
    "settings are recorded in the artefact and explained in notes/exp3_vec2text.md."
)
g.note(
    "This gate validates the committed artefact and recomputes the metric + encryption-audit layer in "
    "the main environment (vec2text inversion needs .venv-vec2text and ~18 min; not re-run here)."
)

g.finish()
