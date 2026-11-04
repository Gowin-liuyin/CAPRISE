"""M11 gate: Vec2Text reconstruction attack -- paper Table II.

Module M11 of docs/MODULE_PLAN.md; paper Section V-B / Table II.

What this gate verifies
-----------------------
1. The full experiment artifact results/exp3_vec2text.json exists and was
   produced under the recorded setting: s=3, beta=0.2, d=768, **at least 50**
   real MS MARCO passages (task-2 raised the sample from 10 to 50), num_steps
   >= 10, the vec2text 0.0.13 GTR corrector, transformers 4.x environment.
   References are constrained to the corrector's 32-token generation budget
   (finding F6/A11: the paper does not state its text-length protocol; longer
   references can never be reconstructed in full).  The candidate pool must
   have been expanded automatically when the initial pool was too small; all
   50 passage ids are recorded and unique.
2. Attack-embedding convention (finding F14): the attack operates on
   vec2text-convention embeddings (T5 mean-pool, no sentence-transformers Dense,
   no L2 normalisation -> empirical norms vary in ~[0.5, 1.0]), NOT on the
   unit-norm data/passage_emb.npy embeddings (which vec2text cannot invert).
3. The saved attack arrays are audited: E (attack embeddings), E' (CAPRISE
   ciphertext) and E_rescaled must satisfy
       max_i || E'_i - s*E_i || < 3*s*beta/8 = 0.225
       E_rescaled_i == E'_i * ||E_i|| / ||E'_i||   (attacker rescaling)
   and the references stored in the npz equal the JSON references.
4. All three attack modes required by ambiguity A11 are present and scored:
   "plaintext", "caprise_raw" (invert e' directly, naive attacker) and
   "caprise_rescaled" (restore the original per-vector norm, strongest attack
   -> privacy upper bound).
5. All five Table II metrics (BLEU 0-100, token P/R/F1, ROUGE-L) are finite and
   in range, per-passage metric components are stored for every passage, and
   the **bootstrap 95% CI** is present for every mode and metric.  The gate
   independently recomputes the CI from the stored hypotheses + references with
   the recorded seed / resample count and requires an exact match.
6. The plaintext baseline must be strong (paper-style BLEU > 40, or -- if the
   MPS budget yields 25-40 -- the deviation must be explicitly recorded in the
   artifact as finding "paper_baseline").  The paper's privacy claim holds
   quantitatively in the paper-closest setting ("caprise_raw": ciphertext fed to
   vec2text without adaptation): BLEU must drop by more than 50% relative to
   plaintext and token F1 by at least 10%; the stronger scale-normalising
   attacker ("caprise_rescaled") must recover substantially more than raw but
   stay below 0.85x the plaintext baseline.  These thresholds are unchanged from
   the 10-passage gate (task-2 only extends coverage; no threshold is relaxed).
7. **Step/beam sensitivity (finding F6/A11)**: the same 5 passages (a subset of
   the 50, passage indices recorded; reduced from 10 to 5 on the Lead's
   instruction because the shared machine ran at ~5-7 loadavg and the 10-passage
   sensitivity exceeded the time budget -- this is a scope adjustment, not a
   threshold relaxation) were re-run in the three modes with
   num_steps=50, sequence_beam_width=10.  The gate checks the config, that the
   references are exactly the main-run subset, that the sensitivity metrics are
   reproducible, that the 20/4 baseline is recomputed on the SAME passages from
   the main run, that the recorded deltas match, and that per-text timing is
   recorded.  It does NOT require a particular sign of the delta (the paper does
   not state its budget); it requires the magnitude/direction to be recorded.
8. Metrics are reproducible: recomputing text_metrics from the stored
   hypotheses + references in this environment reproduces the stored numbers to
   1e-6 -- the artefact is auditable without re-running vec2text.
9. Figure, CSV (with per-passage rows and aggregate CI rows) and human-readable
   notes exist.

The gate deliberately does NOT re-run Vec2Text (needs .venv-vec2text and tens
of minutes); it validates the committed artefact and recomputes the cheap,
deterministic metric/CI/encryption-audit layer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
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

BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260101
BOOTSTRAP_CI_LEVEL = 0.95
SENSITIVITY_N = 5
SENSITIVITY_STEPS = 50
SENSITIVITY_BEAM = 10

g = Gate(
    "M11",
    "vec2text",
    owner="privacy-attacks",
    paper_ref="paper Section V-B / Table II (Vec2Text reconstruction); ambiguities A11, finding F14",
)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _per_passage_recompute(hypotheses: list[str], references: list[str]) -> dict:
    """Independent recomputation of the per-passage metric components (mirrors exp3)."""
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import BLEU

    metric = BLEU(effective_order=True)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    tp, fp, fn, bleu, rouge = [], [], [], [], []
    for hyp, ref in zip(hypotheses, references):
        h = Counter(M._tokens(hyp))
        r = Counter(M._tokens(ref))
        tp.append(int(sum((h & r).values())))
        fp.append(int(sum((h - r).values())))
        fn.append(int(sum((r - h).values())))
        bleu.append(float(metric.sentence_score(hyp, [ref]).score) if hyp.strip() else 0.0)
        rouge.append(float(scorer.score(ref, hyp)["rougeL"].fmeasure))
    precision, recall, f1 = [], [], []
    for t, f_p, f_n in zip(tp, fp, fn):
        p = t / (t + f_p) if (t + f_p) else 0.0
        r = t / (t + f_n) if (t + f_n) else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return {"bleu": bleu, "precision": precision, "recall": recall, "f1": f1, "rouge": rouge,
            "tp": tp, "fp": fp, "fn": fn}


def _bootstrap_recompute(hypotheses: list[str], references: list[str], per_passage: dict,
                         n_resamples: int, seed: int, ci_level: float) -> dict:
    """Independent recomputation of the passage-level bootstrap CI (mirrors exp3)."""
    n = len(hypotheses)
    indices = np.random.default_rng(int(seed)).integers(0, n, size=(int(n_resamples), n))
    tp = np.asarray(per_passage["tp"], dtype=np.float64)
    fp = np.asarray(per_passage["fp"], dtype=np.float64)
    fn = np.asarray(per_passage["fn"], dtype=np.float64)
    rouge = np.asarray(per_passage["rouge"], dtype=np.float64)
    vals = {k: np.empty(int(n_resamples), dtype=np.float64) for k in METRIC_KEYS}
    for b in range(int(n_resamples)):
        ii = indices[b]
        hyps = [hypotheses[i] for i in ii]
        refs = [references[i] for i in ii]
        vals["bleu"][b] = M.bleu(hyps, refs)
        t = float(tp[ii].sum())
        f_p = float(fp[ii].sum())
        f_n = float(fn[ii].sum())
        p = t / (t + f_p) if (t + f_p) else 0.0
        r = t / (t + f_n) if (t + f_n) else 0.0
        vals["precision"][b] = p
        vals["recall"][b] = r
        vals["f1"][b] = 2 * p * r / (p + r) if (p + r) else 0.0
        vals["rouge"][b] = float(rouge[ii].mean())
    alpha = (1.0 - float(ci_level)) / 2.0
    return {
        k: {
            "mean": float(np.mean(v)),
            "low": float(np.percentile(v, 100.0 * alpha)),
            "high": float(np.percentile(v, 100.0 * (1.0 - alpha))),
        }
        for k, v in vals.items()
    }


# --------------------------------------------------------------------------- #
# 1) artifact presence / experimental setting
# --------------------------------------------------------------------------- #
exists = ARTIFACT.exists()
g.check_true("artifact_present", exists, str(ARTIFACT.relative_to(ROOT)))
artifact: dict = {}
if exists:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

setup = artifact.get("setup", {})
n_passages = setup.get("n_passages")
g.check("n_passages_at_least_50", ">=50", n_passages,
        "task-2: sample size raised from 10 to 50 real MS MARCO passages (paper does not state N)",
        passed=isinstance(n_passages, int) and int(n_passages) >= 50)
g.check_true("num_steps_at_least_10", int(setup.get("num_steps", 0)) >= 10,
             f"num_steps={setup.get('num_steps')} (paper does not state it; MPS budget documented in notes)")
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
g.check_true("model_paths_recorded",
             isinstance(setup.get("model_paths"), dict) and "inversion" in setup.get("model_paths", {})
             and "corrector" in setup.get("model_paths", {}),
             f"model_paths={list(setup.get('model_paths', {}))}")

# --------------------------------------------------------------------------- #
# 2) real MS MARCO data + F14 attack-embedding convention
# --------------------------------------------------------------------------- #
passages = artifact.get("passages", {})
texts = passages.get("texts", [])
pids = passages.get("passage_ids", [])
g.check_true("real_m_marco_passages_used",
             len(texts) == n_passages
             and "passages.jsonl" in str(passages.get("source", "")),
             f"source={passages.get('source')}; n_texts={len(texts)}")
g.check("one_reference_per_passage", n_passages, len(texts),
        "original full passage texts stored as BLEU/P/R/F1/ROUGE references")
g.check_true("passage_ids_recorded_and_unique",
             len(pids) == n_passages and len(set(pids)) == len(pids),
             f"n_pids={len(pids)}, unique={len(set(pids))}")
g.check_true("pool_expansion_recorded",
             isinstance(passages.get("pool_size"), int)
             and int(passages["pool_size"]) >= int(passages.get("initial_pool_size", 0)),
             f"initial_pool={passages.get('initial_pool_size')} -> scanned={passages.get('pool_size')} "
             f"(expanded={passages.get('pool_expanded')})")
max_ref_tokens = passages.get("max_ref_tokens")
ref_lens = passages.get("ref_token_lengths") or []
g.check_true("refs_within_corrector_generation_budget",
             isinstance(max_ref_tokens, int) and max_ref_tokens == 32
             and len(ref_lens) == len(texts) and max(ref_lens) <= max_ref_tokens,
             f"max reference tokens={max_ref_tokens}; tokenised lengths min/max="
             f"{min(ref_lens) if ref_lens else None}/{max(ref_lens) if ref_lens else None} "
             f"(vec2text gtr__nq__32 generates <= 32 tokens; F6/A11)")
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
                 f"max|E_rescaled - E'*||E||/||E'|||={max_rescale_dev:.2e}")
    g.check_true("raw_and_rescaled_inputs_differ",
                 float(np.abs(Er - Ep).max()) > 0.1,
                 f"max|E_rescaled - E'|={float(np.abs(Er - Ep).max()):.3f} "
                 f"(raw norm ~{np.linalg.norm(Ep, axis=1).mean():.2f} vs rescaled ~{np.linalg.norm(Er, axis=1).mean():.2f})")
    g.check_true("deterministic_nonces_recorded",
                 bool(enc.get("deterministic_nonces")) and isinstance(enc.get("nonces_sha256"), str)
                 and len(enc.get("nonces_sha256", "")) == 64,
                 f"nonces_sha256={str(enc.get('nonces_sha256'))[:16]}...")

# --------------------------------------------------------------------------- #
# 4) all three attack modes present and scored (+ per-passage metrics, CI)
# --------------------------------------------------------------------------- #
modes = artifact.get("modes", {})
for mode in REQUIRED_MODES:
    payload = modes.get(mode, {})
    hyps = payload.get("hypotheses", [])
    mets = payload.get("metrics", {})
    pp = payload.get("per_passage_metrics", {})
    g.check_true(f"mode_{mode}_present_and_scored",
                 len(hyps) == len(texts) and all(k in mets for k in METRIC_KEYS),
                 f"n_hypotheses={len(hyps)}, metrics={sorted(mets)}")
    g.check_true(f"mode_{mode}_per_passage_metrics_recorded",
                 all(len(pp.get(k, [])) == len(texts) for k in METRIC_KEYS)
                 and all(len(pp.get(k, [])) == len(texts) for k in ("tp", "fp", "fn")),
                 f"per-passage metric component lengths="
                 f"{ {k: len(pp.get(k, [])) for k in METRIC_KEYS + ('tp', 'fp', 'fn')} }")
g.check_true("attack_adaptation_recorded_A11",
             "raw" in json.dumps(artifact.get("definition", {}))
             and "rescaled" in json.dumps(artifact.get("definition", {})),
             "raw-ciphertext and norm-restored attacks both defined (A11)")

boot = artifact.get("bootstrap", {})
g.check_true("bootstrap_config_recorded",
             boot.get("n_resamples") == BOOTSTRAP_N_RESAMPLES
             and boot.get("seed") == BOOTSTRAP_SEED
             and abs(float(boot.get("ci_level", 0.0)) - BOOTSTRAP_CI_LEVEL) < 1e-12,
             f"n_resamples={boot.get('n_resamples')}, seed={boot.get('seed')}, "
             f"ci_level={boot.get('ci_level')} (task-2: 1000 resamples, fixed seed)")

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
        "the MPS budget (20 steps, beam 4, N=50, references calibrated to the corrector's 32-token "
        "generation budget) keeps BLEU below the paper's 83.021 -> deviation recorded in "
        "findings/comparison and explained in notes",
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
# 6) bootstrap 95% CI: present for every mode/metric and independently recomputable
# --------------------------------------------------------------------------- #
boot_present, boot_bad = True, []
for mode in REQUIRED_MODES:
    payload = modes.get(mode, {})
    ci = payload.get("bootstrap_ci", {}).get("metrics", {})
    for k in METRIC_KEYS:
        c = ci.get(k, {})
        hi = 100.0 if k == "bleu" else 1.0
        if not (_finite(c.get("low")) and _finite(c.get("high")) and _finite(c.get("mean"))):
            boot_present = False
            boot_bad.append(f"{mode}.{k}: missing")
            continue
        if not (0.0 <= float(c["low"]) <= float(c["high"]) <= hi):
            boot_present = False
            boot_bad.append(f"{mode}.{k}: [{c['low']}, {c['high']}]")
        if abs(float(c.get("point_estimate", float("nan"))) - float(payload["metrics"][k])) > 1e-9:
            boot_present = False
            boot_bad.append(f"{mode}.{k}: point_estimate mismatch")
g.check_true("bootstrap_ci_present_for_all_modes_and_metrics", boot_present,
             f"bad={boot_bad[:5]}; 3 modes x 5 metrics, 95% CI, 1000 resamples, seed {BOOTSTRAP_SEED}")

boot_repro_ok, worst_ci = True, 0.0
boot_detail = []
for mode in REQUIRED_MODES:
    payload = modes.get(mode, {})
    hyps = payload.get("hypotheses", [])
    if len(hyps) != len(texts):
        boot_repro_ok = False
        boot_detail.append(f"{mode}: hypothesis count mismatch")
        continue
    pp = _per_passage_recompute(hyps, texts)
    stored_pp = payload.get("per_passage_metrics", {})
    for k in ("bleu", "precision", "recall", "f1", "rouge"):
        for a, b in zip(pp[k], stored_pp.get(k, [])):
            d = abs(float(a) - float(b))
            worst_ci = max(worst_ci, d)
            if d > 1e-9:
                boot_repro_ok = False
                boot_detail.append(f"{mode}.per_passage.{k}: {d:.2e}")
    for k in ("tp", "fp", "fn"):
        if list(map(int, pp[k])) != list(map(int, stored_pp.get(k, []))):
            boot_repro_ok = False
            boot_detail.append(f"{mode}.per_passage.{k}: counts differ")
    recomputed = _bootstrap_recompute(hyps, texts, pp, BOOTSTRAP_N_RESAMPLES,
                                      BOOTSTRAP_SEED, BOOTSTRAP_CI_LEVEL)
    stored_ci = payload.get("bootstrap_ci", {}).get("metrics", {})
    for k in METRIC_KEYS:
        for field in ("mean", "low", "high"):
            d = abs(float(recomputed[k][field]) - float(stored_ci.get(k, {}).get(field, float("nan"))))
            worst_ci = max(worst_ci, d)
            if d > 1e-6:
                boot_repro_ok = False
                boot_detail.append(f"{mode}.{k}.{field}: recomputed {recomputed[k][field]:.6f} vs "
                                   f"stored {stored_ci.get(k, {}).get(field)}")
g.check_true("bootstrap_ci_reproducible", boot_repro_ok,
             f"max |recomputed-stored|={worst_ci:.2e} (CI recomputed from hypotheses+references with "
             f"the recorded seed); {boot_detail[:3]}")

# --------------------------------------------------------------------------- #
# 7) metrics reproducibility: recompute text_metrics from stored hypotheses
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
             "per-passage BLEU stored for all modes (figure/CSV data)")

# --------------------------------------------------------------------------- #
# 8) paper Table II reference + comparison recorded
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
comp_entries = ("w/o Enc", "CAPRISE", "CAPRISE_rescaled_attack")
comp_ok = (
    set(comp_entries).issubset(comp.keys())
    and all(
        "delta_ours_minus_paper" in comp.get(entry, {})
        and "ratio_ours_over_paper" in comp.get(entry, {})
        for entry in comp_entries
    )
)
g.check_true("comparison_vs_paper_recorded", comp_ok,
             "ours-vs-paper deltas/ratios stored for the baseline, the paper-closest CAPRISE (raw) "
             "row and the stronger rescaled-attack row (F3/A11)")
g.check_true("comparison_ci_recorded",
             all("bootstrap_ci_95" in comp.get(k, {}) for k in
                 ("w/o Enc", "CAPRISE", "CAPRISE_rescaled_attack")),
             "95% CI stored alongside the paper comparison rows")
g.check_true("findings_recorded",
             {"main_effect", "scale_adaptation_recovery", "paper_baseline", "paper_caprise_row",
              "attack_adaptation", "embedding_convention_F14", "bootstrap_ci",
              "step_beam_sensitivity"}.issubset(kinds),
             f"finding kinds={sorted(k for k in kinds if k)}")

# --------------------------------------------------------------------------- #
# 9) step/beam sensitivity (num_steps=50, sequence_beam_width=10, same 5 passages)
# --------------------------------------------------------------------------- #
sens = artifact.get("sensitivity", {})
sens_ok = bool(sens.get("modes"))
g.check_true("sensitivity_present", sens_ok,
             f"num_steps=50/beam=10 sensitivity on a 5-passage subset; mode keys="
             f"{sorted(sens.get('modes', {}))}")
if sens_ok:
    indices = sens.get("passage_indices", [])
    sens_refs = sens.get("references", [])
    g.check("sensitivity_n_passages", SENSITIVITY_N, len(indices),
            "same 5 passages re-run at num_steps=50, sequence_beam_width=10 "
            "(reduced from 10 due to shared-machine load; direction/magnitude only)",
            passed=len(indices) == SENSITIVITY_N)
    g.check_true("sensitivity_passages_subset_of_main",
                 all(isinstance(i, int) and 0 <= i < len(texts) for i in indices)
                 and len(set(indices)) == len(indices),
                 f"indices={indices}")
    g.check_true("sensitivity_references_match_main_subset",
                 sens_refs == [texts[i] for i in indices],
                 "sensitivity references are exactly the main-run passages at the recorded indices")
    sens_cfg = sens.get("config", {})
    g.check_true("sensitivity_config_50_10",
                 sens_cfg.get("num_steps") == SENSITIVITY_STEPS
                 and sens_cfg.get("sequence_beam_width") == SENSITIVITY_BEAM
                 and sens_cfg.get("s") == 3 and sens_cfg.get("beta") == 0.2,
                 f"config={ {k: sens_cfg.get(k) for k in ('n', 'num_steps', 'sequence_beam_width', 's', 'beta')} }")
    sens_present, sens_bad = True, []
    for mode in REQUIRED_MODES:
        payload = sens.get("modes", {}).get(mode, {})
        hyps = payload.get("hypotheses", [])
        mets = payload.get("metrics", {})
        pp = payload.get("per_passage_metrics", {})
        if len(hyps) != len(sens_refs):
            sens_present = False
            sens_bad.append(f"{mode}: {len(hyps)} hypotheses")
            continue
        if not all(k in mets for k in METRIC_KEYS):
            sens_present = False
            sens_bad.append(f"{mode}: metrics missing")
        if not all(len(pp.get(k, [])) == len(sens_refs) for k in METRIC_KEYS):
            sens_present = False
            sens_bad.append(f"{mode}: per-passage metrics missing")
        if not (_finite(payload.get("per_text_seconds")) and float(payload["per_text_seconds"]) > 0):
            sens_present = False
            sens_bad.append(f"{mode}: per_text_seconds={payload.get('per_text_seconds')}")
        recomputed = M.text_metrics(hyps, sens_refs) if len(hyps) == len(sens_refs) else None
        if recomputed is not None:
            for k in METRIC_KEYS:
                if abs(float(recomputed[k]) - float(mets.get(k, float("nan")))) > 1e-6:
                    sens_present = False
                    sens_bad.append(f"{mode}.{k}: not reproducible")
    g.check_true("sensitivity_all_modes_complete_and_reproducible", sens_present, f"bad={sens_bad[:5]}")

    base_metrics_ok, base_bad = True, []
    base = sens.get("baseline_20_4", {})
    for mode in REQUIRED_MODES:
        hyps = modes.get(mode, {}).get("hypotheses", [])
        if len(hyps) != len(texts):
            base_metrics_ok = False
            base_bad.append(f"{mode}: main hypotheses missing")
            continue
        sub_hyps = [hyps[i] for i in indices]
        sub_refs = [texts[i] for i in indices]
        expected = M.text_metrics(sub_hyps, sub_refs)
        stored = base.get(mode, {}).get("metrics", {})
        for k in METRIC_KEYS:
            d = abs(float(expected[k]) - float(stored.get(k, float("nan"))))
            if d > 1e-6:
                base_metrics_ok = False
                base_bad.append(f"{mode}.{k}: recomputed {expected[k]:.6f} vs stored {stored.get(k)}")
    g.check_true("sensitivity_baseline_20_4_recomputed", base_metrics_ok,
                 f"20/4 baseline recomputed on the SAME 5 passages from the 50-passage run; "
                 f"bad={base_bad[:3]}")

    deltas_ok, delta_bad = True, []
    deltas = sens.get("deltas", {})
    for mode in REQUIRED_MODES:
        for k in METRIC_KEYS:
            entry = deltas.get(mode, {}).get(k, {})
            expected_delta = float(sens["modes"][mode]["metrics"][k]) - float(base[mode]["metrics"][k])
            if not _finite(entry.get("delta")) or abs(float(entry["delta"]) - expected_delta) > 1e-9:
                deltas_ok = False
                delta_bad.append(f"{mode}.{k}")
    g.check_true("sensitivity_deltas_recorded", deltas_ok, f"bad={delta_bad[:5]}")
    g.check_true("sensitivity_headline_recorded",
                 isinstance(sens.get("headline"), str) and "corpus BLEU" in str(sens.get("headline")),
                 f"headline={sens.get('headline')}")
    g.check_true("sensitivity_runtime_recorded",
                 _finite(sens.get("seconds_total")) and float(sens["seconds_total"]) > 0,
                 f"sensitivity_seconds_total={sens.get('seconds_total')}; "
                 f"config={ {k: sens_cfg.get(k) for k in ('n', 'num_steps', 'sequence_beam_width')} }")

# --------------------------------------------------------------------------- #
# 10) figure + csv + notes
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

csv_ok = CSV_PATH.exists() and CSV_PATH.stat().st_size > 200
csv_passage_rows = 0
csv_ci_rows = 0
csv_sens_rows = 0
csv_aggregate_ok = False
if csv_ok:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            section = row.get("section", "")
            if section == "passage":
                csv_passage_rows += 1
            if section.startswith("sensitivity_passage"):
                csv_sens_rows += 1
            if row.get("ci_low") not in (None, ""):
                csv_ci_rows += 1
            if section == "aggregate" and row.get("mode") == "plaintext" and row.get("ci_metric") == "bleu":
                try:
                    csv_aggregate_ok = abs(float(row["bleu"]) - float(plain["bleu"])) < 1e-9
                except Exception:
                    csv_aggregate_ok = False
g.check_true("csv_generated", csv_ok, f"{CSV_PATH.relative_to(ROOT)}")
g.check("csv_per_passage_rows", f">={3 * int(n_passages)}", csv_passage_rows,
        "task-2: per-passage results stored in the CSV (one row per mode x passage)",
        passed=csv_passage_rows >= 3 * int(n_passages))
g.check_true("csv_bootstrap_ci_rows", csv_ci_rows >= 3 * len(METRIC_KEYS),
             f"{csv_ci_rows} aggregate rows carry ci_low/ci_high")
g.check("csv_sensitivity_rows", f">={3 * SENSITIVITY_N}", csv_sens_rows,
        "sensitivity per-passage rows (50/10) plus 20/4 baseline rows",
        passed=csv_sens_rows >= 3 * SENSITIVITY_N)
g.check_true("csv_aggregate_matches_json", csv_aggregate_ok,
             "plaintext aggregate BLEU in the CSV equals the JSON corpus BLEU")
notes_text = NOTES_PATH.read_text(encoding="utf-8") if NOTES_PATH.exists() else ""
g.check_true("notes_written",
             len(notes_text) > 3000 and "Table II" in notes_text and "Vec2Text" in notes_text
             and "A11" in notes_text and ("F14" in notes_text or "attack-convention" in notes_text),
             f"{NOTES_PATH.relative_to(ROOT)} ({len(notes_text)} chars)")
g.check_true("notes_cover_ci_and_sensitivity",
             ("95%" in notes_text or "置信区间" in notes_text)
             and ("bootstrap" in notes_text.lower() or "Bootstrap" in notes_text)
             and ("50" in notes_text and ("num_steps=50" in notes_text or "50/10" in notes_text)),
             "notes document the 50-passage sample, the bootstrap 95% CI and the 50/10 sensitivity")

# --------------------------------------------------------------------------- #
# 10b) plaintext budget study (C2 follow-up): the undisclosed Vec2Text budget
# --------------------------------------------------------------------------- #
BUDGET_PATH = ROOT / "results" / "exp3_plaintext_budget.json"
if BUDGET_PATH.exists():
    bud = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    bsetup = bud.get("setup", {})
    bcomp = bud.get("comparison", {})
    bbudgets = bud.get("budgets", {})
    g.check_true("plaintext_budget_artifact", True, detail=f"{BUDGET_PATH.relative_to(ROOT)}")
    g.check_true(
        "plaintext_budget_both_budgets_on_same_passages",
        set(bbudgets) == {"20/4", "50/10"}
        and bbudgets["20/4"]["num_steps"] == 20 and bbudgets["20/4"]["sequence_beam_width"] == 4
        and bbudgets["50/10"]["num_steps"] == 50 and bbudgets["50/10"]["sequence_beam_width"] == 10
        and int(bsetup.get("n_passages", 0)) >= 10,
        detail=f"n={bsetup.get('n_passages')} passages; budgets={sorted(bbudgets)} "
               f"(20/4 and 50/10 on the same passages)",
    )
    bad_b = []
    for label, payload in bbudgets.items():
        for key in METRIC_KEYS:
            v = payload.get("metrics", {}).get(key)
            lo = payload.get("bootstrap_ci_95", {}).get(key, {}).get("low")
            hi = payload.get("bootstrap_ci_95", {}).get(key, {}).get("high")
            if not (_finite(v) and _finite(lo) and _finite(hi) and lo <= v <= hi):
                bad_b.append((label, key))
    g.check_true("plaintext_budget_metrics_and_ci", not bad_b, detail=f"bad={bad_b[:5]}")
    g.check_true(
        "plaintext_budget_comparison_recorded",
        _finite(bcomp.get("delta_bleu_50_10_minus_20_4"))
        and _finite(bcomp.get("bleu_50_10_minus_paper"))
        and abs(bcomp["paper_plaintext_bleu"] - PAPER_TABLE_II["w/o Enc"]["bleu"]) < 1e-9,
        detail=f"delta BLEU 50/10 - 20/4 = {bcomp.get('delta_bleu_50_10_minus_20_4'):+.3f}; "
               f"50/10 - paper = {bcomp.get('bleu_50_10_minus_paper'):+.3f}; "
               f"CI(50/10) covers paper = {bcomp.get('ci_50_10_covers_paper')}",
    )
    # arithmetic self-consistency (hypotheses are not stored in the budget artefact;
    # the study is reproducible by re-running scripts/plaintext_budget.py with the
    # recorded seed, and its aggregate metrics + 95% CI + per-passage BLEU are stored)
    m20 = bbudgets["20/4"]["metrics"]["bleu"]
    m50 = bbudgets["50/10"]["metrics"]["bleu"]
    ci50 = bbudgets["50/10"]["bootstrap_ci_95"]["bleu"]
    paper_b = PAPER_TABLE_II["w/o Enc"]["bleu"]
    ok_arith = (
        abs(bcomp["bleu_20_4"] - m20) < 1e-9
        and abs(bcomp["bleu_50_10"] - m50) < 1e-9
        and abs(bcomp["delta_bleu_50_10_minus_20_4"] - (m50 - m20)) < 1e-9
        and abs(bcomp["bleu_50_10_minus_paper"] - (m50 - paper_b)) < 1e-9
        and bool(bcomp["ci_50_10_covers_paper"]) == (ci50["low"] <= paper_b <= ci50["high"])
        and (ci50["low"] <= m50 <= ci50["high"])
    )
    g.check_true("plaintext_budget_arithmetic_consistent", ok_arith,
                 detail=f"20/4 BLEU={m20:.3f}, 50/10 BLEU={m50:.3f} (CI {ci50['low']:.3f}-{ci50['high']:.3f}), "
                        f"paper={paper_b:.3f}; deltas and CI-coverage flags match the stored metrics")
else:
    g.check_true("plaintext_budget_artifact", False,
                 detail=f"missing {BUDGET_PATH.relative_to(ROOT)} (C2 follow-up)")

# --------------------------------------------------------------------------- #
# 11) timing / compute budget documented
# --------------------------------------------------------------------------- #
pt = artifact.get("summary", {}).get("per_text_seconds_mean")
g.check_true("per_text_time_recorded", _finite(pt) and 0.0 < float(pt) < 600.0,
             f"mean {pt} s/text on {setup.get('device')}")
rt = artifact.get("runtime", {}).get("seconds_total")
g.check_true("total_runtime_reasonable", _finite(rt) and float(rt) < 7200.0,
             f"total {rt} s (MPS scale difference vs paper A100 documented in notes)")
main_inv = setup.get("inversion_seconds_total")
g.check_true("main_inversion_time_recorded", _finite(main_inv) and float(main_inv) >= 0.0,
             f"main inversion {main_inv}s for {n_passages} passages x 3 modes")

g.note(
    f"Main result (paper-closest row = raw ciphertext): plaintext BLEU={round(plain_bleu, 2)} "
    f"(95% CI {artifact.get('summary', {}).get('plain_bleu_ci95')}) -> "
    f"CAPRISE raw BLEU={round(float(raw.get('bleu', 0)), 2)} "
    f"(95% CI {artifact.get('summary', {}).get('caprise_raw_bleu_ci95')}; "
    f"-{100 * (1 - float(raw.get('bleu', 0)) / plain_bleu):.1f}%); the stronger scale-normalising "
    f"attacker reaches BLEU={round(float(resc.get('bleu', 0)), 2)} "
    f"(95% CI {artifact.get('summary', {}).get('caprise_rescaled_bleu_ci95')}; still below plaintext; "
    f"privacy upper bound, F3/A11). Paper Table II: 83.021 -> 12.425. N={n_passages}."
)
g.note(
    "F14 (critical): Vec2Text gtr-base inverts vec2text-convention embeddings (T5 mean-pool, no Dense, "
    "no L2 normalisation, norms ~0.5-1.0); the M0 sentence-transformers GTR embeddings give garbage even "
    "without encryption. The gate audits that convention via saved arrays + recorded norms."
)
g.note(
    "Bootstrap 95% CI: passage-level nonparametric bootstrap, 1000 resamples, fixed seed 20260101; "
    "the gate recomputes the CI (and per-passage components) from the stored hypotheses + references "
    "in this environment and requires an exact match."
)
g.note(
    "Step/beam sensitivity (F6/A11): the same 5 passages were re-run at num_steps=50, "
    "sequence_beam_width=10 in all three modes; the 20/4 baseline is recomputed on exactly the same "
    "5 passages from the 50-passage run. The Lead reduced the sensitivity sample from 10 to 5 "
    "because the shared machine was loaded (loadavg ~5-7); this is a scope adjustment (direction/"
    "magnitude only), not a threshold relaxation. The gate requires the deltas and the config, not a "
    "particular sign (the paper states neither setting)."
)
g.note(
    "The absolute difference from paper Table II is reported, not fitted: the paper does not state "
    "num_steps / beam width / N / per-passage length (A11) and uses an A100; our MPS-feasible "
    "settings are recorded in the artefact and explained in notes/exp3_vec2text.md. No gate "
    "threshold was relaxed relative to the 10-passage main experiment; coverage was extended (N>=50, CI, 50/10 sensitivity on 5 passages)."
)
g.note(
    "This gate validates the committed artefact and recomputes the metric + CI + encryption-audit "
    "layer in the calling environment (vec2text inversion itself is not re-run)."
)

g.finish()
