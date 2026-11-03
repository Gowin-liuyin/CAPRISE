"""exp3 -- Vec2Text reconstruction attack on CAPRISE-encrypted embeddings (paper Table II).

Paper: arXiv:2601.12331v1, Section V-B / Table II.
    w/o Enc  : BLEU 83.021 | P 0.947 | R 0.950 | F1 0.948 | ROUGE 0.950
    CAPRISE  : BLEU 12.425 | P 0.482 | R 0.498 | F1 0.487 | ROUGE 0.492
    parameter setting: s = 3, beta = 0.2.

Sample-size protocol (task-2)
-----------------------------
The paper does not state its sample size (finding F6/A11).  The first 10-passage
artefact was too small for a stable mean, so the main experiment now uses **50
real MS MARCO passages**, selected by the unchanged rule: the first N passages
of ``data/passages.jsonl`` whose tokenised length fits the corrector's 32-token
generation budget (``max_ref_tokens=32``).  The candidate pool is expanded
automatically (starting at 400 rows) until 50 such passages are found; the
number of rows scanned, the passage ids and the per-passage token lengths are
recorded in the artefact.  Every mode is also reported with a **passage-level
nonparametric bootstrap 95% CI** (1000 resamples, fixed seed 20260101): the
corpus-level metric is recomputed on each resample (sacrebleu corpus BLEU,
micro token P/R/F1, mean ROUGE-L) and the 2.5/97.5 percentiles are reported.

Step/beam sensitivity (finding F6)
----------------------------------
The paper also does not state Vec2Text's ``num_steps`` / ``sequence_beam_width``
(A11/F6).  The main run uses the MPS-feasible ``num_steps=20,
sequence_beam_width=4``; a sensitivity experiment re-runs the first 5 passages
in all three modes with ``num_steps=50, sequence_beam_width=10`` and reports the
metric deltas against the 20/4 baseline measured on exactly the same 5
passages (taken from the 50-passage run, not from the old 10-passage artefact).
The Lead reduced this sensitivity sample from 10 to 5 because the shared machine
ran at loadavg ~5-7 and the 10-passage sensitivity exceeded the time budget; the
sensitivity is used for direction/magnitude only, not for absolute alignment.

Embedding convention (reproduction finding F14 -- critical!)
------------------------------------------------------------
Vec2Text's gtr-base corrector was trained on *vec2text attack-convention*
embeddings: the mean-pooled last hidden state of the frozen GTR-T5-base
encoder, WITHOUT the sentence-transformers Dense projection and WITHOUT L2
normalisation (empirical norms ~0.5-1.0).  The M0 artifact
data/passage_emb.npy instead stores sentence-transformers GTR embeddings
(2_Dense + normalisation, unit norm).  Feeding those to vec2text yields garbage
reconstructions even without encryption, so it cannot be used as the attack
baseline.  We therefore compute the attack-convention embeddings with

    E = corrector.inversion_trainer.call_embedding_model(input_ids, attention_mask)

(tokenise passages with corrector.embedder_tokenizer, max_length=128), exactly
as the Lead-verified F14 recipe.  All three modes then attack these embeddings.

Attack pipeline (threat model of paper Section III-A)
-----------------------------------------------------
The honest-but-curious server holds the stored embedding ciphertext and can
invert it with Vec2Text (Morris et al., 2023):
    1. 50 real MS MARCO passages (data/passages.jsonl, full text as reference);
    2. baseline: invert the plaintext attack-convention embeddings E_i;
    3. CAPRISE (s=3, beta=0.2): encrypt the database vectors,
       e'_i = s*E_i + lambda_i, ||lambda_i|| < 3*s*beta/8 = 0.225 (core EncDB);
    4. two attacker adaptations (paper does not specify how the attack is
       adapted to ciphertexts -- ambiguity A11, see also F14):
         - "caprise_raw"     : invert e'_i directly.  ||e'_i|| ~ s*||E_i|| is far
                               outside the inversion model's input distribution
                               -> naive/weak attacker;
         - "caprise_rescaled": invert e'_i * (||E_i|| / ||e'_i||).  The attacker
                               restores the original per-vector norm (it knows
                               the public embedding convention; assuming it also
                               knows the plaintext norm is the attacker-
                               favourable, i.e. most conservative-for-privacy,
                               assumption).  Strongest attack -> privacy upper
                               bound.
    5. score each reconstruction against the original passage text with
       src/caprise/metrics.py:text_metrics (corpus BLEU 0-100 via sacrebleu
       effective_order, token-level micro P/R/F1, mean ROUGE-L F1).

Environment (vec2text 0.0.13 requires transformers 4.x)
-------------------------------------------------------
    env -u ALL_PROXY HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 \
      .venv-vec2text/bin/python experiments/exp3_vec2text.py            # full: 50x3 + 10x3 sensitivity

    ... --phase main            # only the 50-passage main experiment
    ... --phase sensitivity     # resume: reuse main hypotheses from the artefact
    ... --metrics-only          # recompute metrics/CI/deltas from stored hypotheses
    ... --n 3 --no-sensitivity  # quick pipeline check

The corrector is loaded offline from local HF cache snapshots by
scripts/vec2text_compat.py and placed on the available device (mps here).
Inversion runs in chunks (batch_size=4; chunking does not change per-passage
greedy beam search) -- the MPS device was far slower with a single large batch.
The main artefact is written after the main phase and updated as soon as each
sensitivity mode finishes, so a sensitivity-only resume is possible without
re-running the 50-passage inversion.  The attack-convention embeddings, the
CAPRISE ciphertexts and the rescaled attack inputs are saved in
results/exp3_vec2text_arrays.npz so the M11 gate can audit the whole
encryption/attack chain without re-running vec2text.

Outputs
-------
    results/exp3_vec2text.json        all numbers + hypotheses + per-passage metrics
                                      + bootstrap CI + sensitivity + references
    results/exp3_vec2text_arrays.npz  E (attack convention), E' (CAPRISE), E_rescaled
    results/exp3_vec2text.csv         per-passage rows + aggregate rows with CI
    figures/fig_vec2text.png          paper Table II comparison + CI + sensitivity
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # for scripts.vec2text_compat
sys.path.insert(0, str(ROOT / "src"))

# keep caches / matplotlib config inside the workspace
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from caprise import metrics as M  # noqa: E402
from caprise.caprise import EncDB, KeyGen, noise_radius_adcpe, noise_radius_db  # noqa: E402
from caprise.metrics import _tokens as metric_tokens  # noqa: E402  (same tokenisation as M.token_prf)
from caprise.prf import new_nonce  # noqa: E402

PIPELINE_TAG = "vec2text_t5_meanpool_v1"  # F14: attack-convention embeddings

PAPER_TABLE_II = {
    "w/o Enc": {"bleu": 83.021, "precision": 0.947, "recall": 0.950, "f1": 0.948, "rouge": 0.950},
    "CAPRISE": {"bleu": 12.425, "precision": 0.482, "recall": 0.498, "f1": 0.487, "rouge": 0.492},
}
METRIC_KEYS = ("bleu", "precision", "recall", "f1", "rouge")
MODES = ("plaintext", "caprise_raw", "caprise_rescaled")

# bootstrap 95% CI: passage-level nonparametric bootstrap, fixed seed (task-2)
BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260101
BOOTSTRAP_CI_LEVEL = 0.95

PASSAGES_PATH = ROOT / "data" / "passages.jsonl"
OUT_JSON = ROOT / "results" / "exp3_vec2text.json"
OUT_CSV = ROOT / "results" / "exp3_vec2text.csv"
OUT_FIG = ROOT / "figures" / "fig_vec2text.png"
OUT_NPZ = ROOT / "results" / "exp3_vec2text_arrays.npz"
SOLO_CACHE = ROOT / ".cache" / "exp3_vec2text_smoke"  # smoke files live outside the committed artefacts

MAIN_CACHE_KEYS = ("n", "num_steps", "sequence_beam_width", "seed", "s", "beta", "max_ref_tokens")


# --------------------------------------------------------------------------- #
# deterministic randomness (same nonce derivation convention as exp4)
# --------------------------------------------------------------------------- #
def _rng(seed: int, *tags) -> np.random.Generator:
    h = hashlib.sha256(("|".join(str(t) for t in (seed,) + tags)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


# --------------------------------------------------------------------------- #
# data: real MS MARCO passages (full text references)
# --------------------------------------------------------------------------- #
def select_passages(tok, n: int, max_ref_tokens: int, initial_pool: int = 400,
                    path: Path = PASSAGES_PATH) -> tuple[list[str], dict]:
    """Select ``n`` passages whose tokenised length fits the corrector budget.

    Vec2Text's gtr-base corrector was trained with max_seq_length = 32
    (repo name jxm/gtr__nq__32) and generates at most ~32 tokens, so a reference
    longer than that can never be reconstructed in full and corpus BLEU is
    meaningless for it (finding F6/A11: the paper does not state its text-length
    protocol).  We therefore restrict the sample to passages whose tokens fit
    the model's generation budget -- an honest, documented protocol choice that
    makes the plaintext baseline a valid ceiling for the attack.

    The candidate pool starts at ``initial_pool`` rows and is expanded (same
    rule, one row at a time) until ``n`` passages are found or the file ends.
    """
    if not path.exists():
        raise RuntimeError(
            f"{path.relative_to(ROOT)} missing; run scripts/prepare_data.py first (M0)"
        )
    selected: list[dict] = []
    scanned = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            scanned += 1
            row = json.loads(line)
            text = str(row["text"]).strip()
            ids = tok(text, truncation=False, add_special_tokens=True)["input_ids"]
            if len(ids) <= max_ref_tokens:
                selected.append(
                    {
                        "pid": int(row["pid"]),
                        "text": text,
                        "source": row.get("source"),
                        "ref_token_length": len(ids),
                    }
                )
                if len(selected) >= n:
                    break
    if len(selected) < n:
        raise RuntimeError(
            f"only {len(selected)} passages with <= {max_ref_tokens} tokens in "
            f"{path.relative_to(ROOT)} ({scanned} rows scanned)"
        )
    texts = [row["text"] for row in selected]
    provenance = {
        "source": "data/passages.jsonl (M0 artifact; real MS MARCO passages)",
        "selection": (
            f"first {n} passages of data/passages.jsonl whose tokenised length is <= "
            f"{max_ref_tokens} tokens (the vec2text gtr__nq__32 corrector generation budget; "
            f"F6/A11); the candidate pool starts at {initial_pool} rows and is expanded with the "
            f"same rule until {n} passages are found -- {scanned} rows were scanned"
            + (f" (expanded from {initial_pool})" if scanned > initial_pool else "")
            + ". Full passage text is used as the BLEU/P/R/F1/ROUGE reference (not truncated)."
        ),
        "max_ref_tokens": int(max_ref_tokens),
        "initial_pool_size": int(initial_pool),
        "pool_size": int(scanned),
        "pool_expanded": bool(scanned > initial_pool),
        "passage_ids": [row["pid"] for row in selected],
        "passage_sources": [row.get("source") for row in selected],
        "texts": texts,
        "ref_token_lengths": [int(row["ref_token_length"]) for row in selected],
        "text_token_stats": {
            "min": int(min(row["ref_token_length"] for row in selected)),
            "max": int(max(row["ref_token_length"] for row in selected)),
            "mean": float(np.mean([row["ref_token_length"] for row in selected])),
        },
    }
    return texts, provenance


# --------------------------------------------------------------------------- #
# Vec2Text corrector + attack-convention embeddings (F14)
# --------------------------------------------------------------------------- #
def load_corrector():
    """Offline loader prepared by the Lead (scripts/vec2text_compat.py)."""
    from scripts.vec2text_compat import load_local_corrector

    t0 = time.time()
    corrector = load_local_corrector()
    seconds = time.time() - t0
    dev = next(corrector.model.parameters()).device
    return corrector, str(dev), seconds


def model_paths() -> dict:
    """Record the exact offline snapshots used (task-2: model path in artefact)."""
    try:
        from scripts.vec2text_compat import snapshot_path

        return {
            "inversion": snapshot_path("inversion"),
            "corrector": snapshot_path("corrector"),
            "embedder": snapshot_path("gtr_embedder"),
            "t5_base": snapshot_path("t5-base"),
            "repo_mirror": str(ROOT / "models" / "vec2text"),
        }
    except Exception as exc:  # pragma: no cover - metadata only
        return {"error": str(exc)}


def attack_embeddings(corrector, texts: list[str], max_length: int = 128):
    """F14: vec2text attack-convention embeddings = T5 mean-pool, no Dense, no L2 norm."""
    import torch

    dev = next(corrector.model.parameters()).device
    tok = corrector.embedder_tokenizer
    inputs = tok(
        texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    ).to(dev)
    t0 = time.time()
    with torch.no_grad():
        E = corrector.inversion_trainer.call_embedding_model(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
        )
    E_np = E.detach().float().cpu().numpy().astype(np.float64)
    return E_np, time.time() - t0


# --------------------------------------------------------------------------- #
# CAPRISE encryption of the attack-convention embeddings
# --------------------------------------------------------------------------- #
def encrypt_embeddings(E: np.ndarray, s: int, beta: float, seed: int):
    m, d = E.shape
    key = KeyGen(dim=d, beta=beta, s=s, rng=_rng(seed, "key", d))
    rng = _rng(seed, "nonces", d, m)
    nonces = [new_nonce(rng) for _ in range(m)]
    Ep, _ = EncDB(key, E, nonces=nonces)
    lam = Ep - float(s) * E
    lam_norms = np.linalg.norm(lam, axis=1)
    ep_norms = np.linalg.norm(Ep, axis=1)
    enc = {
        "method": "CAPRISE Algorithm 1 EncDB (src/caprise/caprise.py)",
        "s": int(s),
        "beta": float(beta),
        "d": int(d),
        "noise_radius_db": float(noise_radius_db(s, beta)),
        "noise_radius_adcpe": float(noise_radius_adcpe(s, beta)),
        "noise_bound": float(3.0 * s * beta / 8.0),
        "noise_norm_min": float(lam_norms.min()),
        "noise_norm_max": float(lam_norms.max()),
        "noise_norm_mean": float(lam_norms.mean()),
        "ep_norm_min": float(ep_norms.min()),
        "ep_norm_max": float(ep_norms.max()),
        "ep_norm_mean": float(ep_norms.mean()),
        "nonces_sha256": hashlib.sha256(b"".join(nonces)).hexdigest(),
        "deterministic_nonces": True,
        "max_abs_ep_minus_s_e": float(np.abs(lam).max()),
    }
    return Ep, enc


def build_modes(E: np.ndarray, Ep: np.ndarray) -> dict:
    """The three attack inputs: plaintext, raw ciphertext, rescaled ciphertext."""
    e_norm = np.linalg.norm(E, axis=1, keepdims=True)
    ep_norm = np.linalg.norm(Ep, axis=1, keepdims=True)
    rescaled = Ep * (e_norm / ep_norm)  # attacker restores the original per-vector norm
    return {
        "plaintext": E.astype(np.float32),
        "caprise_raw": Ep.astype(np.float32),
        "caprise_rescaled": rescaled.astype(np.float32),
    }


# --------------------------------------------------------------------------- #
# inversion + metrics
# --------------------------------------------------------------------------- #
def _empty_mps_cache() -> None:
    """Release cached MPS blocks between chunks/modes (beam search allocates a lot)."""
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # pragma: no cover - best effort
        pass


def invert_embeddings_np(corrector, embeddings: np.ndarray, num_steps: int, beam_width: int):
    import torch
    import vec2text

    dev = next(corrector.model.parameters()).device
    E_t = torch.tensor(np.asarray(embeddings, dtype=np.float32)).to(dev)
    t0 = time.time()
    texts = vec2text.invert_embeddings(
        embeddings=E_t,
        corrector=corrector,
        num_steps=num_steps,
        sequence_beam_width=beam_width,
    )
    seconds = time.time() - t0
    return [str(t).strip() for t in texts], seconds


def compute_metrics(hypotheses: list[str], references: list[str]) -> dict:
    mets = M.text_metrics(list(hypotheses), list(references))
    return {k: float(v) for k, v in mets.items()}


def per_passage_scores(hypotheses: list[str], references: list[str]) -> dict:
    """Per-passage metric components (CSV rows, figure, bootstrap resampling).

    * ``tp/fp/fn``: token-count components of micro P/R/F1, so a resampled
      corpus can be scored exactly by summing the resampled counts;
    * ``bleu``: sentence BLEU (sacrebleu effective_order);
    * ``rouge``: per-pair ROUGE-L F1 (mean over pairs == M.rouge_l).
    """
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import BLEU

    metric = BLEU(effective_order=True)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    tp, fp, fn, sentence_bleu, rouge = [], [], [], [], []
    for hyp, ref in zip(hypotheses, references):
        h = Counter(metric_tokens(hyp))
        r = Counter(metric_tokens(ref))
        tp.append(int(sum((h & r).values())))
        fp.append(int(sum((h - r).values())))
        fn.append(int(sum((r - h).values())))
        sentence_bleu.append(float(metric.sentence_score(hyp, [ref]).score) if hyp.strip() else 0.0)
        rouge.append(float(scorer.score(ref, hyp)["rougeL"].fmeasure))
    precision, recall, f1 = [], [], []
    for t, f_p, f_n in zip(tp, fp, fn):
        p = t / (t + f_p) if (t + f_p) else 0.0
        r = t / (t + f_n) if (t + f_n) else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return {
        "bleu": sentence_bleu,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rouge": rouge,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def bootstrap_ci(hypotheses: list[str], references: list[str], per_passage: dict,
                 n_resamples: int = BOOTSTRAP_N_RESAMPLES, seed: int = BOOTSTRAP_SEED,
                 ci_level: float = BOOTSTRAP_CI_LEVEL) -> dict:
    """Passage-level nonparametric bootstrap 95% CI of the corpus-level metrics.

    A fixed seed generates one (n_resamples x n) resampling-index matrix; the
    same matrix is used for every mode.  For each resample the corpus-level
    metric is recomputed from scratch (corpus BLEU needs the texts; token
    P/R/F1 is reconstructible from the per-passage counts; mean ROUGE-L is the
    mean of the resampled per-passage ROUGE values), so the CI is the percentile
    interval of the metric estimator -- not the spread of per-passage scores.
    """
    n = len(hypotheses)
    if n == 0:
        raise ValueError("bootstrap_ci needs at least one passage")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, n, size=(int(n_resamples), n))
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
    low_pct = 100.0 * alpha
    high_pct = 100.0 * (1.0 - alpha)
    return {
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "ci_level": float(ci_level),
        "method": (
            "passage-level nonparametric bootstrap: resample the n passages with replacement "
            "(fixed seed), recompute the corpus-level metric on each resample, report the "
            "2.5/97.5 percentiles; passage weights are uniform and the same resampling matrix "
            "is shared by all modes"
        ),
        "metrics": {
            k: {
                "mean": float(np.mean(v)),
                "low": float(np.percentile(v, low_pct)),
                "high": float(np.percentile(v, high_pct)),
                "std": float(np.std(v, ddof=1)),
            }
            for k, v in vals.items()
        },
    }


def subset_metrics(hypotheses: list[str], references: list[str], indices: list[int]):
    """Metrics (and per-passage scores) on a passage subset -- 20/4 baseline of the sensitivity run."""
    hyps = [hypotheses[i] for i in indices]
    refs = [references[i] for i in indices]
    return compute_metrics(hyps, refs), per_passage_scores(hyps, refs)


def _attach_metrics(payload: dict, hypotheses: list[str], references: list[str],
                    n_resamples: int = BOOTSTRAP_N_RESAMPLES, seed: int = BOOTSTRAP_SEED,
                    ci_level: float = BOOTSTRAP_CI_LEVEL) -> None:
    """Fill a mode payload with metrics, per-passage scores and bootstrap CI."""
    per_passage = per_passage_scores(hypotheses, references)
    metrics = compute_metrics(hypotheses, references)
    boot = bootstrap_ci(hypotheses, references, per_passage, n_resamples, seed, ci_level)
    for k in METRIC_KEYS:
        boot["metrics"][k]["point_estimate"] = float(metrics[k])
    payload["hypotheses"] = list(hypotheses)
    payload["metrics"] = metrics
    payload["per_passage_metrics"] = per_passage
    payload["per_passage_bleu"] = per_passage["bleu"]
    payload["bootstrap_ci"] = boot


# --------------------------------------------------------------------------- #
# figure: paper Table II comparison + CI + step/beam sensitivity
# --------------------------------------------------------------------------- #
MODE_LABELS = {
    "plaintext": "plaintext\n(w/o Enc)",
    "caprise_raw": "CAPRISE raw e'\n(paper Table II analogue)",
    "caprise_rescaled": "CAPRISE rescaled e'\n(strongest attacker)",
}
MODE_COLORS = {"plaintext": "#2ca02c", "caprise_raw": "#ff9896", "caprise_rescaled": "#d62728"}


def make_figure(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = results["modes"]
    paper = results["paper_table_ii"]
    setup = results.get("setup", {})
    order = [m for m in MODES if m in modes]
    n_passages = setup.get("n_passages")
    colors = [MODE_COLORS[m] for m in order]
    labels = [MODE_LABELS[m] for m in order]

    fig, axes = plt.subplots(2, 2, figsize=(17.0, 10.5))

    # (a) BLEU (0-100) + bootstrap 95% CI + paper Table II reference lines
    ax = axes[0, 0]
    xs = np.arange(len(order))
    vals = [modes[m]["metrics"]["bleu"] for m in order]
    yerr = [[], []]
    for m, v in zip(order, vals):
        ci = modes[m].get("bootstrap_ci", {}).get("metrics", {}).get("bleu", {})
        yerr[0].append(max(0.0, v - ci.get("low", v)))
        yerr[1].append(max(0.0, ci.get("high", v) - v))
    ax.bar(xs, vals, color=colors, edgecolor="black", width=0.6, yerr=yerr, capsize=6,
           error_kw={"elinewidth": 1.4, "ecolor": "black"}, label="this reproduction (BLEU + 95% bootstrap CI)")
    ax.axhline(paper["w/o Enc"]["bleu"], color="gray", ls="--", lw=1.2, label="paper w/o Enc (83.021)")
    ax.axhline(paper["CAPRISE"]["bleu"], color="firebrick", ls=":", lw=1.2, label="paper CAPRISE (12.425)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("BLEU (0-100)")
    ax.set_title(f"(a) Vec2Text BLEU: plaintext vs CAPRISE-encrypted\n"
                 f"(N={n_passages} real MS MARCO passages; error bars = 95% bootstrap CI)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5)

    # (b) token P/R/F1 + ROUGE-L with CI
    ax = axes[0, 1]
    mets = [("precision", "P"), ("recall", "R"), ("f1", "F1"), ("rouge", "ROUGE-L")]
    width = 0.8 / len(order)
    for j, mode in enumerate(order):
        ys = [modes[mode]["metrics"][k] for k, _ in mets]
        err = [[], []]
        for k, _ in mets:
            ci = modes[mode].get("bootstrap_ci", {}).get("metrics", {}).get(k, {})
            v = modes[mode]["metrics"][k]
            err[0].append(max(0.0, v - ci.get("low", v)))
            err[1].append(max(0.0, ci.get("high", v) - v))
        ax.bar(np.arange(len(mets)) + j * width - 0.4 + width / 2, ys, width=width,
               color=MODE_COLORS[mode], edgecolor="black",
               yerr=err, capsize=3, error_kw={"elinewidth": 1.0, "ecolor": "black"},
               label=MODE_LABELS[mode].replace("\n", " "))
    for k, _ in mets:
        ax.axhline(paper["w/o Enc"][k], color="gray", ls="--", lw=0.9)
        ax.axhline(paper["CAPRISE"][k], color="firebrick", ls=":", lw=0.9)
    ax.set_xticks(np.arange(len(mets)))
    ax.set_xticklabels([lbl for _, lbl in mets])
    ax.set_ylabel("score (0-1)")
    ax.set_title("(b) token P/R/F1 and ROUGE-L (95% bootstrap CI)\n"
                 "(gray dashed = paper w/o Enc, red dotted = paper CAPRISE)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

    # (c) per-passage BLEU distribution
    ax = axes[1, 0]
    data = [modes[m]["per_passage_bleu"] for m in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    for patch, mode in zip(bp["boxes"], order):
        patch.set_facecolor(MODE_COLORS[mode])
        patch.set_alpha(0.75)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for i, (vals_i, mode) in enumerate(zip(data, order), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals_i))
        ax.scatter(np.full(len(vals_i), i) + jitter, vals_i, s=14, color=MODE_COLORS[mode],
                   edgecolor="black", linewidth=0.3, alpha=0.85, zorder=3)
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels([MODE_LABELS[m].replace("\n", " ") for m in order], fontsize=8, rotation=10)
    ax.set_ylabel("per-passage BLEU")
    ax.set_title(f"(c) per-passage BLEU distribution (N={n_passages}; each dot = 1 passage)")
    ax.grid(axis="y", alpha=0.3)

    # (d) num_steps / beam sensitivity on the same 5 passages
    ax = axes[1, 1]
    sens = results.get("sensitivity")
    if sens and sens.get("modes"):
        s_order = [m for m in MODES if m in sens["modes"]]
        xs = np.arange(len(s_order))
        w = 0.38
        base = [sens["baseline_20_4"][m]["metrics"]["bleu"] for m in s_order]
        fast = [sens["modes"][m]["metrics"]["bleu"] for m in s_order]
        ax.bar(xs - w / 2, base, width=w, color=[MODE_COLORS[m] for m in s_order], alpha=0.45,
               edgecolor="black", hatch="//", label="num_steps=20, beam=4")
        ax.bar(xs + w / 2, fast, width=w, color=[MODE_COLORS[m] for m in s_order],
               edgecolor="black", label="num_steps=50, beam=10")
        for x, b, f in zip(xs, base, fast):
            ax.annotate(f"{f - b:+.2f}", (x, max(b, f) + 1.5), ha="center", fontsize=8)
        for x, b, f in zip(xs, base, fast):
            ax.plot([x - w / 2, x + w / 2], [b, f], color="black", lw=0.8, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels([MODE_LABELS[m].replace("\n", " ") for m in s_order], fontsize=8, rotation=10)
        ax.set_ylabel("BLEU (0-100)")
        ax.set_title(f"(d) step/beam sensitivity on the same {len(sens['passage_indices'])} passages\n"
                     "(labels = BLEU delta 50/10 minus 20/4)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7.5)
    else:
        ax.text(0.5, 0.5, "sensitivity run not included in this artefact",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_axis_off()

    fig.suptitle(
        "Vec2Text reconstruction attack (paper Table II reproduction) -- "
        f"s=3, beta=0.2, N={n_passages}, num_steps={setup.get('num_steps')}, "
        f"beam={setup.get('sequence_beam_width')}, device={setup.get('device')}, "
        f"attack embeddings: {PIPELINE_TAG}; CI: {BOOTSTRAP_N_RESAMPLES} bootstrap resamples (seed {BOOTSTRAP_SEED})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CSV: per-passage rows + aggregate rows with bootstrap CI
# --------------------------------------------------------------------------- #
CSV_HEADER = [
    "section", "mode", "index", "pid", "bleu", "precision", "recall", "f1", "rouge",
    "ci_metric", "ci_low", "ci_high", "hypothesis", "reference",
]


def _csv_metric_row(section: str, mode: str, metrics: dict, ci: dict, k: str) -> list:
    ci_k = (ci or {}).get("metrics", {}).get(k, {})
    return [
        section, mode, "", "", metrics["bleu"], metrics["precision"], metrics["recall"],
        metrics["f1"], metrics["rouge"], k, ci_k.get("low", ""), ci_k.get("high", ""), "", "",
    ]


def write_csv(results: dict, path: Path) -> None:
    """Long CSV: one row per passage + aggregate rows carrying the 95% bootstrap CI."""
    texts = results["passages"]["texts"]
    pids = results["passages"]["passage_ids"]
    rows: list[list] = []
    for mode in MODES:
        payload = results["modes"].get(mode)
        if not payload:
            continue
        pp = payload["per_passage_metrics"]
        for i in range(len(texts)):
            rows.append([
                "passage", mode, i, pids[i], pp["bleu"][i], pp["precision"][i], pp["recall"][i],
                pp["f1"][i], pp["rouge"][i], "", "", "", payload["hypotheses"][i], texts[i],
            ])
        for k in METRIC_KEYS:
            rows.append(_csv_metric_row("aggregate", mode, payload["metrics"], payload.get("bootstrap_ci"), k))
    for name, mt in results["paper_table_ii"].items():
        rows.append([
            "paper_table_ii", name, "", "", mt["bleu"], mt["precision"], mt["recall"],
            mt["f1"], mt["rouge"], "", "", "", "", "",
        ])
    sens = results.get("sensitivity")
    if sens and sens.get("modes"):
        refs = sens["references"]
        for mode in MODES:
            payload = sens["modes"].get(mode)
            if not payload:
                continue
            pp = payload["per_passage_metrics"]
            for j, i in enumerate(sens["passage_indices"]):
                rows.append([
                    "sensitivity_passage", mode, j, sens["passage_ids"][j], pp["bleu"][j],
                    pp["precision"][j], pp["recall"][j], pp["f1"][j], pp["rouge"][j],
                    "", "", "", payload["hypotheses"][j], refs[j],
                ])
            for k in METRIC_KEYS:
                rows.append(_csv_metric_row("sensitivity_aggregate", mode, payload["metrics"],
                                            payload.get("bootstrap_ci"), k))
            base = sens["baseline_20_4"].get(mode)
            if base:
                bpp = base["per_passage_metrics"]
                for j, i in enumerate(sens["passage_indices"]):
                    rows.append([
                        "sensitivity_baseline_passage", mode, j, sens["passage_ids"][j], bpp["bleu"][j],
                        bpp["precision"][j], bpp["recall"][j], bpp["f1"][j], bpp["rouge"][j],
                        "", "", "", results["modes"][mode]["hypotheses"][i], refs[j],
                    ])
                for k in METRIC_KEYS:
                    rows.append(_csv_metric_row("sensitivity_baseline_aggregate", mode, base["metrics"], None, k))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# derived tables: paper comparison, findings, summary
# --------------------------------------------------------------------------- #
def finalize_results(results: dict) -> dict:
    """Recompute the paper comparison, findings and summary from the metrics."""
    modes = results["modes"]
    plain = modes.get("plaintext", {}).get("metrics")
    raw = modes.get("caprise_raw", {}).get("metrics")
    resc = modes.get("caprise_rescaled", {}).get("metrics")

    comparison = {}
    for entry, ours, note in (
        ("w/o Enc", plain, "paper 'w/o Enc' row: plaintext attack-convention embeddings"),
        ("CAPRISE", raw, "paper-closest CAPRISE row: raw ciphertext e' fed to vec2text (no rescaling)"),
        ("CAPRISE_rescaled_attack", resc,
         "strongest-attacker row (not in the paper): e' * ||E||/||e'|| restores the original norm; upper bound on reconstruction"),
    ):
        if ours is None:
            continue
        paper_vals = PAPER_TABLE_II["CAPRISE" if entry.startswith("CAPRISE") else entry]
        comparison[entry] = {
            "ours": ours,
            "paper": paper_vals,
            "note": note,
            "delta_ours_minus_paper": {k: float(ours[k] - paper_vals[k]) for k in METRIC_KEYS},
            "ratio_ours_over_paper": {
                k: float(ours[k] / paper_vals[k]) if paper_vals[k] else None for k in METRIC_KEYS
            },
        }
        ci = modes.get("caprise_rescaled" if entry == "CAPRISE_rescaled_attack" else
                       ("caprise_raw" if entry == "CAPRISE" else "plaintext"), {}).get("bootstrap_ci")
        if ci:
            comparison[entry]["bootstrap_ci_95"] = {k: {"low": ci["metrics"][k]["low"],
                                                        "high": ci["metrics"][k]["high"]}
                                                    for k in METRIC_KEYS}
    results["comparison"] = comparison

    findings = []
    if plain and raw:
        findings.append({
            "kind": "main_effect",
            "row": "CAPRISE raw ciphertext (paper Table II analogue)",
            "plain_bleu": plain["bleu"],
            "caprise_raw_bleu": raw["bleu"],
            "bleu_drop_abs": plain["bleu"] - raw["bleu"],
            "bleu_drop_fraction": (plain["bleu"] - raw["bleu"]) / plain["bleu"] if plain["bleu"] else None,
            "plain_f1": plain["f1"],
            "caprise_raw_f1": raw["f1"],
            "conclusion": "Vec2Text reconstruction collapses when the raw CAPRISE ciphertext is fed "
                          "to the inversion model (paper-closest attack setting): BLEU drops by "
                          "more than half",
        })
    if plain and resc:
        findings.append({
            "kind": "scale_adaptation_recovery",
            "plain_bleu": plain["bleu"],
            "caprise_rescaled_bleu": resc["bleu"],
            "bleu_drop_fraction_vs_plain": (plain["bleu"] - resc["bleu"]) / plain["bleu"] if plain["bleu"] else None,
            "recovery_vs_raw_bleu": (resc["bleu"] - raw["bleu"]) if raw else None,
            "conclusion": "a scale-normalising adversary (e'*||E||/||e'||) recovers much of the "
                          "reconstruction ability -- the strongest attacker, reported as the privacy "
                          "upper bound (F3/A11); CAPRISE still degrades it relative to plaintext",
        })
    n = int(results.get("setup", {}).get("n_passages", 0))
    boot_cfg = results.get("bootstrap", {})
    if plain:
        findings.append({
            "kind": "paper_baseline",
            "ours_plain_bleu": plain["bleu"],
            "paper_plain_bleu": PAPER_TABLE_II["w/o Enc"]["bleu"],
            "delta": plain["bleu"] - PAPER_TABLE_II["w/o Enc"]["bleu"],
            "n_passages": n,
            "note": "paper does not state num_steps/beam/N/passage length (A11); our MPS budget "
                    f"(20 steps, beam 4, N={n}) and the 32-token reference calibration may explain "
                    "the delta",
        })
    if raw:
        findings.append({
            "kind": "paper_caprise_row",
            "row": "CAPRISE raw ciphertext (paper Table II analogue)",
            "ours_caprise_bleu": raw["bleu"],
            "paper_caprise_bleu": PAPER_TABLE_II["CAPRISE"]["bleu"],
            "delta": raw["bleu"] - PAPER_TABLE_II["CAPRISE"]["bleu"],
            "rescaled_attack_bleu": resc["bleu"] if resc else None,
        })
    if raw and resc:
        findings.append({
            "kind": "attack_adaptation",
            "raw_bleu": raw["bleu"],
            "rescaled_bleu": resc["bleu"],
            "conclusion": "raw ciphertext (norm ~ s*||E||) is out of the inversion model's input "
                          "distribution and is much weaker than restoring the original norm; the "
                          "rescaled attack is the conservative (attacker-favourable) main setting",
            "rescaled_attack_stronger": bool(resc["bleu"] > raw["bleu"]),
        })
    findings.append({
        "kind": "embedding_convention_F14",
        "attack_convention": PIPELINE_TAG,
        "ep_norm_mean": results.get("encryption", {}).get("ep_norm_mean"),
        "conclusion": "Vec2Text gtr-base operates on T5 mean-pooled (un-Densed, un-normalised) "
                      "embeddings; using sentence-transformers GTR embeddings as attack input "
                      "produces garbage even without encryption (F14), so the attack baseline "
                      "here is the correct vec2text convention.",
    })
    ci_ok = all(modes.get(m, {}).get("bootstrap_ci", {}).get("metrics") for m in MODES if m in modes)
    if ci_ok:
        def ci_bounds(mode):
            ci = modes[mode]["bootstrap_ci"]["metrics"]
            return {k: [ci[k]["low"], ci[k]["high"]] for k in METRIC_KEYS}
        findings.append({
            "kind": "bootstrap_ci",
            "n_passages": n,
            "n_resamples": boot_cfg.get("n_resamples", BOOTSTRAP_N_RESAMPLES),
            "seed": boot_cfg.get("seed", BOOTSTRAP_SEED),
            "ci_level": boot_cfg.get("ci_level", BOOTSTRAP_CI_LEVEL),
            "plain_bleu_ci95": ci_bounds("plaintext")["bleu"] if "plaintext" in modes else None,
            "caprise_raw_bleu_ci95": ci_bounds("caprise_raw")["bleu"] if "caprise_raw" in modes else None,
            "caprise_rescaled_bleu_ci95": ci_bounds("caprise_rescaled")["bleu"] if "caprise_rescaled" in modes else None,
            "conclusion": "50-passage means are reported with a passage-level nonparametric bootstrap "
                          "95% CI (1000 resamples, fixed seed); the CAPRISE-vs-plaintext gap is far "
                          "larger than the CI widths, so the main privacy effect is not a small-sample "
                          "artefact",
        })
    sens = results.get("sensitivity")
    if sens and sens.get("modes") and sens.get("deltas"):
        findings.append({
            "kind": "step_beam_sensitivity",
            "n_passages": len(sens.get("passage_indices", [])),
            "baseline": sens.get("baseline_config"),
            "sensitivity": sens.get("config"),
            "bleu_delta": {m: sens["deltas"][m]["bleu"]["delta"] for m in sens["deltas"]},
            "f1_delta": {m: sens["deltas"][m]["f1"]["delta"] for m in sens["deltas"]},
            "headline": sens.get("headline"),
            "conclusion": "num_steps=50/beam=10 vs 20/4 changes the reconstructed-text metrics only "
                          "modestly on the same passages (see deltas); the direction and size bound "
                          "how much of the paper-vs-reproduction gap could be due to the unspecified "
                          "Vec2Text budget (F6/A11) -- it does not explain the full gap",
        })
    results["findings"] = findings

    def _ci_tuple(mode, key):
        ci = modes.get(mode, {}).get("bootstrap_ci", {}).get("metrics", {}).get(key, {})
        if "low" not in ci:
            return None
        return [ci["low"], ci["high"]]

    results["summary"] = {
        "n_passages": n,
        "main_paper_comparison_row": "CAPRISE raw ciphertext (paper Table II analogue; paper does not specify A11 adaptation)",
        "plain_bleu": plain["bleu"] if plain else None,
        "plain_bleu_ci95": _ci_tuple("plaintext", "bleu"),
        "caprise_raw_bleu": raw["bleu"] if raw else None,
        "caprise_raw_bleu_ci95": _ci_tuple("caprise_raw", "bleu"),
        "caprise_rescaled_bleu": resc["bleu"] if resc else None,
        "caprise_rescaled_bleu_ci95": _ci_tuple("caprise_rescaled", "bleu"),
        "per_text_seconds_mean": float(
            np.mean([payload["per_text_seconds"] for payload in modes.values()])
        ) if modes else None,
        "attack_convention": PIPELINE_TAG,
        "bootstrap": boot_cfg or None,
        "sensitivity_bleu_delta": (
            {m: sens["deltas"][m]["bleu"]["delta"] for m in sens["deltas"]}
            if sens and sens.get("deltas") else None
        ),
    }
    return results


# --------------------------------------------------------------------------- #
# experiment driver
# --------------------------------------------------------------------------- #
def _main_cache_cfg(cfg: dict) -> dict:
    return {k: cfg[k] for k in MAIN_CACHE_KEYS}


def _sensitivity_cache_cfg(cfg: dict) -> dict:
    s = cfg["sensitivity"]
    return {
        "n": s["n"],
        "num_steps": s["num_steps"],
        "sequence_beam_width": s["sequence_beam_width"],
        "seed": cfg["seed"],
        "s": cfg["s"],
        "beta": cfg["beta"],
        "max_ref_tokens": cfg["max_ref_tokens"],
    }


def _artifact_main_cfg(artifact: dict) -> dict | None:
    cfg = artifact.get("config", {})
    try:
        return {k: cfg[k] for k in MAIN_CACHE_KEYS}
    except KeyError:
        return None


def _artifact_sensitivity_cfg(artifact: dict) -> dict | None:
    cfg = artifact.get("sensitivity", {}).get("config", {})
    keys = ("n", "num_steps", "sequence_beam_width", "seed", "s", "beta", "max_ref_tokens")
    try:
        return {k: cfg[k] for k in keys}
    except KeyError:
        return None


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _rel(path: Path) -> str:
    """Workspace-relative path for log lines; absolute path for files outside ROOT (smoke runs)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict, status: str) -> None:
    payload["status"] = status
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resume_modes(artifact: dict, cfg: dict, n: int, phase: str) -> dict:
    """Reuse completed main-mode hypotheses from a previous artefact (same config)."""
    if not cfg.get("use_cache", True) or not artifact:
        return {}
    if _artifact_main_cfg(artifact) != _main_cache_cfg(cfg):
        return {}
    resume = {}
    for mode in cfg["modes"]:
        payload = artifact.get("modes", {}).get(mode, {})
        if len(payload.get("hypotheses", [])) == n:
            resume[mode] = payload
    if phase == "sensitivity" and len(resume) != len(cfg["modes"]):
        raise RuntimeError(
            "phase=sensitivity needs a complete main artefact with the same config; "
            f"found modes {sorted(resume)} of {list(cfg['modes'])}"
        )
    return resume


def _resume_sensitivity(artifact: dict, cfg: dict) -> dict:
    if not cfg.get("use_cache", True) or not artifact:
        return {}
    sens = artifact.get("sensitivity", {})
    if _artifact_sensitivity_cfg(artifact) != _sensitivity_cache_cfg(cfg):
        return {}
    n = int(cfg["sensitivity"]["n"])
    out = {}
    for mode in cfg["modes"]:
        payload = sens.get("modes", {}).get(mode, {})
        if len(payload.get("hypotheses", [])) == n:
            out[mode] = payload
    return out


def run_experiment(cfg: dict, out_json: Path) -> dict:
    t_start = time.time()
    n = int(cfg["n"])
    seed = int(cfg["seed"])
    s = int(cfg["s"])
    beta = float(cfg["beta"])
    num_steps = int(cfg["num_steps"])
    beam = int(cfg["sequence_beam_width"])
    wanted_modes = list(cfg["modes"])
    phase = str(cfg.get("phase", "all"))
    bootstrap_cfg = cfg.get("bootstrap", {})
    n_resamples = int(bootstrap_cfg.get("n_resamples", BOOTSTRAP_N_RESAMPLES))
    boot_seed = int(bootstrap_cfg.get("seed", BOOTSTRAP_SEED))
    ci_level = float(bootstrap_cfg.get("ci_level", BOOTSTRAP_CI_LEVEL))

    previous = _read_json(out_json) if phase != "main" or cfg.get("use_cache", True) else {}
    resume_main = _resume_modes(previous, cfg, n, phase)
    resume_sens = _resume_sensitivity(previous, cfg) if phase != "main" else {}

    results: dict = {
        "experiment": "exp3_vec2text",
        "paper_ref": "paper Section V-B / Table II (Vec2Text reconstruction attack)",
        "definition": {
            "bleu": "corpus BLEU, 0-100 (sacrebleu with effective_order=True)",
            "precision/recall/f1": "token-level micro-averaged P/R/F1 over the corpus (0-1)",
            "rouge": "mean ROUGE-L F1 (0-1)",
            "bootstrap_ci": (
                "passage-level nonparametric bootstrap (1000 resamples, fixed seed 20260101); the "
                "corpus-level metric is recomputed on each resample and the 2.5/97.5 percentiles are "
                "reported (shared resampling matrix across modes)"
            ),
            "attack_embedding_convention": (
                "vec2text attack-convention embeddings E = T5 encoder last_hidden_state mean-pool "
                "(corrector.inversion_trainer.call_embedding_model), NO sentence-transformers Dense "
                "projection and NO L2 normalisation -- finding F14. data/passage_emb.npy (ST GTR "
                "embeddings, Dense+normalised) is NOT used as the attack input because vec2text "
                "reconstructs garbage from it even without encryption."
            ),
            "attack_modes": {
                "plaintext": "invert the plaintext attack-convention embedding E_i (paper 'w/o Enc')",
                "caprise_raw": "invert the CAPRISE ciphertext e'_i directly (norm ~ s*||E_i|| ~ 1.5-3, out of the inversion model's input distribution; naive attacker)",
                "caprise_rescaled": "invert e'_i * (||E_i||/||e'_i||): attacker restores the original per-vector norm; strongest attack, used for the privacy upper bound",
            },
            "references": "original MS MARCO passage texts (full text, not truncated)",
        },
        "setup": {
            "n_passages": n,
            "num_steps": num_steps,
            "sequence_beam_width": beam,
            "seed": seed,
            "s": s,
            "beta": beta,
            "d": None,
            "device": "unknown",
            "corrector_load_seconds": None,
            "attack_embedding_convention": PIPELINE_TAG,
            "vec2text": "0.0.13 (jxm/gtr__nq__32 inversion + jxm/gtr__nq__32__correct corrector, offline local snapshots)",
            "max_ref_tokens": int(cfg.get("max_ref_tokens", 32)),
            "batch_size": int(cfg.get("batch_size", 4)),
            "environments": {
                "python": sys.version.split()[0],
                "run_with": "must be .venv-vec2text/bin/python (transformers 4.x); main .venv has transformers 5.x and is incompatible with vec2text 0.0.13",
                "hf_home": os.environ.get("HF_HOME"),
                "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            },
            "bootstrap": {
                "n_resamples": n_resamples,
                "seed": boot_seed,
                "ci_level": ci_level,
            },
        },
        "passages": {},
        "embeddings": {},
        "encryption": {},
        "modes": {},
        "bootstrap": {
            "n_resamples": n_resamples,
            "seed": boot_seed,
            "ci_level": ci_level,
            "statistic": "corpus-level metric recomputed on passage-level bootstrap resamples",
            "same_resample_indices_for_all_modes": True,
        },
        "sensitivity": {},
        "paper_table_ii": PAPER_TABLE_II,
        "comparison": {},
        "findings": [],
        "config": cfg,
    }

    # ---- corrector (offline) ----
    corrector, dev, load_s = load_corrector()
    results["setup"]["device"] = dev
    results["setup"]["corrector_load_seconds"] = round(load_s, 2)
    results["setup"]["model_paths"] = model_paths()
    try:
        import torch

        results["setup"]["torch"] = torch.__version__
        results["setup"]["device_family"] = "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:  # pragma: no cover
        pass
    print(f"[exp3] corrector loaded in {load_s:.1f}s on {dev}", flush=True)

    # ---- passage selection (corrector generation budget, F6/A11) ----
    texts, provenance = select_passages(
        corrector.embedder_tokenizer, n, int(cfg.get("max_ref_tokens", 32)),
        int(cfg.get("initial_pool_size", 400)),
    )
    results["passages"] = provenance
    print(f"[exp3] selected {n} passages (<= {provenance['max_ref_tokens']} tokens) by scanning "
          f"{provenance['pool_size']} rows (initial pool {provenance['initial_pool_size']})", flush=True)

    # ---- attack-convention embeddings (F14) + CAPRISE encryption ----
    E, emb_seconds = attack_embeddings(corrector, texts)
    e_norms = np.linalg.norm(E, axis=1)
    results["setup"]["d"] = int(E.shape[1])
    results["embeddings"] = {
        "convention": PIPELINE_TAG,
        "description": "mean-pooled T5 last_hidden_state (vec2text attack convention); no Dense, no L2 normalisation",
        "norm_min": float(e_norms.min()),
        "norm_max": float(e_norms.max()),
        "norm_mean": float(e_norms.mean()),
        "seconds": round(emb_seconds, 2),
        "sha256": hashlib.sha256(E.astype(np.float32).tobytes()).hexdigest(),
    }
    print(f"[exp3] attack-convention embeddings: ||E|| in [{e_norms.min():.3f}, {e_norms.max():.3f}] "
          f"(mean {e_norms.mean():.3f})", flush=True)

    Ep, enc = encrypt_embeddings(E, s, beta, seed)
    results["encryption"] = enc

    mode_emb = build_modes(E, Ep)
    mode_emb = {k: v for k, v in mode_emb.items() if k in wanted_modes}
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        E_attack=E.astype(np.float32),
        E_prime=Ep.astype(np.float32),
        E_rescaled=build_modes(E, Ep)["caprise_rescaled"].astype(np.float32),
        references=np.array(texts, dtype=object),
    )
    results["embeddings"]["arrays_file"] = str(OUT_NPZ.relative_to(ROOT))
    results["embeddings"]["arrays_sha256"] = hashlib.sha256(OUT_NPZ.read_bytes()).hexdigest()
    print(f"[exp3] wrote {OUT_NPZ.relative_to(ROOT)} (E, E', E_rescaled + references)", flush=True)

    # ---- main inversion: n passages x 3 modes at num_steps/beam ----
    # Inversion is done in chunks: batch=20 with beam=4 on the MPS device was far
    # slower than the per-text scaling suggested; chunking keeps the per-step beam
    # batch small.  Greedy beam search per passage is independent of the batch, so
    # chunking does not change the reconstruction quality.
    batch_size = max(1, int(cfg.get("batch_size", 4)))
    t_invert0 = time.time()
    for mode, emb in mode_emb.items():
        reused = resume_main.get(mode)
        if reused is not None:
            hypotheses = [str(h) for h in reused["hypotheses"]]
            seconds = float(reused.get("seconds", 0.0) or 0.0)
            payload = {}
            _attach_metrics(payload, hypotheses, texts, n_resamples, boot_seed, ci_level)
            payload["seconds"] = seconds
            payload["per_text_seconds"] = seconds / n if n else None
            payload["reused_from_artifact"] = True
            results["modes"][mode] = payload
            print(f"[exp3] mode {mode}: reused {n} hypotheses from existing artefact "
                  f"(recorded {seconds:.1f}s)", flush=True)
            continue
        print(f"[exp3] mode {mode}: inverting {n} embeddings in chunks of {batch_size} "
              f"(num_steps={num_steps}, beam={beam})...", flush=True)
        hypotheses: list[str] = []
        seconds = 0.0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            _empty_mps_cache()
            chunk_hyps, chunk_s = invert_embeddings_np(corrector, emb[start:end], num_steps, beam)
            hypotheses.extend(chunk_hyps)
            seconds += chunk_s
            print(f"[exp3] mode {mode}: {end}/{n} done, chunk {chunk_s:.1f}s "
                  f"(total {seconds:.1f}s, {seconds / end:.2f}s/text)", flush=True)
        payload = {}
        _attach_metrics(payload, hypotheses, texts, n_resamples, boot_seed, ci_level)
        payload["seconds"] = seconds
        payload["per_text_seconds"] = seconds / n if n else None
        payload["reused_from_artifact"] = False
        results["modes"][mode] = payload
        print(f"[exp3] mode {mode}: {seconds:.1f}s ({seconds / n:.2f}s/text); "
              f"BLEU={payload['metrics']['bleu']:.3f} "
              f"CI95=[{payload['bootstrap_ci']['metrics']['bleu']['low']:.3f}, "
              f"{payload['bootstrap_ci']['metrics']['bleu']['high']:.3f}]", flush=True)
    results["setup"]["this_invocation_inversion_seconds"] = round(time.time() - t_invert0, 2)
    # total compute time of the main inversion (sums per-mode wall time, so it is
    # preserved when a later invocation resumes from cached hypotheses)
    results["setup"]["inversion_seconds_total"] = round(
        sum(float(payload.get("seconds", 0.0)) for payload in results["modes"].values()), 2
    )

    # write the main artefact before the (long) sensitivity phase so a sensitivity
    # crash cannot lose the 50-passage result
    finalize_results(results)
    results["runtime"] = {
        "seconds_total": round(time.time() - t_start, 1),
        "main_phase_seconds": round(time.time() - t_start, 1),
        "main_compute_seconds": round(float(results["setup"]["inversion_seconds_total"]), 1),
        "compute_seconds_total": round(float(results["setup"]["inversion_seconds_total"]), 1),
        "sensitivity_phase_seconds": None,
        "main_resumed_modes": sorted(resume_main),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "device": dev,
    }
    _write_json(out_json, results, "main_complete_sensitivity_pending" if cfg.get("run_sensitivity") else "complete")
    print(f"[exp3] main artefact written -> {_rel(out_json)}", flush=True)

    # ---- sensitivity: steps/beam 20/4 -> 50/10 on the same 5 passages ----
    if cfg.get("run_sensitivity"):
        sens_cfg = cfg["sensitivity"]
        sens_n = int(sens_cfg["n"])
        sens_steps = int(sens_cfg["num_steps"])
        sens_beam = int(sens_cfg["sequence_beam_width"])
        sens_batch = max(1, int(sens_cfg.get("batch_size", batch_size)))
        if sens_n > n:
            raise RuntimeError(f"sensitivity n={sens_n} exceeds main n={n}")
        indices = list(range(sens_n))
        sens_refs = [texts[i] for i in indices]
        sens_pids = [provenance["passage_ids"][i] for i in indices]
        sens = {
            "purpose": "num_steps / sequence_beam_width sensitivity (paper does not state them; F6/A11)",
            "passage_indices": indices,
            "passage_ids": sens_pids,
            "references": sens_refs,
            "config": {
                "n": sens_n,
                "num_steps": sens_steps,
                "sequence_beam_width": sens_beam,
                "seed": seed,
                "s": s,
                "beta": beta,
                "max_ref_tokens": int(cfg.get("max_ref_tokens", 32)),
            },
            "baseline_config": {
                "num_steps": num_steps,
                "sequence_beam_width": beam,
                "source": "same passages, taken from the 50-passage main run",
            },
            "modes": {},
            "baseline_20_4": {},
            "deltas": {},
            "headline": None,
        }
        t_sens0 = time.time()
        for mode in wanted_modes:
            reused = resume_sens.get(mode)
            if reused is not None:
                hypotheses = [str(h) for h in reused["hypotheses"]]
                seconds = float(reused.get("seconds", 0.0) or 0.0)
                payload = {}
                _attach_metrics(payload, hypotheses, sens_refs, n_resamples, boot_seed, ci_level)
                payload["seconds"] = seconds
                payload["per_text_seconds"] = seconds / sens_n if sens_n else None
                payload["reused_from_artifact"] = True
                sens["modes"][mode] = payload
                print(f"[exp3] sensitivity {mode}: reused {sens_n} hypotheses from artefact", flush=True)
                continue
            print(f"[exp3] sensitivity {mode}: inverting {sens_n} embeddings in chunks of "
                  f"{sens_batch} (num_steps={sens_steps}, beam={sens_beam})...", flush=True)
            hypotheses = []
            seconds = 0.0
            for start in range(0, sens_n, sens_batch):
                end = min(start + sens_batch, sens_n)
                _empty_mps_cache()
                chunk_hyps, chunk_s = invert_embeddings_np(
                    corrector, mode_emb[mode][indices][start:end], sens_steps, sens_beam
                )
                hypotheses.extend(chunk_hyps)
                seconds += chunk_s
                print(f"[exp3] sensitivity {mode}: {end}/{sens_n} done, chunk {chunk_s:.1f}s "
                      f"(total {seconds:.1f}s, {seconds / end:.2f}s/text)", flush=True)
            payload = {}
            _attach_metrics(payload, hypotheses, sens_refs, n_resamples, boot_seed, ci_level)
            payload["seconds"] = seconds
            payload["per_text_seconds"] = seconds / sens_n if sens_n else None
            payload["reused_from_artifact"] = False
            sens["modes"][mode] = payload
            # incremental write so a later sensitivity mode crash keeps earlier ones
            results["sensitivity"] = sens
            _write_json(out_json, finalize_results(results), "sensitivity_in_progress")
            print(f"[exp3] sensitivity {mode}: {seconds:.1f}s ({seconds / sens_n:.2f}s/text); "
                  f"BLEU={payload['metrics']['bleu']:.3f}", flush=True)

        # baseline 20/4 on exactly the same passages (from the 50-passage run)
        sens["baseline_20_4"] = {}
        deltas = {}
        for mode in wanted_modes:
            base_metrics, base_pp = subset_metrics(results["modes"][mode]["hypotheses"], texts, indices)
            sens["baseline_20_4"][mode] = {
                "metrics": base_metrics,
                "per_passage_metrics": base_pp,
                "source": "50-passage main run, same passages as the sensitivity subset (num_steps=20, beam=4)",
            }
            deltas[mode] = {
                k: {
                    "sensitivity_50_10": float(sens["modes"][mode]["metrics"][k]),
                    "baseline_20_4": float(base_metrics[k]),
                    "delta": float(sens["modes"][mode]["metrics"][k] - base_metrics[k]),
                }
                for k in METRIC_KEYS
            }
        sens["deltas"] = deltas
        sens["headline"] = (
            f"on the same {sens_n} passages, changing num_steps 20->{sens_steps} and "
            f"sequence_beam_width 4->{sens_beam} changes corpus BLEU by "
            f"{deltas['plaintext']['bleu']['delta']:+.3f} (plaintext), "
            f"{deltas.get('caprise_raw', {}).get('bleu', {}).get('delta', float('nan')):+.3f} (CAPRISE raw), "
            f"{deltas.get('caprise_rescaled', {}).get('bleu', {}).get('delta', float('nan')):+.3f} "
            f"(CAPRISE rescaled)"
        )
        sens["seconds_total"] = round(time.time() - t_sens0, 1)
        results["sensitivity"] = sens
        print(f"[exp3] sensitivity headline: {sens['headline']}", flush=True)

        finalize_results(results)
        results["runtime"]["seconds_total"] = round(time.time() - t_start, 1)
        results["runtime"]["sensitivity_phase_seconds"] = round(time.time() - t_sens0, 1)
        results["runtime"]["sensitivity_compute_seconds"] = round(
            sum(float(payload.get("seconds", 0.0)) for payload in sens["modes"].values()), 1
        )
        results["runtime"]["compute_seconds_total"] = round(
            float(results["setup"]["inversion_seconds_total"])
            + float(results["runtime"]["sensitivity_compute_seconds"]), 1
        )
        results["runtime"]["sensitivity_resumed_modes"] = sorted(resume_sens)
        _write_json(out_json, results, "complete")
        print(f"[exp3] final artefact written -> {_rel(out_json)}", flush=True)
    else:
        _write_json(out_json, finalize_results(results), "complete")

    return results


# --------------------------------------------------------------------------- #
# metrics-only mode (no vec2text import; usable from the main venv / M11 gate)
# --------------------------------------------------------------------------- #
def recompute_metrics_from_artifact(path: Path) -> dict:
    results = json.loads(path.read_text(encoding="utf-8"))
    passages = results["passages"]
    texts = passages["texts"]
    boot_cfg = results.get("bootstrap", {})
    n_resamples = int(boot_cfg.get("n_resamples", BOOTSTRAP_N_RESAMPLES))
    boot_seed = int(boot_cfg.get("seed", BOOTSTRAP_SEED))
    ci_level = float(boot_cfg.get("ci_level", BOOTSTRAP_CI_LEVEL))
    for mode, payload in results["modes"].items():
        hyps = payload.get("hypotheses", [])
        if len(hyps) != len(texts):
            raise RuntimeError(f"{mode}: {len(hyps)} hypotheses vs {len(texts)} references")
        _attach_metrics(payload, hyps, texts, n_resamples, boot_seed, ci_level)
    sens = results.get("sensitivity")
    if sens and sens.get("modes"):
        indices = sens["passage_indices"]
        refs = [texts[i] for i in indices]
        for mode, payload in sens["modes"].items():
            hyps = payload.get("hypotheses", [])
            if len(hyps) != len(refs):
                raise RuntimeError(f"sensitivity {mode}: {len(hyps)} hypotheses vs {len(refs)} references")
            _attach_metrics(payload, hyps, refs, n_resamples, boot_seed, ci_level)
        for mode in list(sens["modes"]):
            base_metrics, base_pp = subset_metrics(results["modes"][mode]["hypotheses"], texts, indices)
            sens["baseline_20_4"][mode] = {
                "metrics": base_metrics,
                "per_passage_metrics": base_pp,
                "source": "50-passage main run, same passages as the sensitivity subset (num_steps=20, beam=4)",
            }
            sens["deltas"][mode] = {
                k: {
                    "sensitivity_50_10": float(sens["modes"][mode]["metrics"][k]),
                    "baseline_20_4": float(base_metrics[k]),
                    "delta": float(sens["modes"][mode]["metrics"][k] - base_metrics[k]),
                }
                for k in METRIC_KEYS
            }
        sens["headline"] = (
            f"on the same {len(indices)} passages, changing num_steps "
            f"{results['setup']['num_steps']}->{sens['config']['num_steps']} and "
            f"sequence_beam_width {results['setup']['sequence_beam_width']}->"
            f"{sens['config']['sequence_beam_width']} changes corpus BLEU by "
            f"{sens['deltas']['plaintext']['bleu']['delta']:+.3f} (plaintext), "
            f"{sens['deltas'].get('caprise_raw', {}).get('bleu', {}).get('delta', float('nan')):+.3f} (CAPRISE raw), "
            f"{sens['deltas'].get('caprise_rescaled', {}).get('bleu', {}).get('delta', float('nan')):+.3f} "
            f"(CAPRISE rescaled)"
        )
    return finalize_results(results)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def default_cfg() -> dict:
    return {
        "n": 50,
        "num_steps": 20,
        "sequence_beam_width": 4,
        "seed": 20260101,
        "s": 3,
        "beta": 0.2,
        "modes": list(MODES),
        "use_cache": True,
        "batch_size": 4,
        "max_ref_tokens": 32,
        "initial_pool_size": 400,
        "phase": "all",
        "run_sensitivity": True,
        "sensitivity": {"n": 5, "num_steps": 50, "sequence_beam_width": 10, "batch_size": 2},
        "bootstrap": {
            "n_resamples": BOOTSTRAP_N_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "ci_level": BOOTSTRAP_CI_LEVEL,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="exp3: Vec2Text reconstruction attack (paper Table II)")
    ap.add_argument("--smoke", action="store_true", help="3 passages quick check (no sensitivity)")
    ap.add_argument("--metrics-only", action="store_true",
                    help="recompute metrics/CI/deltas from stored hypotheses+references (no vec2text)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--num-steps", type=int, default=None)
    ap.add_argument("--beam", type=int, default=None)
    ap.add_argument("--modes", type=str, default=None, help="comma separated")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None, help="inversion chunk size (default 4)")
    ap.add_argument("--max-ref-tokens", type=int, default=None,
                    help="max reference tokens (corrector generation budget; default 32)")
    ap.add_argument("--pool-size", type=int, default=None,
                    help="initial candidate pool size; expanded automatically until n passages found")
    ap.add_argument("--phase", choices=("all", "main", "sensitivity"), default=None,
                    help="all = 50x3 + sensitivity; main = only the 50x3; sensitivity = resume 50x3 from artefact")
    ap.add_argument("--sensitivity-n", type=int, default=None)
    ap.add_argument("--sensitivity-steps", type=int, default=None)
    ap.add_argument("--sensitivity-beam", type=int, default=None)
    ap.add_argument("--sensitivity-batch-size", type=int, default=None)
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--out-json", type=str, default=str(OUT_JSON))
    ap.add_argument("--out-csv", type=str, default=str(OUT_CSV))
    ap.add_argument("--out-fig", type=str, default=str(OUT_FIG))
    args = ap.parse_args(argv)

    if args.metrics_only:
        path = Path(args.out_json)
        if not path.exists():
            print(f"[exp3] artifact {path} does not exist", file=sys.stderr)
            return 1
        results = recompute_metrics_from_artifact(path)
        _write_json(path, results, "complete")
        write_csv(results, Path(args.out_csv))
        make_figure(results, Path(args.out_fig))
        for mode, payload in results["modes"].items():
            mt = payload["metrics"]
            ci = payload["bootstrap_ci"]["metrics"]["bleu"]
            print(f"[exp3] {mode}: BLEU={mt['bleu']:.3f} "
                  f"CI95=[{ci['low']:.3f}, {ci['high']:.3f}] P={mt['precision']:.3f} "
                  f"R={mt['recall']:.3f} F1={mt['f1']:.3f} ROUGE={mt['rouge']:.3f}")
        print(f"[exp3] metrics/CI recomputed -> {_rel(path)}")
        return 0

    cfg = default_cfg()
    if args.smoke:
        cfg.update({"n": 3, "modes": list(MODES), "use_cache": False, "run_sensitivity": False})
    if args.n is not None:
        cfg["n"] = args.n
    if args.num_steps is not None:
        cfg["num_steps"] = args.num_steps
    if args.beam is not None:
        cfg["sequence_beam_width"] = args.beam
    if args.modes:
        cfg["modes"] = [m.strip() for m in args.modes.split(",") if m.strip()]
    if args.no_cache:
        cfg["use_cache"] = False
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.max_ref_tokens is not None:
        cfg["max_ref_tokens"] = args.max_ref_tokens
    if args.pool_size is not None:
        cfg["initial_pool_size"] = args.pool_size
    if args.sensitivity_n is not None:
        cfg["sensitivity"]["n"] = args.sensitivity_n
    if args.sensitivity_steps is not None:
        cfg["sensitivity"]["num_steps"] = args.sensitivity_steps
    if args.sensitivity_beam is not None:
        cfg["sensitivity"]["sequence_beam_width"] = args.sensitivity_beam
    if args.sensitivity_batch_size is not None:
        cfg["sensitivity"]["batch_size"] = args.sensitivity_batch_size
    if args.no_sensitivity:
        cfg["run_sensitivity"] = False
    if args.phase is not None:
        cfg["phase"] = args.phase
        if args.phase == "main":
            cfg["run_sensitivity"] = False
        if args.phase == "sensitivity":
            cfg["run_sensitivity"] = True

    if args.smoke:
        out_json = SOLO_CACHE.with_suffix(".json")
        out_csv = SOLO_CACHE.with_suffix(".csv")
        out_fig = SOLO_CACHE.with_suffix(".png")
    else:
        out_json = Path(args.out_json)
        out_csv = Path(args.out_csv)
        out_fig = Path(args.out_fig)

    print(f"[exp3] config: {json.dumps(cfg)}", flush=True)
    t0 = time.time()
    results = run_experiment(cfg, out_json)
    print(f"[exp3] run finished in {time.time() - t0:.1f}s", flush=True)

    write_csv(results, out_csv)
    print(f"[exp3] wrote {out_csv.relative_to(ROOT) if out_csv.is_relative_to(ROOT) else out_csv}")
    make_figure(results, out_fig)
    print(f"[exp3] wrote {out_fig.relative_to(ROOT) if out_fig.is_relative_to(ROOT) else out_fig}")

    for mode, payload in results["modes"].items():
        mt = payload["metrics"]
        ci = payload["bootstrap_ci"]["metrics"]["bleu"]
        print(f"[exp3] {mode}: BLEU={mt['bleu']:.3f} CI95=[{ci['low']:.3f}, {ci['high']:.3f}] "
              f"P={mt['precision']:.3f} R={mt['recall']:.3f} F1={mt['f1']:.3f} "
              f"ROUGE={mt['rouge']:.3f} ({payload['per_text_seconds']:.2f}s/text)", flush=True)
    if results.get("sensitivity", {}).get("headline"):
        print(f"[exp3] {results['sensitivity']['headline']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
