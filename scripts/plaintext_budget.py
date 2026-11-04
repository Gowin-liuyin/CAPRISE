"""C2 follow-up: is the Table II *plaintext* gap a Vec2Text budget artefact?

Motivation (finding F6/A11): the paper gives BLEU 83.021 for plaintext GTR
embeddings but never states the Vec2Text budget (num_steps / beam width) or the
sample size.  On 50 passages with num_steps=20, beam=4 this reproduction gets
69.875.  The 5-passage sensitivity (50/10 vs 20/4) raised plaintext BLEU by
+21.7 on that tiny subset, which is suggestive but too noisy to attribute the
13-BLEU gap.

This script measures the *same* 20 real MS MARCO passages (token length <= 32,
the corrector's generation budget) at both budgets:

    20/4   (main experiment budget)
    50/10  (larger budget, batch=1 on MPS)

and reports corpus BLEU/P/R/F1/ROUGE-L with passage-level 95% bootstrap CI.
The comparison isolates the budget effect from sample-selection effects.

Run:  .venv-vec2text/bin/python scripts/plaintext_budget.py
Writes: results/exp3_plaintext_budget.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "exp3_plaintext_budget.json"
N_PASSAGES = 20
BUDGETS = (("20/4", 20, 4, 4), ("50/10", 50, 10, 1))  # (label, steps, beam, batch)
MAX_REF_TOKENS = 32
SEED = 20260101


def _load_exp3():
    spec = importlib.util.spec_from_file_location("exp3_vec2text_mod", ROOT / "experiments" / "exp3_vec2text.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    t0 = time.time()
    exp3 = _load_exp3()
    corrector, dev, load_s = exp3.load_corrector()
    print(f"[budget] corrector loaded in {load_s:.1f}s on {dev}", flush=True)

    texts, provenance = exp3.select_passages(
        corrector.embedder_tokenizer, N_PASSAGES, MAX_REF_TOKENS, 400
    )
    print(f"[budget] selected {len(texts)} passages (<= {MAX_REF_TOKENS} tokens) "
          f"by scanning {provenance['pool_size']} rows", flush=True)

    E, emb_s = exp3.attack_embeddings(corrector, texts)
    norms = np.linalg.norm(E, axis=1)
    print(f"[budget] attack-convention embeddings: ||E|| in [{norms.min():.3f},{norms.max():.3f}] "
          f"mean {norms.mean():.3f} ({emb_s:.1f}s)", flush=True)

    runs: dict = {}
    for label, steps, beam, batch in BUDGETS:
        print(f"[budget] budget {label}: inverting {len(texts)} embeddings "
              f"(num_steps={steps}, beam={beam}, batch={batch}) ...", flush=True)
        hyps: list[str] = []
        t_start = time.time()
        for i0 in range(0, len(texts), batch):
            chunk = E[i0:i0 + batch]
            chunk_hyps, _chunk_s = exp3.invert_embeddings_np(corrector, chunk, steps, beam)
            hyps.extend(chunk_hyps)
            exp3._empty_mps_cache()
            done = min(i0 + batch, len(texts))
            print(f"[budget]   {label}: {done}/{len(texts)} done, "
                  f"{(time.time() - t_start) / done:.1f}s/text", flush=True)
        per_text = (time.time() - t_start) / len(texts)
        payload: dict = {"hypotheses": hyps, "per_text_seconds": round(per_text, 3)}
        exp3._attach_metrics(payload, hyps, texts, exp3.BOOTSTRAP_N_RESAMPLES,
                             exp3.BOOTSTRAP_SEED, exp3.BOOTSTRAP_CI_LEVEL)
        runs[label] = payload
        print(f"[budget] budget {label}: BLEU={payload['metrics']['bleu']:.3f} "
              f"F1={payload['metrics']['f1']:.3f} ({per_text:.1f}s/text)", flush=True)

    base = runs["20/4"]["metrics"]
    big = runs["50/10"]["metrics"]
    paper_bleu = exp3.PAPER_TABLE_II["w/o Enc"]["bleu"]
    result = {
        "experiment": "exp3_plaintext_budget",
        "paper_ref": "Table II 'w/o Enc' row (BLEU 83.021); budget not stated (F6/A11)",
        "question": "is the plaintext BLEU gap (ours 69.9 vs paper 83.0 on N=50) a Vec2Text budget artefact?",
        "setup": {
            "n_passages": len(texts),
            "selection": provenance["selection"],
            "max_ref_tokens": MAX_REF_TOKENS,
            "attack_embedding_convention": exp3.PIPELINE_TAG,
            "device": dev,
            "corrector_load_seconds": round(load_s, 2),
            "seed": SEED,
            "passage_ids": provenance["passage_ids"],
            "ref_token_lengths": provenance["ref_token_lengths"],
            "embedding_norm_min": float(norms.min()),
            "embedding_norm_max": float(norms.max()),
            "embedding_norm_mean": float(norms.mean()),
        },
        "budgets": {
            label: {
                "num_steps": _s,
                "sequence_beam_width": _b,
                "batch": _ba,
                "metrics": runs[label]["metrics"],
                "bootstrap_ci_95": {k: {"low": v["low"], "high": v["high"]}
                                    for k, v in runs[label]["bootstrap_ci"]["metrics"].items()},
                "per_text_seconds": runs[label]["per_text_seconds"],
                "per_passage_bleu": runs[label]["per_passage_bleu"],
            }
            for (label, _s, _b, _ba) in BUDGETS
        },
        "comparison": {
            "paper_plaintext_bleu": paper_bleu,
            "bleu_20_4": base["bleu"],
            "bleu_50_10": big["bleu"],
            "delta_bleu_50_10_minus_20_4": big["bleu"] - base["bleu"],
            "bleu_50_10_minus_paper": big["bleu"] - paper_bleu,
            "bleu_20_4_minus_paper": base["bleu"] - paper_bleu,
            "ci_50_10_covers_paper": bool(
                runs["50/10"]["bootstrap_ci"]["metrics"]["bleu"]["low"] <= paper_bleu
                <= runs["50/10"]["bootstrap_ci"]["metrics"]["bleu"]["high"]
            ),
            "note": (
                "same passages, same attack-convention embeddings; only num_steps/beam change. "
                "The delta bounds how much of the paper-vs-reproduction plaintext gap the "
                "undisclosed budget can explain (F6/A11)."
            ),
        },
        "runtime_seconds": round(time.time() - t0, 1),
        "hardware": {
            "platform": sys.platform,
            "machine": __import__("platform").machine(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[budget] wrote {OUT.relative_to(ROOT)}", flush=True)
    print(f"[budget] BLEU 20/4 = {base['bleu']:.3f}, 50/10 = {big['bleu']:.3f}, "
          f"paper = {paper_bleu:.3f} (delta {big['bleu'] - paper_bleu:+.3f}); "
          f"CI(50/10) covers paper: {result['comparison']['ci_50_10_covers_paper']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
