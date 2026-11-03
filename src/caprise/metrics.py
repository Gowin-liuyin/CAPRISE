"""Metrics used by the reproduction experiments.

Retrieval: Recall@k (w.r.t. plaintext top-k / qrels), MRR@k, nDCG@k.
Text reconstruction (paper Table II): BLEU (0-100), token Precision/Recall/F1
(0-1), ROUGE-L F1 (0-1).
Vector-analysis attack (paper Fig. 5): ASR@t = |top_t(cipher) & top_t(plain)| / t,
plus Kendall tau / Spearman rank correlation as auxiliary leakage measures.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "recall_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "bleu",
    "token_prf",
    "rouge_l",
    "text_metrics",
    "top_t_overlap",
    "kendall_tau",
    "spearman",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #


def recall_at_k(retrieved: Sequence, reference: Iterable, k: int | None = None) -> float:
    """|retrieved[:k] & reference| / |reference|."""
    ref = set(reference)
    if not ref:
        return float("nan")
    chunk = list(retrieved[:k]) if k is not None else list(retrieved)
    return len(set(chunk) & ref) / len(ref)


def mrr_at_k(retrieved: Sequence, relevant: Iterable, k: int | None = None) -> float:
    """Reciprocal rank of the first relevant document (0 if none in top-k)."""
    rel = set(relevant)
    chunk = list(retrieved[:k]) if k is not None else list(retrieved)
    for rank, doc in enumerate(chunk, start=1):
        if doc in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence, qrels: dict, k: int) -> float:
    """nDCG@k with graded relevance ``qrels`` (id -> gain)."""
    dcg = 0.0
    for i, doc in enumerate(list(retrieved)[:k], start=1):
        gain = float(qrels.get(doc, 0.0))
        if gain > 0:
            dcg += (2.0**gain - 1.0) / math.log2(i + 1.0)
    ideal = sorted((float(g) for g in qrels.values() if g > 0), reverse=True)[:k]
    idcg = sum((2.0**g - 1.0) / math.log2(i + 1.0) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------- #
# Text reconstruction metrics (paper Table II)
# --------------------------------------------------------------------------- #


def bleu(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Corpus BLEU (sacrebleu, 0-100), matching the paper's BLEU scale."""
    from sacrebleu.metrics import BLEU

    metric = BLEU(effective_order=True)
    return float(metric.corpus_score(list(hypotheses), [list(references)]).score)


def token_prf(hypotheses: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Micro-averaged token-level Precision / Recall / F1 over the corpus."""
    tp = fp = fn = 0
    for hyp, ref in zip(hypotheses, references):
        h = Counter(_tokens(hyp))
        r = Counter(_tokens(ref))
        tp += sum((h & r).values())
        fp += sum((h - r).values())
        fn += sum((r - h).values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_l(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Mean ROUGE-L F1 (0-1)."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    vals = [scorer.score(ref, hyp)["rougeL"].fmeasure for hyp, ref in zip(hypotheses, references)]
    return float(np.mean(vals)) if vals else 0.0


def text_metrics(hypotheses: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """All reconstruction metrics in the paper's Table II layout."""
    prf = token_prf(hypotheses, references)
    return {
        "bleu": bleu(hypotheses, references),
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "rouge": rouge_l(hypotheses, references),
    }


# --------------------------------------------------------------------------- #
# Vector-analysis attack metrics (paper Fig. 5)
# --------------------------------------------------------------------------- #


def top_t_overlap(plain_ranking: Sequence[int], cipher_ranking: Sequence[int], t: int = 10) -> float:
    """ASR@t: fraction of plaintext top-t neighbours recovered in ciphertext top-t."""
    plain = set(list(plain_ranking)[:t])
    cipher = set(list(cipher_ranking)[:t])
    if not plain:
        return float("nan")
    return len(plain & cipher) / len(plain)


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy.stats import kendalltau

    return float(kendalltau(a, b).statistic)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)
