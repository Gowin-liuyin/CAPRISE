"""M0 gate: MS MARCO data + gtr-t5-base embeddings (data module).

Protocol under test
-------------------
Paper §V experimental setup: MS MARCO passage ranking dev (small) + gtr-t5-base
(768-d, L2-normalised) embeddings.

Gate criteria (from docs/MODULE_PLAN.md M0 + Lead instruction)
---------------------------------------------------------------
1. data/ artifacts present and non-empty: passages.jsonl, passage_emb.npy,
   passage_ids.npy, queries.jsonl, query_emb.npy, query_ids.npy, qrels.tsv,
   meta.json.
2. scale: m >= 20,000 passages; n_queries >= 200.
3. cleanliness: embeddings finite (no NaN/Inf); rows L2-normalised
   (| ||v|| - 1 | < 1e-3); row counts agree across npy/jsonl/ids; dim == 768.
4. qrels coverage: every included eval qid has >= 1 relevant passage inside the
   corpus (coverage == 100%), qrels pairs reported.
5. plaintext sanity: >= 200 queries, MRR@10 / Recall@10 / nDCG@10 vs qrels
   recomputed from the artifacts (real numbers; the Lead's expected GTR band is
   0.2 <= MRR@10 <= 0.5).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from gates.common import Gate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_data import retrieval_metrics  # noqa: E402  (same metric code as prepare_data)

g = Gate("M0", "data", owner="data-retrieval",
         paper_ref="§V Experimental setup (MS MARCO passage dev + gtr-t5-base)")

# --------------------------------------------------------------------------- #
# 1. files present and non-empty
# --------------------------------------------------------------------------- #
REQUIRED = [
    "passages.jsonl", "passage_emb.npy", "passage_ids.npy",
    "queries.jsonl", "query_emb.npy", "query_ids.npy",
    "qrels.tsv", "meta.json",
]
missing = [f for f in REQUIRED if not (DATA / f).exists() or (DATA / f).stat().st_size == 0]
g.check_true("01_files_present_nonempty", not missing,
             f"missing/empty={missing}; extra={'eval_qids.json' if (DATA/'eval_qids.json').exists() else 'NONE'}")

meta = json.loads((DATA / "meta.json").read_text())
eval_qids = json.loads((DATA / "eval_qids.json").read_text())

p_emb = np.load(DATA / "passage_emb.npy")
p_ids = np.load(DATA / "passage_ids.npy")
q_emb = np.load(DATA / "query_emb.npy")
q_ids = np.load(DATA / "query_ids.npy")
m, dim = p_emb.shape
nq = q_emb.shape[0]

n_passages_jsonl = sum(1 for _ in open(DATA / "passages.jsonl", encoding="utf-8"))
n_queries_jsonl = sum(1 for _ in open(DATA / "queries.jsonl", encoding="utf-8"))
qrels_rows = [line.rstrip("\n").split("\t") for line in open(DATA / "qrels.tsv", encoding="utf-8") if line.strip()]
qrels = [(int(r[0]), int(r[2])) for r in qrels_rows]

g.check_true("02_meta_reports_same_scale",
             meta.get("m_passages") == m and meta.get("n_queries") == nq,
             f"meta m={meta.get('m_passages')} nq={meta.get('n_queries')} vs npy m={m} nq={nq}")

# --------------------------------------------------------------------------- #
# 2. scale
# --------------------------------------------------------------------------- #
g.check("03_m_ge_20000", expected="m >= 20000", actual=int(m), passed=m >= 20_000, detail=f"m={m}")
g.check("04_n_queries_ge_200", expected="n_queries >= 200", actual=int(nq), passed=nq >= 200, detail=f"n_queries={nq}")

# --------------------------------------------------------------------------- #
# 3. cleanliness
# --------------------------------------------------------------------------- #
p_finite = bool(np.isfinite(p_emb).all())
q_finite = bool(np.isfinite(q_emb).all())
g.check_true("05_no_nan_inf", p_finite and q_finite,
             f"passage finite={p_finite}, query finite={q_finite}")

p_norms = np.linalg.norm(p_emb.astype(np.float64), axis=1)
q_norms = np.linalg.norm(q_emb.astype(np.float64), axis=1)
max_dev = float(max(np.abs(p_norms - 1).max(), np.abs(q_norms - 1).max()))
g.check_true("06_l2_normalised_dev_lt_1e-3", max_dev < 1e-3,
             f"max | ||v||-1 | = {max_dev:.3e} (passage max dev {np.abs(p_norms-1).max():.2e}, "
             f"query max dev {np.abs(q_norms-1).max():.2e})")

g.check_true("07_row_counts_consistent",
             m == len(p_ids) == n_passages_jsonl and nq == len(q_ids) == n_queries_jsonl,
             f"passages: emb={m} ids={len(p_ids)} jsonl={n_passages_jsonl}; "
             f"queries: emb={nq} ids={len(q_ids)} jsonl={n_queries_jsonl}")
g.check_true("08_embedding_dim_768", dim == 768, f"dim={dim}")
g.check_true("09_ids_unique", len(set(int(x) for x in p_ids)) == m and len(set(int(x) for x in q_ids)) == nq,
             "passage_ids and query_ids are unique")

# text alignment: non-empty texts, same order as ids
bad_text = 0
jsonl_pids = []
with open(DATA / "passages.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        jsonl_pids.append(int(rec["pid"]))
        if not rec.get("text", "").strip():
            bad_text += 1
g.check_true("10_texts_nonempty_and_aligned",
             bad_text == 0 and jsonl_pids == [int(x) for x in p_ids],
             f"empty_texts={bad_text}; jsonl/ids order identical={jsonl_pids == [int(x) for x in p_ids]}")

# --------------------------------------------------------------------------- #
# 4. qrels coverage
# --------------------------------------------------------------------------- #
corpus_pids = set(int(x) for x in p_ids)
qrels_in_corpus = [(q, p) for q, p in qrels if p in corpus_pids]
rel_by_q: dict[int, set[int]] = {}
for q, p in qrels_in_corpus:
    rel_by_q.setdefault(q, set()).add(p)
eval_set = set(int(q) for q in eval_qids)
covered = {q for q in eval_set if q in rel_by_q}
coverage = len(covered) / max(1, len(eval_set))
g.check_true("11_qrels_coverage_100pct", coverage == 1.0,
             f"eval qids with >=1 relevant passage in corpus: {len(covered)}/{len(eval_set)} "
             f"({coverage*100:.2f}%), qrels pairs in corpus={len(qrels_in_corpus)}, qids={len(rel_by_q)}")

# --------------------------------------------------------------------------- #
# 5. plaintext sanity retrieval (recomputed from artifacts)
# --------------------------------------------------------------------------- #
pid2row = {int(p): i for i, p in enumerate(p_ids)}
rel_pairs_idx = [(q, pid2row[p]) for q, p in qrels_in_corpus if q in eval_set and p in pid2row]
t0 = time.time()
metrics = retrieval_metrics(q_emb, p_emb, q_ids, rel_pairs_idx)
dt = time.time() - t0

# random-ranking baseline (analytic for 1 relevant among m):
# E[1/rank * 1{rank<=10}] ~ (1/m) * H_10
random_mrr_baseline = float((1.0 / m) * sum(1.0 / r for r in range(1, 11)))

g.check("12_sanity_queries_ge_200", expected=">=200 evaluated", actual=int(metrics["n_queries_evaluated"]),
        passed=metrics["n_queries_evaluated"] >= 200,
        detail=f"{metrics['n_queries_evaluated']} queries, {dt:.1f}s")
g.check("13_mrr10_in_expected_band", expected="0.2 <= MRR@10 <= 0.5", actual=round(metrics["MRR@10"], 4),
        passed=0.2 <= metrics["MRR@10"] <= 0.5,
        detail=f"MRR@10={metrics['MRR@10']:.4f}; GTR-T5-base full-corpus MS MARCO dev MRR@10 is ~0.39 "
               f"(this corpus keeps BM25 hard negatives, so a value in the band is expected)")
g.check("14_recall10_above_random", expected="Recall@10 > 0.5", actual=round(metrics["Recall@10"], 4),
        passed=metrics["Recall@10"] > 0.5, detail=f"Recall@10={metrics['Recall@10']:.4f}")
g.check("15_mrr_beats_random_baseline", expected=f"> {random_mrr_baseline:.2e} (random ranking)",
        actual=round(metrics["MRR@10"], 4), passed=metrics["MRR@10"] > 100 * random_mrr_baseline,
        detail=f"random-ranking baseline MRR@10 ~ {random_mrr_baseline:.2e}")

# the original task-1 "pure uniform random filler" control is stored in data_sanity.json
sanity_path = RESULTS / "data_sanity.json"
if sanity_path.exists():
    sanity = json.loads(sanity_path.read_text())
    control = sanity.get("random_fill_only_control", {})
    g.note(f"plaintext sanity (primary, with BM25 hard negatives): "
           f"MRR@10={metrics['MRR@10']:.4f}, Recall@10={metrics['Recall@10']:.4f}, "
           f"nDCG@10={metrics['nDCG@10']:.4f}, Recall@100={metrics['Recall@100']:.4f}")
    if control:
        g.note(f"control (original spec: uniform random filler only, hard negatives removed): "
               f"MRR@10={control.get('MRR@10')}, Recall@10={control.get('Recall@10')}, "
               f"n={control.get('n_queries_evaluated')}")
else:
    g.note("results/data_sanity.json missing (run scripts/prepare_data.py without --skip-gate-inputs)")

g.note(f"corpus composition: {meta.get('corpus_composition')}")
g.note(f"embedding: {meta.get('embedding_model')} dim={dim}, device={meta.get('hardware', {}).get('device')}, "
       f"dtype={meta.get('hardware', {}).get('embedding_dtype')}")
g.note(f"embedding latency: {RESULTS.joinpath('embed_timing.json').exists() and json.loads((RESULTS/'embed_timing.json').read_text()).get('batch_ms_mean')} ms / batch=128 queries "
       f"(paper: 79.52 ms on A100 GPU)")

res = g.finish()
print(json.dumps({k: v for k, v in res.items() if k in ("module", "name", "pass", "duration_s")}, indent=2))
sys.exit(0 if res["pass"] else 1)
