"""M18 / exp2 -- retrieval accuracy of ppRAG on real MS MARCO data.

Paper reference: §V (the paper claims "high retrieval accuracy" but reports no
retrieval table/figure; this experiment is the supplementary accuracy study
defined in docs/MODULE_PLAN.md M18 and the Lead's exp2 design).

Design
------
* Database: data/ (m = 50,000 real MS MARCO passages, gtr-t5-base 768-d L2).
* Queries: the 1,000 eval queries of data/eval_qids.json (100 % qrels coverage).
* Reference: plaintext exact top-10 (PlaintextDB).
* Encrypted pipelines (all with local DecDB + exact-plaintext-distance re-ranking,
  k = 10):
    (a) CAPRISE, k' = k          (strict Algorithm 2 without DistanceDP)
    (b) CAPRISE, k' = 15, 20     (practical oversampling)
    (c) CAPRISE + DistanceDP(r = 0.02, 0.05), k' from Theorem 1
        (src/caprise/theorem.py:k_prime(k, m, n, r))
    (d) ADCPE (SAP baseline), k' = k, 20
    (e) beta sensitivity {0.05, 0.1, 0.2, 0.4} for CAPRISE (k' = k)
* Metrics:
    - Recall@10 relative to the plaintext top-10 (set overlap / 10);
    - qrels MRR@10 / Recall@10 / nDCG@10 of the encrypted top-10;
    - mean k'; per-query end-to-end latency; server search latency;
    - Phase-1 upload time / encryption throughput.

Run
---
    cd "code/CAPRISE复现"
    env -u ALL_PROXY .venv/bin/python experiments/exp2_retrieval.py            # full: 50k x 1000
    env -u ALL_PROXY .venv/bin/python experiments/exp2_retrieval.py --n-queries 50   # quick

Outputs: results/exp2_retrieval.json, results/exp2_retrieval.csv,
         figures/fig_retrieval.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caprise.aescipher import AESDocumentCipher  # noqa: E402
from caprise.caprise import ADCPE, KeyGen, noise_radius_db  # noqa: E402
from caprise.pprag import PPRAGClient, PPRAGServer  # noqa: E402
from caprise.server import PlaintextDB  # noqa: E402
from caprise.theorem import k_prime as theorem_k_prime  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

K = 10
S = 3
DIM = 768


# --------------------------------------------------------------------------- #
# data / metric helpers
# --------------------------------------------------------------------------- #
def load_data(n_queries: int, m: int | None):
    p_emb = np.load(DATA / "passage_emb.npy")
    p_ids = np.load(DATA / "passage_ids.npy")
    q_emb = np.load(DATA / "query_emb.npy")
    q_ids = np.load(DATA / "query_ids.npy")
    q_texts = [json.loads(line)["text"] for line in open(DATA / "queries.jsonl", encoding="utf-8")]
    texts: list[str] = []
    with open(DATA / "passages.jsonl", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    mm = len(p_ids) if m is None else min(m, len(p_ids))
    nq = min(n_queries, len(q_ids))
    rel_by_q: dict[int, set[int]] = {}
    for line in open(DATA / "qrels.tsv", encoding="utf-8"):
        if not line.strip():
            continue
        qid, _, pid, _ = line.split("\t")
        rel_by_q.setdefault(int(qid), set()).add(int(pid))
    return (p_emb[:mm].astype(np.float64), p_ids[:mm].astype(np.int64), texts[:mm],
            q_emb[:nq].astype(np.float64), q_ids[:nq].astype(np.int64), q_texts[:nq],
            rel_by_q, mm, nq)


def qrels_metrics(ids_per_query, qids, rel_by_q) -> dict:
    mrrs, recs, ndcgs = [], [], []
    for ids, qid in zip(ids_per_query, qids):
        rel = rel_by_q.get(int(qid))
        if not rel:
            continue
        rank = next((r for r, i in enumerate(ids, start=1) if int(i) in rel), None)
        mrrs.append(1.0 / rank if rank else 0.0)
        recs.append(1.0 if rank else 0.0)
        ndcgs.append(1.0 / math.log2(rank + 1) if rank else 0.0)
    n = max(1, len(mrrs))
    return {
        "n_queries_with_qrels": len(mrrs),
        "MRR@10": float(np.mean(mrrs)) if mrrs else 0.0,
        "Recall@10": float(np.mean(recs)) if recs else 0.0,
        "nDCG@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def recall_vs_plaintext(ids_per_query, plain_ids_per_query) -> float:
    return float(np.mean([
        len(set(int(x) for x in a) & set(int(x) for x in b)) / K
        for a, b in zip(ids_per_query, plain_ids_per_query)
    ]))


def latency_stats(times_ms: list[float]) -> dict:
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "total_s": float(arr.sum() / 1000.0),
        "n": int(arr.size),
    }


# --------------------------------------------------------------------------- #
# encrypted-pipeline runners
# --------------------------------------------------------------------------- #
def run_caprise_config(key, E, texts, pids, Q, k=K, dp_radius=0.0,
                       k_prime_override=None, server=None, client=None):
    """Run one CAPRISE retrieval configuration; returns ids/k'/latencies/upload stats."""
    if server is None:
        server = PPRAGServer()
        client = PPRAGClient(key)
        t0 = time.time()
        client.phase1_upload(server, texts=texts, embeddings=E, ids=pids.tolist())
        upload_s = time.time() - t0
    else:
        upload_s = 0.0

    # instrument server-side search latency
    search_times: list[float] = []
    orig_search = server.db.search

    def timed_search(q_prime, kk):
        t = time.perf_counter()
        res = orig_search(q_prime, kk)
        search_times.append((time.perf_counter() - t) * 1000.0)
        return res

    server.db.search = timed_search

    ids_per_q, kprimes, lat = [], [], []
    for q in Q:
        t = time.perf_counter()
        res = client.phase2_retrieve(server, q, k=k, dp_radius=dp_radius, k_prime=k_prime_override)
        lat.append((time.perf_counter() - t) * 1000.0)
        ids_per_q.append([int(x) for x in res.ids])
        kprimes.append(int(res.k_prime_requested))

    server.db.search = orig_search
    return {
        "ids": ids_per_q,
        "k_prime_mean": float(np.mean(kprimes)),
        "k_prime_set": sorted(set(kprimes)),
        "latency_ms": latency_stats(lat),
        "server_search_ms": latency_stats(search_times),
        "upload_s": upload_s,
        "enc_throughput_vec_s": (len(pids) / upload_s) if upload_s > 0 else None,
    }


def run_adcpe_config(key, E, texts, pids, Q, k=K, k_prime=None):
    """ADCPE / SAP baseline (same pipeline shape, uniform noise radius s*beta/4)."""
    m = E.shape[0]
    aes = AESDocumentCipher()
    t0 = time.time()
    E_prime = np.empty_like(E)
    nonces: list[bytes] = []
    for i in range(m):
        E_prime[i], r = ADCPE.Enc(key, E[i])
        nonces.append(r)
    enc_s = time.time() - t0
    server = PPRAGServer()
    t0 = time.time()
    server.upload(E_prime, nonces, [aes.encrypt(t) for t in texts], ids=pids.tolist())
    upload_s = enc_s + (time.time() - t0)

    kp = k if k_prime is None else int(k_prime)
    ids_per_q, lat, search_times = [], [], []
    for q in Q:
        t = time.perf_counter()
        q_prime, _ = ADCPE.EncQ(key, q)
        found = server.db.search(q_prime, kp)
        search_times.append(0.0)  # included in the end-to-end latency below
        E_hat = np.asarray([ADCPE.Dec(key, server.db.E_prime[i], server.db.nonces[i])
                            for i in found.indices])
        d = np.linalg.norm(E_hat - q[None, :], axis=1)
        order = np.argsort(d, kind="stable")[:k]
        ids_per_q.append([int(found.ids[i]) for i in order])
        lat.append((time.perf_counter() - t) * 1000.0)

    return {
        "ids": ids_per_q,
        "k_prime_mean": float(kp),
        "k_prime_set": [kp],
        "latency_ms": latency_stats(lat),
        "server_search_ms": {"note": "not separately instrumented for ADCPE"},
        "upload_s": upload_s,
        "enc_throughput_vec_s": m / enc_s,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-queries", type=int, default=1000)
    ap.add_argument("--m", type=int, default=None, help="corpus subset size (default: all 50000)")
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    t_start = time.time()

    E, pids, texts, Q, qids, qtexts, rel_by_q, m, nq = load_data(args.n_queries, args.m)
    print(f"[exp2] m={m}, n_queries={nq}, d={DIM}, k={K}, s={S}", flush=True)

    # ---------------- plaintext baseline ---------------- #
    plain_db = PlaintextDB(E, ids=pids.tolist())
    plain_ids: list[list[int]] = []
    plain_lat: list[float] = []
    for q in Q:
        t = time.perf_counter()
        res = plain_db.search(q, K)
        plain_lat.append((time.perf_counter() - t) * 1000.0)
        plain_ids.append([int(x) for x in res.ids])
    plain_metrics = qrels_metrics(plain_ids, qids, rel_by_q)
    plaintext_record = {
        "config": "plaintext",
        "method": "plaintext",
        "beta": None,
        "dp_radius": 0.0,
        "k_prime_setting": f"k'=k={K}",
        "k_prime_mean": float(K),
        "recall@10_vs_plaintext": 1.0,
        "qrels": plain_metrics,
        "latency_ms": latency_stats(plain_lat),
        "server_search_ms": latency_stats(plain_lat),
        "upload_s": 0.0,
        "enc_throughput_vec_s": None,
    }
    print(f"[exp2] plaintext: qrels MRR@10={plain_metrics['MRR@10']:.4f}, "
          f"Recall@10={plain_metrics['Recall@10']:.4f}, {np.mean(plain_lat):.3f} ms/query", flush=True)

    configs: list[dict] = [plaintext_record]
    rng = np.random.default_rng(args.seed)

    def add_config(name, method, beta, dp_radius, k_prime_setting, result, extra_note=""):
        rec = {
            "config": name,
            "method": method,
            "beta": beta,
            "dp_radius": dp_radius,
            "k_prime_setting": k_prime_setting,
            "k_prime_mean": result["k_prime_mean"],
            "k_prime_set": result["k_prime_set"],
            "recall@10_vs_plaintext": recall_vs_plaintext(result["ids"], plain_ids),
            "qrels": qrels_metrics(result["ids"], qids, rel_by_q),
            "latency_ms": result["latency_ms"],
            "server_search_ms": result["server_search_ms"],
            "upload_s": result["upload_s"],
            "enc_throughput_vec_s": result["enc_throughput_vec_s"],
        }
        if extra_note:
            rec["note"] = extra_note
        configs.append(rec)
        print(f"[exp2] {name:<28} recall@10(plain)={rec['recall@10_vs_plaintext']:.4f}  "
              f"qrels MRR@10={rec['qrels']['MRR@10']:.4f}  "
              f"k'={rec['k_prime_mean']:.1f}  {rec['latency_ms']['mean']:.2f} ms/q", flush=True)
        return rec

    # ---------------- CAPRISE, beta = args.beta (main block) ---------------- #
    key = KeyGen(dim=DIM, beta=args.beta, s=S, rng=rng)
    t0 = time.time()
    server = PPRAGServer()
    client = PPRAGClient(key)
    client.phase1_upload(server, texts=texts, embeddings=E, ids=pids.tolist())
    upload_main_s = time.time() - t0
    print(f"[exp2] CAPRISE Phase-1 upload (m={m}): {upload_main_s:.2f}s "
          f"({m/upload_main_s:.0f} vec/s incl. AES)", flush=True)

    def caprise_cfg(name, dp_radius, k_prime_override, setting, note=""):
        res = run_caprise_config(key, E, texts, pids, Q, k=K, dp_radius=dp_radius,
                                 k_prime_override=k_prime_override, server=server, client=client)
        res["upload_s"] = upload_main_s if not configs[1:] else 0.0
        res["enc_throughput_vec_s"] = (m / upload_main_s) if not configs[1:] else None
        return add_config(name, "CAPRISE", args.beta, dp_radius, setting, res, note)

    caprise_cfg("caprise_kp=k", 0.0, None, f"k'=k={K}", "strict Algorithm 2, no DP")
    caprise_cfg("caprise_kp=15", 0.0, 15, "k'=15", "oversampling")
    caprise_cfg("caprise_kp=2k", 0.0, 2 * K, f"k'=2k={2*K}", "oversampling")

    r02 = 0.02
    r05 = 0.05
    kp_r02 = int(math.ceil(theorem_k_prime(K, m, DIM, r02)))
    kp_r05 = int(math.ceil(theorem_k_prime(K, m, DIM, r05)))
    caprise_cfg(f"caprise_dp_r{r02}_theorem-kp", r02, None, "k'=Theorem1",
                f"DistanceDP r={r02}; k'=ceil(Theorem1)={kp_r02}")
    caprise_cfg(f"caprise_dp_r{r05}_theorem-kp", r05, None, "k'=Theorem1",
                f"DistanceDP r={r05}; k'=ceil(Theorem1)={kp_r05}")

    # ---------------- ADCPE baseline ---------------- #
    key_ad = KeyGen(dim=DIM, beta=args.beta, s=S, rng=np.random.default_rng(args.seed + 1))
    ad_k = run_adcpe_config(key_ad, E, texts, pids, Q, k=K, k_prime=K)
    add_config("adcpe_kp=k", "ADCPE", args.beta, 0.0, f"k'=k={K}", ad_k,
               "SAP baseline (uniform noise radius s*beta/4)")
    del server  # free ciphertext memory before the beta sweep
    ad_2k = run_adcpe_config(key_ad, E, texts, pids, Q, k=K, k_prime=2 * K)
    add_config("adcpe_kp=2k", "ADCPE", args.beta, 0.0, f"k'=2k={2*K}", ad_2k)

    # ---------------- beta sensitivity (CAPRISE, k'=k) ---------------- #
    beta_sweep = {}
    for beta in (0.05, 0.1, 0.2, 0.4):
        if abs(beta - args.beta) < 1e-12:
            rec = next(c for c in configs if c["config"] == "caprise_kp=k")
            beta_sweep[beta] = rec
            continue
        kb = KeyGen(dim=DIM, beta=beta, s=S, rng=np.random.default_rng(args.seed + int(beta * 1000)))
        srv = PPRAGServer()
        cli = PPRAGClient(kb)
        t0 = time.time()
        cli.phase1_upload(srv, texts=texts, embeddings=E, ids=pids.tolist())
        up_s = time.time() - t0
        res = run_caprise_config(kb, E, texts, pids, Q, k=K, dp_radius=0.0,
                                 server=srv, client=cli)
        res["upload_s"] = up_s
        res["enc_throughput_vec_s"] = m / up_s
        rec = add_config(f"caprise_beta{beta}_kp=k", "CAPRISE", beta, 0.0, f"k'=k={K}", res,
                         "beta sensitivity (smaller beta -> less noise -> more accurate)")
        beta_sweep[beta] = rec

    # ---------------- save JSON ---------------- #
    meta = {
        "experiment": "exp2_retrieval (M18): retrieval accuracy of ppRAG / CAPRISE on MS MARCO",
        "paper_ref": "§V Experimental evaluation (accuracy claim; paper gives no retrieval table)",
        "dataset": "MS MARCO passage dev small + gtr-t5-base (data/, m=%d)" % m,
        "m": m,
        "n_queries": nq,
        "k": K,
        "dim": DIM,
        "s": S,
        "beta_default": args.beta,
        "seed": args.seed,
        "rerank": "local DecDB + exact plaintext L2 distance to the original query (SPEC.md §3)",
        "metrics": ["Recall@10 vs plaintext top-10", "qrels MRR@10/Recall@10/nDCG@10",
                    "mean k'", "per-query end-to-end latency", "Phase-1 upload throughput"],
        "theorem_k_prime": {"r=0.02": kp_r02, "r=0.05": kp_r05,
                            "formula": "ceil(k_prime(k, m, n, r)) with delta_alpha_ratio=1.0"},
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": __import__("os").cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "device": "CPU (Apple M5, no CUDA); embeddings precomputed on MPS fp16",
        },
        "note_crypto_nondeterminism": "fresh 128-bit nonces per encryption (os.urandom); "
                                      "numbers vary by ~1e-3 across runs; key generation seeded",
        "total_wall_s": round(time.time() - t_start, 1),
        "command": " ".join([sys.executable] + sys.argv),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out = {"meta": meta, "plaintext": plaintext_record, "configs": configs,
           "beta_sensitivity": sorted(beta_sweep.items())}
    tag = args.out_tag
    json_path = RESULTS / f"exp2_retrieval{tag}.json"
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- save CSV ---------------- #
    csv_path = RESULTS / f"exp2_retrieval{tag}.csv"
    fields = ["config", "method", "beta", "dp_radius", "k_prime_setting", "k_prime_mean",
              "recall@10_vs_plaintext", "qrels_MRR@10", "qrels_Recall@10", "qrels_nDCG@10",
              "latency_ms_mean", "latency_ms_median", "latency_ms_p95", "server_search_ms_mean",
              "upload_s", "enc_throughput_vec_s"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in configs:
            w.writerow({
                "config": c["config"], "method": c["method"], "beta": c["beta"],
                "dp_radius": c["dp_radius"], "k_prime_setting": c["k_prime_setting"],
                "k_prime_mean": round(c["k_prime_mean"], 3),
                "recall@10_vs_plaintext": round(c["recall@10_vs_plaintext"], 6),
                "qrels_MRR@10": round(c["qrels"]["MRR@10"], 6),
                "qrels_Recall@10": round(c["qrels"]["Recall@10"], 6),
                "qrels_nDCG@10": round(c["qrels"]["nDCG@10"], 6),
                "latency_ms_mean": round(c["latency_ms"]["mean"], 4),
                "latency_ms_median": round(c["latency_ms"]["median"], 4),
                "latency_ms_p95": round(c["latency_ms"]["p95"], 4),
                "server_search_ms_mean": round(c["server_search_ms"].get("mean", float("nan")), 4)
                if "mean" in c["server_search_ms"] else "",
                "upload_s": round(c["upload_s"], 4) if c["upload_s"] else "",
                "enc_throughput_vec_s": round(c["enc_throughput_vec_s"], 1) if c["enc_throughput_vec_s"] else "",
            })

    # ---------------- figure ---------------- #
    make_figure(configs, beta_sweep, FIGURES / f"fig_retrieval{tag}.png")

    print(f"[exp2] wrote {json_path.relative_to(ROOT)}, {csv_path.relative_to(ROOT)}, "
          f"figures/fig_retrieval{tag}.png  ({time.time()-t_start:.0f}s total)", flush=True)
    return 0


def make_figure(configs: list[dict], beta_sweep: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    enc = [c for c in configs if c["method"] != "plaintext"]
    labels = [c["config"].replace("caprise_", "CAPRISE\n").replace("adcpe_", "ADCPE\n") for c in enc]
    x = np.arange(len(enc))

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

    # (a) Recall@10 vs plaintext
    ax = axes[0]
    rec = [c["recall@10_vs_plaintext"] for c in enc]
    bars = ax.bar(x, rec, color="#4C72B0")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 1.02)
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Recall@10 vs plaintext top-10")
    ax.set_title("(a) encrypted retrieval fidelity")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, rec):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.01, f"{v:.3f}", ha="center", fontsize=6)

    # (b) qrels MRR@10
    ax = axes[1]
    mrr = [c["qrels"]["MRR@10"] for c in enc]
    pmrr = next(c["qrels"]["MRR@10"] for c in configs if c["method"] == "plaintext")
    bars = ax.bar(x, mrr, color="#DD8452")
    ax.axhline(pmrr, color="black", ls="--", lw=1, label=f"plaintext {pmrr:.3f}")
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("qrels MRR@10")
    ax.set_title("(b) end-to-end retrieval quality (qrels)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, mrr):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=6)

    # (c) beta sensitivity
    ax = axes[2]
    betas = sorted(beta_sweep)
    rv = [beta_sweep[b]["recall@10_vs_plaintext"] for b in betas]
    qm = [beta_sweep[b]["qrels"]["MRR@10"] for b in betas]
    ax.plot(betas, rv, "o-", label="Recall@10 vs plaintext")
    ax.plot(betas, qm, "s-", label="qrels MRR@10")
    ax.set_xlabel("beta (CAPRISE distance parameter)")
    ax.set_ylabel("metric")
    ax.set_title("(c) beta sensitivity (CAPRISE, k'=k)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
