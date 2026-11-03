#!/usr/bin/env python
"""M0 data module: MS MARCO passage dev data + gtr-t5-base embeddings.

Reproduction project: CAPRISE / ppRAG (Efficient Privacy-Preserving RAG with
Distance-Preserving Encryption, arXiv:2601.12331).

Pipeline (idempotent / cacheable, safe to re-run):
  1. raw: qrels.dev.small.tsv, queries.dev.small.tsv (official MS MARCO dev
     small), collection.tsv from collectionandqueries.tar.gz (Azure CDN, fetched
     with parallel HTTP range requests and resumable part files).
  2. corpus: take the first N_EVAL queries of qrels.dev.small (default 1000),
     put ALL of their relevant passages into the corpus, fill the remaining
     slots with a deterministic uniform sample (seeded blake2b hash, heap
     selection) of the full MS MARCO passage collection -> m passages.
     Corpus row order = [relevant passages sorted by pid] + [filler sorted by
     pid], so every prefix of the corpus contains all relevant passages
     (handy for m-scaling experiments downstream).
  3. embed passages & queries with sentence-transformers/gtr-t5-base
     (L2-normalised, float32).
  4. sanity: plaintext top-10 retrieval on the eval queries -> MRR@10 /
     Recall@10 / Recall@100 / nDCG@10 (results/data_sanity.json).
  5. timing: gtr-t5-base query embedding latency, batch=128, CPU
     (results/embed_timing.json, for the Phase-1 overhead comparison).
  6. data/meta.json with counts / model / timings / hardware.

Usage
-----
    cd "code/CAPRISE复现"
    env -u ALL_PROXY HF_HOME="$PWD/.cache/huggingface" \
        .venv/bin/python scripts/prepare_data.py --m 100000

Outputs (all under the project root):
    data/passage_emb.npy   (m, 768) float32, L2-normalised
    data/passage_ids.npy   (m,)     int64
    data/passages.jsonl    m lines  {"pid": int, "text": str}
    data/query_emb.npy     (nq, 768) float32, L2-normalised
    data/query_ids.npy     (nq,)    int64
    data/queries.jsonl     nq lines {"qid": int, "text": str}
    data/qrels.tsv         qid \t 0 \t pid \t 1  (only pids inside the corpus)
    data/eval_qids.json    list of eval qids with guaranteed qrels coverage
    data/meta.json
    results/data_sanity.json, results/embed_timing.json
    data/raw/              raw downloads / caches (tar, extracted tsv, model)
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# IMPORTANT: the environment exports ALL_PROXY=socks5://...; huggingface_hub /
# httpx abort with "SOCKS proxy ... socksio not installed".  Drop only that
# variable (HTTP(S)_PROXY is fine for both httpx and curl).  Must happen before
# transformers / sentence-transformers are imported.
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
RESULTS = ROOT / "results"

TAR_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"
TAR_SIZE = 1_057_717_952  # true Content-Length of collectionandqueries.tar.gz (verified by gzip -t)
TAR_NAME = "collectionandqueries.tar.gz"
QRELS_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv"
BM25_TAR_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.dev.tar.gz"
BM25_TAR_SIZE = 687_414_398  # top1000.dev.tar.gz: BM25 top-1000 candidates for passage-dev queries
BM25_MEMBER = "top1000.dev"
MODEL_NAME = "sentence-transformers/gtr-t5-base"
LOCAL_MODEL_DIR = RAW / "gtr-t5-base"
MODEL_FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "1_Pooling/config.json",
    "2_Dense/config.json",
    "2_Dense/model.safetensors",
    "model.safetensors",
]
N_COLLECTION_LINES = 8_841_823  # official MS MARCO passage collection size


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def log(*a):
    print(f"[prepare_data {time.strftime('%H:%M:%S')}]", *a, flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, check=False, **kw)


def http_download(url: str, dest: Path, expect_size: int | None = None, retries: int = 4) -> None:
    """Small-file download with curl (respects HTTP(S)_PROXY from env)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        r = run(
            [
                "curl", "-sSL", "--max-time", "120", "--retry", "3", "--retry-delay", "2",
                "-o", str(tmp), url,
            ]
        )
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            if expect_size is None or tmp.stat().st_size == expect_size:
                tmp.replace(dest)
                return
        log(f"  download attempt {attempt} failed for {url}")
        time.sleep(3 * attempt)
    raise RuntimeError(f"could not download {url}")


def download_parallel(url: str, dest: Path, size: int, n_parts: int = 16, workers: int = 8) -> None:
    """Resumable parallel HTTP-range download (parts kept under data/raw/tarparts).

    The Azure CDN throttles a single connection (~0.4-0.8 MB/s here) but a
    handful of ranged connections give ~5 MB/s in aggregate.
    """
    parts_dir = RAW / "tarparts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    chunk = (size + n_parts - 1) // n_parts
    jobs = []
    for i in range(n_parts):
        s = i * chunk
        e = min(s + chunk - 1, size - 1)
        if s <= e:
            jobs.append((i, s, e))

    def fetch(job):
        i, s, e = job
        out = parts_dir / f"part_{i:03d}"
        want = e - s + 1
        if out.exists() and out.stat().st_size == want:
            return
        for attempt in range(1, 7):
            run(
                [
                    "curl", "-sSL", "--max-time", "1800", "--retry", "3", "--retry-delay", "2",
                    "-r", f"{s}-{e}", "-o", str(out), url,
                ]
            )
            if out.exists() and out.stat().st_size == want:
                return
            log(f"  part {i} attempt {attempt} incomplete, retrying")
            time.sleep(2 * attempt)
        raise RuntimeError(f"part {i} of {url} failed")

    log(f"downloading {url} ({size/1e9:.3f} GB) in {len(jobs)} ranges x {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fetch, jobs))

    # assemble + verify
    tmp = dest.with_suffix(dest.suffix + ".assembling")
    with open(tmp, "wb") as fo:
        for i, _, _ in jobs:
            with open(parts_dir / f"part_{i:03d}", "rb") as fi:
                shutil.copyfileobj(fi, fo, length=8 << 20)
    if tmp.stat().st_size != size:
        tmp.unlink()
        raise RuntimeError(f"assembled size mismatch for {dest}")
    tmp.replace(dest)
    log(f"  assembled {dest} ({dest.stat().st_size} bytes); gzip integrity check...")
    r = run(["gzip", "-t", str(dest)])
    if r.returncode != 0:
        dest.unlink()
        raise RuntimeError("gzip integrity check failed")
    shutil.rmtree(parts_dir, ignore_errors=True)
    log("  gzip OK")


def extract_tar_members(tar_path: Path, names: list[str], outdir: Path) -> None:
    """Extract selected small members from the (streaming) tar.gz archive."""
    outdir.mkdir(parents=True, exist_ok=True)
    wanted = set(names)
    got = set()
    with tarfile.open(tar_path, "r|gz") as tf:
        for mem in tf:
            base = os.path.basename(mem.name)
            if base in wanted and base not in got:
                f = tf.extractfile(mem)
                assert f is not None
                (outdir / base).write_bytes(f.read())
                got.add(base)
                log(f"  extracted {base} ({ (outdir/base).stat().st_size } bytes)")
    if got != wanted:
        raise RuntimeError(f"missing tar members: {wanted - got}")


def iter_collection_lines(tar_path: Path):
    """Stream collection.tsv out of the tar.gz (no 3.2 GB file on disk)."""
    with tarfile.open(tar_path, "r|gz") as tf:
        for mem in tf:
            if os.path.basename(mem.name) == "collection.tsv":
                f = tf.extractfile(mem)
                assert f is not None
                yield from f
                return
    raise RuntimeError("collection.tsv not found in archive")


def read_qrels(path: Path):
    """qrels.dev.small.tsv -> (ordered unique qids, list of (qid, pid))."""
    qids: list[int] = []
    seen: set[int] = set()
    pairs: list[tuple[int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            qid, pid = int(parts[0]), int(parts[2])
            pairs.append((qid, pid))
            if qid not in seen:
                seen.add(qid)
                qids.append(qid)
    return qids, pairs


# --------------------------------------------------------------------------- #
# step 1: raw files
# --------------------------------------------------------------------------- #
def ensure_raw(args) -> tuple[Path, Path, Path]:
    RAW.mkdir(parents=True, exist_ok=True)
    tar_path = RAW / TAR_NAME

    # --- qrels (small, direct download) ---
    qrels_path = DATA / "qrels.dev.small.tsv"
    if not (qrels_path.exists() and qrels_path.stat().st_size > 50_000):
        log("downloading qrels.dev.small.tsv")
        http_download(QRELS_URL, qrels_path)
    # sanity: 7437 qrels pairs / 6980 qids
    qids, pairs = read_qrels(qrels_path)
    log(f"qrels: {len(pairs)} pairs, {len(qids)} qids")
    if len(qids) < 6000:
        raise RuntimeError("qrels file looks corrupt")

    # --- queries.dev.small.tsv: only ships inside the tar.gz ---
    queries_path = DATA / "queries.dev.small.tsv"
    need_queries = True
    if queries_path.exists() and queries_path.stat().st_size > 100_000:
        n = sum(1 for _ in open(queries_path, "rb"))
        if n >= 6000:
            need_queries = False
            log(f"queries present: {n} lines")
    # --- collection archive ---
    need_tar = not (tar_path.exists() and tar_path.stat().st_size == TAR_SIZE)
    if need_tar:
        if tar_path.exists():
            log(f"removing incomplete archive {tar_path} ({tar_path.stat().st_size} bytes)")
            tar_path.unlink()
        download_parallel(TAR_URL, tar_path, TAR_SIZE, n_parts=args.tar_parts, workers=args.tar_workers)
    if need_queries or need_tar:
        log("extracting queries.dev.small.tsv (+ qrels for verification) from archive")
        extract_tar_members(tar_path, ["queries.dev.small.tsv", "qrels.dev.small.tsv"], RAW)
        # official archive copy wins over earlier direct download only if valid
        tq = RAW / "queries.dev.small.tsv"
        if not (queries_path.exists() and queries_path.stat().st_size > 100_000):
            shutil.copyfile(tq, queries_path)
        tqr = RAW / "qrels.dev.small.tsv"
        if tqr.exists() and tqr.stat().st_size > 50_000 and tqr.stat().st_size != qrels_path.stat().st_size:
            log("note: tar qrels differs in size from direct qrels; keeping direct download")
    n = sum(1 for _ in open(queries_path, "rb"))
    log(f"queries.dev.small.tsv: {n} lines")
    if n < 6000:
        raise RuntimeError("queries file corrupt")
    return qrels_path, queries_path, tar_path


# --------------------------------------------------------------------------- #
# step 2: corpus selection (qrels-forced + BM25 hard negatives + random filler)
# --------------------------------------------------------------------------- #
def ensure_bm25_negatives(
    qrels_path: Path,
    queries_path: Path,
    n_eval: int,
    n_per_query: int,
    seed: int,
    tar_parts: int,
    tar_workers: int,
) -> dict[int, list[int]]:
    """Approximate BM25 top-N candidates per eval query from top1000.dev.

    ``top1000.dev.tar.gz`` ships one line per (query, candidate) pair::

        qid \t pid \t query \t passage_text

    ordered by pid, with **no rank column**.  We therefore recover a BM25-like
    ranking by scoring every candidate passage of every eval query with the
    fraction of query tokens it contains, and keep the top ``n_per_query``
    pids per query.  This yields real (lexically matched) hard negatives.
    """
    qids, _ = read_qrels(qrels_path)
    eval_qids = qids[:n_eval]
    cache = RAW / f"bm25_negs_q{n_eval}_n{n_per_query}_s{seed}.json"
    if cache.exists():
        log(f"using cached BM25 hard-negative selection {cache.name}")
        return {int(k): v for k, v in json.load(open(cache)).items()}

    tar = RAW / "top1000.dev.tar.gz"
    if not (tar.exists() and tar.stat().st_size == BM25_TAR_SIZE):
        download_parallel(BM25_TAR_URL, tar, BM25_TAR_SIZE, n_parts=tar_parts, workers=tar_workers)

    qtext: dict[int, str] = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            a, b = line.rstrip("\n").split("\t", 1)
            qtext[int(a)] = b
    eval_set = set(int(q) for q in eval_qids)
    qsets = {q: set(qtext[q].lower().split()) for q in eval_qids}
    keep = max(n_per_query * 3, n_per_query)
    heaps: dict[int, list] = {q: [] for q in eval_qids}  # (score, -pid) min-heaps
    n_lines = 0
    t0 = time.time()
    with tarfile.open(tar, "r|gz") as tf:
        for mem in tf:
            if os.path.basename(mem.name) != BM25_MEMBER:
                continue
            f = tf.extractfile(mem)
            assert f is not None
            for line in f:
                n_lines += 1
                parts = line.rstrip(b"\n").split(b"\t", 3)
                if len(parts) != 4:
                    continue
                qid = int(parts[0])
                if qid not in eval_set:
                    continue
                pid = int(parts[1])
                toks = set(parts[3].decode("utf-8", "replace").lower().split())
                qs = qsets[qid]
                score = len(qs & toks) / max(1, len(qs))
                h = heaps[qid]
                if len(h) < keep:
                    heapq.heappush(h, (score, -pid))
                elif (score, -pid) > h[0]:
                    heapq.heapreplace(h, (score, -pid))
            break  # the member we want is the first entry of the archive
    log(f"scanned {n_lines} top1000.dev lines in {time.time()-t0:.1f}s")
    selected: dict[int, list[int]] = {}
    for q in eval_qids:
        ranked = sorted(heaps[q], key=lambda x: (-x[0], -x[1]))[:n_per_query]
        selected[q] = [int(-negpid) for score, negpid in ranked if score > 0]
    cache.write_text(json.dumps({str(k): v for k, v in selected.items()}))
    n_sel = sum(len(v) for v in selected.values())
    log(f"BM25 hard negatives: {n_sel} pids for {len(selected)} eval queries (top {n_per_query}/query)")
    return selected


def select_corpus(
    qrels_path: Path,
    tar_path: Path,
    m: int,
    n_eval: int,
    seed: int,
    hard_negs: dict[int, list[int]] | None = None,
    n_hard_per_query: int = 0,
):
    """Return (corpus list[(pid, text, source)], eval_qids, qrels pairs kept, stats).

    Corpus composition, row order:
      [all qrels-relevant passages of the eval queries, sorted by pid]
      [BM25 hard negatives for the eval queries, sorted by pid]        (optional)
      [uniform random filler, sorted by pid]
    -> every prefix contains all relevant passages (m-scaling experiments).
    """
    qids, pairs = read_qrels(qrels_path)
    eval_qids = qids[:n_eval]
    eval_set = set(eval_qids)
    rel_pids = {pid for qid, pid in pairs if qid in eval_set}

    hard_pids_by_q: dict[int, list[int]] = {}
    if hard_negs and n_hard_per_query > 0:
        for q in eval_qids:
            cand = [pid for pid in hard_negs.get(q, []) if pid not in rel_pids]
            if cand:
                hard_pids_by_q[q] = cand[:n_hard_per_query]
    hard_pids = {pid for v in hard_pids_by_q.values() for pid in v}
    needed = rel_pids | hard_pids
    log(f"eval queries: {len(eval_qids)}; relevant pids: {len(rel_pids)}; "
        f"BM25 hard negatives: {len(hard_pids)}")

    cache = RAW / f"corpus_selected_m{m}_q{n_eval}_s{seed}_h{n_hard_per_query}x{len(hard_pids)}.jsonl"
    if cache.exists():
        log(f"using cached corpus selection {cache.name}")
        corpus = []
        with open(cache, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                corpus.append((int(rec["pid"]), rec["text"], rec.get("source", "random_fill")))
        if len(corpus) == m:
            return corpus, eval_qids, pairs, {"cached": True}

    fill_k = m - len(needed)
    if fill_k < 0:
        raise RuntimeError(f"m={m} smaller than #forced passages={len(needed)}")
    seed_b = str(seed).encode()
    heap: list[tuple[int, int, bytes]] = []  # (-hash, pid, text bytes)
    needed_text: dict[int, str] = {}
    n_lines = 0
    t0 = time.time()
    for line in iter_collection_lines(tar_path):
        n_lines += 1
        pid_b, sep, text_b = line.partition(b"\t")
        if not sep:
            continue
        pid = int(pid_b)
        if pid in needed:
            needed_text[pid] = text_b.decode("utf-8", "replace")
        elif fill_k > 0:
            h = int.from_bytes(hashlib.blake2b(seed_b + pid_b, digest_size=8).digest(), "big")
            if len(heap) < fill_k:
                heapq.heappush(heap, (-h, pid, text_b))
            elif h < -heap[0][0]:
                heapq.heapreplace(heap, (-h, pid, text_b))
        if n_lines % 2_000_000 == 0:
            log(f"  scanned {n_lines/1e6:.1f}M collection rows ({time.time()-t0:.0f}s)")
    log(f"scanned {n_lines} collection rows in {time.time()-t0:.1f}s")
    if n_lines != N_COLLECTION_LINES:
        log(f"WARNING: collection has {n_lines} lines (expected {N_COLLECTION_LINES})")
    missing = needed - set(needed_text)
    if missing:
        raise RuntimeError(f"{len(missing)} forced pids missing from collection: {sorted(missing)[:5]}")

    corpus: list[tuple[int, str, str]] = []
    corpus += [(pid, needed_text[pid], "qrels") for pid in sorted(rel_pids)]
    corpus += [(pid, needed_text[pid], "bm25_hard_neg") for pid in sorted(hard_pids)]
    fill = sorted(((pid, text) for _, pid, text in heap), key=lambda x: x[0])
    corpus += [(pid, text.decode("utf-8", "replace"), "random_fill") for pid, text in fill]
    assert len(corpus) == m, (len(corpus), m)

    with open(cache, "w", encoding="utf-8") as f:
        for pid, text, source in corpus:
            f.write(json.dumps({"pid": pid, "text": text, "source": source}, ensure_ascii=False) + "\n")
    log(f"corpus selected: {len(corpus)} passages -> {cache.name}")
    return corpus, eval_qids, pairs, {"cached": False, "collection_lines": n_lines,
                                      "n_hard_pids": len(hard_pids), "n_rel_pids": len(rel_pids)}


# --------------------------------------------------------------------------- #
# step 3: embeddings
# --------------------------------------------------------------------------- #
def _pick_device_dtype(args):
    import torch

    dev = args.device
    if dev == "auto":
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if args.dtype == "auto":
        dt = torch.float16 if dev == "mps" else torch.float32
    else:
        dt = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    return dev, dt


def load_model(args):
    """Load gtr-t5-base.  NOTE: the published model.safetensors is float16 and
    sentence-transformers 6.2 keeps fp16 on CPU, which is ~10x slower than
    fp32 there; we force a dtype accordingly: fp16 on Apple MPS (validated
    against CPU fp32: min cosine 0.9998) and fp32 on CPU."""
    import torch

    torch.set_num_threads(args.num_threads)
    from sentence_transformers import SentenceTransformer

    dev, dt = _pick_device_dtype(args)
    log(f"loading gtr-t5-base on {dev} (dtype={dt})")
    if LOCAL_MODEL_DIR.exists() and (LOCAL_MODEL_DIR / "model.safetensors").exists():
        model = SentenceTransformer(str(LOCAL_MODEL_DIR), device=dev, model_kwargs={"dtype": dt})
    else:
        model = SentenceTransformer(MODEL_NAME, device=dev, model_kwargs={"dtype": dt})
    return model


def encode_all(model, sentences: list[str], batch_size: int):
    import numpy as np

    t0 = time.time()
    emb = model.encode(
        sentences,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    dt = time.time() - t0
    emb = np.asarray(emb, dtype=np.float32)
    return emb, dt


# --------------------------------------------------------------------------- #
# step 4: plaintext sanity retrieval
# --------------------------------------------------------------------------- #
def retrieval_metrics(q_emb, p_emb, qids, qrels, k_values=(10, 100)):
    """Plaintext exact top-k search over the corpus; returns dict of IR metrics.

    `qrels` is a list of (qid, corpus_row_index) pairs.
    """
    import numpy as np

    nq = q_emb.shape[0]
    rel_by_q = {}
    for qid, row in qrels:
        rel_by_q.setdefault(qid, set()).add(row)
    rr10 = 0.0
    rec10 = 0.0
    rec100 = 0.0
    ndcg10 = 0.0
    n_eval = 0
    chunk = 100
    for s in range(0, nq, chunk):
        e = min(s + chunk, nq)
        sims = q_emb[s:e] @ p_emb.T  # (b, m)
        # top-100
        k = min(max(k_values), sims.shape[1])
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        for row in range(e - s):
            qid = int(qids[s + row])
            rel = rel_by_q.get(qid)
            if not rel:
                continue
            n_eval += 1
            cols = part[row]
            order = cols[np.argsort(-sims[row, cols])]
            # map to pids outside loop (vectorized would need array)
            ranks = None
            for r, idx in enumerate(order, start=1):
                if int(idx) in rel:
                    ranks = r
                    break
            if ranks is not None:
                rr10 += 1.0 / ranks if ranks <= 10 else 0.0
                if ranks <= 10:
                    rec10 += 1.0
                    ndcg10 += 1.0 / np.log2(ranks + 1)
                if ranks <= 100:
                    rec100 += 1.0
    return {
        "n_queries_evaluated": n_eval,
        "MRR@10": rr10 / n_eval,
        "Recall@10": rec10 / n_eval,
        "Recall@100": rec100 / n_eval,
        "nDCG@10": ndcg10 / n_eval,
    }


def sanity_examples(model, p_emb, qids, qtexts, pids, qrels, n_examples=5, topk=10):
    import numpy as np

    rel_by_q = {}
    for qid, row in qrels:
        rel_by_q.setdefault(qid, set()).add(row)
    # encode the first n_examples query texts (prefer ones with qrels coverage)
    sel = [i for i, q in enumerate(qids) if q in rel_by_q][:n_examples]
    texts = [qtexts[i] for i in sel]
    ex_emb = np.asarray(
        model.encode(texts, batch_size=8, normalize_embeddings=True, convert_to_numpy=True),
        dtype=np.float32,
    )
    sims = ex_emb @ p_emb.T
    examples = []
    for row, i in enumerate(sel):
        qid = int(qids[i])
        order = np.argsort(-sims[row])[:topk]
        top = [
            {"rank": r + 1, "pid": int(pids[idx]), "relevant": int(idx) in rel_by_q.get(qid, set())}
            for r, idx in enumerate(order)
        ]
        examples.append({"qid": qid, "query": qtexts[i], "top10": top})
    return examples


# --------------------------------------------------------------------------- #
# step 5: embedding latency benchmark
# --------------------------------------------------------------------------- #
def embed_timing(model, queries: list[str], batch_size: int, model_path,
                 runs: int = 3, batch_sizes: tuple[int, ...] | None = None):
    """Query embedding latency (input for the Phase-1 overhead comparison).

    Paper reference: generating embeddings for a batch of 128 queries takes
    ~79.52 ms on an NVIDIA A100 GPU.  We report the measured Apple-MPS (fp16)
    numbers for batch=128 (and smaller batches for reference), plus a CPU/fp32
    reference measured on the same machine.
    """
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    dev = str(model.device)
    wanted = [batch_size] if batch_sizes is None else list(dict.fromkeys([batch_size] + list(batch_sizes)))

    def _time(m, bs):
        n = min(bs, len(queries))
        batch = queries[:n]
        m.encode(batch, batch_size=n, normalize_embeddings=True, convert_to_numpy=True)  # warmup
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            m.encode(batch, batch_size=n, normalize_embeddings=True, convert_to_numpy=True)
            ts.append((time.perf_counter() - t0) * 1000.0)
        return ts, n

    configs = {}
    for bs in wanted:
        ts, n = _time(model, bs)
        configs[f"batch_{n}"] = {
            "n_queries": n,
            "ms_runs": [round(t, 2) for t in ts],
            "ms_mean": round(float(np.mean(ts)), 2),
            "ms_median": round(float(np.median(ts)), 2),
            "per_query_ms": round(float(np.mean(ts)) / n, 3),
        }
    primary_n = min(batch_size, len(queries))
    primary = configs[f"batch_{primary_n}"]

    single = []
    for _ in range(5):
        t0 = time.perf_counter()
        model.encode([queries[0]], batch_size=1, normalize_embeddings=True, convert_to_numpy=True)
        single.append((time.perf_counter() - t0) * 1000.0)
    out = {
        "device": dev,
        "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        "torch_num_threads": torch.get_num_threads(),
        "batch_size": primary_n,
        "batch_ms_runs": primary["ms_runs"],
        "batch_ms_mean": primary["ms_mean"],
        "batch_ms_median": primary["ms_median"],
        "per_query_ms": primary["per_query_ms"],
        "single_query_ms_mean": round(float(np.mean(single)), 2),
        "runs": runs,
        "configs": configs,
        "paper_reference": "paper: batch=128 query embeddings take ~79.52 ms on an NVIDIA A100 GPU (Phase-1 overhead experiment, Appendix V-1)",
    }
    # CPU float32 reference on the same queries (fp16 is catastrophic on CPU)
    try:
        import os as _os
        _os.environ["HF_HUB_OFFLINE"] = "1"
        mc = SentenceTransformer(str(model_path), device="cpu", model_kwargs={"dtype": torch.float32})
        cpu_cfg = {}
        for bs in wanted:
            tc, n = _time(mc, bs)
            cpu_cfg[f"batch_{n}"] = {
                "n_queries": n,
                "ms_runs": [round(t, 2) for t in tc],
                "ms_mean": round(float(np.mean(tc)), 2),
                "per_query_ms": round(float(np.mean(tc)) / n, 3),
            }
        out["cpu_fp32_reference"] = {"configs": cpu_cfg}
        del mc
    except Exception as exc:  # pragma: no cover
        out["cpu_fp32_reference"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=50_000, help="corpus size (default 50000; paper uses m=100000)")
    ap.add_argument("--n-eval-queries", type=int, default=1000, help="first N qrels queries whose relevant passages are guaranteed in corpus")
    ap.add_argument("--hard-negs-per-query", type=int, default=20,
                    help="BM25 hard negatives per eval query forced into the corpus (0 = pure uniform random filler, the original task-1 spec)")
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto", help="embedding device (auto: mps when available)")
    ap.add_argument("--dtype", choices=["auto", "float16", "float32", "bfloat16"], default="auto", help="embedding dtype (auto: fp16 on mps, fp32 on cpu)")
    ap.add_argument("--num-threads", type=int, default=min(10, os.cpu_count() or 8))
    ap.add_argument("--tar-parts", type=int, default=16)
    ap.add_argument("--tar-workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="redo embedding + sanity even if outputs exist")
    ap.add_argument("--skip-gate-inputs", action="store_true", help="debug: skip sanity/embedding timing outputs")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---------- step 1+2: corpus ----------
    qrels_path, queries_path, tar_path = ensure_raw(args)
    hard_negs = None
    if args.hard_negs_per_query > 0:
        hard_negs = ensure_bm25_negatives(
            qrels_path, queries_path, args.n_eval_queries, args.hard_negs_per_query,
            args.seed, args.tar_parts, args.tar_workers,
        )
    corpus, eval_qids, qrels_pairs, corpus_stats = select_corpus(
        qrels_path, tar_path, args.m, args.n_eval_queries, args.seed,
        hard_negs=hard_negs, n_hard_per_query=args.hard_negs_per_query,
    )
    pids = [c[0] for c in corpus]
    ptexts = [c[1] for c in corpus]
    sources = [c[2] for c in corpus]
    pid_set = set(pids)
    from collections import Counter
    source_counts = dict(Counter(sources))
    log(f"corpus sources: {source_counts}")

    # queries: only eval queries (those with guaranteed qrels coverage)
    qtext_by_id: dict[int, str] = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            qtext_by_id[int(parts[0])] = parts[1]
    missing_q = [q for q in eval_qids if q not in qtext_by_id]
    if missing_q:
        raise RuntimeError(f"{len(missing_q)} eval qids missing from queries.dev.small.tsv: {missing_q[:5]}")
    qids = [int(q) for q in eval_qids]
    qtexts = [qtext_by_id[q] for q in qids]

    # qrels restricted to corpus pids
    keep_pairs = [(qid, pid) for qid, pid in qrels_pairs if pid in pid_set]
    rel_pids_all = {pid for _, pid in keep_pairs}
    covered_qids = {qid for qid, _ in keep_pairs}
    coverage = len(set(eval_qids) & covered_qids) / len(eval_qids)
    log(f"qrels in corpus: {len(keep_pairs)} pairs, {len(covered_qids)} qids, "
        f"eval coverage={coverage*100:.1f}%")

    # ---------- write tabular artifacts ----------
    emb_path = DATA / "passage_emb.npy"
    need_embed = args.force or not (
        emb_path.exists() and (DATA / "passage_ids.npy").exists() and (DATA / "passages.jsonl").exists()
        and (DATA / "query_emb.npy").exists() and (DATA / "query_ids.npy").exists()
        and (DATA / "queries.jsonl").exists()
    )
    if need_embed:
        with open(DATA / "passages.jsonl", "w", encoding="utf-8") as f:
            for pid, text, source in corpus:
                f.write(json.dumps({"pid": pid, "text": text, "source": source}, ensure_ascii=False) + "\n")
        with open(DATA / "queries.jsonl", "w", encoding="utf-8") as f:
            for qid, text in zip(qids, qtexts):
                f.write(json.dumps({"qid": qid, "text": text}, ensure_ascii=False) + "\n")
        with open(DATA / "qrels.tsv", "w", encoding="utf-8") as f:
            for qid, pid in keep_pairs:
                f.write(f"{qid}\t0\t{pid}\t1\n")
        with open(DATA / "eval_qids.json", "w", encoding="utf-8") as f:
            json.dump(qids, f)
        import numpy as np

        np.save(DATA / "passage_ids.npy", np.asarray(pids, dtype=np.int64))
        np.save(DATA / "query_ids.npy", np.asarray(qids, dtype=np.int64))
        np.save(DATA / "corpus_sources.npy", np.asarray(sources))
        log(f"wrote passages.jsonl ({len(corpus)}), queries.jsonl ({len(qids)}), qrels.tsv ({len(keep_pairs)})")
        t_model = time.time()
        model = load_model(args)
        model_load_s = time.time() - t_model
        log(f"model loaded in {model_load_s:.1f}s")
        log(f"encoding {len(ptexts)} passages (batch={args.batch_size}) ...")
        p_emb, p_sec = encode_all(model, ptexts, args.batch_size)
        log(f"passage encode: {p_sec:.1f}s ({len(ptexts)/p_sec:.1f} passages/s)")
        log(f"encoding {len(qtexts)} queries ...")
        q_emb, q_sec = encode_all(model, qtexts, args.batch_size)
        log(f"query encode: {q_sec:.1f}s")
        assert p_emb.shape == (len(corpus), 768), p_emb.shape
        assert q_emb.shape == (len(qids), 768), q_emb.shape
        np.save(emb_path, p_emb.astype(np.float32))
        np.save(DATA / "query_emb.npy", q_emb.astype(np.float32))
        passage_encode_s = p_sec
        query_encode_s = q_sec
        # free memory
        del p_emb, q_emb
    else:
        log("embeddings already exist, skipping embedding step")
        model = load_model(args)
        model_load_s = 0.0
        passage_encode_s = query_encode_s = None

    import numpy as np

    p_emb = np.load(emb_path, mmap_mode="r")
    q_emb = np.load(DATA / "query_emb.npy")
    pids_arr = np.load(DATA / "passage_ids.npy")
    qids_arr = np.load(DATA / "query_ids.npy")

    # ---------- step 4: sanity ----------
    sanity_path = RESULTS / "data_sanity.json"
    timing_path = RESULTS / "embed_timing.json"
    if args.skip_gate_inputs:
        log("skipping sanity/timing outputs (--skip-gate-inputs)")
    else:
        if args.force or not sanity_path.exists():
            log("computing plaintext retrieval sanity metrics ...")
            t0 = time.time()
            sources_arr = np.load(DATA / "corpus_sources.npy")
            p_emb_np = np.asarray(p_emb)
            q_emb_np = np.asarray(q_emb)
            # map pid -> corpus row index
            pid2row = {int(p): i for i, p in enumerate(pids_arr)}
            rel_pairs_idx = [(qid, pid2row[pid]) for qid, pid in keep_pairs]
            metrics = retrieval_metrics(q_emb_np, p_emb_np, qids_arr, rel_pairs_idx)
            # control variant: original task-1 spec = pure uniform random filler
            # (drop the BM25 hard negatives); same queries, same relevant passages.
            keep_rows = np.where(sources_arr != "bm25_hard_neg")[0]
            row_of_pid = {int(pids_arr[orig]): j for j, orig in enumerate(keep_rows)}
            rel_pairs_sub = [(qid, row_of_pid[pid]) for qid, pid in keep_pairs if pid in row_of_pid]
            metrics_random_only = retrieval_metrics(q_emb_np, p_emb_np[keep_rows], qids_arr, rel_pairs_sub)
            examples = sanity_examples(
                model, p_emb_np, qids_arr, qtexts, pids_arr, rel_pairs_idx, n_examples=5
            )
            metrics["corpus_size"] = int(len(pids_arr))
            metrics["corpus_composition"] = source_counts
            metrics["qrels_pairs_in_corpus"] = int(len(keep_pairs))
            metrics["random_fill_only_control"] = metrics_random_only
            metrics["examples"] = examples
            metrics["note"] = (
                "primary = exact plaintext top-k over the m-passage corpus. The corpus contains "
                "the qrels-relevant passages of the eval queries plus BM25 hard negatives "
                "(top1000.dev, query-token-overlap ranked) plus uniform random filler, which "
                "makes the retrieval task non-trivial (like MS MARCO reranking). "
                "random_fill_only_control repeats the evaluation on the original task-1 spec "
                "corpus (uniform random filler only, hard negatives removed); as expected, MRR "
                "there is much higher because all remaining distractors are random passages "
                "(the full-8.8M-collection MS MARCO dev MRR@10 of GTR-T5-base is ~0.39)."
            )
            metrics["elapsed_s"] = round(time.time() - t0, 2)
            with open(sanity_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            log(f"plaintext sanity: {json.dumps({k: v for k, v in metrics.items() if k != 'examples'})}")
        else:
            log("data_sanity.json exists, skipping (use --force to redo)")

        if args.force or not timing_path.exists():
            log("measuring query embedding latency (batch=128) ...")
            timing = embed_timing(model, qtexts, 128, LOCAL_MODEL_DIR, batch_sizes=(args.batch_size,))
            timing["hardware"] = {
                "platform": platform.platform(),
                "cpu": platform.processor() or "Apple M5",
                "torch": __import__("torch").__version__,
                "sentence_transformers": __import__("sentence_transformers").__version__,
            }
            with open(timing_path, "w", encoding="utf-8") as f:
                json.dump(timing, f, indent=2)
            log(f"embed timing: {timing['batch_ms_mean']} ms / {timing['batch_size']} queries "
                f"({timing['per_query_ms']} ms/query, {timing['device']}/{timing['dtype']})")

    # ---------- step 6: meta ----------
    import numpy as np

    meta = {
        "dataset": "MS MARCO passage ranking dev (small)",
        "embedding_model": MODEL_NAME,
        "embedding_dim": int(np.load(emb_path, mmap_mode="r").shape[1]),
        "normalized": True,
        "dtype": "float32",
        "m_passages": int(len(pids_arr)),
        "n_queries": int(len(qids_arr)),
        "n_eval_queries": int(len(set(eval_qids))),
        "corpus_composition": {
            "qrels_relevant": int(source_counts.get("qrels", 0)),
            "bm25_hard_neg": int(source_counts.get("bm25_hard_neg", 0)),
            "random_fill": int(source_counts.get("random_fill", 0)),
            "hard_negs_per_query": int(args.hard_negs_per_query),
            "note": "row order = [qrels-relevant passages sorted by pid] + [BM25 hard negatives sorted by pid] + [random filler sorted by pid]; any corpus prefix of length >= qrels_relevant+bm25_hard_neg contains all relevant passages. Passages.jsonl carries a per-row 'source' field and data/corpus_sources.npy matches it.",
        },
        "corpus_fill": {
            "source": "MS MARCO passage collection (collection.tsv, 8,841,823 passages) from collectionandqueries.tar.gz",
            "method": f"deterministic uniform sample by blake2b(seed:pid) heap selection, seed={args.seed}",
            "collection_lines": corpus_stats.get("collection_lines", N_COLLECTION_LINES),
            "cached_selection": bool(corpus_stats.get("cached")),
            "hard_negatives": {
                "source": BM25_TAR_URL + f" (member {BM25_MEMBER})",
                "selection": f"top {args.hard_negs_per_query} per eval query by fraction of query tokens contained in the candidate passage (file has no rank column)",
                "n_pids": int(source_counts.get("bm25_hard_neg", 0)),
            } if args.hard_negs_per_query > 0 else None,
        },
        "qrels": {
            "file": "data/qrels.tsv",
            "n_pairs": int(len(keep_pairs)),
            "n_qids": int(len(covered_qids)),
            "eval_coverage": float(coverage),
            "format": "qid \\t 0 \\t pid \\t 1 (only pids present in corpus)",
        },
        "raw_sources": {
            "qrels": QRELS_URL,
            "queries": "queries.dev.small.tsv from " + TAR_URL,
            "collection": TAR_URL,
            "tar_size_bytes": TAR_SIZE,
            "bm25_hard_negatives": BM25_TAR_URL if args.hard_negs_per_query > 0 else None,
        },
        "files": {
            "passage_emb.npy": [int(len(pids_arr)), int(np.load(emb_path, mmap_mode="r").shape[1])],
            "passage_ids.npy": [int(len(pids_arr))],
            "corpus_sources.npy": [int(len(pids_arr))],
            "passages.jsonl": int(len(pids_arr)),
            "query_emb.npy": [int(len(qids_arr)), 768],
            "query_ids.npy": [int(len(qids_arr))],
            "queries.jsonl": int(len(qids_arr)),
            "qrels.tsv": int(len(keep_pairs)),
            "eval_qids.json": int(len(qids_arr)),
        },
        "timing_s": {
            "total_wall": round(time.time() - t_start, 1),
            "model_load": round(model_load_s, 1),
            "passage_encode_s": round(passage_encode_s, 1) if passage_encode_s else None,
            "query_encode_s": round(query_encode_s, 1) if query_encode_s else None,
            "passages_per_s": round(len(pids_arr) / passage_encode_s, 1) if passage_encode_s else None,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "sentence_transformers": __import__("sentence_transformers").__version__,
            "device": str(model.device),
            "embedding_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
            "torch_num_threads": args.num_threads,
        },
        "embedding_timing": "results/embed_timing.json",
        "sanity": "results/data_sanity.json",
        "command": " ".join([sys.executable] + sys.argv),
        "args": vars(args),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(DATA / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"wrote data/meta.json; total wall time {time.time()-t_start:.1f}s")
    log("DONE")


if __name__ == "__main__":
    main()
