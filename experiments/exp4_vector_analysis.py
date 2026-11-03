"""exp4 -- vector-analysis attack ASR on CAPRISE vs ADCPE (paper Fig. 5).

Paper setting (PAPER_ANALYSIS.md Section 9.5, ambiguity A12/A13)
---------------------------------------------------------------
The honest-but-curious server holds the encrypted database and the public
nonces.  The paper picks *a vector from the database* as the test query, then
computes its distance to all other database vectors -- first in plaintext,
then on the ADCPE-encrypted and CAPRISE-encrypted databases -- and compares the
induced rankings.  ASR (Attack Success Rate) measures how consistent the
ciphertext ranking is with the plaintext ranking (lower = more privacy).

The paper gives no formula for ASR (ambiguity A12), so this reproduction defines
and reports:

    ASR@t = |top_t(ciphertext distances) & top_t(plaintext distances)| / t
            averaged over |Q| random database-vector queries, t in {1,5,10}
            (main metric; lower is better for privacy)

Auxiliary leakage measures (also lower = less leakage):

    Kendall tau / Spearman rho between the full ciphertext distance vector and
    the full plaintext distance vector (m-1 entries, query itself excluded);
    distance-reconstruction error mean |d_cipher / s - d_plain|, i.e. how
    accurately the server can recover the absolute plaintext distances after
    removing the known/common scale s (in a real deployment the adversary does
    not know s, but dividing by a *common* positive constant does not change
    the ranking; this error is a scale-invariant measure of distance leakage
    and, per finding A19, the quantity where CAPRISE's larger DB-side noise
    must show up).

Attack model / implementation notes
-----------------------------------
* The adversary sees only (e'_i, nonce_i), never the plaintext vectors nor
  (s, K).  The test query is the stored ciphertext e'_j itself (as in the
  paper's Fig. 5), NOT an EncQ ciphertext: this is the in-database structure
  analysis attack of paper Section IV.
* Both methods use the *same* plaintext database E, the same key (s=3, beta as
  configured) and the same query indices; only the noise radius differs:
  CAPRISE DB radius 3*s*beta/8 = 0.225, ADCPE/SAP radius s*beta/4 = 0.150.
* Deterministic nonces (derived from the run seed) are fed to EncDB /
  ADCPE.Enc so the whole experiment is bit-for-bit reproducible; the crypto
  formulas are the core implementations from src/caprise/caprise.py.
* Dimension plan (ambiguity A13): d=768 uses real gtr-t5-base MS MARCO passage
  embeddings when available (falling back to synthetic isotropic vectors);
  d=1536/3072 use L2-normalised i.i.d. Gaussian vectors (GTR only emits 768
  dims).  We additionally run a d=768 *synthetic* point so that the dimension
  trend (768 -> 1536 -> 3072) is measured on one consistent distribution and
  the real-data point can be compared against it directly.

Outputs
-------
    results/exp4_vector_analysis.json    machine-readable results
    results/exp4_vector_analysis.csv     main table as CSV
    figures/fig5_vector_analysis.png     4-panel figure

Run
---
    .venv/bin/python experiments/exp4_vector_analysis.py            # full
    .venv/bin/python experiments/exp4_vector_analysis.py --quick    # fast
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caprise import metrics as M  # noqa: E402
from caprise.caprise import (  # noqa: E402
    ADCPE,
    EncDB,
    KeyGen,
    noise_radius_adcpe,
    noise_radius_db,
)
from caprise.prf import new_nonce  # noqa: E402

TS_DEFAULT = (1, 5, 10)
METHODS = ("ADCPE", "CAPRISE")

# Paper Fig. 5 reference values, digitised from the paper source figure
# ../../output/pdf/source/caprise/figures/exp_asr.png (bar heights, +-1 pp).
# Paper axis: "Embedding Size" in {64,128,256,312,384,512,768,896,1024,1536,1792,2048};
# paper axis: "ASR (%)"; method labels in the published figure are "CONPRISE"
# (typo for CAPRISE) vs "ADCPE".  PAPER_ANALYSIS.md previously recorded "no ASR
# values given" (A12) because the numbers are only in the figure image.
PAPER_FIG5 = {
    "source": "figures/exp_asr.png in the paper LaTeX source (digitised bar heights, +-1 pp)",
    "dims": [64, 128, 256, 312, 384, 512, 768, 896, 1024, 1536, 1792, 2048],
    "CAPRISE": [53.5, 60.6, 70.2, 73.3, 75.1, 76.3, 83.1, 84.6, 84.8, 86.3, 87.5, 88.8],
    "ADCPE": [65.8, 76.5, 78.2, 78.7, 78.5, 85.8, 87.5, 87.3, 88.3, 91.9, 91.7, 94.6],
}


# --------------------------------------------------------------------------- #
# Deterministic randomness helpers
# --------------------------------------------------------------------------- #


def _rng(seed: int, *tags) -> np.random.Generator:
    """Sub-stream generator: (seed, tags) -> independent reproducible stream."""
    h = hashlib.sha256(("|".join(str(t) for t in (seed,) + tags)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def _nonces(rng: np.random.Generator, m: int) -> list[bytes]:
    return [new_nonce(rng) for _ in range(m)]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def _l2norm(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return X / n


def synthetic_db(m: int, d: int, seed: int) -> np.ndarray:
    """L2-normalised i.i.d. standard Gaussian rows (isotropic on the sphere)."""
    rng = _rng(seed, "synthetic", d, m)
    return _l2norm(rng.standard_normal((m, d)))


def load_real_db(m: int, seed: int) -> tuple[np.ndarray | None, dict]:
    """Real gtr-t5-base MS MARCO passage embeddings (d=768), if obtainable.

    Preference order:
      1. data/passage_emb.npy produced by scripts/prepare_data.py (M0);
      2. a cached mini-embedding file under .cache/exp4/ (built once from the
         real MS MARCO corpus texts in data/raw/corpus_selected_*.jsonl with
         the same sentence-transformers/gtr-t5-base model);
      3. otherwise report the real-data point as skipped.

    Returns (E_unit_norm float64 (m,768), provenance dict) or (None, meta).
    """
    meta: dict = {"dim": 768}
    emb_path = ROOT / "data" / "passage_emb.npy"
    if emb_path.exists():
        E = np.load(emb_path).astype(np.float64)
        source = "data/passage_emb.npy"
    else:
        cache_dir = ROOT / ".cache" / "exp4"
        cache = cache_dir / f"real_gtr_passage_emb_{m}_{seed}.npy"
        if cache.exists():
            E = np.load(cache).astype(np.float64)
            source = str(cache.relative_to(ROOT))
        else:
            corpus_files = sorted((ROOT / "data" / "raw").glob("corpus_selected_*.jsonl"))
            if not corpus_files:
                meta["status"] = "skipped"
                meta["reason"] = "no data/passage_emb.npy and no data/raw/corpus_selected_*.jsonl"
                return None, meta
            texts = []
            with open(corpus_files[0], "r", encoding="utf-8") as f:
                for line in f:
                    texts.append(json.loads(line)["text"])
                    if len(texts) >= m:
                        break
            if len(texts) < m:
                meta["status"] = "skipped"
                meta["reason"] = f"only {len(texts)} corpus texts available"
                return None, meta
            model_dir = ROOT / "data" / "raw" / "gtr-t5-base"
            model_name = str(model_dir) if (model_dir / "config.json").exists() else "sentence-transformers/gtr-t5-base"
            from sentence_transformers import SentenceTransformer

            t0 = time.time()
            model = SentenceTransformer(model_name, device="cpu")
            E = model.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float64)
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache, E.astype(np.float32))
            source = str(cache.relative_to(ROOT))
            meta["embedding_seconds"] = round(time.time() - t0, 2)
            meta["embedding_model"] = model_name
            del model
    E = _l2norm(E)
    n_total = int(E.shape[0])
    selected_rows = None
    if n_total > m:
        rng = _rng(seed, "real", m)
        idx = rng.choice(n_total, size=m, replace=False)
        idx.sort()
        E = E[idx]
        selected_rows = [int(i) for i in idx[:10]]
    # dataset provenance from M0 artifact, if present
    ds_meta = None
    meta_path = ROOT / "data" / "meta.json"
    if meta_path.exists():
        try:
            full_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ds_meta = {k: full_meta.get(k) for k in ("dataset", "embedding_model", "m_passages", "normalized", "corpus_composition")}
        except Exception:  # pragma: no cover
            ds_meta = None
    meta.update(
        status="ok",
        source=source,
        m=int(E.shape[0]),
        dim=int(E.shape[1]),
        available_rows=n_total,
        selected_rows_first10=selected_rows,
        dataset=ds_meta,
        corpus=(
            "MS MARCO passage dev subset (M0 scripts/prepare_data.py; 50k real gtr-t5-base embeddings)"
            if emb_path.exists()
            else "MS MARCO passage corpus (real texts)"
        ),
    )
    return E, meta


# --------------------------------------------------------------------------- #
# Encryption (deterministic nonces; core formulas from src/caprise/caprise.py)
# --------------------------------------------------------------------------- #


def encrypt_database(method: str, key, E: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Encrypt the database with CAPRISE or ADCPE/SAP using deterministic nonces."""
    m = E.shape[0]
    nonces = _nonces(rng, m)
    if method == "CAPRISE":
        Ep, _ = EncDB(key, E, nonces=nonces)
        return Ep
    if method == "ADCPE":
        # ADCPE.Enc is the core per-vector SAP encryption (radius s*beta/4);
        # ADCPE.EncDB is the same loop with fresh os.urandom nonces.  We call
        # Enc with deterministic nonces purely for reproducibility.
        return np.stack([ADCPE.Enc(key, e, nonce=r)[0] for e, r in zip(E, nonces)])
    raise ValueError(f"unknown method {method!r}")


def make_key(d: int, beta: float, s: int, seed: int):
    return KeyGen(dim=d, beta=beta, s=s, rng=_rng(seed, "key", d, beta))


# --------------------------------------------------------------------------- #
# The attack: server-side ranking of plaintext vs ciphertext distances
# --------------------------------------------------------------------------- #


def _sq_dists(X: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Squared L2 distances between rows of Q (nq,d) and X (m,d)."""
    xn = np.einsum("ij,ij->i", X, X)
    qn = np.einsum("ij,ij->i", Q, Q)
    D2 = qn[:, None] + xn[None, :] - 2.0 * (Q @ X.T)
    return np.clip(D2, 0.0, None)


def plaintext_gap_stats(E: np.ndarray, query_idx: np.ndarray, ts=(1, 5, 10, 50)) -> dict:
    """Nearest-neighbour distance gap statistics of the *plaintext* database.

    The vector-analysis attack can only flip ranks when the encryption noise is
    comparable to the gaps between consecutive plaintext distances.  We record
    mean/median gaps at ranks t in {1,5,10,50} so that the ASR levels (and the
    small CAPRISE-vs-ADCPE delta) can be explained quantitatively: only the
    low-rank gaps (rank 5+) are of the order of the noise-induced distance
    perturbation sigma ~ sqrt(2)*R/sqrt(d) (R = noise radius).
    """
    E = np.asarray(E, dtype=np.float64)
    nq = len(query_idx)
    D2 = _sq_dists(E, E[query_idx])
    D2[np.arange(nq), query_idx] = np.inf
    D = np.sort(np.sqrt(D2), axis=1)[:, :-1]  # drop the self entry (inf sorts last)
    out: dict = {
        "nn_distance_mean": float(D[:, 0].mean()),
        "nn_distance_median": float(np.median(D[:, 0])),
        "mean_pairwise_distance": float(D.mean()),
    }
    for t in ts:
        if t < D.shape[1]:
            gaps = D[:, t] - D[:, t - 1]
            out[f"gap_rank{t}"] = float(gaps.mean())
            out[f"gap_rank{t}_median"] = float(np.median(gaps))
    return out


def attack_metrics(
    E: np.ndarray,
    Ep: np.ndarray,
    query_idx: np.ndarray,
    s: int,
    ts: tuple[int, ...] = TS_DEFAULT,
) -> dict:
    """Vector-analysis attack metrics for one encrypted database.

    The adversary sees Ep (and nonces, unused here) and takes the stored
    ciphertext e'_j (``Ep[query_idx]``) as the test query.  Ground truth is the
    plaintext distance ranking from e_j = E[query_idx] to the other E[i].
    """
    E = np.asarray(E, dtype=np.float64)
    Ep = np.asarray(Ep, dtype=np.float64)
    nq = len(query_idx)
    rows = np.arange(nq)

    D2p = _sq_dists(E, E[query_idx])
    D2c = _sq_dists(Ep, Ep[query_idx])
    # exclude the query vector itself: distance 0 in both spaces (it would
    # otherwise trivially occupy rank 1 in the plaintext and ciphertext top-1)
    D2p[rows, query_idx] = np.inf
    D2c[rows, query_idx] = np.inf

    d_plain = np.sqrt(D2p)
    d_cipher = np.sqrt(D2c) / float(s)  # remove the common scale s (rank-invariant)

    rank_plain = np.argsort(d_plain, axis=1, kind="stable")
    rank_cipher = np.argsort(d_cipher, axis=1, kind="stable")

    out: dict = {}
    per_query = {f"asr@{t}": [] for t in ts}
    for t in ts:
        for i in range(nq):
            per_query[f"asr@{t}"].append(M.top_t_overlap(rank_plain[i], rank_cipher[i], t))

    kendall, spearman, dist_err = [], [], []
    keep = np.ones(E.shape[0], dtype=bool)
    for i in range(nq):
        keep[:] = True
        keep[query_idx[i]] = False  # drop the query itself (its entry is inf)
        dp_i = d_plain[i][keep]
        dc_i = d_cipher[i][keep]
        kendall.append(M.kendall_tau(dp_i, dc_i))
        spearman.append(M.spearman(dp_i, dc_i))
        dist_err.append(float(np.abs(dc_i - dp_i).mean()))

    for t in ts:
        vals = np.asarray(per_query[f"asr@{t}"], dtype=np.float64)
        out[f"asr@{t}"] = float(vals.mean())
        out[f"asr@{t}_std"] = float(vals.std(ddof=1)) if nq > 1 else 0.0
    out["kendall"] = float(np.mean(kendall))
    out["kendall_std"] = float(np.std(kendall, ddof=1)) if nq > 1 else 0.0
    out["spearman"] = float(np.mean(spearman))
    out["spearman_std"] = float(np.std(spearman, ddof=1)) if nq > 1 else 0.0
    out["dist_err"] = float(np.mean(dist_err))
    out["dist_err_std"] = float(np.std(dist_err, ddof=1)) if nq > 1 else 0.0
    out["dist_err_rel"] = float(np.mean(dist_err) / float(np.mean(d_plain[np.isfinite(d_plain)])))
    out["n_queries"] = int(nq)
    out["m"] = int(E.shape[0])
    out["dim"] = int(E.shape[1])
    return out


# --------------------------------------------------------------------------- #
# Experiment driver
# --------------------------------------------------------------------------- #


def _summarise_cell(
    E: np.ndarray, query_idx: np.ndarray, beta: float, s: int, seed: int, tag: str
) -> dict:
    """Encrypt with both methods (same key/data/query idx) and measure the attack."""
    d = E.shape[1]
    key = make_key(d, beta, s, seed)
    cell: dict = {"radius": {"CAPRISE": noise_radius_db(s, beta), "ADCPE": noise_radius_adcpe(s, beta)}}
    # theoretical std of the noise-induced distance perturbation, in plaintext
    # units: sigma = sqrt(2) * (R/s) / sqrt(d)   (two independent noise vectors)
    sigma_plain = {
        "CAPRISE": float(np.sqrt(2.0) * (noise_radius_db(s, beta) / s) / np.sqrt(d)),
        "ADCPE": float(np.sqrt(2.0) * (noise_radius_adcpe(s, beta) / s) / np.sqrt(d)),
    }
    gap = plaintext_gap_stats(E, query_idx)
    cell["noise_sigma_plain"] = sigma_plain
    cell["plaintext_gap_stats"] = gap
    for method in METHODS:
        t0 = time.time()
        Ep = encrypt_database(method, key, E, _rng(seed, "enc", method, tag))
        enc_s = time.time() - t0
        mets = attack_metrics(E, Ep, query_idx, s)
        mets["encrypt_seconds"] = round(enc_s, 3)
        mets["noise_sigma_plain"] = sigma_plain[method]
        for t in (1, 5, 10, 50):
            if f"gap_rank{t}" in gap:
                mets[f"gap_over_sigma_rank{t}"] = gap[f"gap_rank{t}"] / sigma_plain[method]
        # sanity: ciphertext must not be the plaintext (no accidental leak/scale)
        mets["max_abs_ep_minus_s_e"] = float(np.abs(Ep - s * E).max())
        cell[method] = mets
    return cell


def run_experiment(cfg: dict) -> dict:
    t_start = time.time()
    seed = int(cfg["seed"])
    s = int(cfg["s"])
    beta = float(cfg["beta"])
    m = int(cfg["m"])
    nq = int(cfg["nq"])
    ts = tuple(cfg.get("ts", TS_DEFAULT))

    results: dict = {
        "experiment": "exp4_vector_analysis",
        "paper_ref": "CAPRISE paper Fig. 5 / PAPER_ANALYSIS.md Section 9.5 (ambiguities A12, A13, A19)",
        "definition": {
            "ASR@t": "|top_t(ciphertext distances) & top_t(plaintext distances)| / t, averaged over queries; lower = more privacy",
            "kendall": "Kendall tau between the full plaintext and ciphertext distance vectors (m-1 entries; lower = less leakage)",
            "spearman": "Spearman rho between the same two vectors (lower = less leakage)",
            "dist_err": "mean |d_cipher/s - d_plain| over all pairs; lower bound on absolute-distance leakage (scale removed)",
            "attack_query": "one stored ciphertext e'_j from the encrypted database (server-visible), NOT an EncQ ciphertext",
            "test_query_count": nq,
        },
        "setup": {
            "s": s,
            "beta": beta,
            "m": m,
            "nq": nq,
            "seed": seed,
            "dims_synthetic": list(cfg.get("synth_dims", [768, 1536, 3072])),
            "real_dim": cfg.get("real_dim", 768),
            "ts": list(ts),
            "deterministic_nonces": True,
            "note": (
                "d=768 real = gtr-t5-base MS MARCO passage embeddings (L2 normalised); "
                "d=1536/3072 synthetic = L2-normalised i.i.d. Gaussian (GTR emits only 768 dims, A13); "
                "a synthetic d=768 control makes the dimension trend distribution-consistent."
            ),
        },
        "selfcheck": {},
        "main": {"synthetic": {}, "real_gtr": {}},
        "paper_dims_synthetic": {},
        "paper_fig5_reference": PAPER_FIG5,
        "m_sweep": {},
        "real_m_sweep": {},
        "normalization_control": {},
        "beta_sweep": {},
        "findings": [],
    }

    # ---------------- pipeline self-check: noiseless ciphertext -> ASR == 1 ----------------
    d0 = int(cfg.get("selfcheck_dim", 768))
    m0 = min(int(cfg.get("selfcheck_m", 1000)), m)
    E0 = synthetic_db(m0, d0, seed)
    q0 = _rng(seed, "selfcheck_q").choice(m0, size=min(nq, m0), replace=False)
    err0 = attack_metrics(E0, s * E0, q0, s, ts)
    results["selfcheck"] = {
        "description": "E' = s*E (no noise): ciphertext ranking must equal plaintext ranking",
        "asr@1": err0["asr@1"],
        "asr@5": err0["asr@5"],
        "asr@10": err0["asr@10"],
        "kendall": err0["kendall"],
        "dist_err": err0["dist_err"],
        "max_abs_ep_minus_s_e": 0.0,
    }

    # ---------------- main table: synthetic dims + real d=768 ----------------
    for d in cfg.get("synth_dims", [768, 1536, 3072]):
        E = synthetic_db(m, d, seed)
        q = _rng(seed, "q", d, m).choice(m, size=nq, replace=False)
        cell = _summarise_cell(E, q, beta, s, seed, tag=f"synth_{d}")
        cell["source"] = "synthetic_gaussian"
        cell["n_queries"] = int(nq)
        results["main"]["synthetic"][str(d)] = cell
        print(
            f"[exp4] synthetic d={d}: ADCPE ASR@10={cell['ADCPE']['asr@10']:.3f} "
            f"CAPRISE ASR@10={cell['CAPRISE']['asr@10']:.3f}",
            flush=True,
        )

    real_dim = cfg.get("real_dim", 768)
    if real_dim:
        E_real, meta = load_real_db(m, seed)
        if E_real is not None and E_real.shape[1] == real_dim:
            q = _rng(seed, "q", "real", real_dim, m).choice(E_real.shape[0], size=nq, replace=False)
            cell = _summarise_cell(E_real, q, beta, s, seed, tag="real_768")
            cell["source"] = "real_gtr"
            cell["provenance"] = meta
            results["main"]["real_gtr"][str(real_dim)] = cell
            print(
                f"[exp4] real GTR d={real_dim}: ADCPE ASR@10={cell['ADCPE']['asr@10']:.3f} "
                f"CAPRISE ASR@10={cell['CAPRISE']['asr@10']:.3f}  (source: {meta.get('source')})",
                flush=True,
            )
        else:
            results["main"]["real_gtr"] = {"status": "skipped", "reason": meta.get("reason", "unavailable")}

    # ---------------- paper Fig.5 dimension grid (same embedding-size points as the figure) ----------------
    # The paper varies the embedding size over {64,...,2048}.  GTR only emits 768
    # dims, so for this controlled trend we use L2-normalised Gaussian vectors at
    # exactly the paper's sizes (A13: the paper does not state which embedding
    # model produces each size; we hold the distribution fixed to isolate the
    # dimension effect).
    paper_dims = cfg.get("paper_dims") or []
    paper_asr = {}
    for d in paper_dims:
        E = synthetic_db(m, int(d), seed)
        q = _rng(seed, "q", int(d), m).choice(m, size=nq, replace=False)
        cell = _summarise_cell(E, q, beta, s, seed, tag=f"paperedims_{d}")
        cell["source"] = "synthetic_gaussian_normalized"
        results["paper_dims_synthetic"][str(d)] = cell
        paper_asr[str(d)] = {mm: cell[mm]["asr@10"] for mm in METHODS}
        print(
            f"[exp4] paper-dim d={d}: ADCPE ASR@10={cell['ADCPE']['asr@10']:.3f} "
            f"CAPRISE ASR@10={cell['CAPRISE']['asr@10']:.3f}",
            flush=True,
        )

    # ---------------- normalisation control (A4): unnormalised Gaussian DB ----------------
    # Does omitting L2 normalisation (a plausible reading of the paper, A4) explain
    # the paper's strong ASR-vs-dimension trend?  With unnormalised embeddings the
    # norm grows like sqrt(d) while the noise radius stays fixed at s*beta/8 etc., so
    # the relative noise shrinks with d and ASR should *increase* with d, but from a
    # point that is already ~1.0 at large d.  Measured to decide the question.
    for d in cfg.get("norm_control_dims", []) or []:
        rng = _rng(seed, "synthetic_unnorm", int(d), m)
        E = rng.standard_normal((m, int(d)))  # NOTE: no L2 normalisation
        q = _rng(seed, "q", int(d), m, "unnorm").choice(m, size=nq, replace=False)
        cell = _summarise_cell(E, q, beta, s, seed, tag=f"unnorm_{d}")
        cell["source"] = "synthetic_gaussian_unnormalized"
        results["normalization_control"][str(d)] = cell
        print(
            f"[exp4] unnormalised d={d}: ADCPE ASR@10={cell['ADCPE']['asr@10']:.3f} "
            f"CAPRISE ASR@10={cell['CAPRISE']['asr@10']:.3f}",
            flush=True,
        )

    # ---------------- real-GTR m sweep: database density effect ----------------
    # The paper does not state m for Fig. 5; larger m makes nearest neighbours
    # denser (smaller distance gaps), which should lower ASR on real embeddings.
    # This tests whether the remaining gap between our m=5000 real point and the
    # paper's numbers is explained by database size.
    for m_i in cfg.get("real_m_sweep", []) or []:
        try:
            E_m, meta_m = load_real_db(int(m_i), seed)
            if E_m is None:
                results["real_m_sweep"][str(m_i)] = {"status": "skipped", "reason": meta_m.get("reason", "unavailable")}
                continue
            q = _rng(seed, "q", "real_m", int(m_i)).choice(E_m.shape[0], size=min(nq, E_m.shape[0]), replace=False)
            cell = _summarise_cell(E_m, q, beta, s, seed, tag=f"real_m_{m_i}")
            cell["source"] = "real_gtr"
            cell["provenance"] = meta_m
            results["real_m_sweep"][str(m_i)] = cell
            print(
                f"[exp4] real m={m_i}: ADCPE ASR@10={cell['ADCPE']['asr@10']:.3f} "
                f"CAPRISE ASR@10={cell['CAPRISE']['asr@10']:.3f} "
                f"nn={cell['plaintext_gap_stats']['nn_distance_mean']:.3f}",
                flush=True,
            )
            del E_m
        except MemoryError:  # pragma: no cover
            results["real_m_sweep"][str(m_i)] = {"status": "skipped", "reason": "MemoryError"}
            print(f"[exp4] real m={m_i}: skipped (MemoryError)", flush=True)

    # ---------------- m sweep at d=768 synthetic ----------------
    if cfg.get("m_sweep"):
        for m_i in cfg["m_sweep"]:
            E = synthetic_db(int(m_i), 768, seed)
            q = _rng(seed, "q", 768, int(m_i)).choice(int(m_i), size=min(nq, int(m_i)), replace=False)
            cell = _summarise_cell(E, q, beta, s, seed, tag=f"msweep_{m_i}")
            cell["source"] = "synthetic_gaussian"
            results["m_sweep"][str(m_i)] = cell
            print(
                f"[exp4] m={m_i}: ADCPE ASR@10={cell['ADCPE']['asr@10']:.3f} "
                f"CAPRISE ASR@10={cell['CAPRISE']['asr@10']:.3f}",
                flush=True,
            )

    # ---------------- beta sweep at d=768 synthetic ----------------
    if cfg.get("beta_sweep"):
        for beta_i in cfg["beta_sweep"]:
            E = synthetic_db(m, 768, seed)
            q = _rng(seed, "q", 768, m).choice(m, size=nq, replace=False)
            cell = _summarise_cell(E, q, float(beta_i), s, seed, tag=f"betasweep_{beta_i}")
            cell["source"] = "synthetic_gaussian"
            cell["beta"] = float(beta_i)
            results["beta_sweep"][str(beta_i)] = cell
            print(
                f"[exp4] beta={beta_i}: ADCPE ASR@10={cell['ADCPE']['asr@10']:.3f} "
                f"CAPRISE ASR@10={cell['CAPRISE']['asr@10']:.3f}",
                flush=True,
            )

    # ---------------- findings (auto-derived, honest reporting) ----------------
    findings = []
    synth = results["main"]["synthetic"]
    for d, cell in synth.items():
        delta = cell["CAPRISE"]["asr@10"] - cell["ADCPE"]["asr@10"]
        findings.append(
            {
                "kind": "main_cell",
                "dim": int(d),
                "source": "synthetic_gaussian",
                "asr10_adcpe": cell["ADCPE"]["asr@10"],
                "asr10_caprise": cell["CAPRISE"]["asr@10"],
                "delta_caprise_minus_adcpe": delta,
                "caprise_lower_asr10": bool(delta < 0),
            }
        )
    for d, cell in results["main"]["real_gtr"].items():
        if isinstance(cell, dict) and "ADCPE" in cell:
            delta = cell["CAPRISE"]["asr@10"] - cell["ADCPE"]["asr@10"]
            findings.append(
                {
                    "kind": "main_cell",
                    "dim": int(d),
                    "source": "real_gtr",
                    "asr10_adcpe": cell["ADCPE"]["asr@10"],
                    "asr10_caprise": cell["CAPRISE"]["asr@10"],
                    "delta_caprise_minus_adcpe": delta,
                    "caprise_lower_asr10": bool(delta < 0),
                }
            )
    # distance-error ratio (finding A19: the *measurable* CAPRISE advantage)
    for d, cell in list(synth.items()) + [
        (d, c) for d, c in results["main"]["real_gtr"].items() if isinstance(c, dict) and "ADCPE" in c
    ]:
        ratio = cell["CAPRISE"]["dist_err"] / cell["ADCPE"]["dist_err"] if cell["ADCPE"]["dist_err"] else float("nan")
        findings.append(
            {
                "kind": "dist_err_ratio",
                "dim": int(d),
                "source": cell.get("source"),
                "caprise_dist_err": cell["CAPRISE"]["dist_err"],
                "adcpe_dist_err": cell["ADCPE"]["dist_err"],
                "ratio": ratio,
                "theory_ratio_noise_radii": 1.5,
            }
        )
    # paper Fig.5 comparison at the common d=768 point (synthetic, normalised)
    for src, section in (("synthetic_gaussian", synth), ("paper_dims_synthetic", results["paper_dims_synthetic"])):
        cell = section.get("768")
        if not isinstance(cell, dict) or "ADCPE" not in cell:
            continue
        i = PAPER_FIG5["dims"].index(768)
        for method in METHODS:
            findings.append(
                {
                    "kind": "paper_fig5_comparison",
                    "dim": 768,
                    "source": src,
                    "method": method,
                    "ours_asr10": cell[method]["asr@10"],
                    "paper_asr10": PAPER_FIG5[method][i] / 100.0,
                    "abs_diff_pp": abs(cell[method]["asr@10"] - PAPER_FIG5[method][i] / 100.0) * 100,
                }
            )
    # real-m sweep trend (database density explanation for the residual gap)
    mvals = sorted(int(k) for k, v in results["real_m_sweep"].items() if isinstance(v, dict) and "ADCPE" in v)
    if len(mvals) >= 2:
        findings.append(
            {
                "kind": "real_m_density",
                "m_values": mvals,
                "asr10_adcpe": [results["real_m_sweep"][str(mm)]["ADCPE"]["asr@10"] for mm in mvals],
                "asr10_caprise": [results["real_m_sweep"][str(mm)]["CAPRISE"]["asr@10"] for mm in mvals],
                "nn_distance": [results["real_m_sweep"][str(mm)]["plaintext_gap_stats"]["nn_distance_mean"] for mm in mvals],
                "interpretation": "ASR drops as the database grows denser (smaller NN gaps): the paper does not report m for Fig. 5, so its absolute ASR level is not identifiable from the text.",
            }
        )
    # normalisation control outcome
    norm_ctrl = results["normalization_control"]
    if norm_ctrl:
        findings.append(
            {
                "kind": "normalization_control",
                "dims": sorted(int(k) for k in norm_ctrl),
                "asr10_adcpe": [norm_ctrl[k]["ADCPE"]["asr@10"] for k in sorted(norm_ctrl, key=int)],
                "asr10_caprise": [norm_ctrl[k]["CAPRISE"]["asr@10"] for k in sorted(norm_ctrl, key=int)],
                "interpretation": "unnormalised vectors make the fixed noise radius negligible at large d (ASR->1); this cannot reproduce the paper's low ASR at small dimensions, so A4 is not the explanation of the paper's trend.",
            }
        )
    n_rev = sum(1 for f in findings if f["kind"] == "main_cell" and not f["caprise_lower_asr10"])
    results["findings"] = findings
    results["summary"] = {
        "n_main_cells": sum(1 for f in findings if f["kind"] == "main_cell"),
        "n_cells_caprise_asr10_lower": sum(1 for f in findings if f["kind"] == "main_cell" and f["caprise_lower_asr10"]),
        "n_cells_caprise_asr10_not_lower": n_rev,
        "caprise_lower_asr10_trend": bool(n_rev == 0),
        "mean_asr10_adcpe": float(
            np.mean([f["asr10_adcpe"] for f in findings if f["kind"] == "main_cell"])
        ),
        "mean_asr10_caprise": float(
            np.mean([f["asr10_caprise"] for f in findings if f["kind"] == "main_cell"])
        ),
    }
    if results["paper_dims_synthetic"]:
        pds = results["paper_dims_synthetic"]
        results["summary"]["paper_dims_caprise_lower_all"] = bool(
            all(pds[k]["CAPRISE"]["asr@10"] < pds[k]["ADCPE"]["asr@10"] for k in pds)
        )
        results["summary"]["paper_dims_asr10_caprise_mean"] = float(
            np.mean([pds[k]["CAPRISE"]["asr@10"] for k in pds])
        )
        results["summary"]["paper_dims_asr10_adcpe_mean"] = float(
            np.mean([pds[k]["ADCPE"]["asr@10"] for k in pds])
        )
        results["summary"]["paper_fig5_asr10_caprise_mean"] = float(np.mean(PAPER_FIG5["CAPRISE"]) / 100.0)
        results["summary"]["paper_fig5_asr10_adcpe_mean"] = float(np.mean(PAPER_FIG5["ADCPE"]) / 100.0)

    results["runtime"] = {
        "seconds": round(time.time() - t_start, 2),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
    }
    return results


# --------------------------------------------------------------------------- #
# Figure (mirrors paper Fig. 5: ASR of the vector-analysis attack vs dimension)
# --------------------------------------------------------------------------- #


def make_figure(results: dict, path: Path) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    synth = results["main"]["synthetic"]
    dims = sorted(int(d) for d in synth)

    def cell_get(section, d):
        return section[str(d)]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    colors = {"ADCPE": "#1f77b4", "CAPRISE": "#d62728"}
    markers = {"ADCPE": "o", "CAPRISE": "s"}

    # (a) ASR@1/5/10 vs dimension (synthetic trend) + real d=768 markers
    ax = axes[0, 0]
    for method in METHODS:
        for t, ls in ((1, ":"), (5, "--"), (10, "-")):
            ys = [cell_get(synth, d)[method][f"asr@{t}"] for d in dims]
            ax.plot(dims, ys, ls, color=colors[method], marker=markers[method], ms=5,
                    label=f"{method} ASR@{t}")
    real = results["main"]["real_gtr"]
    for d, cell in real.items():
        if isinstance(cell, dict) and "ADCPE" in cell:
            for method in METHODS:
                first = True
                for t in (1, 5, 10):
                    ax.scatter([int(d)], [cell[method][f"asr@{t}"]], marker="*", s=230,
                               color=colors[method], edgecolor="black", zorder=5, alpha=0.9,
                               label=(f"{method} real GTR" if first else None))
                    first = False
    ax.set_xscale("log")
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("embedding dimension n")
    ax.set_ylabel("ASR@t  (lower = more privacy)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(a) vector-analysis ASR vs dimension\n(paper Fig. 5; synthetic trend, * = real GTR d=768)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower left")

    # (b) Kendall tau / Spearman rho vs dimension
    ax = axes[0, 1]
    for method in METHODS:
        ax.plot(dims, [cell_get(synth, d)[method]["kendall"] for d in dims], "-",
                color=colors[method], marker=markers[method], label=f"{method} Kendall $\\tau$")
        ax.plot(dims, [cell_get(synth, d)[method]["spearman"] for d in dims], ":",
                color=colors[method], marker=markers[method], ms=4, label=f"{method} Spearman $\\rho$")
    for d, cell in real.items():
        if isinstance(cell, dict) and "ADCPE" in cell:
            for method in METHODS:
                ax.scatter([int(d)], [cell[method]["kendall"]], marker="*", s=180,
                           color=colors[method], edgecolor="black", zorder=5)
    ax.set_xscale("log")
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("embedding dimension n")
    ax.set_ylabel("rank correlation (higher = more leakage)")
    ax.set_title("(b) full-ranking agreement")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # (c) distance-reconstruction error vs dimension
    ax = axes[1, 0]
    for method in METHODS:
        ax.plot(dims, [cell_get(synth, d)[method]["dist_err"] for d in dims], "-",
                color=colors[method], marker=markers[method], label=f"{method} dist. error")
    for d, cell in real.items():
        if isinstance(cell, dict) and "ADCPE" in cell:
            for method in METHODS:
                ax.scatter([int(d)], [cell[method]["dist_err"]], marker="*", s=180,
                           color=colors[method], edgecolor="black", zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("embedding dimension n")
    ax.set_ylabel("mean |d_cipher/s - d_plain|")
    ax.set_title("(c) distance-reconstruction error\n(finding A19: measurable CAPRISE advantage)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # (d) beta sweep at d=768 synthetic
    ax = axes[1, 1]
    bs = results.get("beta_sweep", {})
    if bs:
        betas = sorted(float(b) for b in bs)
        for method in METHODS:
            ax.plot(betas, [bs[str(b)][method]["asr@10"] for b in betas], "-",
                    color=colors[method], marker=markers[method], label=f"{method} ASR@10")
            ax.plot(betas, [bs[str(b)][method]["dist_err"] * 10 for b in betas], ":",
                    color=colors[method], marker=markers[method], ms=4,
                    label=f"{method} 10x dist. error")
        ax.set_xlabel("beta (noise scale)")
        ax.legend(fontsize=8)
    ax.set_ylabel("ASR@10 / scaled error")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(d) privacy knob beta (d=768 synthetic)")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "CAPRISE vs ADCPE -- vector-analysis attack (paper Fig. 5 reproduction)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CSV of the main table
# --------------------------------------------------------------------------- #


def write_csv(results: dict, path: Path) -> None:
    rows = ["source,dim,method,m,beta,asr@1,asr@1_std,asr@5,asr@5_std,asr@10,asr@10_std,kendall,spearman,dist_err,dist_err_rel"]
    for section in ("synthetic", "real_gtr"):
        for d, cell in results["main"][section].items():
            if not isinstance(cell, dict) or "ADCPE" not in cell:
                continue
            for method in METHODS:
                c = cell[method]
                rows.append(
                    f"{cell.get('source','')},{d},{method},{c['m']},{results['setup']['beta']},"
                    f"{c['asr@1']:.6f},{c['asr@1_std']:.6f},{c['asr@5']:.6f},{c['asr@5_std']:.6f},"
                    f"{c['asr@10']:.6f},{c['asr@10_std']:.6f},{c['kendall']:.6f},{c['spearman']:.6f},"
                    f"{c['dist_err']:.8f},{c['dist_err_rel']:.6f}"
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def default_config() -> dict:
    return {
        "s": 3,
        "beta": 0.2,
        "m": 5000,
        "nq": 100,
        "seed": 20260101,
        "synth_dims": [768, 1536, 3072],
        "real_dim": 768,
        "ts": [1, 5, 10],
        "m_sweep": [2000, 5000, 10000],
        "beta_sweep": [0.1, 0.2, 0.4],
        "selfcheck_dim": 768,
        "selfcheck_m": 1000,
    }


def quick_config() -> dict:
    """Reduced, fast configuration used by the M10 gate to re-verify the pipeline."""
    return {
        "s": 3,
        "beta": 0.2,
        "m": 1500,
        "nq": 25,
        "seed": 20260101,
        "synth_dims": [768, 1536],
        "real_dim": 0,
        "ts": [1, 5, 10],
        "m_sweep": [],
        "beta_sweep": [],
        "selfcheck_dim": 768,
        "selfcheck_m": 600,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="exp4: vector-analysis ASR (paper Fig. 5)")
    ap.add_argument("--quick", action="store_true", help="reduced config (pipeline check)")
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--nq", type=int, default=None)
    ap.add_argument("--dims", type=str, default=None, help="comma separated synthetic dims")
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--out-json", type=str, default=str(ROOT / "results" / "exp4_vector_analysis.json"))
    ap.add_argument("--out-csv", type=str, default=str(ROOT / "results" / "exp4_vector_analysis.csv"))
    ap.add_argument("--out-fig", type=str, default=str(ROOT / "figures" / "fig5_vector_analysis.png"))
    args = ap.parse_args(argv)

    cfg = quick_config() if args.quick else default_config()
    if args.m is not None:
        cfg["m"] = args.m
        cfg["m_sweep"] = []
    if args.nq is not None:
        cfg["nq"] = args.nq
    if args.dims is not None:
        cfg["synth_dims"] = [int(x) for x in args.dims.split(",")]
    if args.no_real:
        cfg["real_dim"] = 0

    print(f"[exp4] config: {json.dumps(cfg)}", flush=True)
    t0 = time.time()
    results = run_experiment(cfg)
    results["config"] = cfg
    print(f"[exp4] run finished in {time.time() - t0:.1f}s", flush=True)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[exp4] wrote {out_json.relative_to(ROOT)}")

    write_csv(results, Path(args.out_csv))
    print(f"[exp4] wrote {Path(args.out_csv).relative_to(ROOT)}")
    make_figure(results, Path(args.out_fig))
    print(f"[exp4] wrote {Path(args.out_fig).relative_to(ROOT)}")

    s = results["summary"]
    print(
        f"[exp4] main cells: {s['n_main_cells']}, CAPRISE ASR@10 lower in "
        f"{s['n_cells_caprise_asr10_lower']}, trend={s['caprise_lower_asr10_trend']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
