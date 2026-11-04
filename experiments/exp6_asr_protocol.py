"""ASR protocol study (supplementary): can the paper's Fig.5 curve be reproduced?

The paper's Fig.5 shows ASR rising from ~53% to ~89% (CAPRISE) and ~66% to ~95%
(ADCPE) as the "embedding size" grows from 64 to 2048, but the paper does not say
which data/embedding protocol produced those 12 sizes (GTR emits 768 dims only;
finding F11/A13).  This script tests the natural candidate protocols on real MS
MARCO GTR embeddings and compares every cell with the digitised paper values:

  H1_proj_renorm   random Gaussian projection 768->d, then L2 normalisation
                   (Johnson-Lindenstrauss; vectors stay on the unit sphere)
  H2_trunc_raw     first min(d,768) coordinates, NO renormalisation
                   (norm shrinks as sqrt(d/768) below 768; zero-padded above)
  H3_trunc_renorm  first min(d,768) coordinates, then L2 normalisation
                   (zero-padded above 768)

Attack protocol follows the paper: a database vector is used as the test query,
the database is encrypted with CAPRISE (s=3, beta=0.2) and ADCPE (s*beta/4), the
distances are recomputed in ciphertext space and ranked, and ASR@t is the top-t
overlap with the plaintext ranking.

Run:  .venv/bin/python experiments/exp6_asr_protocol.py
Writes: results/exp6_asr_protocol.json
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caprise.caprise import ADCPE, EncDB, KeyGen  # noqa: E402
from caprise.metrics import kendall_tau, top_t_overlap  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "exp6_asr_protocol.json"
EMB = ROOT / "data" / "passage_emb.npy"
EMB_BACKUP = ROOT / "data" / "backup_m50k" / "passage_emb.npy"

PAPER_DIMS = (64, 128, 256, 312, 384, 512, 768, 896, 1024, 1536, 1792, 2048)
S, BETA = 3, 0.2
NQ = 100
M_MAIN = 5000
M_SWEEP = (5000, 20000, 100000)
M_SWEEP_DIMS = (64, 768)
SEED = 20260101


def paper_fig5() -> dict:
    art = RESULTS / "exp4_vector_analysis.json"
    if art.exists():
        d = json.loads(art.read_text(encoding="utf-8"))
        ref = d.get("paper_fig5_reference")
        if ref:
            return {"dims": ref["dims"], "CAPRISE": ref["CAPRISE"], "ADCPE": ref["ADCPE"]}
    # fallback: values digitised in the M10 notes
    return {
        "dims": list(PAPER_DIMS),
        "CAPRISE": [53.5, 60.6, 70.2, 73.3, 75.1, 76.3, 83.1, 84.6, 84.8, 86.3, 87.5, 88.8],
        "ADCPE": [65.8, 76.5, 78.2, 78.7, 78.5, 85.8, 87.5, 87.3, 88.3, 91.9, 91.7, 94.6],
    }


def load_embeddings() -> tuple[np.ndarray, str]:
    for path in (EMB, EMB_BACKUP):
        if path.exists():
            E = np.load(path).astype(np.float32)
            return E, str(path.relative_to(ROOT))
    raise SystemExit("no passage embeddings found")


def project(E: np.ndarray, d: int, seed: int) -> np.ndarray:
    """Gaussian random projection 768->d, then L2 normalise."""
    rng = np.random.default_rng(seed + d)
    R = rng.standard_normal((E.shape[1], d), dtype=np.float32) / math.sqrt(d)
    P = E @ R
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return P.astype(np.float32)


def truncate(E: np.ndarray, d: int, renorm: bool) -> np.ndarray:
    """First min(d,768) coordinates; zero-pad above 768; optionally renormalise."""
    if d <= E.shape[1]:
        T = E[:, :d].copy()
    else:
        T = np.zeros((E.shape[0], d), dtype=np.float32)
        T[:, : E.shape[1]] = E
    if renorm:
        norms = np.linalg.norm(T, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        T = T / norms
    return T.astype(np.float32)


def asr_cell(E: np.ndarray, key, nq: int, seed: int) -> dict:
    """Encrypt the DB with CAPRISE and ADCPE, rank by ciphertext distance, ASR@t."""
    rng = np.random.default_rng(seed)
    m = E.shape[0]
    q_idx = rng.choice(m, size=min(nq, m), replace=False)
    E64 = E.astype(np.float64)

    # deterministic nonces: one stream per row, shared by both methods
    raw = np.random.default_rng(seed + 7).integers(0, 1 << 63, size=(m, 2), dtype=np.int64)
    nonces = [raw[i].tobytes() for i in range(m)]

    Ep, _ = EncDB(key, E64, nonces=nonces)
    Ea = np.empty_like(Ep)
    for i in range(m):
        Ea[i], _ = ADCPE.Enc(key, E64[i], nonce=nonces[i])

    def _metrics(prime: np.ndarray) -> dict:
        asr = {1: [], 5: [], 10: []}
        taus = []
        for j in q_idx:
            d_plain = np.linalg.norm(E64 - E64[j], axis=1)
            d_cipher = np.linalg.norm(prime - prime[j], axis=1)
            d_plain[j] = np.inf
            d_cipher[j] = np.inf
            order_p = np.argsort(d_plain, kind="stable")
            order_c = np.argsort(d_cipher, kind="stable")
            for t in asr:
                asr[t].append(top_t_overlap(order_p, order_c, t))
            taus.append(kendall_tau(d_plain[order_p], d_cipher[order_p]))
        return {
            **{f"asr@{t}": float(np.mean(v)) for t, v in asr.items()},
            "asr@10_std": float(np.std(asr[10])),
            "kendall": float(np.mean(taus)),
            "n_queries": int(len(q_idx)),
            "m": int(m),
            "dim": int(E.shape[1]),
        }

    cap = _metrics(Ep)
    adc = _metrics(Ea)
    gap = adc["asr@10"] - cap["asr@10"]
    return {"CAPRISE": cap, "ADCPE": adc, "asr10_gap_adcpe_minus_caprise": gap}


def compare_to_paper(cells: dict, paper: dict) -> dict:
    rows = []
    for d in PAPER_DIMS:
        key = str(d)
        if key not in cells:
            continue
        i = paper["dims"].index(d)
        rows.append(
            {
                "d": d,
                "paper_caprise": paper["CAPRISE"][i],
                "paper_adcpe": paper["ADCPE"][i],
                "our_caprise": cells[key]["CAPRISE"]["asr@10"] * 100.0,
                "our_adcpe": cells[key]["ADCPE"]["asr@10"] * 100.0,
                "delta_caprise_pp": cells[key]["CAPRISE"]["asr@10"] * 100.0 - paper["CAPRISE"][i],
                "delta_adcpe_pp": cells[key]["ADCPE"]["asr@10"] * 100.0 - paper["ADCPE"][i],
            }
        )
    if not rows:
        return {"rows": []}
    our_c = np.asarray([r["our_caprise"] for r in rows])
    our_a = np.asarray([r["our_adcpe"] for r in rows])
    pap_c = np.asarray([r["paper_caprise"] for r in rows])
    pap_a = np.asarray([r["paper_adcpe"] for r in rows])
    from scipy.stats import spearmanr

    return {
        "rows": rows,
        "mean_abs_delta_caprise_pp": float(np.mean(np.abs(our_c - pap_c))),
        "mean_abs_delta_adcpe_pp": float(np.mean(np.abs(our_a - pap_a))),
        "spearman_caprise_vs_paper": float(spearmanr(our_c, pap_c).statistic),
        "spearman_adcpe_vs_paper": float(spearmanr(our_a, pap_a).statistic),
        "our_caprise_range_pp": [float(our_c.min()), float(our_c.max())],
        "paper_caprise_range_pp": [float(pap_c.min()), float(pap_c.max())],
    }


def main() -> int:
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    E_full, source = load_embeddings()
    paper = paper_fig5()
    print(f"[exp6] real GTR embeddings: {E_full.shape} from {source}", flush=True)

    rng = np.random.default_rng(SEED)
    idx_main = rng.choice(E_full.shape[0], size=M_MAIN, replace=False)
    E_main = E_full[np.sort(idx_main)]
    key = KeyGen(dim=768, beta=BETA, s=S)

    protocols = {
        "H1_proj_renorm": lambda d: project(E_main, d, SEED),
        "H2_trunc_raw": lambda d: truncate(E_main, d, renorm=False),
        "H3_trunc_renorm": lambda d: truncate(E_main, d, renorm=True),
    }
    cells: dict = {}
    comparisons: dict = {}
    for name, fn in protocols.items():
        cells[name] = {}
        print(f"[exp6] protocol {name} ...", flush=True)
        for d in PAPER_DIMS:
            Ed = fn(d)
            key_d = KeyGen(dim=d, beta=BETA, s=S)
            cell = asr_cell(Ed, key_d, NQ, SEED)
            cells[name][str(d)] = cell
            print(f"[exp6]   {name} d={d:5d}: CAPRISE ASR@10={cell['CAPRISE']['asr@10']:.3f} "
                  f"ADCPE={cell['ADCPE']['asr@10']:.3f}", flush=True)
        comparisons[name] = compare_to_paper(cells[name], paper)

    # m sweep on a few dims (projection protocol + raw truncation at d=768 == real data)
    m_sweep: dict = {}
    for d in M_SWEEP_DIMS:
        m_sweep[str(d)] = {}
        for m in M_SWEEP:
            if m > E_full.shape[0]:
                continue
            rng_m = np.random.default_rng(SEED + m)
            sub = E_full[np.sort(rng_m.choice(E_full.shape[0], size=m, replace=False))]
            Ed = sub if d == 768 else project(sub, d, SEED)
            key_d = KeyGen(dim=d, beta=BETA, s=S)
            cell = asr_cell(Ed, key_d, min(NQ, 50), SEED)
            m_sweep[str(d)][str(m)] = cell
            print(f"[exp6]   m={m:6d} d={d:5d}: CAPRISE={cell['CAPRISE']['asr@10']:.3f} "
                  f"ADCPE={cell['ADCPE']['asr@10']:.3f}", flush=True)

    # beta sweep on the truncation protocol (the paper does not state beta for Fig.5)
    beta_sweep: dict = {}
    for beta in (0.2, 0.3, 0.4):
        cells_b: dict = {}
        for d in PAPER_DIMS:
            Ed = truncate(E_main, d, renorm=False)
            cells_b[str(d)] = asr_cell(Ed, KeyGen(dim=d, beta=beta, s=S), 60, SEED)
        beta_sweep[str(beta)] = {
            "cells": cells_b,
            "comparison_to_paper": compare_to_paper(cells_b, paper),
        }
        cmp_b = beta_sweep[str(beta)]["comparison_to_paper"]
        print(f"[exp6] beta={beta}: mean|dC|={cmp_b['mean_abs_delta_caprise_pp']:.1f}pp "
              f"rhoC={cmp_b['spearman_caprise_vs_paper']:.3f} | mean|dA|="
              f"{cmp_b['mean_abs_delta_adcpe_pp']:.1f}pp rhoA={cmp_b['spearman_adcpe_vs_paper']:.3f}",
              flush=True)

    summary = {
        name: {
            "mean_abs_delta_caprise_pp": comparisons[name]["mean_abs_delta_caprise_pp"],
            "spearman_caprise_vs_paper": comparisons[name]["spearman_caprise_vs_paper"],
            "our_caprise_range_pp": comparisons[name]["our_caprise_range_pp"],
        }
        for name in protocols
    }
    best = min(summary, key=lambda k: summary[k]["mean_abs_delta_caprise_pp"])
    result = {
        "experiment": "exp6_asr_protocol",
        "paper_ref": "arXiv:2601.12331 Fig.5 / Sec. V-C (ASR vs embedding size); data protocol not stated (F11/A13)",
        "question": (
            "Which data protocol reproduces the paper's Fig.5 ASR trend (CAPRISE 53.5->88.8%, "
            "ADCPE 65.8->94.6% over the 12 sizes 64..2048)?"
        ),
        "setup": {
            "source": source,
            "n_corpus": int(E_full.shape[0]),
            "m_main": M_MAIN,
            "n_queries": NQ,
            "s": S,
            "beta": BETA,
            "seed": SEED,
            "protocols": {
                "H1_proj_renorm": "Gaussian projection 768->d then L2 normalise",
                "H2_trunc_raw": "first min(d,768) coords, no renormalisation, zero-pad above 768",
                "H3_trunc_renorm": "first min(d,768) coords, L2 normalise, zero-pad above 768",
            },
            "attack": "DB vector as test query; encrypt DB with CAPRISE (3sβ/8) and ADCPE (sβ/4); rank ciphertext distances; ASR@t = top-t overlap",
        },
        "paper_fig5": paper,
        "cells": cells,
        "comparison_to_paper": comparisons,
        "beta_sweep_truncation": beta_sweep,
        "m_sweep": m_sweep,
        "summary": {
            "per_protocol": summary,
            "best_protocol_by_mean_abs_delta": best,
            "paper_caprise_range_pp": comparisons[best]["paper_caprise_range_pp"],
            "our_caprise_range_pp": comparisons[best]["our_caprise_range_pp"],
            "note": (
                "The paper's protocol is unknown (F11). A protocol reproduces the trend only if its "
                "range is comparable to the paper's (~53->89 pp) and the Spearman correlation across "
                "the 12 sizes is high; the synthetic-uniform protocol (exp4) is flat by construction."
            ),
        },
        "runtime_seconds": round(time.time() - t0, 1),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[exp6] wrote {OUT}; best protocol by mean|Δ| = {best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
