"""M15 gate: ppRAG Phase 2 (Algorithm 2 lines 10-21) -- encrypted retrieval.

Protocol under test
-------------------
Client:  e_q --DistanceDP--> e~_q --EncQ--> e'_q   (only e'_q is transmitted)
Server:  Database_Search(e'_q, k') -> top-k' ciphertext ids + nonces
Client:  DecDB the k' candidates -> ReRank by plaintext distance -> top-k docs.

Gate criteria (docs/MODULE_PLAN.md M15)
---------------------------------------
1. only the ciphertext query e'_q reaches the server (never the plaintext e_q);
   k' is derived from Theorem 1 when DistanceDP is active.
2. local decryption + plaintext re-ranking reproduces exactly the top-k among the
   k' candidates returned by the server (deterministic given the candidates).
3. retrieval utility on real MS MARCO data: recall@10 vs plaintext top-10 is high
   for CAPRISE(k'=k), improves (or is not worse) with small oversampling, and stays
   high under DistanceDP(r=0.02) with Theorem-1 k'.
4. timings: plaintext search vs encrypted pipeline per query.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from gates.common import Gate, add_src_to_path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
add_src_to_path()

from caprise.caprise import DecDB_one, KeyGen, noise_radius_q  # noqa: E402
from caprise.pprag import PPRAGClient, PPRAGServer  # noqa: E402
from caprise.server import PlaintextDB  # noqa: E402
from caprise.theorem import k_prime as theorem_k_prime  # noqa: E402

M_SUB = 8000      # same real-data subset as M14 (contains all 1053 qrels passages)
N_Q = 100
K = 10
KEY = dict(dim=768, beta=0.2, s=3)

g = Gate("M15", "phase2", owner="data-retrieval",
         paper_ref="Algorithm 2 lines 10-21 / Phase 2 retrieve / Theorem 1")


def load_real_subset(m_sub: int):
    p_emb = np.load(DATA / "passage_emb.npy")
    p_ids = np.load(DATA / "passage_ids.npy")
    q_emb = np.load(DATA / "query_emb.npy")
    q_ids = np.load(DATA / "query_ids.npy")
    texts: list[str] = []
    with open(DATA / "passages.jsonl", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    m = min(m_sub, len(p_ids))
    return (p_emb[:m].astype(np.float64), p_ids[:m], texts[:m],
            q_emb[:N_Q].astype(np.float64), q_ids[:N_Q])


E, ids, texts, Q, qids = load_real_subset(M_SUB)
m, d = E.shape

key = KeyGen(dim=KEY["dim"], beta=KEY["beta"], s=KEY["s"])
server = PPRAGServer()
client = PPRAGClient(key)
t0 = time.time()
client.phase1_upload(server, texts=texts, embeddings=E, ids=ids.tolist())
upload_s = time.time() - t0

plain = PlaintextDB(E, ids=ids)
plain_sets = [set(int(x) for x in plain.search(q, K).ids) for q in Q]

# spy on the server-side search to prove only e'_q is transmitted
calls: list[dict] = []
_orig_search = server.db.search


def _spy_search(q_prime, k):
    res = _orig_search(q_prime, k)
    calls.append({
        "q_prime": np.array(q_prime, copy=True),
        "k": int(k),
        "indices": np.array(res.indices, copy=True),
        "ids": list(res.ids),
        "distances": np.array(res.distances, copy=True),
    })
    return res


server.db.search = _spy_search


def recall_vs_plaintext(sets):
    return float(np.mean([len(a & b) / K for a, b in zip(plain_sets, sets)]))


def run_config(dp_radius, k_prime=None):
    sets = []
    times = []
    radii = []
    kprimes = []
    for q in Q:
        t0 = time.perf_counter()
        res = client.phase2_retrieve(server, q, k=K, dp_radius=dp_radius, k_prime=k_prime)
        times.append((time.perf_counter() - t0) * 1000.0)
        sets.append(set(int(x) for x in res.ids))
        radii.append(res.perturbation_radius)
        kprimes.append(res.k_prime_requested)
    return sets, np.mean(times), radii, kprimes


def decrypt_candidates(call: dict):
    """Client-side decryption of the candidates of one search call."""
    return np.asarray([DecDB_one(key, server.db.E_prime[i], server.db.nonces[i]) for i in call["indices"]])


# --------------------------------------------------------------------------- #
# config (b): CAPRISE, no DP, k' = k  (strict Algorithm 2 without DP)
# --------------------------------------------------------------------------- #
n_calls_before = len(calls)
sets_b, ms_b, _, kpr_b = run_config(0.0)
rec_b = recall_vs_plaintext(sets_b)
g.check_true("01_caprise_no_dp_k_equal_k_works", all(len(s) == K for s in sets_b),
             f"{len(sets_b)} retrievals x k={K}; recall@10 vs plaintext = {rec_b:.4f}")
g.check_true("02_no_dp_uses_k_prime_equal_k", all(x == K for x in kpr_b), f"k' values = {sorted(set(kpr_b))}")

# --------------------------------------------------------------------------- #
# config (c): small oversampling k' = k + 10
# --------------------------------------------------------------------------- #
sets_c, ms_c, _, kpr_c = run_config(0.0, k_prime=K + 10)
rec_c = recall_vs_plaintext(sets_c)
g.check_true("03_oversampling_k_plus_10", all(x == K + 10 for x in kpr_c),
             f"k'={K+10} requested; recall@10 = {rec_c:.4f} (vs {rec_b:.4f} at k'=k)")

# --------------------------------------------------------------------------- #
# config (d): CAPRISE + DistanceDP(r=0.02), k' from Theorem 1
# --------------------------------------------------------------------------- #
R_DP = 0.02
sets_d, ms_d, radii_d, kpr_d = run_config(R_DP)
rec_d = recall_vs_plaintext(sets_d)
expected_kp = int(math.ceil(theorem_k_prime(K, m, d, R_DP)))
g.check_true("04_dp_k_prime_from_theorem1", all(x == expected_kp for x in kpr_d),
             f"r={R_DP}: k'={sorted(set(kpr_d))} == ceil(Theorem1(k={K}, m={m}, n={d}, r={R_DP})) = {expected_kp}")
g.check_true("05_dp_perturbation_radius_reported", float(np.allclose(radii_d, R_DP, atol=1e-9)),
             f"realised radii = {sorted(set(round(r, 6) for r in radii_d))}")

# --------------------------------------------------------------------------- #
# 1. only ciphertext query is transmitted
# --------------------------------------------------------------------------- #
new_calls = calls[n_calls_before:]
sample = new_calls[0]
q0 = Q[0]
q_prime = sample["q_prime"]
R_q = noise_radius_q(key.s, key.beta)
norm_ok = abs(float(np.linalg.norm(q_prime)) - key.s * float(np.linalg.norm(q0))) < R_q + 1e-9
g.check_true("06_only_ciphertext_query_transmitted",
             q_prime.shape == (d,) and not np.allclose(q_prime, q0) and not np.allclose(q_prime, key.s * q0) and norm_ok,
             f"server received q' (d={q_prime.shape[0]}), | ||q'|| - s||q|| | <= sβ/8={R_q:.4f}; "
             f"||q'||={np.linalg.norm(q_prime):.4f}, s||q||={key.s*np.linalg.norm(q0):.4f}")
g.check_true("07_all_search_calls_are_ciphertext",
             all(c["q_prime"].shape == (d,) for c in new_calls) and len(new_calls) == N_Q * 3,
             f"{len(new_calls)} server search calls, all with a (d,) ciphertext query")

# --------------------------------------------------------------------------- #
# 2. local decryption + plaintext re-rank reproduces the returned top-k
# --------------------------------------------------------------------------- #
# one controlled call: the candidate set returned by the server is decrypted and
# re-ranked by exact plaintext distance; the client's own output must match it.
q_t = Q[0]
n_before = len(calls)
res_t = client.phase2_retrieve(server, q_t, k=K, dp_radius=0.0, k_prime=K + 10)
call_t = calls[n_before]
cand_t = decrypt_candidates(call_t)
plain_d = np.linalg.norm(cand_t - q_t[None, :], axis=1)
order = np.argsort(plain_d, kind="stable")[:K]
expected_ids = [int(call_t["ids"][i]) for i in order]
dist_match = bool(np.allclose(np.asarray(res_t.rerank_distances), plain_d[order], atol=1e-12))
id_match = expected_ids == [int(x) for x in res_t.ids]
g.check_true("08_rerank_reproducible", id_match and dist_match,
             f"manual DecDB+plaintext rerank over the server's k'={K+10} candidates reproduces the "
             f"returned top-{K} ids exactly (ids_match={id_match}, distances_match={dist_match})")

# --------------------------------------------------------------------------- #
# 3. retrieval utility (real MS MARCO data)
# --------------------------------------------------------------------------- #
g.check_true("09_no_dp_recall_high", rec_b >= 0.70,
             f"CAPRISE k'=k recall@10 vs plaintext = {rec_b:.4f} (encryption-noise boundary flips)")
g.check_true("10_oversampling_helps", rec_c >= rec_b - 1e-9,
             f"recall: k'={K} -> {rec_b:.4f}; k'={K+10} -> {rec_c:.4f}")
g.check_true("11_dp_recall_high", rec_d >= 0.80,
             f"CAPRISE+DistanceDP(r={R_DP}), k'={expected_kp}: recall@10 vs plaintext = {rec_d:.4f}")

# --------------------------------------------------------------------------- #
# 4. timings
# --------------------------------------------------------------------------- #
t0 = time.time()
for q in Q:
    plain.search(q, K)
plain_ms = (time.time() - t0) / len(Q) * 1000.0
g.note(f"Phase-2 on real data (m={m}, d={d}, k={K}, {N_Q} queries): "
       f"plaintext search {plain_ms:.3f} ms/query; "
       f"encrypted pipeline (EncQ+search+DecDB+rerank): k'=k {ms_b:.3f} ms, "
       f"k'={K+10} {ms_c:.3f} ms, DP(r={R_DP}, k'={expected_kp}) {ms_d:.3f} ms")
g.note(f"recall@10 vs plaintext: CAPRISE(k'=k)={rec_b:.4f}, CAPRISE(k'={K+10})={rec_c:.4f}, "
       f"CAPRISE+DP(r={R_DP}, k'={expected_kp})={rec_d:.4f}")
g.note(f"Phase-1 upload (re-used from protocol): m={m}, {upload_s:.2f}s")

res = g.finish()
print(json.dumps({k: v for k, v in res.items() if k in ("module", "name", "pass")}, indent=2))
sys.exit(0 if res["pass"] else 1)
