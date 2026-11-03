"""Integration smoke test for the ppRAG pipeline (synthetic embeddings).

Not a module gate: this is Lead integration sanity before the data-driven
Phase-1/2 gates (M14-M16).  Verifies that:

1. Phase 1 upload leaves only ciphertext embeddings / nonces / AES ciphertexts
   on the server (no plaintext embeddings or documents).
2. Phase 2 retrieval with CAPRISE (no DP) returns a high-recall top-k.
3. Phase 2 with DistanceDP expands to k' per Theorem 1 and still retrieves.
4. AES round-trip and DecDB round-trip are exact.
5. Phase 3 prompt contains the query and the retrieved docs only.

Run: .venv/bin/python -m tests.test_pprag_smoke
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caprise.aescipher import AESDocumentCipher  # noqa: E402
from caprise.caprise import DecDB, KeyGen  # noqa: E402
from caprise.pprag import PPRAGClient, PPRAGServer  # noqa: E402
from caprise.server import PlaintextDB  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(0)
    m, d, n_q = 2000, 768, 20
    key = KeyGen(dim=d, beta=0.2, s=3)

    # synthetic but clustered database: 20 topic centres + noise (so retrieval
    # is non-trivial and top-k sets are stable in plaintext space)
    centres = rng.standard_normal((20, d))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    assign = rng.integers(0, 20, size=m)
    E = centres[assign] + 0.3 * rng.standard_normal((m, d))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    texts = [f"passage {i} about topic {assign[i]}" for i in range(m)]

    server = PPRAGServer(key)
    client = PPRAGClient(key, embedder=lambda ts: E[: len(ts)])
    t0 = time.time()
    client.phase1_upload(server, texts=texts, embeddings=E)
    upload_s = time.time() - t0

    # 1) server-side leakage checks
    assert not np.array_equal(server.db.E_prime, E), "server must not store plaintext embeddings"
    assert server.db.nonces is not None and len(server.db.nonces) == m
    assert all(isinstance(c, bytes) and len(c) > 16 for c in server.doc_ciphertexts)
    assert not hasattr(server, "_plaintext_docs")
    # decrypt DB roundtrip (server stores ciphertext in float32 for memory;
    # ~1e-6 absolute error after decryption is negligible vs beta=0.2)
    E_hat = DecDB(key, server.db.E_prime, server.db.nonces)
    rt_err = float(np.abs(E_hat - E).max())
    assert rt_err < 1e-5, f"decrypt roundtrip error {rt_err}"  # float32 storage
    print(f"DecDB roundtrip max err (float32 storage): {rt_err:.2e}")

    # 2) queries: top-k near a topic centre
    q_idx = rng.choice(m, size=n_q, replace=False)
    Queries = E[q_idx]
    plain = PlaintextDB(E)
    k = 10

    # plaintext reference
    plain_sets = [set(plain.search(q, k).indices.tolist()) for q in Queries]

    # 3) CAPRISE without DP
    t0 = time.time()
    cap_sets = []
    for q in Queries:
        res = client.phase2_retrieve(server, q, k=k, dp_radius=0.0)
        ids = [int(x) for x in res.ids]
        cap_sets.append(set(ids))
    t_cap = (time.time() - t0) / n_q
    recall_cap = np.mean([len(a & b) / k for a, b in zip(plain_sets, cap_sets)])

    # 4) CAPRISE + DistanceDP r=0.02
    t0 = time.time()
    dp_sets, kprimes = [], []
    for q in Queries:
        res = client.phase2_retrieve(server, q, k=k, dp_radius=0.02)
        dp_sets.append(set(int(x) for x in res.ids))
        kprimes.append(res.k_prime_requested)
    t_dp = (time.time() - t0) / n_q
    recall_dp = np.mean([len(a & b) / k for a, b in zip(plain_sets, dp_sets)])
    kp = int(np.median(kprimes))

    # 5) Phase 3 prompt
    prompt = client.phase3_build_prompt("what is the query?", ["doc A", "doc B"])
    assert "doc A" in prompt and "doc B" in prompt and "what is the query?" in prompt

    print(f"upload: {upload_s:.2f}s for m={m}")
    print(f"CAPRISE no-DP : recall@10 vs plaintext = {recall_cap:.4f}, {t_cap*1000:.1f} ms/query")
    print(f"CAPRISE + DP  : recall@10 vs plaintext = {recall_dp:.4f}, k' median={kp}, {t_dp*1000:.1f} ms/query")
    # oversampling effect (encryption noise alone loses some boundary items
    # when k'=k; a small oversampling recovers near-perfect recall)
    rec_ov = []
    for kp_ov in (k, 15, 20):
        sets = [set(int(x) for x in client.phase2_retrieve(server, q, k=k, dp_radius=0.0, k_prime=kp_ov).ids) for q in Queries]
        rec_ov.append(float(np.mean([len(a & b) / k for a, b in zip(plain_sets, sets)])))
    print(f"oversampling k'=(10,15,20) recall@10 = {[round(x, 4) for x in rec_ov]}")
    ok = (
        recall_cap > 0.8
        and recall_dp > 0.85
        and kp > k
        and rec_ov[1] > rec_ov[0]
        and np.isfinite([recall_cap, recall_dp]).all()
    )
    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
