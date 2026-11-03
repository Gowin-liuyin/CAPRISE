"""M14 gate: ppRAG Phase 1 (Algorithm 2 lines 1-9) -- encrypted upload.

Protocol under test
-------------------
Data owner side:  e_i = gtr(doc_i); (e'_i, r_i) = EncDB(key, e_i);
                  x'_i = AES-GCM(k_AES, doc_i)
Cloud side (honest-but-curious): stores {(e'_i, r_i, x'_i)} only.
The data owner must be able to decrypt everything back (DecDB round-trip,
AES round-trip).

Gate criteria (docs/MODULE_PLAN.md M14)
---------------------------------------
1. server holds only ciphertext embeddings + nonces + AES ciphertext documents;
   in particular it must NOT hold the CAPRISE secret key (s, K), plaintext
   embeddings, or plaintext documents.
2. one nonce per ciphertext row, 128-bit, distinct.
3. DecDB round-trip recovers the embeddings (float32 storage error < 1e-5).
4. AES-GCM round-trip recovers every document exactly.
5. per-row database noise respects the paper's bound ||E'-sE|| < 3*s*beta/8.
6. ids / order preserved; upload statistics reported (m, dim, timings).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from gates.common import Gate, add_src_to_path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
add_src_to_path()

from caprise.aescipher import AESDocumentCipher  # noqa: E402
from caprise.caprise import DecDB, EncDB, KeyGen, noise_radius_db  # noqa: E402
from caprise.pprag import PPRAGClient, PPRAGServer  # noqa: E402

M_SUB = 8000          # real corpus rows used by this gate (all qrels passages are the first 1053 rows)
KEY = dict(dim=768, beta=0.2, s=3)

g = Gate("M14", "phase1", owner="data-retrieval",
         paper_ref="Algorithm 2 lines 1-9 / Phase 1 upload")


def load_real_subset(m_sub: int):
    p_emb = np.load(DATA / "passage_emb.npy")
    p_ids = np.load(DATA / "passage_ids.npy")
    texts: list[str] = []
    with open(DATA / "passages.jsonl", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    m = min(m_sub, len(p_ids))
    return p_emb[:m].astype(np.float64), p_ids[:m], texts[:m]


E, ids, texts = load_real_subset(M_SUB)
m, d = E.shape
g.check_true("00_real_data_loaded", m >= 2000 and d == 768,
             f"real MS MARCO corpus subset: m={m}, d={d}, ids[:3]={ids[:3].tolist()}")

key = KeyGen(dim=KEY["dim"], beta=KEY["beta"], s=KEY["s"])
server = PPRAGServer(key)
client = PPRAGClient(key)

t0 = time.time()
stats = client.phase1_upload(server, texts=texts, embeddings=E, ids=ids.tolist())
upload_s = time.time() - t0

# --------------------------------------------------------------------------- #
# 1. server-side leakage checks
# --------------------------------------------------------------------------- #
server_attrs = set(vars(server).keys())
unexpected = server_attrs - {"key", "db", "doc_ciphertexts"}
g.check_true("01_server_has_only_protocol_fields", not unexpected,
             f"server attributes = {sorted(server_attrs)} (+ db.nonces / db.E_prime inside db)")

# the honest-but-curious server must not own the CAPRISE secret key (s, K)
has_key = hasattr(server, "key")
g.check_true("02_server_holds_no_secret_key", not has_key,
             "PPRAGServer must not store the client's CryptoKey (s, K): the cloud only needs "
             "ciphertexts + nonces for the top-k' search; DecDB stays client-side"
             if has_key else "server has no CryptoKey attribute")

E_prime = server.db.E_prime
g.check_true("03_server_stores_ciphertext_not_plaintext",
             not np.allclose(E_prime.astype(np.float64), E) and not np.allclose(E_prime, KEY["s"] * E),
             f"E' shape={E_prime.shape} dtype={E_prime.dtype}; differs from E and from sE")

plaintext_leaks = [i for i, t in enumerate(texts[:200]) if t.encode("utf-8") in b"".join(server.doc_ciphertexts[:200])]
g.check_true("04_no_plaintext_documents_on_server", not plaintext_leaks,
             f"checked first 200 documents; plaintext leaks={len(plaintext_leaks)}")

doc_ok = all(isinstance(c, bytes) and len(c) >= 16 for c in server.doc_ciphertexts)
g.check_true("05_documents_are_aes_ciphertexts", doc_ok and len(server.doc_ciphertexts) == m,
             f"{len(server.doc_ciphertexts)} AES-GCM ciphertext blobs, all bytes, len>=16")

# --------------------------------------------------------------------------- #
# 2. nonces
# --------------------------------------------------------------------------- #
nonces = server.db.nonces
nonce_ok = nonces is not None and len(nonces) == m and all(len(n) == 16 for n in nonces) and len(set(nonces)) == m
g.check_true("06_one_distinct_128bit_nonce_per_row", nonce_ok,
             f"len={0 if nonces is None else len(nonces)}, unique={0 if nonces is None else len(set(nonces))}/{m}, "
             f"bytes={sorted({len(n) for n in (nonces or [])})}")

# --------------------------------------------------------------------------- #
# 3. DecDB round-trip (float32 ciphertext storage in EncryptedVectorDB)
# --------------------------------------------------------------------------- #
t0 = time.time()
E_hat = DecDB(key, server.db.E_prime, server.db.nonces)
dec_s = time.time() - t0
rt_max = float(np.abs(E_hat - E).max())
rt_rel = float(np.linalg.norm(E_hat - E) / np.linalg.norm(E))
g.check_true("07_decdb_roundtrip", rt_max < 1e-5,
             f"max|E_hat-E|={rt_max:.2e} (<1e-5), relative={rt_rel:.2e}, {dec_s:.2f}s for m={m} "
             f"({m/dec_s:.0f} vec/s)")

# --------------------------------------------------------------------------- #
# 4. AES-GCM round-trip
# --------------------------------------------------------------------------- #
cipher = client.aes_cipher
t0 = time.time()
dec_texts = [cipher.decrypt_text(c) for c in server.doc_ciphertexts]
aes_s = time.time() - t0
mis = sum(1 for a, b in zip(dec_texts, texts) if a != b)
g.check_true("08_aes_roundtrip_exact", mis == 0,
             f"{m - mis}/{m} documents recovered exactly ({m/aes_s:.0f} docs/s)")

# --------------------------------------------------------------------------- #
# 5. database noise bound (Eq. 7): ||lambda_i|| < 3*s*beta/8
# --------------------------------------------------------------------------- #
noise = E_prime.astype(np.float64) - key.s * E
noise_norms = np.linalg.norm(noise, axis=1)
R_db = noise_radius_db(key.s, key.beta)
g.check_true("09_db_noise_bound", bool((noise_norms < R_db).all()),
             f"max||lambda||={noise_norms.max():.6f} < R_db=3sβ/8={R_db:.6f} "
             f"(violations={int((noise_norms >= R_db).sum())})")

# --------------------------------------------------------------------------- #
# 6. ids / order preserved + stats
# --------------------------------------------------------------------------- #
g.check_true("10_ids_preserved", list(server.db.ids) == ids.tolist(),
             f"server ids == upload ids (first5={list(server.db.ids)[:5]})")
g.check_true("11_upload_stats", stats.get("m") == m and stats.get("dim") == d,
             f"phase1_upload returned {stats}")

g.note(f"Phase-1 on real data: m={m}, d={d}, EncDB+AES upload {upload_s:.2f}s "
       f"({m/upload_s:.0f} docs/s end-to-end); DecDB {m/dec_s:.0f} vec/s; AES decrypt {m/aes_s:.0f} docs/s")
g.note("The paper does not report Phase-1 absolute timings beyond the 128-query embedding batch; "
       "hardware here is Apple M5 CPU (no GPU).")
g.note(f"encryption key: s={key.s}, beta={key.beta}, dim={key.dim} (paper secure setting)")

res = g.finish()
print(json.dumps({k: v for k, v in res.items() if k in ("module", "name", "pass")}, indent=2))
sys.exit(0 if res["pass"] else 1)
