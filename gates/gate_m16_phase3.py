"""M16 gate: ppRAG Phase 3 (Algorithm 2 lines 22-27) -- prompt assembly.

Protocol under test
-------------------
input  = query q + retrieved plaintext documents {x_1..x_k}
prompt = "Answer the question using only the retrieved context below.
          Context: [1] x_1 ... [k] x_k
          Question: q
          Answer:"
(LLM generation itself is out of scope: no GPU / no hosted LLM, see SPEC.md §3.)

Gate criteria (docs/MODULE_PLAN.md M16)
---------------------------------------
1. the prompt contains the query and exactly the k documents retrieved in Phase 2,
   in retrieval order, with [1]..[k] markers;
2. it contains no documents that were not retrieved (no whole-database leakage);
3. the prompt is exactly the template filled with query + docs (no other data,
   no embeddings/numbers leak); documents are decrypted AES plaintexts that
   belong to the corpus;
4. the server side never sees the prompt / query / plaintext docs (Phase 2
   boundaries preserved).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from gates.common import Gate, add_src_to_path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
add_src_to_path()

from caprise.caprise import KeyGen  # noqa: E402
from caprise.pprag import PPRAGClient, PPRAGServer  # noqa: E402

M_SUB = 4000   # gate speed: Phase-1 upload on real data (all 1053 qrels passages are in rows 0..1052)
N_Q = 20
K = 10
KEY = dict(dim=768, beta=0.2, s=3)

g = Gate("M16", "phase3", owner="data-retrieval",
         paper_ref="Algorithm 2 lines 22-27 / Phase 3 prompting the LLM")

p_emb = np.load(DATA / "passage_emb.npy")
p_ids = np.load(DATA / "passage_ids.npy")
q_emb = np.load(DATA / "query_emb.npy")
q_ids = np.load(DATA / "query_ids.npy")
qtexts = [json.loads(line)["text"] for line in open(DATA / "queries.jsonl", encoding="utf-8")]
texts: list[str] = []
with open(DATA / "passages.jsonl", encoding="utf-8") as f:
    for line in f:
        texts.append(json.loads(line)["text"])

m = min(M_SUB, len(p_ids))
E = p_emb[:m].astype(np.float64)
ids = p_ids[:m]
docs_all = texts[:m]
Q = q_emb[:N_Q].astype(np.float64)
Q_text = qtexts[:N_Q]

key = KeyGen(dim=KEY["dim"], beta=KEY["beta"], s=KEY["s"])
server = PPRAGServer()
client = PPRAGClient(key)
client.phase1_upload(server, texts=docs_all, embeddings=E, ids=ids.tolist())

# Phase 2 retrieval for N_Q queries -> Phase 3 prompt
retrieved = []
for q in Q:
    res = client.phase2_retrieve(server, q, k=K, dp_radius=0.0)
    retrieved.append(res)

r0 = retrieved[0]
docs0 = list(r0.docs)
g.check_true("00_phase2_to_phase3_handoff", len(docs0) == K and all(isinstance(t, str) for t in docs0),
             f"{len(docs0)} plaintext docs returned by Phase 2 for query {int(q_ids[0])}")

prompt = client.phase3_build_prompt(Q_text[0], docs0)

# --------------------------------------------------------------------------- #
# 1. prompt contains query + exactly the retrieved docs, in order
# --------------------------------------------------------------------------- #
g.check_true("01_prompt_contains_query", Q_text[0] in prompt,
             f"query {Q_text[0][:60]!r} present")

all_docs_present = all(f"[{i + 1}] {d}" in prompt for i, d in enumerate(docs0))
order_ok = [prompt.find(f"[{i + 1}] {d}") for i, d in enumerate(docs0)] == sorted(
    prompt.find(f"[{i + 1}] {d}") for i, d in enumerate(docs0))
g.check_true("02_prompt_contains_all_retrieved_docs_in_order", all_docs_present and order_ok,
             f"{len(docs0)}/{K} docs present with [i] markers in retrieval order")

markers = [f"[{i + 1}]" for i in range(K)]
g.check_true("03_exactly_k_documents", all(m in prompt for m in markers) and f"[{K + 1}]" not in prompt,
             f"markers [1]..[{K}] present, [{K + 1}] absent")

# --------------------------------------------------------------------------- #
# 2. no non-retrieved document leaks into the prompt
# --------------------------------------------------------------------------- #
retrieved_set = set(d.strip() for d in docs0)
rng = np.random.default_rng(0)
candidates = [docs_all[i] for i in rng.choice(m, size=200, replace=False)]
non_retrieved = [d for d in candidates if d.strip() not in retrieved_set][:20]
leaks = [d for d in non_retrieved if d.strip() in prompt]
g.check_true("04_no_non_retrieved_documents", not leaks,
             f"checked {len(non_retrieved)} random non-retrieved corpus docs; leaks={len(leaks)} "
             f"(prompt must contain only the retrieved documents)")

# --------------------------------------------------------------------------- #
# 3. prompt is exactly the template filled with query + docs
# --------------------------------------------------------------------------- #
context = "\n\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs0))
expected = (
    "Answer the question using only the retrieved context below.\n\n"
    f"Context:\n{context}\n\nQuestion: {Q_text[0]}\nAnswer:"
)
g.check_true("05_prompt_exact_template", prompt == expected,
             f"prompt == template(query, retrieved docs) exactly; len={len(prompt)} chars "
             f"(query {len(Q_text[0])} + {len(docs0)} docs {sum(len(d) for d in docs0)} chars)")
g.check_true("06_no_extra_payload", prompt.endswith("Answer:") and "e'" not in prompt and "embedding" not in prompt.lower(),
             "ends with 'Answer:'; no ciphertext/embedding identifiers leaked")

# --------------------------------------------------------------------------- #
# 4. retrieved docs are real decrypted corpus documents; server saw none of them
# --------------------------------------------------------------------------- #
corpus_text_set = set(t.strip() for t in docs_all)
in_corpus = sum(1 for d in docs0 if d.strip() in corpus_text_set)
g.check_true("07_docs_are_decrypted_corpus_texts", in_corpus == K,
             f"{in_corpus}/{K} decrypted docs match corpus texts")

server_blob = b"".join(server.doc_ciphertexts)
server_attrs = set(vars(server).keys())
server_plain_leak = Q_text[0].encode() in server_blob or any(d.encode() in server_blob for d in docs0[:50])
g.check_true("08_server_never_sees_prompt_plaintext", not server_plain_leak and "prompt" not in server_attrs,
             f"server only has {sorted(server_attrs)}; query/documents not present in its AES blobs")

# --------------------------------------------------------------------------- #
# 5. batch of real prompts (all N_Q queries) behave the same; empty-context edge
# --------------------------------------------------------------------------- #
ok_batch = 0
for res, qt in zip(retrieved, Q_text):
    p = client.phase3_build_prompt(qt, res.docs)
    if qt in p and all(f"[{i + 1}] {d}" in p for i, d in enumerate(res.docs)) and len(res.docs) == K:
        ok_batch += 1
g.check_true("09_batch_prompts_ok", ok_batch == N_Q, f"{ok_batch}/{N_Q} prompts correctly assembled")

empty_prompt = client.phase3_build_prompt("empty?", [])
g.check_true("10_empty_context_edge_case", "empty?" in empty_prompt and "Context:\n\n" in empty_prompt,
             "empty retrieval list produces a well-formed prompt")

g.note(f"Phase-3 gate on real data: Phase-1 upload m={m}, Phase-2 k={K} over {N_Q} queries, "
       f"prompt template filled with decrypted documents only")
g.note("LLM generation is out of scope (SPEC.md §3); prompt assembly is the verifiable Phase-3 object.")

res = g.finish()
print(json.dumps({k: v for k, v in res.items() if k in ("module", "name", "pass")}, indent=2))
sys.exit(0 if res["pass"] else 1)
