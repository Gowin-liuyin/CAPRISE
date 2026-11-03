"""ppRAG pipeline (paper Algorithm 2): Phase 1 upload, Phase 2 retrieve, Phase 3
augment & generate.

Design notes (see SPEC.md section 3):

* The cloud server is honest-but-curious; it holds only CAPRISE ciphertext
  embeddings, their nonces, and AES-GCM ciphertext documents.
* Phase 2 runs DistanceDP on the query embedding, then EncQ, then a top-k'
  search in ciphertext space, then local decryption and re-ranking.  Because the
  paper does not specify the ReRank model, we re-rank candidates by the *exact
  plaintext distance* to the original query embedding; this isolates the
  retrieval loss caused by encryption + DP from any re-ranker model loss.
* k' is derived from Theorem 1 (``caprise.theorem.k_prime``) using the realised
  DP perturbation radius, unless the caller passes an explicit k'.
* Phase 3 only assembles the augmented prompt; LLM generation is out of scope
  for this reproduction (no GPU / no hosted LLM), and the retrieval quality is
  the object under study.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .aescipher import AESDocumentCipher
from .caprise import CryptoKey, DecDB_one, EncDB, EncQ
from .dp import distance_dp
from .server import EncryptedVectorDB
from .theorem import k_prime as theorem_k_prime

__all__ = ["PPRAGServer", "PPRAGClient", "RetrievalResult"]


class PPRAGServer:
    """Honest-but-curious cloud: ciphertext embeddings + AES ciphertext docs."""

    def __init__(self, key: CryptoKey):
        self.key = key
        self.db: EncryptedVectorDB | None = None
        self.doc_ciphertexts: list[bytes] = []

    def upload(
        self,
        E_prime: np.ndarray,
        nonces: Sequence[bytes],
        doc_ciphertexts: Sequence[bytes],
        ids: Sequence | None = None,
    ) -> None:
        if len(doc_ciphertexts) != E_prime.shape[0]:
            raise ValueError("expected one ciphertext document per embedding")
        self.db = EncryptedVectorDB(E_prime, nonces=nonces, ids=ids)
        self.doc_ciphertexts = list(doc_ciphertexts)


@dataclass
class RetrievalResult:
    ids: list
    docs: list
    cipher_distances: np.ndarray
    rerank_distances: np.ndarray
    k_prime_requested: int
    perturbation_radius: float
    stats: dict = field(default_factory=dict)


class PPRAGClient:
    """User side: owns the CAPRISE key and the AES key, embeds/encrypts, and
    performs local decryption + re-ranking."""

    def __init__(
        self,
        key: CryptoKey,
        embedder: Callable[[Sequence[str]], np.ndarray] | None = None,
        aes_cipher: AESDocumentCipher | None = None,
    ):
        self.key = key
        self.embedder = embedder
        self.aes_cipher = aes_cipher if aes_cipher is not None else AESDocumentCipher()
        # local copies the data owner naturally keeps (for evaluation only; not
        # uploaded to the server)
        self._plaintext_docs: list[str] = []
        self._plaintext_embeddings: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Phase 1: data preparation and upload
    # ------------------------------------------------------------------ #
    def phase1_upload(
        self,
        server: PPRAGServer,
        texts: Sequence[str],
        embeddings: np.ndarray | None = None,
        ids: Sequence | None = None,
    ) -> dict:
        if embeddings is None:
            if self.embedder is None:
                raise ValueError("provide embeddings or an embedder callable")
            embeddings = self.embedder(texts)
        E = np.asarray(embeddings, dtype=np.float64)
        E_prime, nonces = EncDB(self.key, E)
        ciphertexts = [self.aes_cipher.encrypt(t) for t in texts]
        server.upload(E_prime, nonces, ciphertexts, ids=ids)
        self._plaintext_docs = list(texts)
        self._plaintext_embeddings = E
        return {"m": E.shape[0], "dim": E.shape[1]}

    # ------------------------------------------------------------------ #
    # Phase 2: query and retrieval
    # ------------------------------------------------------------------ #
    def phase2_retrieve(
        self,
        server: PPRAGServer,
        query_embedding: np.ndarray,
        k: int = 10,
        dp_radius: float | str | None = 0.0,
        eps: float | None = None,
        k_prime: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> RetrievalResult:
        """Encrypted retrieval returning the decrypted top-k documents."""
        if server.db is None:
            raise RuntimeError("server has no uploaded data")
        q = np.asarray(query_embedding, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.key.dim:
            raise ValueError(f"query dim {q.shape[0]} != key dim {self.key.dim}")

        m = len(server.db)
        r_used = 0.0
        q_for_enc = q
        if dp_radius in (None, 0, 0.0):
            q_for_enc = q
        else:
            q_for_enc, r_used = distance_dp(
                q,
                radius="gamma" if dp_radius == "gamma" else float(dp_radius),
                eps=eps,
                rng=rng,
            )

        if k_prime is None:
            # Theorem 1 with the realised angular perturbation ~ radial r
            k_prime = int(math.ceil(theorem_k_prime(k, m, self.key.dim, r_used))) if r_used > 0 else k

        q_prime, q_nonce = EncQ(self.key, q_for_enc)
        found = server.db.search(q_prime, k_prime)

        # local decryption of candidate embeddings
        E_hat = np.empty((len(found.indices), self.key.dim), dtype=np.float64)
        for j, i in enumerate(found.indices):
            E_hat[j] = DecDB_one(self.key, server.db.E_prime[i], server.db.nonces[i])

        # ReRank: exact plaintext distance to the original query (SPEC.md 3)
        rerank_d = np.linalg.norm(E_hat - q[None, :], axis=1)
        order = np.argsort(rerank_d, kind="stable")[:k]
        chosen = order
        ids_out = [found.ids[i] for i in chosen]
        docs_out = [self.aes_cipher.decrypt_text(server.doc_ciphertexts[found.indices[i]]) for i in chosen]
        return RetrievalResult(
            ids=ids_out,
            docs=docs_out,
            cipher_distances=found.distances[chosen],
            rerank_distances=rerank_d[chosen],
            k_prime_requested=int(k_prime),
            perturbation_radius=float(r_used),
            stats={
                "m": m,
                "k": k,
                "k_prime": int(k_prime),
                "candidates": len(found.indices),
                "dp_radius": dp_radius if not isinstance(dp_radius, str) else "gamma",
                "eps": eps,
            },
        )

    # ------------------------------------------------------------------ #
    # Phase 3: augment & generate
    # ------------------------------------------------------------------ #
    @staticmethod
    def phase3_build_prompt(query: str, retrieved_docs: Sequence[str]) -> str:
        context = "\n\n".join(f"[{i + 1}] {d}" for i, d in enumerate(retrieved_docs))
        return (
            "Answer the question using only the retrieved context below.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
