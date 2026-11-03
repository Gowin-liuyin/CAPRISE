"""Vector stores: plaintext baseline and CAPRISE-encrypted server-side store.

The server is honest-but-curious: it stores only ciphertext embeddings plus the
per-vector nonces, and answers top-k' nearest-neighbour queries in the encrypted
space (paper Algorithm 2, ``Database_Search``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = ["SearchResult", "PlaintextDB", "EncryptedVectorDB"]


@dataclass
class SearchResult:
    indices: np.ndarray        # row indices into the stored matrix
    distances: np.ndarray      # squared L2 distances in the search space
    ids: list


def _topk_sqdist(q: np.ndarray, M: np.ndarray, k: int, row_norms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-k by squared L2 distance using the Gram-trick; vectorised."""
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, M.shape[0])
    # ||q - m||^2 = ||m||^2 - 2 q.m + ||q||^2  (||q||^2 is a constant shift)
    scores = row_norms - 2.0 * (M @ q)
    idx = np.argpartition(scores, k - 1)[:k]
    idx = idx[np.argsort(scores[idx], kind="stable")]
    return idx, scores[idx] + float(q @ q)


class PlaintextDB:
    """Plaintext reference store (ground truth for retrieval quality)."""

    def __init__(self, embeddings: np.ndarray, ids: Sequence | None = None):
        E = np.asarray(embeddings, dtype=np.float64)
        if E.ndim != 2:
            raise ValueError("embeddings must be (m, d)")
        self.embeddings = E
        self.ids = list(range(E.shape[0])) if ids is None else list(ids)
        self.row_norms = np.einsum("ij,ij->i", E, E)

    def search(self, q: np.ndarray, k: int) -> SearchResult:
        idx, d2 = _topk_sqdist(np.asarray(q, dtype=np.float64), self.embeddings, k, self.row_norms)
        return SearchResult(indices=idx, distances=d2, ids=[self.ids[i] for i in idx])

    def __len__(self) -> int:
        return self.embeddings.shape[0]


class EncryptedVectorDB:
    """Server-side store of CAPRISE ciphertext embeddings (or ADCPE ones)."""

    def __init__(
        self,
        E_prime: np.ndarray,
        nonces: Sequence[bytes] | None = None,
        ids: Sequence | None = None,
        dtype=np.float32,
    ):
        M = np.asarray(E_prime, dtype=dtype)
        if M.ndim != 2:
            raise ValueError("E_prime must be (m, d)")
        self.E_prime = M
        self.row_norms = np.einsum("ij,ij->i", M, M, dtype=np.float64)
        self.nonces = None if nonces is None else list(nonces)
        self.ids = list(range(M.shape[0])) if ids is None else list(ids)

    def search(self, q_prime: np.ndarray, k: int) -> SearchResult:
        q = np.asarray(q_prime, dtype=self.E_prime.dtype)
        idx, d2 = _topk_sqdist(q, self.E_prime, k, self.row_norms)
        return SearchResult(indices=idx, distances=d2, ids=[self.ids[i] for i in idx])

    def __len__(self) -> int:
        return self.E_prime.shape[0]
