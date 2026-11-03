"""CAPRISE -- Conditional Approximate Distance-Comparison-Preserving Symmetric
Encryption, plus the ADCPE (SAP) baseline it is compared against.

Formulas follow Algorithm 1 of arXiv:2601.12331 verbatim (LaTeX source in
../../output/pdf/source/caprise/ppRAG_FLLM.tex, lines 211-258):

    KeyGen   : s <-$ S, K <-$ {0,1}^k, return (s, K)
    EncDB    : r <-$ {0,1}^l
               randbits1 || randbits2 <- PRF(K, r)
               n <- N(0, I_d)(randbits1),  u <- U_{0,1}(randbits2)
               lambda_ei <- (3/8) * (n * s * beta / ||n||) * u^(1/d)
               e'_i <- s*e_i + lambda_ei ; return (e'_i, r)
    DecDB    : rebuild lambda_ei from (K, r); e_i <- (e'_i - lambda_ei) / s
    EncQ     : eta_eq <- (1/8) * (n * s * beta / ||n||) * u^(1/d)
               e'_q <- s*e_q + eta_eq ; return (e'_q, r)

Noise bounds (paper Eq. 7): ||eta_eq|| < s*beta/8, ||lambda_ei|| < 3*s*beta/8.

ADCPE baseline (Fuchsbauer et al., 2022, "Scale-And-Perturb" SAP):
    lambda_m <- (s*beta/4) * (n/||n||) * u^(1/d)
    c <- s*m + lambda_m
i.e. the standard ADCPE scheme whose noise radius is s*beta/4 for every vector.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .prf import NONCE_BYTES, PRF_KEY_BYTES, coins, new_nonce, prf

FLOAT_DTYPE = np.float64

# --------------------------------------------------------------------------- #
# Keys and parameters
# --------------------------------------------------------------------------- #


@dataclass
class CryptoKey:
    """CAPRISE secret key (s, K) together with public parameters (beta, dim)."""

    s: int
    K: bytes
    beta: float
    dim: int

    def __post_init__(self) -> None:
        if self.s <= 0:
            raise ValueError("scaling factor s must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if len(self.K) != PRF_KEY_BYTES:
            raise ValueError(f"K must be {PRF_KEY_BYTES} bytes")


def KeyGen(
    dim: int,
    beta: float = 0.2,
    s: int | None = None,
    keyspace: tuple[int, int] = (1, 1 << 16),
    rng: np.random.Generator | None = None,
) -> CryptoKey:
    """Sample (s, K).  s is sampled uniformly from the integer keyspace unless
    given explicitly (the paper's privacy experiments fix s = 3)."""
    if s is None:
        lo, hi = keyspace
        if not (1 <= lo <= hi):
            raise ValueError("invalid keyspace")
        s = int(np.random.default_rng().integers(lo, hi + 1)) if rng is None else int(rng.integers(lo, hi + 1))
    if rng is None:
        K = os.urandom(PRF_KEY_BYTES)
    else:
        K = rng.integers(0, 1 << 64, size=2, dtype=np.uint64).tobytes()
    return CryptoKey(s=int(s), K=K, beta=float(beta), dim=int(dim))


# --------------------------------------------------------------------------- #
# Noise radii
# --------------------------------------------------------------------------- #


def noise_radius_db(s: int, beta: float) -> float:
    """Database-side radius: 3*s*beta/8 (paper Eq. 7)."""
    return 3.0 * s * beta / 8.0


def noise_radius_q(s: int, beta: float) -> float:
    """Query-side radius: s*beta/8 (paper Eq. 7)."""
    return s * beta / 8.0


def noise_radius_adcpe(s: int, beta: float) -> float:
    """ADCPE / SAP noise radius: s*beta/4 (Fuchsbauer et al. 2022, Sec. 5.1)."""
    return s * beta / 4.0


# --------------------------------------------------------------------------- #
# Noise sampling
# --------------------------------------------------------------------------- #


def _noise(key: CryptoKey, nonce: bytes, radius: float) -> np.ndarray:
    """Sample lambda from the d-ball of radius ``radius`` (uniform in the ball).

    Direction from a standard Gaussian, radius via u^(1/d) with u ~ U(0,1):
    this is the standard trick to sample a point uniformly inside a d-ball
    (also used by ADCPE, Sec. 5.1).  Norm is strictly < radius because u in (0,1).
    """
    n, u = coins(key.K, nonce, key.dim)
    norm = np.linalg.norm(n)
    # guard against the probability-zero degenerate direction
    while norm == 0.0:
        nonce = os.urandom(NONCE_BYTES)
        n, u = coins(key.K, nonce, key.dim)
        norm = np.linalg.norm(n)
    return (radius * (u ** (1.0 / key.dim)) / norm) * n


def sample_noise(key: CryptoKey, nonce: bytes, radius: float) -> np.ndarray:
    """Public helper (used by tests / experiments) for one noise sample."""
    return _noise(key, nonce, radius)


# --------------------------------------------------------------------------- #
# CAPRISE: EncDB / DecDB / EncQ
# --------------------------------------------------------------------------- #


def EncDB_one(key: CryptoKey, e: np.ndarray) -> tuple[np.ndarray, bytes]:
    """Encrypt a single database embedding e (d,) -> (e', r)."""
    e = np.asarray(e, dtype=FLOAT_DTYPE)
    if e.shape != (key.dim,):
        raise ValueError(f"expected shape ({key.dim},), got {e.shape}")
    r = new_nonce()
    lam = _noise(key, r, noise_radius_db(key.s, key.beta))
    return key.s * e + lam, r


def EncDB(
    key: CryptoKey, E: np.ndarray, nonces: Sequence[bytes] | None = None
) -> tuple[np.ndarray, list[bytes]]:
    """Encrypt a matrix of database embeddings (m, d) -> (E', [r_i]).

    ``nonces`` may be supplied for reproducibility of tests; normally omitted.
    """
    E = np.asarray(E, dtype=FLOAT_DTYPE)
    if E.ndim != 2 or E.shape[1] != key.dim:
        raise ValueError(f"expected shape (m, {key.dim}), got {E.shape}")
    m = E.shape[0]
    if nonces is not None:
        if len(nonces) != m:
            raise ValueError("nonces length mismatch")
        rs = list(nonces)
    else:
        rs = [new_nonce() for _ in range(m)]
    radius = noise_radius_db(key.s, key.beta)
    E_prime = np.empty_like(E)
    for i in range(m):
        E_prime[i] = key.s * E[i] + _noise(key, rs[i], radius)
    return E_prime, rs


def DecDB_one(key: CryptoKey, e_prime: np.ndarray, r: bytes) -> np.ndarray:
    """Decrypt one ciphertext embedding."""
    e_prime = np.asarray(e_prime, dtype=FLOAT_DTYPE)
    lam = _noise(key, r, noise_radius_db(key.s, key.beta))
    return (e_prime - lam) / key.s


def DecDB(
    key: CryptoKey, E_prime: np.ndarray, nonces: Sequence[bytes]
) -> np.ndarray:
    """Decrypt a matrix of ciphertext embeddings (m, d)."""
    E_prime = np.asarray(E_prime, dtype=FLOAT_DTYPE)
    if E_prime.ndim != 2 or E_prime.shape[1] != key.dim:
        raise ValueError(f"expected shape (m, {key.dim}), got {E_prime.shape}")
    if len(nonces) != E_prime.shape[0]:
        raise ValueError("nonces length mismatch")
    radius = noise_radius_db(key.s, key.beta)
    out = np.empty_like(E_prime)
    for i in range(E_prime.shape[0]):
        out[i] = (E_prime[i] - _noise(key, nonces[i], radius)) / key.s
    return out


def EncQ(key: CryptoKey, e_q: np.ndarray) -> tuple[np.ndarray, bytes]:
    """Encrypt a query embedding e_q (d,) -> (e'_q, r)."""
    e_q = np.asarray(e_q, dtype=FLOAT_DTYPE)
    if e_q.shape != (key.dim,):
        raise ValueError(f"expected shape ({key.dim},), got {e_q.shape}")
    r = new_nonce()
    eta = _noise(key, r, noise_radius_q(key.s, key.beta))
    return key.s * e_q + eta, r


def DecQ(key: CryptoKey, e_q_prime: np.ndarray, r: bytes) -> np.ndarray:
    """Decrypt a query ciphertext (convenience for tests)."""
    eta = _noise(key, r, noise_radius_q(key.s, key.beta))
    return (np.asarray(e_q_prime, dtype=FLOAT_DTYPE) - eta) / key.s


# --------------------------------------------------------------------------- #
# ADCPE (SAP) baseline
# --------------------------------------------------------------------------- #


class ADCPE:
    """ADCPE / Scale-And-Perturb baseline (Fuchsbauer et al., 2022).

    Identical structure to CAPRISE except that every vector (database or query)
    uses the same noise radius s*beta/4.  Consequently ADCPE preserves distance
    comparisons among stored database embeddings too, which is exactly the
    leakage CAPRISE is designed to remove.
    """

    @staticmethod
    def Enc(key: CryptoKey, e: np.ndarray, nonce: bytes | None = None) -> tuple[np.ndarray, bytes]:
        """Encrypt one vector (database or query, ADCPE treats them alike)."""
        e = np.asarray(e, dtype=FLOAT_DTYPE)
        r = new_nonce() if nonce is None else nonce
        lam = _noise(key, r, noise_radius_adcpe(key.s, key.beta))
        return key.s * e + lam, r

    @staticmethod
    def EncDB(key: CryptoKey, E: np.ndarray) -> np.ndarray:
        E = np.asarray(E, dtype=FLOAT_DTYPE)
        if E.ndim != 2 or E.shape[1] != key.dim:
            raise ValueError(f"expected shape (m, {key.dim}), got {E.shape}")
        radius = noise_radius_adcpe(key.s, key.beta)
        E_prime = np.empty_like(E)
        for i in range(E.shape[0]):
            E_prime[i] = key.s * E[i] + _noise(key, new_nonce(), radius)
        return E_prime

    @staticmethod
    def EncQ(key: CryptoKey, e_q: np.ndarray) -> tuple[np.ndarray, bytes]:
        e_q = np.asarray(e_q, dtype=FLOAT_DTYPE)
        r = new_nonce()
        eta = _noise(key, r, noise_radius_adcpe(key.s, key.beta))
        return key.s * e_q + eta, r

    @staticmethod
    def Dec(key: CryptoKey, e_prime: np.ndarray, r: bytes) -> np.ndarray:
        lam = _noise(key, r, noise_radius_adcpe(key.s, key.beta))
        return (np.asarray(e_prime, dtype=FLOAT_DTYPE) - lam) / key.s
