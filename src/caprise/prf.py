"""PRF instantiation for CAPRISE (TapeGen-style HMAC-SHA256 counter mode).

The paper only requires PRF : {0,1}^k x {0,1}^l -> {0,1}^* (Sec. III-B1 and
Algorithm 1).  We instantiate it with HMAC-SHA256 in counter mode, the same
construction family used by the TapeGen PRF (Bellare et al., CRYPTO 2009) that
the ADCPE paper (Fuchsbauer et al., 2022) suggests.

Two design points that make decryption possible (see SPEC.md 2.2):

* ``coins(K, r, d)`` deterministically re-derives the Gaussian direction ``n``
  and the uniform scalar ``u`` from the nonce ``r``.  The Gaussian vector is
  drawn by a PCG64 generator seeded with 128 PRF bits; the uniform scalar is
  the next 128 PRF bits mapped into (0, 1).
* The nonce ``r`` is a public value stored/transmitted next to the ciphertext.
"""

from __future__ import annotations

import hashlib
import hmac
import os

import numpy as np

PRF_KEY_BYTES = 16  # k = 128 bit
NONCE_BYTES = 16    # l = 128 bit


def prf(key: bytes, msg: bytes, outlen: int) -> bytes:
    """PRF(K, msg) -> outlen pseudorandom bytes (HMAC-SHA256 counter mode)."""
    if outlen <= 0:
        raise ValueError("outlen must be positive")
    out = bytearray()
    counter = 0
    while len(out) < outlen:
        block = hmac.new(key, msg + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:outlen])


def new_nonce(rng: np.random.Generator | None = None) -> bytes:
    """Fresh nonce r <-$ {0,1}^l."""
    if rng is None:
        return os.urandom(NONCE_BYTES)
    return rng.integers(0, 1 << 64, size=2, dtype=np.uint64).tobytes()


def coins(key: bytes, nonce: bytes, dim: int) -> tuple[np.ndarray, float]:
    """Derive (n, u) from PRF(K, r); n in R^dim standard Gaussian, u in (0,1)."""
    bits = prf(key, nonce, 32)
    randbits1, randbits2 = bits[:16], bits[16:]

    # randbits1 -> deterministic standard Gaussian direction.
    seed = int.from_bytes(randbits1, "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    n = rng.standard_normal(dim)

    # randbits2 -> deterministic uniform scalar in (0, 1).
    u = (int.from_bytes(randbits2, "big") + 1) / float(1 << 128)
    return n, u


def unit_direction(n: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(n)
    if norm == 0.0:  # probability zero, but keep the sampler total
        raise ValueError("degenerate Gaussian direction has zero norm")
    return n / norm
