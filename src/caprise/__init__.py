"""CAPRISE / ppRAG reproduction package.

Paper: "Efficient Privacy-Preserving Retrieval Augmented Generation with
Distance-Preserving Encryption" (arXiv:2601.12331).

See ../../SPEC.md for the exact formulas and implementation conventions.
"""

from .caprise import (
    CryptoKey,
    KeyGen,
    EncDB,
    DecDB,
    EncQ,
    ADCPE,
    noise_radius_db,
    noise_radius_q,
    noise_radius_adcpe,
)
from .prf import prf, coins, new_nonce

__all__ = [
    "CryptoKey",
    "KeyGen",
    "EncDB",
    "DecDB",
    "EncQ",
    "ADCPE",
    "noise_radius_db",
    "noise_radius_q",
    "noise_radius_adcpe",
    "prf",
    "coins",
    "new_nonce",
]

__version__ = "0.1.0"
