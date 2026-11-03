"""AES-GCM document encryption for ppRAG Phase 1/2 (paper Algorithm 2).

The paper says the user uploads ``x'_i <- Enc_AES(k_AES, x_i)`` and the cloud
returns the encrypted documents; the client decrypts locally.  AES-256-GCM is a
standard authenticated-encryption instantiation.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

__all__ = ["AESKey", "AESDocumentCipher"]

DEFAULT_KEY_BYTES = 32  # AES-256
NONCE_BYTES = 12        # GCM standard nonce length


@dataclass
class AESKey:
    key: bytes

    @classmethod
    def generate(cls) -> "AESKey":
        return cls(AESGCM.generate_key(bit_length=DEFAULT_KEY_BYTES * 8))


class AESDocumentCipher:
    """AES-256-GCM wrapper; ciphertext blob = nonce || ciphertext."""

    def __init__(self, key: AESKey | bytes | None = None):
        if key is None:
            key = AESKey.generate()
        if isinstance(key, AESKey):
            key = key.key
        if len(key) not in (16, 24, 32):
            raise ValueError("AES key must be 16/24/32 bytes")
        self._aead = AESGCM(key)
        self.key = key

    def encrypt(self, plaintext: str | bytes) -> bytes:
        data = plaintext.encode("utf-8") if isinstance(plaintext, str) else plaintext
        nonce = os.urandom(NONCE_BYTES)
        return nonce + self._aead.encrypt(nonce, data, None)

    def decrypt(self, blob: bytes) -> bytes:
        nonce, ct = blob[:NONCE_BYTES], blob[NONCE_BYTES:]
        return self._aead.decrypt(nonce, ct, None)

    def decrypt_text(self, blob: bytes) -> str:
        return self.decrypt(blob).decode("utf-8")


def b64encode(blob: bytes) -> str:
    return base64.b64encode(blob).decode("ascii")


def b64decode(text: str) -> bytes:
    return base64.b64decode(text)
