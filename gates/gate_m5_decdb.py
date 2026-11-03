"""M5 gate: DecDB protocol (Algorithm 1 lines 15-22).

Checks
------
1. Determinism: decrypting twice gives bit-identical results.
2. Correctness: exact inverse of EncDB (max abs error < 1e-9).
3. Dependency on the nonce: a wrong nonce must NOT recover the plaintext.
4. Batch and single-vector decryption agree.
"""

from __future__ import annotations

import numpy as np

from gates.common import Gate, add_src_to_path

add_src_to_path()
from caprise.caprise import DecDB, DecDB_one, EncDB, KeyGen  # noqa: E402

g = Gate("M5", "decdb", owner="lead", paper_ref="Algorithm 1 lines 15-22")

key = KeyGen(dim=768, beta=0.2, s=3)
rng = np.random.default_rng(5)
E = rng.standard_normal((200, 768))
E /= np.linalg.norm(E, axis=1, keepdims=True)
E_prime, nonces = EncDB(key, E)

E_hat1 = DecDB(key, E_prime, nonces)
E_hat2 = DecDB(key, E_prime, nonces)
g.check_true("decrypt_deterministic", bool(np.array_equal(E_hat1, E_hat2)))
err = float(np.abs(E_hat1 - E).max())
g.check_true("exact_inverse_lt_1e9", err < 1e-9, f"max|e_hat-e|={err:.2e}")

# wrong nonce -> wrong plaintext (the noise cannot be cancelled)
wrong = list(nonces)
wrong[0] = bytes(len(nonces[0]))
bad = DecDB(key, E_prime[:1], wrong[:1])[0]
dist_wrong = float(np.linalg.norm(bad - E[0]))
g.check_true("wrong_nonce_not_recovered", dist_wrong > 0.1, f"||e_bad-e||={dist_wrong:.3f} (noise R=0.225, /s)")

# batch and single agree
single = DecDB_one(key, E_prime[3], nonces[3])
g.check_true("batch_equals_single", bool(np.allclose(single, E_hat1[3], rtol=0, atol=0)))

errs = []
try:
    DecDB(key, E_prime, nonces[:-1])
except ValueError:
    errs.append("nonce_count")
g.check("nonce_count_mismatch_rejected", ["nonce_count"], errs)

g.note("DecDB replays PRF(K, r) to rebuild lambda; correctness needs (K, r).")
g.note(f"Measurement scale: wrong-nonce error {dist_wrong:.3f} vs noise radius/s = {0.225/3:.3f}.")
g.finish()
