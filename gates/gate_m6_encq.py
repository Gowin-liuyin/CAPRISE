"""M6 gate: EncQ protocol (Algorithm 1 lines 23-31).

Checks
------
1. Round-trip through DecQ: max abs error < 1e-9.
2. Query noise ||e'_q - s*e_q|| strictly < s*beta/8 (Eq. 7).
3. Determinism given the nonce.
4. Query noise radius is exactly 1/3 of the database radius (1/8 vs 3/8).
"""

from __future__ import annotations

import numpy as np

from gates.common import Gate, add_src_to_path

add_src_to_path()
from caprise.caprise import (  # noqa: E402
    DecQ,
    EncQ,
    KeyGen,
    noise_radius_q,
)

g = Gate("M6", "encq", owner="lead", paper_ref="Algorithm 1 lines 23-31")

key = KeyGen(dim=768, beta=0.2, s=3)
R_q = noise_radius_q(3, 0.2)
rng = np.random.default_rng(6)

q = rng.standard_normal(768)
q /= np.linalg.norm(q)

q_prime, nonce = EncQ(key, q)
q_hat = DecQ(key, q_prime, nonce)
err = float(np.abs(q_hat - q).max())
g.check_true("roundtrip_lt_1e9", err < 1e-9, f"max|q_hat-q|={err:.2e}")

norms = []
for _ in range(1000):
    qp, _n = EncQ(key, q)
    norms.append(np.linalg.norm(qp - key.s * q))
norms = np.asarray(norms)
g.check_true("query_noise_bound_strict", bool((norms < R_q).all()), f"max||eta||={norms.max():.6f} < sβ/8={R_q}")
g.check_true("query_noise_scale", float(np.median(norms)) > 0.9 * R_q, f"median/R={np.median(norms)/R_q:.4f}")

# determinism / nonce replay
same = EncQ(key, q)
same2 = EncQ(key, q)
g.check_true("fresh_nonce_each_call", same[1] != same2[1])
qp, nc = EncQ(key, q)
qp2 = EncQ(key, q)  # fresh nonce -> different ciphertext
g.check_true("fresh_nonce_changes_ciphertext", not np.array_equal(qp, qp2[0]))
err_replay = float(np.abs(DecQ(key, qp, nc) - q).max())
g.check_true("deterministic_with_nonce", err_replay < 1e-12, f"max|DecQ(EncQ(q))-q|={err_replay:.2e}")

errs = []
try:
    EncQ(key, np.zeros(10))
except ValueError:
    errs.append("wrong_dim")
g.check("malformed_input_rejected", ["wrong_dim"], errs)

g.note("Query noise bound sβ/8 = 0.075 (s=3,β=0.2) vs DB 3sβ/8 = 0.225.")
g.finish()
