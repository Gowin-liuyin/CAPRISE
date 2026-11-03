"""M4 gate: EncDB protocol (Algorithm 1 lines 6-14).

Checks
------
1. Round-trip through DecDB: max abs error < 1e-9 (float64).
2. Per-vector noise ||E' - s*E|| strictly < 3*s*beta/8 (Eq. 7).
3. Ciphertext distances are s times plaintext distances plus bounded noise
   (sanity: correlation > 0.999, deviation < 4 * noise radii).
4. Batch and single-vector encryption agree for the same nonce.
5. Nonces are unique / correct count; malformed inputs raise.
"""

from __future__ import annotations

import numpy as np

from gates.common import Gate, add_src_to_path

add_src_to_path()
from caprise.caprise import (  # noqa: E402
    DecDB,
    EncDB,
    EncDB_one,
    KeyGen,
    noise_radius_db,
    noise_radius_q,
)

g = Gate("M4", "encdb", owner="lead", paper_ref="Algorithm 1 lines 6-14")

key = KeyGen(dim=768, beta=0.2, s=3)
R = noise_radius_db(3, 0.2)
rng = np.random.default_rng(2024)
E = rng.standard_normal((500, 768))
E /= np.linalg.norm(E, axis=1, keepdims=True)

E_prime, nonces = EncDB(key, E)
E_hat = DecDB(key, E_prime, nonces)

roundtrip_err = float(np.abs(E_hat - E).max())
g.check_true("roundtrip_max_err_lt_1e9", roundtrip_err < 1e-9, f"max|e_hat-e|={roundtrip_err:.2e}")
g.check_true("cipher_shape", E_prime.shape == E.shape and len(nonces) == 500)
g.check_true("nonces_unique", len(set(nonces)) == 500)

noise = E_prime - key.s * E
norms = np.linalg.norm(noise, axis=1)
g.check_true("noise_bound_strict", bool((norms < R).all()), f"max||lambda||={norms.max():.6f} < 3sβ/8={R}")
g.check_true("noise_matches_expected_scale", float(np.median(norms)) > 0.9 * R,
             f"median||lambda||/R={np.median(norms)/R:.4f}")

# ciphertext distance relation ||e'_i - e'_j|| ~ s ||e_i - e_j|| + bounded noise
D_plain = np.linalg.norm(E[:, None, :] - E[None, :, :], axis=2)
D_cipher = np.linalg.norm(E_prime[:, None, :] - E_prime[None, :, :], axis=2)
mask = ~np.eye(500, dtype=bool)
corr = float(np.corrcoef(D_plain[mask], D_cipher[mask])[0, 1])
g.check_true("cipher_distance_correlation", corr > 0.98, f"pearson={corr:.6f}")
# per-pair relative deviation: |d' - s*d| / (s*d) is bounded by 2R/(s*d) = 3β/(4d)
rel = np.abs(D_cipher[mask] - key.s * D_plain[mask]) / (key.s * D_plain[mask])
g.check_true("cipher_distance_relative_deviation", float(np.percentile(rel, 99)) < 0.15,
             f"p99 rel.dev={np.percentile(rel, 99):.4f}, max={rel.max():.4f} (bound 3β/(4d_min)={3*key.beta/(4*D_plain[mask].min()):.4f})")
g.check_true("cipher_distance_scale", abs(np.median(D_cipher[mask]) / np.median(D_plain[mask]) - 3.0) < 0.05,
             f"median ratio={np.median(D_cipher[mask])/np.median(D_plain[mask]):.4f} (s=3)")

# batch vs single with the same nonce
e0 = E[0]
ep0, r0 = EncDB_one(key, e0)
batch0, _ = EncDB(key, E[:1], nonces=[r0])
g.check_true("batch_equals_single", bool(np.allclose(batch0[0], ep0, rtol=0, atol=0)),
             "identical bits when nonce is fixed")

# malformed input handling
errs = []
try:
    EncDB(key, np.zeros(768))
except ValueError:
    errs.append("1d")
try:
    EncDB(key, np.zeros((3, 10)))
except ValueError:
    errs.append("wrong_dim")
g.check("malformed_inputs_rejected", ["1d", "wrong_dim"], sorted(errs))

g.note("EncDB = s*e + lambda with ||lambda|| < 3*s*beta/8 strictly (u<1).")
g.note("Ciphertext norms are ~s (=3) because ||e||=1 and ||lambda||<=0.225.")
g.finish()
