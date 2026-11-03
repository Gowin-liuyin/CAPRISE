"""M1 gate: PRF protocol (Algorithm 1 line 8: randbits1||randbits2 <- PRF(K,r)).

Protocol under test
-------------------
* ``prf(K, r, outlen)`` is deterministic, keyed, and returns exactly ``outlen``
  pseudorandom bytes (HMAC-SHA256 counter mode).
* ``coins(K, r, d)`` deterministically derives the Gaussian direction ``n``
  and the uniform scalar ``u`` consumed by lines 9-10 of Algorithm 1.
* The derived n is a standard Gaussian sample stream and u is U(0,1).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kstest, norm

from gates.common import Gate, add_src_to_path

add_src_to_path()
from caprise.prf import PRF_KEY_BYTES, coins, prf  # noqa: E402

g = Gate("M1", "prf", owner="lead", paper_ref="Algorithm 1 line 8 / SPEC 2.2")

K1 = bytes(range(PRF_KEY_BYTES))
K2 = bytes(range(1, PRF_KEY_BYTES + 1))
msg = b"nonce-0000000000"

ctrl = prf(K1, msg, 32)
g.check_true("prf_deterministic", ctrl == prf(K1, msg, 32), "same (K,r) -> same bits")
g.check_true("prf_key_sensitivity", ctrl != prf(K2, msg, 32), "different K -> different bits")
g.check_true("prf_msg_sensitivity", ctrl != prf(K1, b"nonce-0000000001", 32))
g.check("prf_output_length", 32, len(prf(K1, msg, 32)))
g.check_true("prf_arbitrary_output_length", len(prf(K1, msg, 7)) == 7 and len(prf(K1, msg, 100)) == 100)
g.check_true("prf_min_entropy_smoke", len(set(prf(K1, i.to_bytes(8, "big"), 16) for i in range(500))) == 500)

# --- coins(): deterministic (n, u) derivation ------------------------------
n1, u1 = coins(K1, msg, 768)
n2, u2 = coins(K1, msg, 768)
g.check_true("coins_deterministic", np.array_equal(n1, n2) and u1 == u2)
g.check_true("coins_shape", n1.shape == (768,))
g.check_true("u_strictly_interior", 0.0 < u1 < 1.0, f"u={u1:.6f}")

# statistical checks over many nonces (d=256 to keep it fast)
rng = np.random.default_rng(12345)
N, U = [], []
for _ in range(3000):
    r = rng.integers(0, 1 << 63, dtype=np.uint64).tobytes()
    n, u = coins(K1, r, 256)
    N.append(n)
    U.append(u)
N = np.asarray(N)
U = np.asarray(U)

g.check_true("gaussian_mean_near_0", abs(float(N.mean())) < 0.01, f"mean={N.mean():.5f}")
g.check_true("gaussian_std_near_1", abs(float(N.std()) - 1.0) < 0.02, f"std={N.std():.5f}")
# Kolmogorov-Smirnov against N(0,1) (subsample for independence of the test)
p_norm = kstest(N.ravel()[::7], norm().cdf).pvalue
g.check_true("gaussian_ks_test", p_norm > 0.01, f"KS p={p_norm:.4f}")
g.check_true("uniform_mean_near_half", abs(float(U.mean()) - 0.5) < 0.02, f"mean={U.mean():.5f}")
p_unif = kstest(U, "uniform").pvalue
g.check_true("uniform_ks_test", p_unif > 0.01, f"KS p={p_unif:.4f}")

g.note("PRF instantiation: HMAC-SHA256 counter mode (TapeGen-style), k=128 bit.")
g.note("n drawn by PCG64 seeded with 128 PRF bits; u = (128 PRF bits + 1)/2^128 in (0,1).")
g.finish()
