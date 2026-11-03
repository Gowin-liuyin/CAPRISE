"""M9 gate: ADCPE / SAP baseline (Fuchsbauer et al. 2022, Sec. 5.1).

    Enc_SAP(m) = s*m + lambda_m,   ||lambda_m|| < s*beta/4
    lambda_m = (s*beta/4) * (u/||u||) * x0^(1/d),  u ~ N(0,I_d), x0 ~ U(0,1)

Checks
------
1. Round-trip Dec(Enc(m)) = m.
2. Noise bound ||c - s*m|| < s*beta/4 strictly.
3. β-DCP property: ||x-y|| < ||y-z|| - β => ||f(x)-f(y)|| < ||f(y)-f(z)||
   for every random triple (this is what CAPRISE gives up for DB vectors).
"""

from __future__ import annotations

import numpy as np

from gates.common import Gate, add_src_to_path

add_src_to_path()
from caprise.caprise import ADCPE, KeyGen, noise_radius_adcpe  # noqa: E402

g = Gate("M9", "adcpe", owner="lead", paper_ref="ADCPE (SAP) Sec.5.1 / SPEC 2.1")

key = KeyGen(dim=768, beta=0.2, s=3)
R = noise_radius_adcpe(3, 0.2)
rng = np.random.default_rng(17)
beta = key.beta
g.check_true("radius_formula", abs(R - 3 * 0.2 / 4) < 1e-15, f"sβ/4={R}")

# round-trip + noise bound
m = rng.standard_normal(768)
m /= np.linalg.norm(m)
c, r = ADCPE.Enc(key, m)
m_hat = ADCPE.Dec(key, c, r)
err = float(np.abs(m_hat - m).max())
g.check_true("roundtrip_lt_1e9", err < 1e-9, f"max err={err:.2e}")
norms = np.asarray([np.linalg.norm(ADCPE.Enc(key, m)[0] - key.s * m) for _ in range(500)])
g.check_true("noise_bound_sbeta4", bool((norms < R).all()), f"max||lambda||={norms.max():.6f} < {R}")
g.check_true("noise_bound_stricter_than_caprise_db", R < 3 * 3 * 0.2 / 8,
             f"ADCPE {R} < CAPRISE DB {3*3*0.2/8}")

# β-DCP: ||x-y|| < ||y-z|| - beta => ||f(x)-f(y)|| < ||f(y)-f(z)||
# Construct premise-satisfying triples (isotropic random vectors are too
# concentrated to ever have gaps > beta, see M7).
TARGET = 20_000
D = key.dim


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _perp(base, v):
    v = v - np.einsum("ij,ij->i", v, base)[:, None] * base
    return _unit(v)


def _rot(base, dist, v):
    theta = 2.0 * np.arcsin(dist / 2.0)
    return np.cos(theta)[:, None] * base + np.sin(theta)[:, None] * _perp(base, v)


Y = _unit(rng.standard_normal((TARGET, D)))
d_xy = rng.uniform(0.3, 1.0, size=TARGET)
margin = rng.uniform(1e-6, 0.05, size=TARGET)
d_yz = d_xy + beta + margin
X = _rot(Y, d_xy, rng.standard_normal((TARGET, D)))
Z = _rot(Y, d_yz, rng.standard_normal((TARGET, D)))
premise = np.linalg.norm(X - Y, axis=1) < np.linalg.norm(Y - Z, axis=1) - beta
g.check_true("premise_holds_by_construction", bool(premise.all()), f"{TARGET} triples")
n = len(X)

FX = np.stack([ADCPE.Enc(key, x)[0] for x in X])
FY = np.stack([ADCPE.Enc(key, y)[0] for y in Y])
FZ = np.stack([ADCPE.Enc(key, z)[0] for z in Z])
viol = int((np.linalg.norm(FX - FY, axis=1) >= np.linalg.norm(FY - FZ, axis=1)).sum())
g.check("dcp_violations", 0, viol, f"{viol}/{n} β-DCP violations")

g.note("ADCPE/SAP baseline verified; its noise radius sβ/4 < 3sβ/8 (CAPRISE DB) is exactly "
       "why ADCPE leaks stored-vector structure to vector analysis.")
g.finish()
