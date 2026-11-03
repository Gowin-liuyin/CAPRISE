"""Independent numeric audit of the *intermediate* inequalities of the CAPRISE
proof (paper Eqs. (4)-(20)), one check per proof step.

The reproduction gates verify the two *conclusions* of the claim:

    Eq.(2)  distance-comparison preservation query<->DB   -> gates/gate_m7_eq2.py
    Eq.(3)  non-preservation among DB embeddings          -> gates/gate_m8_eq3.py

This script additionally checks every intermediate inequality of the paper's
proof under the exact implementation of Algorithm 1, so that "the proof as
printed, step by step" is auditable.  It is a verification artifact (Lead-side
cross-check), not part of the reproduction pipeline.

Layout of the people proof (LaTeX source ppRAG_FLLM.tex, lines 288-367):

    (4)  premises: ||e_q-e_1|| < ||e_q-e_2|| - beta  AND
                   ||e_1-e_2|| < ||e_1-e_3|| - beta
    (5)  e'_q = s e_q + eta            (implementation identity, EncQ)
    (6)  e'_i = s e_i + lambda_i       (implementation identity, EncDB)
    (7)  ||eta|| < s*beta/8, ||lambda|| < 3*s*beta/8
    (8)  ||e'_q-e'_1|| <= ||eta|| + s||e_q-e_1|| + ||lambda_1||
    (9)  < ||eta|| + ||lambda_1|| + s(||e_q-e_2|| - beta)
    (10) < s*beta/8 + 3*s*beta/8 + ||s e_q - s e_2|| - s*beta
    (11) < ||s e_q - s e_2|| - (||eta|| + ||lambda_2||)
         <= ||s e_q - s e_2|| - ||eta - lambda_2||
    (12) <= ||(s e_q - s e_2) - (eta - lambda_2)|| = ||e'_q - e'_2||
    (13) ||e'_1-e'_2|| < ||lambda_1|| + ||lambda_2|| + s(||e_1-e_3|| - beta)
    (14) < 3*s*beta/8 + 3*s*beta/8 + ||s e_1 - s e_3|| - s*beta
    (15) = ||s e_1 - s e_3|| - s*beta/4 - delta,  delta > 0 exists
    (16) ||e'_1-e'_3|| = ||(s e_1 - s e_3) + (lambda_1 - lambda_3)||
    (17) >= ||s e_1 - s e_3|| - ||lambda_1 - lambda_3||
    (18) >  ||s e_1 - s e_3|| - (||lambda_1|| + ||lambda_3||)
    (19) >  ||s e_1 - s e_3|| - 3*s*beta/4
    (20) = ||s e_1 - s e_3|| - 3*s*beta/4 + delta', delta' > 0 exists
    conclusion: delta + delta' < s*beta/2  =>  ||e'_1-e'_2|| > ||e'_1-e'_3||

Run:  .venv/bin/python scripts/check_proof_chain.py
Writes results/proof_chain_check.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caprise.caprise import DecDB_one, EncDB_one, EncQ, KeyGen, sample_noise  # noqa: E402
from caprise.prf import new_nonce  # noqa: E402

DIM = 768
BETA = 0.2
S = 3
N_TRIPLES = 4000
SEED = 20260118


def _noise_norm(key, nonce, radius):
    return float(np.linalg.norm(sample_noise(key, nonce, radius)))


def main() -> int:
    rng = np.random.default_rng(SEED)
    key = KeyGen(dim=DIM, beta=BETA, s=S)

    rows_q = []   # Eq.(8)-(12) per (e_q, e_1, e_2)
    rows_db = []  # Eq.(13)-(20) per (e_1, e_2, e_3)

    for _ in range(N_TRIPLES):
        # ---- construct a query triple satisfying premise (4a) -------------
        # All vectors are unit-norm; a chord of length d corresponds to the
        # polar angle theta = 2*asin(d/2) (unit sphere).
        e_q = rng.standard_normal(DIM)
        e_q /= np.linalg.norm(e_q)
        u = rng.standard_normal(DIM)
        u -= u.dot(e_q) * e_q
        u /= np.linalg.norm(u)
        d01 = 0.05
        th1 = 2.0 * math.asin(d01 / 2.0)
        e_1 = math.cos(th1) * e_q + math.sin(th1) * u
        d02 = d01 + BETA + 1e-3                      # gap > beta by construction
        th2 = 2.0 * math.asin(min(d02 / 2.0, 0.999999))
        e_2 = math.cos(th2) * e_q + math.sin(th2) * u
        e_1 /= np.linalg.norm(e_1)
        e_2 /= np.linalg.norm(e_2)
        if not (np.linalg.norm(e_q - e_1) < np.linalg.norm(e_q - e_2) - BETA):
            continue

        q_p, q_r = EncQ(key, e_q)
        e1_p, r1 = EncDB_one(key, e_1)
        e2_p, r2 = EncDB_one(key, e_2)
        eta = q_p - S * e_q
        lam1 = e1_p - S * e_1
        lam2 = e2_p - S * e_2
        n_eta, n_lam1, n_lam2 = map(float, (np.linalg.norm(x) for x in (eta, lam1, lam2)))
        d_q1 = float(np.linalg.norm(e_q - e_1))
        d_q2 = float(np.linalg.norm(e_q - e_2))
        lhs = float(np.linalg.norm(q_p - e1_p))
        rhs = float(np.linalg.norm(q_p - e2_p))

        rows_q.append(
            {
                "eq7_eta_lt": n_eta < S * BETA / 8,
                "eq7_lambda_lt": (n_lam1 < 3 * S * BETA / 8) and (n_lam2 < 3 * S * BETA / 8),
                "eq5_identity": float(np.max(np.abs(q_p - (S * e_q + eta)))) < 1e-12,
                "eq6_identity": float(np.max(np.abs(e1_p - (S * e_1 + lam1)))) < 1e-12,
                "eq8": lhs <= n_eta + S * d_q1 + n_lam1 + 1e-12,
                "eq9": n_eta + S * d_q1 + n_lam1 < n_eta + n_lam1 + S * (d_q2 - BETA) + 1e-12,
                "eq10": n_eta + n_lam1 + S * (d_q2 - BETA)
                        < (S * BETA / 8 + 3 * S * BETA / 8 + np.linalg.norm(S * e_q - S * e_2) - S * BETA) + 1e-12,
                "eq11": np.linalg.norm(S * e_q - S * e_2) - (S * BETA / 8 + 3 * S * BETA / 8)
                        < np.linalg.norm(S * e_q - S * e_2) - (n_eta + n_lam2) + 1e-12,
                "eq11b": np.linalg.norm(S * e_q - S * e_2) - n_eta - n_lam2
                        <= np.linalg.norm(S * e_q - S * e_2) - float(np.linalg.norm(eta - lam2)) + 1e-12,
                # paper Eq.(12) as printed uses (se_q - se_2) - (eta - lambda), but
                # e'_q - e'_2 = (se_q - se_2) + (eta - lambda).  Check both:
                "eq12_as_printed_equals_cipher_dist":
                    abs(float(np.linalg.norm((S * e_q - S * e_2) - (eta - lam2))) - rhs) < 1e-12,
                "eq12_corrected_plus_equals_cipher_dist":
                    abs(float(np.linalg.norm((S * e_q - S * e_2) + (eta - lam2))) - rhs) < 1e-12,
                "eq12_reverse_triangle_plus":
                    float(np.linalg.norm((S * e_q - S * e_2) + (eta - lam2)))
                    >= float(np.linalg.norm(S * e_q - S * e_2)) - float(np.linalg.norm(eta - lam2)) - 1e-12,
                "conclusion": lhs < rhs,
            }
        )

        # ---- database triple satisfying premise (4b) ----------------------
        # Construct on the unit sphere with chord distances:
        #   ||a-b|| = base,  ||a-e_3|| = base + beta + 1e-3  (> ||a-b|| + beta)
        a = rng.standard_normal(DIM)
        a /= np.linalg.norm(a)
        w = rng.standard_normal(DIM)
        w -= w.dot(a) * a
        w /= np.linalg.norm(w)
        base = float(rng.uniform(0.02, 0.20))
        th_b = 2.0 * math.asin(base / 2.0)
        b = math.cos(th_b) * a + math.sin(th_b) * w
        d_ac = base + BETA + 1e-3
        th_c = 2.0 * math.asin(min(d_ac / 2.0, 0.999999))
        e_3 = math.cos(th_c) * a + math.sin(th_c) * w
        b /= np.linalg.norm(b)
        e_3 /= np.linalg.norm(e_3)
        if not (np.linalg.norm(a - b) < np.linalg.norm(a - e_3) - BETA):
            continue
        lam_a = sample_noise(key, new_nonce(), 3 * S * BETA / 8)
        lam_b = sample_noise(key, new_nonce(), 3 * S * BETA / 8)
        lam_c = sample_noise(key, new_nonce(), 3 * S * BETA / 8)
        a_p, b_p, c_p = S * a + lam_a, S * b + lam_b, S * e_3 + lam_c
        vac = float(np.linalg.norm(S * a - S * e_3))
        lhs2 = float(np.linalg.norm(a_p - b_p))
        rhs2 = float(np.linalg.norm(a_p - c_p))
        delta = vac - S * BETA / 4 - lhs2
        delta_p = rhs2 - (vac - 3 * S * BETA / 4)
        rows_db.append(
            {
                "eq13": lhs2 < float(np.linalg.norm(lam_a)) + float(np.linalg.norm(lam_b))
                        + S * (float(np.linalg.norm(a - e_3)) - BETA) + 1e-12,
                "eq14": float(np.linalg.norm(lam_a)) + float(np.linalg.norm(lam_b))
                        + S * (float(np.linalg.norm(a - e_3)) - BETA)
                        < 3 * S * BETA / 8 + 3 * S * BETA / 8 + vac - S * BETA + 1e-12,
                "eq15_delta_positive": delta > 0.0,
                "eq16_identity": float(np.max(np.abs((a_p - c_p) - ((S * a - S * e_3) + (lam_a - lam_c))))) < 1e-12,
                "eq17": float(np.linalg.norm(a_p - c_p))
                        >= vac - float(np.linalg.norm(lam_a - lam_c)) - 1e-12,
                "eq18": vac - float(np.linalg.norm(lam_a - lam_c))
                        > vac - (float(np.linalg.norm(lam_a)) + float(np.linalg.norm(lam_c))) - 1e-12,
                "eq19": rhs2 >= vac - 3 * S * BETA / 4 - 1e-12,
                "eq20_delta_prime_positive": delta_p > 0.0,
                "conclusion_condition_delta_sum_lt_half_sbeta": (delta + delta_p) < S * BETA / 2,
                "conclusion_inverted": lhs2 > rhs2,
            }
        )

    def summarize(rows, keys):
        out = {}
        for k in keys:
            vals = [r[k] for r in rows]
            if all(isinstance(v, (bool, np.bool_)) for v in vals):
                out[k] = {"n": len(vals), "violations": int(sum(not v for v in vals))}
            else:
                arr = np.asarray(vals, dtype=float)
                out[k] = {"n": len(vals), "min": float(arr.min()), "mean": float(arr.mean())}
        return out

    q_keys = ["eq7_eta_lt", "eq7_lambda_lt", "eq5_identity", "eq6_identity",
              "eq8", "eq9", "eq10", "eq11", "eq11b",
              "eq12_as_printed_equals_cipher_dist",
              "eq12_corrected_plus_equals_cipher_dist",
              "eq12_reverse_triangle_plus", "conclusion"]
    db_keys = ["eq13", "eq14", "eq15_delta_positive", "eq16_identity", "eq17",
               "eq18", "eq19", "eq20_delta_prime_positive",
               "conclusion_condition_delta_sum_lt_half_sbeta", "conclusion_inverted"]
    summary = {
        "n_query_triples": len(rows_q),
        "n_db_triples": len(rows_db),
        "query_chain": summarize(rows_q, q_keys),
        "db_chain": summarize(rows_db, db_keys),
    }
    # number of DB triples for which the extra condition delta+delta'<s*beta/2 fires
    summary["db_chain"]["n_delta_condition_fires"] = int(
        sum(r["conclusion_condition_delta_sum_lt_half_sbeta"] for r in rows_db)
    )
    summary["db_chain"]["n_inverted_when_condition_fires"] = int(
        sum(
            r["conclusion_condition_delta_sum_lt_half_sbeta"] and r["conclusion_inverted"]
            for r in rows_db
        )
    )
    checked = {k: v for k, v in summary["query_chain"].items()
               if k != "eq12_as_printed_equals_cipher_dist"}
    # The last two db_chain fields measure *how often* the paper's extra
    # condition delta+delta' < s*beta/2 fires under random noise (finding F2 /
    # A19) -- they are observations, not pass/fail checks.
    observed_keys = {"conclusion_condition_delta_sum_lt_half_sbeta", "conclusion_inverted"}
    all_ok = all(
        v.get("violations", 0) == 0
        for v in list(checked.values()) + [
            x for k, x in summary["db_chain"].items()
            if isinstance(x, dict) and k not in observed_keys
        ]
    )
    summary["pass"] = bool(all_ok)
    summary["paper_ref"] = "arXiv:2601.12331v1 Eqs.(4)-(20); LaTeX ppRAG_FLLM.tex lines 288-367"
    summary["settings"] = {"dim": DIM, "beta": BETA, "s": S, "n_triples": N_TRIPLES, "seed": SEED}

    out = ROOT / "results" / "proof_chain_check.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
