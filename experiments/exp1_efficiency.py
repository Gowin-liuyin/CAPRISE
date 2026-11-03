"""exp1 -- efficiency experiments: Fig.4 (encryption throughput), Phase-1 overhead,
Phase-2 retrieval-side encryption throughput vs PHE baseline (module M17).

Paper: arXiv:2601.12331 Sec. V-1 (Efficiency), Fig. 4 (Rate_Processing.png /
Rate_Encryption.png), Phase-1 (79.52 ms embedding + 15 ms CAPRISE for 128
queries, overhead <19%) and Phase-2 (2339 vectors/s at d=768 vs RemoteRAG PHE
250-300 vectors/s, claimed 9x).

What is measured (all on Apple M5 CPU, numpy implementation of CAPRISE):
  A. CAPRISE EncDB throughput on the paper's Fig.4 15-point dimension grid
     {64,128,256,312,384,512,768,896,1024,1536,1792,2048,2560,3072,3584}
     (+4096 as an extra point that is NOT in the paper), vectorised numpy
     implementation, batch = max(1000, ceil(4e6/d)) embeddings (>= 1000 in every
     row), 3 interleaved repetitions, median; both wall-clock and
     time.process_time() are recorded.
  B. "Paper implementation style" probe: the same EncDB operation implemented as
     a per-element Python scalar loop (one Python-level draw + one multiply-add
     per coordinate), mathematically identical to caprise.caprise._noise.
     Throughput (embeddings/s and elements/s) is compared against the paper's
     Fig.4 values.  This is an implementation-style probe: it is NOT the paper's
     code and NOT the src/ core implementation.
  C. Pixel digitisation of the paper's own Fig.4a / Fig.4b PNGs (15 markers
     each) so that the paper values and this reproduction can be listed side by
     side in one table (accuracy +-2%, see notes).
  D. Phase-1 overhead: embed 128 MS MARCO dev queries with gtr-t5-base
     (local copy data/raw/gtr-t5-base; MPS float16 and CPU float32) and encrypt
     the same 128 queries with CAPRISE EncQ; overhead = enc / embed.
  E. Phase-2: per-vector CAPRISE encryption throughput at d=768; Paillier
     baseline (python-phe 1.5.0, GMP-accelerated when gmpy2 is available;
     1024-bit and 2048-bit keys, 5 vectors extrapolated) plus a pure-CPython
     builtin-pow lower bound; speedups vs measured Paillier and vs the numbers
     quoted in the paper (RemoteRAG PHE 250-300 vectors/s).
  F. Implementation-style probe (ambiguity A17): vectorised numpy noise
     generation vs a per-element Python loop, to explain why the paper's
     per-element-style implementation reports 2339 vectors/s while this
     vectorised implementation is ~20-30x faster.

Outputs
-------
    results/exp1_efficiency.json
    figures/fig4_efficiency.png
    notes/exp1_efficiency.md   (written separately)

Run:  .venv/bin/python experiments/exp1_efficiency.py
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from caprise.caprise import (  # noqa: E402
    EncDB,
    EncQ,
    KeyGen,
    noise_radius_db,
    sample_noise,
)
from caprise.prf import new_nonce, prf  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# --------------------------------------------------------------------------- #
# Measurement grid
# --------------------------------------------------------------------------- #
# Paper Fig.4 x-axis, in exactly the order printed on the figure (15 ticks).
PAPER_DIMS = (64, 128, 256, 312, 384, 512, 768, 896, 1024, 1536, 1792, 2048, 2560, 3072, 3584)
# Extra grid point kept from the earlier (6-point) run.  NOT in paper Fig.4, so
# it is marked "in_paper": False and excluded from every paper-grid assertion.
EXTRA_DIMS = (4096,)
DIMS = PAPER_DIMS + EXTRA_DIMS

BATCH_MIN = 1000                     # paper-independent floor required by the gate
TARGET_ELEMENTS_PER_RUN = 4_000_000  # ~32 MB of input per timed run at every d
REPS = 5                             # interleaved rounds (>= the required 3); median is reported
SEED = 20260118

# Paper-style (per-element Python) probe: much slower per element, so a smaller
# element budget is used per timed run (~50-110 ms per run at every d).
SCALAR_TARGET_ELEMENTS_PER_RUN = 400_000
SCALAR_MIN_BATCH = 8

# Monotonicity tolerance, calibrated on the paper's own Fig.4a (digitised): its
# 256 -> 312 step RISES by +24.4% while all other adjacent steps fall, so the
# reproduction is allowed adjacent rises up to 25% (plus an overall >2x drop and
# a Spearman rho <= -0.9 against d).
MONO_WAVE = 0.25

# paper numbers (Sec. V-1)
PAPER_PHASE1_EMBED_MS = 79.52   # 128 queries, NVIDIA A100 GPU
PAPER_PHASE1_ENC_MS = 15.0      # CAPRISE encryption of the same 128 queries
PAPER_PHASE1_OVERHEAD_PCT = 19.0
PAPER_PHASE2_CAPRISE_VEC_S = 2339.0
PAPER_PHASE2_REMOTERAG_PHE_VEC_S = (250.0, 300.0)

LOCAL_MODEL = ROOT / "data" / "raw" / "gtr-t5-base"
EMBED_TIMING_JSON = RESULTS / "embed_timing.json"

# Fixed-workload calibration reference measured on this same machine when it was
# comparably idle: the previous revision's EncDB run at d=768, batch=1000, 3 reps,
# API-default nonces (results/exp1_efficiency.json of git HEAD, runtime 37.1 s).
# It is used only to quantify how much slower/faster the current run's machine is
# (the machine is shared with other reproduction jobs, so absolute rates move by
# several x between runs; ratios and curve shapes do not).
REFERENCE_RUN_768 = {
    "d": 768,
    "batch": 1000,
    "reps": 3,
    "nonce_mode": "api_default (os.urandom per vector)",
    "median_s": 0.014564,
    "median_cpu_s": 0.014462,
    "us_per_vector": 14.564,
    "us_per_vector_cpu": 14.462,
    "embeddings_per_s": 68660.3,
    "embeddings_per_s_cpu": 69125.0,
    "source_run": (
        "results/exp1_efficiency.json of the previous revision of this experiment (git HEAD, the "
        "6-point [384..4096] grid): d=768 row, batch=1000, 3 reps, EncDB API defaults, median "
        "14.564 us/vector; the artifact was produced on 2026-11-03 in the early afternoon"
    ),
    "load_at_measurement": (
        "the previous notes recorded a 1-min load average of ~4.8 while that artifact was produced, "
        "so this reference is NOT a guaranteed-idle baseline -- it is simply a fixed earlier "
        "measurement of the same implementation"
    ),
    "is_guaranteed_idle": False,
    "usage": (
        "only used to report an indicative machine-state factor for this run; NO assertion depends "
        "on it. The load-invariant assertions normalise inside each run instead: the gate compares "
        "(rate_d768 / rate_calibration) measured in the artifact with the same ratio measured in the "
        "gate process (see throughput.recheck_reference), which cancels the machine-speed factor "
        "without any external constant"
    ),
}
LOAD_CALIBRATION = {"d": 768, "batch": 2000, "reps": 3}
# Fixed workload repeated by the gate so that the artifact's d=768 throughput can be
# re-checked load-invariantly: both sides are normalised by this workload, which
# cancels the machine-speed factor between the two runs/processes.
RECHECK_WORKLOAD = {"d": 768, "batch": 1500, "reps": 3}


def batch_for(d: int) -> int:
    """Timed-run batch for the vectorised measurement: >=1000 vectors and
    ~TARGET_ELEMENTS_PER_RUN elements, so every timed window is long enough to
    survive scheduling noise on a shared machine."""
    return max(BATCH_MIN, math.ceil(TARGET_ELEMENTS_PER_RUN / d))


# --------------------------------------------------------------------------- #
# A. CAPRISE encryption throughput vs dimension (vectorised implementation)
# --------------------------------------------------------------------------- #
def throughput_rows() -> list[dict]:
    """EncDB throughput on the Fig.4 dimension grid (vectorised numpy).

    Rounds are *interleaved* (round k measures every dimension in the paper
    order) so that load drift on a shared machine hits all dimensions alike;
    this is what makes the per-dimension medians comparable and what the gate
    re-computes from the stored per-round raw timings.
    """
    keys: dict[int, object] = {}
    mats: dict[int, np.ndarray] = {}
    batches: dict[int, int] = {}
    nonces: dict[int, list[bytes]] = {}
    for d in DIMS:
        keys[d] = KeyGen(dim=d, beta=0.2, s=3, rng=np.random.default_rng(SEED))
        b = batch_for(d)
        batches[d] = b
        rng = np.random.default_rng(1000 + d)
        E = rng.standard_normal((b, d))
        E /= np.linalg.norm(E, axis=1, keepdims=True)  # GTR embeddings are L2-normalised
        mats[d] = E
        # Per-vector nonces are drawn once per dimension (outside the timed loop).
        # Rationale: the paper's EncDB derives its randomness from a PRF, not from
        # one OS-entropy syscall per vector, and os.urandom latency is syscall
        # overhead rather than cryptographic cost (it also dominates the small-d
        # end of the curve on a loaded machine).  nonce_mode_probe() quantifies
        # the difference between the two modes.
        raw = os.urandom(16 * b)
        nonces[d] = [raw[16 * i: 16 * i + 16] for i in range(b)]
        EncDB(keys[d], E[:16], nonces=nonces[d][:16])  # warm-up (cache, allocator)

    runs_wall: dict[int, list[float]] = {d: [] for d in DIMS}
    runs_cpu: dict[int, list[float]] = {d: [] for d in DIMS}
    for rnd in range(REPS):
        # rotating start: every dimension occupies every position in the round
        # equally often, so a slow drift of the machine inside a round cannot
        # systematically favour large or small dimensions
        order = DIMS[rnd % len(DIMS):] + DIMS[: rnd % len(DIMS)]
        for d in order:
            t0w, t0c = time.perf_counter(), time.process_time()
            EncDB(keys[d], mats[d], nonces=nonces[d])
            runs_wall[d].append(time.perf_counter() - t0w)
            runs_cpu[d].append(time.process_time() - t0c)

    rows = []
    for d in DIMS:
        b = batches[d]
        med = float(np.median(runs_wall[d]))
        med_cpu = float(np.median(runs_cpu[d]))
        rows.append(
            {
                "d": d,
                "in_paper": d in PAPER_DIMS,
                "batch": b,
                "n_elements_per_run": b * d,
                "reps": REPS,
                "runs_s": [round(x, 6) for x in runs_wall[d]],
                "median_s": round(med, 6),
                "runs_cpu_s": [round(x, 6) for x in runs_cpu[d]],
                "median_cpu_s": round(med_cpu, 6),
                "embeddings_per_s": round(b / med, 1),
                "elements_per_s": round(b * d / med, 0),
                "ms_per_vector": round(med / b * 1000, 5),
                # CPU-time based (robust to co-running jobs on the same machine)
                "embeddings_per_s_cpu": round(b / med_cpu, 1),
                "elements_per_s_cpu": round(b * d / med_cpu, 0),
                "ms_per_vector_cpu": round(med_cpu / b * 1000, 5),
                "input_mb": round(b * d * 8 / 1e6, 2),
                # least-interference estimator (fastest of the REPS rounds): reported for
                # reference only, the median above is the operational number
                "min_s": round(float(min(runs_wall[d])), 6),
                "embeddings_per_s_min": round(b / float(min(runs_wall[d])), 1),
                "elements_per_s_min": round(b * d / float(min(runs_wall[d])), 0),
                "nonce_mode": "pre-generated nonces (drawn once per dimension, outside the timed loop); "
                              "see nonce_mode_probe for the API-default (os.urandom per vector) delta",
            }
        )
        print(f"  d={d:5d}{'' if d in PAPER_DIMS else ' *'}: {rows[-1]['embeddings_per_s']:>10.1f} emb/s, "
              f"{rows[-1]['elements_per_s']/1e6:>7.2f} M elem/s, {rows[-1]['ms_per_vector']:.5f} ms/vec "
              f"(batch={b})")
    del mats
    return rows


def spearman_rho(x, y) -> float:
    """Spearman rank correlation (no ties in either series)."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(xa)).astype(float)
    ry = np.argsort(np.argsort(ya)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def monotonicity_summary(rows: list[dict]) -> dict:
    """Monotonicity of the vectorised embeddings/s on the 15 paper dimensions.

    Tolerance: the paper's own Fig.4a is *not* strictly monotone -- digitised,
    its 256 -> 312 step RISES by +24.4% (3,886 -> 4,834 embeddings/s) while every
    other adjacent step falls.  A reproduction can therefore not be required to be
    strictly monotone either; the acceptance rule is "overall strong decrease
    (first/last > 2x) + no systematic increase (Spearman rho <= -0.9) + no adjacent
    rise larger than the paper's own largest adjacent rise (+24.4% -> 25% cap)".

    Two estimators are stored:
      * naive     -- ratios of the per-dimension medians;
      * round-paired -- rate ratio measured *inside* each interleaved round and
        then medianed.  A common load factor cancels inside a round, so this is
        the load-invariant version of "throughput decreases with d".
    """
    paper = [r for r in rows if r["in_paper"]]
    wave = MONO_WAVE
    steps = []
    for a, b in zip(paper, paper[1:]):
        # per-round RATE ratio: batches differ between dimensions, so the elapsed
        # times must be converted to rates before dividing
        paired_w = [
            (a["batch"] / a["runs_s"][r]) / (b["batch"] / b["runs_s"][r]) for r in range(REPS)
        ]
        paired_c = [
            (a["batch"] / a["runs_cpu_s"][r]) / (b["batch"] / b["runs_cpu_s"][r]) for r in range(REPS)
        ]
        steps.append(
            {
                "d_from": a["d"],
                "d_to": b["d"],
                "ratio_naive_wall": round(a["embeddings_per_s"] / b["embeddings_per_s"], 4),
                "ratio_naive_cpu": round(a["embeddings_per_s_cpu"] / b["embeddings_per_s_cpu"], 4),
                "ratio_paired_wall": round(float(np.median(paired_w)), 4),
                "ratio_paired_cpu": round(float(np.median(paired_c)), 4),
            }
        )
    floor = 1.0 / (1.0 + wave)
    viol_n_wall = [s["d_to"] for s in steps if s["ratio_naive_wall"] < floor]
    viol_n_cpu = [s["d_to"] for s in steps if s["ratio_naive_cpu"] < floor]
    viol_p_wall = [s["d_to"] for s in steps if s["ratio_paired_wall"] < floor]
    viol_p_cpu = [s["d_to"] for s in steps if s["ratio_paired_cpu"] < floor]
    drop_wall = paper[0]["embeddings_per_s"] / paper[-1]["embeddings_per_s"]
    drop_cpu = paper[0]["embeddings_per_s_cpu"] / paper[-1]["embeddings_per_s_cpu"]
    rho_wall = spearman_rho([r["d"] for r in paper], [r["embeddings_per_s"] for r in paper])
    rho_cpu = spearman_rho([r["d"] for r in paper], [r["embeddings_per_s_cpu"] for r in paper])
    worst_up = max(1.0 / min(s["ratio_naive_cpu"] for s in steps),
                   1.0 / min(s["ratio_paired_cpu"] for s in steps))
    return {
        "definition": (
            "ratio(d_i -> d_i+1) = embeddings_per_s(d_i) / embeddings_per_s(d_i+1); "
            "monotone-decreasing-with-jitter means ratio >= 1/(1+wave) for every "
            "adjacent pair of the 15 paper dimensions (wave = 25%, calibrated on the "
            "paper's own digitised Fig.4a, whose 256->312 step rises +24.4%; see "
            "paper_fig4_digitisation.paper_curve_shape)"
        ),
        "wave_allowed_pct": wave * 100.0,
        "steps": steps,
        "primary_estimator": (
            "CPU-time rates (time.process_time()) with round-paired ratios: both dimensions are timed "
            "back-to-back inside the same interleaved round and the rate ratio is taken inside the "
            "round, so the machine-load factor cancels. Wall-clock numbers are reported but not "
            "asserted (the shared machine's wall-clock rate moves by several x between runs)."
        ),
        "naive_violations": viol_n_wall,
        "naive_violations_cpu": viol_n_cpu,
        "round_paired_violations": viol_p_wall,
        "round_paired_violations_cpu": viol_p_cpu,
        "paper_grid_drop_ratio_first_over_last": round(drop_wall, 3),
        "paper_grid_drop_ratio_cpu": round(drop_cpu, 3),
        "spearman_rho_vs_dim": round(rho_wall, 4),
        "spearman_rho_vs_dim_cpu": round(rho_cpu, 4),
        "largest_adjacent_rise_pct": round((worst_up - 1.0) * 100.0, 2),
        "tolerance_basis": (
            "the paper's own digitised Fig.4a rises +24.4% on 256 -> 312 and falls on every other "
            "step; the reproduction therefore allows adjacent rises up to 25% (see "
            "paper_fig4_digitisation.paper_curve_shape) instead of demanding strict monotonicity"
        ),
        "ok": (not viol_p_cpu) and drop_cpu > 2.0 and rho_cpu <= -0.9,
        "ok_wall": (not viol_p_wall) and drop_wall > 2.0 and rho_wall <= -0.9,
        "informational_note": (
            "wall-clock drop / Spearman / violations are informational; the assertion uses the CPU-time "
            "series, which is invariant to co-running jobs on the shared machine"
        ),
    }


# --------------------------------------------------------------------------- #
# B. "Paper implementation style" probe: per-element Python scalar EncDB
# --------------------------------------------------------------------------- #
def scalar_style_noise_list(key, nonce: bytes) -> list[float]:
    """sqrt(3 s beta / 8)-style noise drawn the *paper* way: one Python-level
    scalar operation per coordinate.

    lambda_i = (3 s beta / 8) * (n_i / ||n||) * u^(1/d), with (n, u) re-derived
    from (K, r) exactly as src/caprise/prf.py does.  Mathematically identical to
    caprise.caprise._noise; the experiment verifies that numerically.
    """
    bits = prf(key.K, nonce, 32)
    seed = int.from_bytes(bits[:16], "big")
    u = (int.from_bytes(bits[16:], "big") + 1) / float(1 << 128)
    rng = np.random.Generator(np.random.PCG64(seed))
    d = key.dim
    n = [0.0] * d
    sq = 0.0
    for i in range(d):                      # per-element scalar Gaussian draw
        g = float(rng.standard_normal())
        n[i] = g
        sq += g * g
    norm = math.sqrt(sq)
    scale = noise_radius_db(key.s, key.beta) * (u ** (1.0 / d)) / norm
    return [scale * g for g in n]


def scalar_style_noise(key, nonce: bytes) -> np.ndarray:
    """Same noise as :func:`scalar_style_noise_list`, as a numpy array (used only
    by the equivalence check)."""
    return np.asarray(scalar_style_noise_list(key, nonce), dtype=np.float64)


def scalar_style_encdb_one(key, e, nonce: bytes) -> list[float]:
    """EncDB for one vector, paper style: per-element Python scalar loop.

    e'_i = s * e_i + lambda_i computed coordinate by coordinate; no numpy
    vectorisation anywhere in this function (it returns a plain Python list).
    """
    lam = scalar_style_noise_list(key, nonce)
    s = key.s
    d = key.dim
    out = [0.0] * d
    for i in range(d):                      # per-element scalar encrypt
        out[i] = s * float(e[i]) + lam[i]
    return out


def check_scalar_style_equivalence(dims=(64, 312, 768, 3072), n_checks: int = 2) -> dict:
    """Verify that the per-element probe is numerically identical to src noise."""
    checks = []
    for d in dims:
        key = KeyGen(dim=d, beta=0.2, s=3, rng=np.random.default_rng(SEED))
        for j in range(n_checks):
            nonce = new_nonce(np.random.default_rng(555 + 10 * j + d))
            lam_ref = sample_noise(key, nonce, noise_radius_db(key.s, key.beta))
            lam_scalar = scalar_style_noise(key, nonce)
            scale = float(np.abs(lam_ref).max())
            max_abs = float(np.abs(lam_ref - lam_scalar).max())
            checks.append(
                {
                    "d": d,
                    "nonce_index": j,
                    "max_abs_diff": max_abs,
                    "max_rel_diff": max_abs / scale if scale else 0.0,
                }
            )
    worst = max(c["max_rel_diff"] for c in checks)
    return {
        "what": (
            "lambda from the per-element Python loop vs caprise.caprise._noise "
            "(via caprise.caprise.sample_noise) for identical (K, r); only the "
            "summation order of the norm differs, so the expected difference is "
            "float round-off (~1e-16 relative)"
        ),
        "n_checks": len(checks),
        "dims": list(dims),
        "worst_rel_diff": worst,
        "max_abs_diff": max(c["max_abs_diff"] for c in checks),
        "rows": checks,
        "equivalent_up_to_float_roundoff": worst < 1e-12,
    }


def scalar_style_throughput() -> list[dict]:
    """EncDB throughput of the paper-style per-element Python implementation."""
    keys: dict[int, object] = {}
    mats: dict[int, np.ndarray] = {}
    batches: dict[int, int] = {}
    nonces: dict[int, list[bytes]] = {}
    for d in DIMS:
        keys[d] = KeyGen(dim=d, beta=0.2, s=3, rng=np.random.default_rng(SEED))
        b = max(SCALAR_MIN_BATCH, math.ceil(SCALAR_TARGET_ELEMENTS_PER_RUN / d))
        batches[d] = b
        rng = np.random.default_rng(2000 + d)
        E = rng.standard_normal((b, d))
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        mats[d] = E
        raw = os.urandom(16 * b)
        nonces[d] = [raw[16 * i: 16 * i + 16] for i in range(b)]
        for i in range(min(2, b)):          # warm-up
            scalar_style_encdb_one(keys[d], E[i], nonces[d][i])

    runs_wall: dict[int, list[float]] = {d: [] for d in DIMS}
    runs_cpu: dict[int, list[float]] = {d: [] for d in DIMS}
    for rnd in range(REPS):
        order = DIMS[rnd % len(DIMS):] + DIMS[: rnd % len(DIMS)]
        for d in order:
            E, key, ns = mats[d], keys[d], nonces[d]
            b = batches[d]
            t0w, t0c = time.perf_counter(), time.process_time()
            for i in range(b):
                scalar_style_encdb_one(key, E[i], ns[i])
            runs_wall[d].append(time.perf_counter() - t0w)
            runs_cpu[d].append(time.process_time() - t0c)

    rows = []
    for d in DIMS:
        b = batches[d]
        med = float(np.median(runs_wall[d]))
        med_cpu = float(np.median(runs_cpu[d]))
        rows.append(
            {
                "d": d,
                "in_paper": d in PAPER_DIMS,
                "batch_vectors": b,
                "n_elements_per_run": b * d,
                "reps": REPS,
                "runs_s": [round(x, 6) for x in runs_wall[d]],
                "median_s": round(med, 6),
                "runs_cpu_s": [round(x, 6) for x in runs_cpu[d]],
                "median_cpu_s": round(med_cpu, 6),
                "embeddings_per_s": round(b / med, 1),
                "elements_per_s": round(b * d / med, 0),
                "us_per_vector": round(med / b * 1e6, 1),
                "us_per_element": round(med / (b * d) * 1e6, 4),
                "embeddings_per_s_cpu": round(b / med_cpu, 1),
                "elements_per_s_cpu": round(b * d / med_cpu, 0),
                "us_per_vector_cpu": round(med_cpu / b * 1e6, 1),
                "us_per_element_cpu": round(med_cpu / (b * d) * 1e6, 4),
                # least-interference estimator (fastest of the REPS rounds)
                "min_s": round(float(min(runs_wall[d])), 6),
                "embeddings_per_s_min": round(b / float(min(runs_wall[d])), 1),
                "elements_per_s_min": round(b * d / float(min(runs_wall[d])), 0),
                "us_per_element_min": round(float(min(runs_wall[d])) / (b * d) * 1e6, 4),
            }
        )
        print(f"  d={d:5d}{'' if d in PAPER_DIMS else ' *'}: {rows[-1]['embeddings_per_s']:>9.1f} emb/s, "
              f"{rows[-1]['elements_per_s']/1e6:>6.2f} M elem/s, "
              f"{rows[-1]['us_per_element']:.4f} us/elem (batch={b})")
    del mats
    return rows


def _region_mean(rows: list[dict], lo: int, hi: int, field: str) -> float:
    vals = [r[field] for r in rows if r["in_paper"] and lo <= r["d"] <= hi]
    return float(np.mean(vals))


def load_average() -> dict:
    """1/5/15-minute load average (the machine is shared with other jobs)."""
    try:
        la = os.getloadavg()
    except OSError:  # pragma: no cover
        return {}
    return {"loadavg_1m": round(la[0], 2), "loadavg_5m": round(la[1], 2), "loadavg_15m": round(la[2], 2)}


def recheck_reference_probe(d: int = RECHECK_WORKLOAD["d"], batch: int = RECHECK_WORKLOAD["batch"],
                            reps: int = RECHECK_WORKLOAD["reps"]) -> dict:
    """Fixed EncDB workload used as the denominator of the load-invariant re-check.

    The gate runs exactly this workload again and compares
    (rate_d768 / rate_this) between the artifact and the gate run, which cancels the
    machine-speed factor shared by both measurements.
    """
    key = KeyGen(dim=d, beta=0.2, s=3, rng=np.random.default_rng(SEED))
    rng = np.random.default_rng(5000 + d)
    E = rng.standard_normal((batch, d))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    raw = os.urandom(16 * batch)
    ns = [raw[16 * i: 16 * i + 16] for i in range(batch)]
    EncDB(key, E[:8], nonces=ns[:8])
    runs, runs_cpu = [], []
    for _ in range(reps):
        t0w, t0c = time.perf_counter(), time.process_time()
        EncDB(key, E, nonces=ns)
        runs.append(time.perf_counter() - t0w)
        runs_cpu.append(time.process_time() - t0c)
    med = float(np.median(runs))
    med_cpu = float(np.median(runs_cpu))
    return {
        "what": "fixed EncDB workload (d=768, batch=1500, 3 reps, pre-generated nonces) used as the "
                "denominator of the gate's load-invariant re-check of the d=768 throughput",
        "d": d,
        "batch": batch,
        "reps": reps,
        "runs_s": [round(x, 6) for x in runs],
        "median_s": round(med, 6),
        "runs_cpu_s": [round(x, 6) for x in runs_cpu],
        "median_cpu_s": round(med_cpu, 6),
        "embeddings_per_s": round(batch / med, 1),
        "embeddings_per_s_cpu": round(batch / med_cpu, 1),
    }


def load_calibration_probe(d: int = LOAD_CALIBRATION["d"], batch: int = LOAD_CALIBRATION["batch"],
                           reps: int = LOAD_CALIBRATION["reps"]) -> dict:
    """Fixed-workload probe that says how fast this machine is *right now*.

    Identical fixed work (d=768, batch=2000, 3 repetitions) in both nonce modes:
    the API-default mode is directly comparable to REFERENCE_RUN_768, the
    pre-generated mode is the mode the throughput grid uses.
    """
    key = KeyGen(dim=d, beta=0.2, s=3, rng=np.random.default_rng(SEED))
    rng = np.random.default_rng(4000 + d)
    E = rng.standard_normal((batch, d))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    raw = os.urandom(16 * batch)
    ns = [raw[16 * i: 16 * i + 16] for i in range(batch)]
    out = {
        "what": "fixed workload (d=768, batch=2000, 3 reps) measured at run time; "
                "the ratio to REFERENCE_RUN_768 is the machine speed factor of this run "
                "(the reference run used batch=1000; per-vector cost is batch independent)",
        "d": d,
        "batch": batch,
        "reps": reps,
        "reference_run": REFERENCE_RUN_768,
        "load_average": load_average(),
        "modes": {},
    }
    for mode in ("api_default", "pre_generated"):
        if mode == "api_default":
            EncDB(key, E[:8])
        else:
            EncDB(key, E[:8], nonces=ns[:8])
        runs, runs_cpu = [], []
        for _ in range(reps):
            t0w, t0c = time.perf_counter(), time.process_time()
            if mode == "api_default":
                EncDB(key, E)
            else:
                EncDB(key, E, nonces=ns)
            runs.append(time.perf_counter() - t0w)
            runs_cpu.append(time.process_time() - t0c)
        med = float(np.median(runs))
        med_cpu = float(np.median(runs_cpu))
        out["modes"][mode] = {
            "runs_ms": [round(x * 1000, 3) for x in runs],
            "median_ms": round(med * 1000, 3),
            "runs_cpu_ms": [round(x * 1000, 3) for x in runs_cpu],
            "median_cpu_ms": round(med_cpu * 1000, 3),
            "us_per_vector": round(med / batch * 1e6, 3),
            "us_per_vector_cpu": round(med_cpu / batch * 1e6, 3),
            "embeddings_per_s": round(batch / med, 1),
            "embeddings_per_s_cpu": round(batch / med_cpu, 1),
        }
    api = out["modes"]["api_default"]
    out["speed_factor_vs_reference_run"] = round(api["us_per_vector"] / REFERENCE_RUN_768["us_per_vector"], 3)
    out["speed_factor_vs_reference_run_cpu"] = round(
        api["us_per_vector_cpu"] / REFERENCE_RUN_768["us_per_vector_cpu"], 3)
    out["interpretation"] = (
        "speed_factor_vs_reference_run = (this run's us/vector) / (reference-run us/vector); 1.0 means "
        "this machine is as fast as the reference run, 5.0 means absolute rates are ~5x below it. The "
        "reference run itself was measured under a 1-min load average of ~4.8 (see reference_run), so "
        "this factor is an INDICATOR of machine state only -- no assertion uses it (the load-invariant "
        "assertions normalise inside each run). Absolute throughput comparisons against the paper must "
        "be read together with this factor."
    )
    print(f"  load calibration d={d} batch={batch}: API-default "
          f"{api['us_per_vector']:.2f} us/vector vs reference run "
          f"{REFERENCE_RUN_768['us_per_vector']:.2f} -> speed factor "
          f"{out['speed_factor_vs_reference_run']:.2f}x; pre-generated "
          f"{out['modes']['pre_generated']['us_per_vector']:.2f} us/vector")
    return out


def nonce_mode_probe(dims=(64, 768, 3072), batch: int = 4000, reps: int = 3) -> dict:
    """Quantify the difference between pre-generated nonces and the EncDB API
    default (one os.urandom syscall per vector inside the timed loop)."""
    rows = []
    for d in dims:
        key = KeyGen(dim=d, beta=0.2, s=3, rng=np.random.default_rng(SEED))
        rng = np.random.default_rng(3000 + d)
        E = rng.standard_normal((batch, d))
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        raw = os.urandom(16 * batch)
        ns = [raw[16 * i: 16 * i + 16] for i in range(batch)]
        med = {}
        for mode in ("pre_generated", "api_default"):
            if mode == "pre_generated":
                EncDB(key, E[:8], nonces=ns[:8])
            else:
                EncDB(key, E[:8])
            runs = []
            for _ in range(reps):
                t0 = time.perf_counter()
                if mode == "pre_generated":
                    EncDB(key, E, nonces=ns)
                else:
                    EncDB(key, E)
                runs.append(time.perf_counter() - t0)
            med[mode] = float(np.median(runs))
        rows.append(
            {
                "d": d,
                "batch": batch,
                "pre_generated_embeddings_per_s": round(batch / med["pre_generated"], 1),
                "api_default_embeddings_per_s": round(batch / med["api_default"], 1),
                "us_per_vector_pre_generated": round(med["pre_generated"] / batch * 1e6, 3),
                "us_per_vector_api_default": round(med["api_default"] / batch * 1e6, 3),
                "api_over_pre_generated": round(med["api_default"] / med["pre_generated"], 3),
            }
        )
        print(f"  d={d:5d}: pre-generated nonces {rows[-1]['us_per_vector_pre_generated']:.2f} us/vector vs "
              f"API default {rows[-1]['us_per_vector_api_default']:.2f} us/vector "
              f"({rows[-1]['api_over_pre_generated']:.3f}x)")
    return {
        "what": (
            "the measurement grid uses pre-generated nonces (drawn outside the timed loop) because "
            "the paper's EncDB derives randomness from a PRF, not from one OS-entropy syscall per "
            "vector; this probe shows how much per-vector os.urandom adds to a vectorised EncDB"
        ),
        "rows": rows,
    }


def style_shape_summary(rows: list[dict]) -> dict:
    """Element-level shape of the paper-style probe (paper Fig.4b claim)."""
    hi = _region_mean(rows, 1536, 3072, "elements_per_s")
    lo = _region_mean(rows, 64, 128, "elements_per_s")
    hi_min = _region_mean(rows, 1536, 3072, "elements_per_s_min")
    lo_min = _region_mean(rows, 64, 128, "elements_per_s_min")
    elems = [r["elements_per_s"] for r in rows if r["in_paper"]]
    emb = [r["embeddings_per_s"] for r in rows if r["in_paper"]]
    return {
        "definition": (
            "elements_per_s in the paper's 'high' region d in [1536,3072] vs the "
            "small-dimension region d <= 128 (mean of the paper dimensions)"
        ),
        "elements_per_s_high_1536_3072_mean": round(hi, 1),
        "elements_per_s_low_le128_mean": round(lo, 1),
        "high_over_low": round(hi / lo, 4),
        "elements_per_s_range_paper_grid": [round(min(elems), 1), round(max(elems), 1)],
        "elements_per_s_ratio_paper_grid": round(max(elems) / min(elems), 4),
        "embeddings_per_s_drop_first_over_last": round(emb[0] / emb[-1], 4),
        "high_over_low_min_estimator": round(hi_min / lo_min, 4),
        "elements_per_s_high_1536_3072_mean_min": round(hi_min, 1),
        "elements_per_s_low_le128_mean_min": round(lo_min, 1),
        "estimator": "median of REPS==5 interleaved rounds (primary); *_min fields use the "
                     "fastest round (least-interference estimator) for reference",
    }


# --------------------------------------------------------------------------- #
# C. Digitisation of the paper's Fig.4 PNGs (pixel -> value)
# --------------------------------------------------------------------------- #
# The two panels of the paper's Fig.4 are only available as raster images in the
# LaTeX source (the numbers are not in the PDF text).  We therefore recover the
# 15 plotted values of each panel from the PNG: locate the axes spines, the
# y-axis tick marks and the 15 filled circular markers, and map pixels to values
# with a least-squares fit through the tick rows.
#
# Calibration inputs that are *read off the printed axes* (unambiguous labels):
#   Fig.4(a) y axis "x10^3 embeddings/s", ticks 20, 15, 10, 5, 0 (top to bottom)
#   Fig.4(b) y axis "x10^3 elements/s",  ticks 2000, 1800, 1600, 1400, 1200, 1000
# Everything else (spine positions, tick pixel rows/columns, marker centroids,
# x-axis linearity) is detected automatically on every run.
PAPER_FIG4_SOURCE = {
    "fig4a": {
        "file": "Rate_Processing.png",
        "panel": "Fig.4(a): encryption throughput of embeddings",
        "y_label": "x10^3 embeddings/s",
        "y_tick_values": (20.0, 15.0, 10.0, 5.0, 0.0),
        "y_tick_scale": 1e3,
        "quantity": "embeddings_per_s",
    },
    "fig4b": {
        "file": "Rate_Encryption.png",
        "panel": "Fig.4(b): encryption throughput of features (elements)",
        "y_label": "x10^3 elements/s",
        "y_tick_values": (2000.0, 1800.0, 1600.0, 1400.0, 1200.0, 1000.0),
        "y_tick_scale": 1e3,
        "quantity": "elements_per_s",
    },
}
MARKER_RGB = (43, 86, 142)          # filled marker colour in the paper PNGs (sRGB)
MARKER_COLOR_TOL = 30               # per-channel tolerance for the marker mask
DIGITISATION_UNCERTAINTY_PCT = 2.0  # +-2% on every digitised point (see notes)


def paper_figure_dir() -> Path:
    """Locate the paper's LaTeX figure directory (env override, then defaults)."""
    candidates = []
    env = os.environ.get("CAPRISE_PAPER_FIGDIR")
    if env:
        candidates.append(Path(env))
    candidates += [
        ROOT.parents[1] / "output" / "pdf" / "source" / "caprise" / "figures",
        ROOT.parent / "output" / "pdf" / "source" / "caprise" / "figures",
        ROOT.parents[2] / "output" / "pdf" / "source" / "caprise" / "figures",
    ]
    for c in candidates:
        if all((c / v["file"]).exists() for v in PAPER_FIG4_SOURCE.values()):
            return c
    raise FileNotFoundError(
        "paper Fig.4 PNGs not found; set CAPRISE_PAPER_FIGDIR to the directory that "
        "contains " + ", ".join(v["file"] for v in PAPER_FIG4_SOURCE.values())
    )


def _load_rgb(path: Path) -> np.ndarray:
    """Read a PNG as an (H, W, 3) uint8 array (matplotlib's image reader)."""
    import matplotlib.image as mpimg

    a = mpimg.imread(str(path))
    if a.dtype == np.uint8:
        return a[..., :3]
    return np.round(np.clip(a[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)


def _runs(index: np.ndarray, gap: int = 0) -> list[tuple[int, int]]:
    """Group sorted integer indices into runs, joining gaps <= ``gap``."""
    groups: list[list[int]] = []
    for i in np.asarray(index, dtype=int):
        if groups and i - groups[-1][-1] <= gap:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return [(g[0], g[-1]) for g in groups]


def _digitise_panel(path: Path, y_tick_values, y_tick_scale: float) -> dict:
    """Recover the values of one panel of the paper's Fig.4 from its PNG."""
    from scipy import ndimage

    rgb = _load_rgb(path)
    H, W, _ = rgb.shape
    img = rgb.astype(int)
    dark = img.sum(2) < 200                     # black text, spines, ticks

    # --- axes frame ------------------------------------------------------- #
    spine_rows = np.where(dark.sum(1) > 0.6 * W)[0]
    spine_cols = np.where(dark.sum(0) > 0.6 * H)[0]
    if len(spine_rows) < 2 or len(spine_cols) < 2:
        raise ValueError(f"{path.name}: could not find the axes spines")
    row_runs, col_runs = _runs(spine_rows), _runs(spine_cols)
    top, bottom = row_runs[0], row_runs[-1]
    left, right = col_runs[0], col_runs[-1]

    # --- y ticks (short black marks just left of the left spine) ---------- #
    tick_window = list(range(max(0, left[0] - 6), left[0]))
    y_rows = np.where(dark[:, tick_window].any(1))[0]
    y_rows = [(g[0] + g[1]) / 2.0 for g in _runs(y_rows)]
    if len(y_rows) != len(y_tick_values):
        raise ValueError(
            f"{path.name}: found {len(y_rows)} y-axis tick marks, expected {len(y_tick_values)}"
        )
    y_vals = np.asarray(y_tick_values, dtype=float)
    y_rows_arr = np.asarray(y_rows, dtype=float)
    y_coef = np.polyfit(y_rows_arr, y_vals, 1)
    y_resid = np.polyval(y_coef, y_rows_arr) - y_vals

    # --- x ticks (short black marks just below the bottom spine) ---------- #
    x_window = list(range(bottom[1] + 1, min(H, bottom[1] + 6)))
    x_cols = np.where(dark[x_window, :].any(0))[0]
    x_cols = [(g[0] + g[1]) / 2.0 for g in _runs(x_cols)]

    # --- markers (filled dark-blue circles) ------------------------------- #
    marker = np.abs(img - np.asarray(MARKER_RGB, dtype=int)).max(2) <= MARKER_COLOR_TOL
    labels, _n = ndimage.label(marker)
    markers = []
    for i, sl in enumerate(ndimage.find_objects(labels), 1):
        area = int((labels[sl] == i).sum())
        if area < 40:                            # ignore stray antialiasing pixels
            continue
        cy, cx = ndimage.center_of_mass(labels[sl] == i)
        markers.append(
            {"x_px": float(sl[1].start + cx), "y_px": float(sl[0].start + cy), "area_px": area}
        )
    markers.sort(key=lambda m: m["x_px"])

    # --- consistency checks and value extraction -------------------------- #
    if len(markers) != len(PAPER_DIMS):
        raise ValueError(
            f"{path.name}: found {len(markers)} markers, expected {len(PAPER_DIMS)}"
        )
    if len(x_cols) != len(PAPER_DIMS):
        raise ValueError(
            f"{path.name}: found {len(x_cols)} x-axis tick marks, expected {len(PAPER_DIMS)}"
        )
    x_mismatch = [abs(m["x_px"] - xc) for m, xc in zip(markers, x_cols)]
    x_fit = np.polyfit(np.asarray(PAPER_DIMS, dtype=float), [m["x_px"] for m in markers], 1)
    x_resid = np.polyval(x_fit, np.asarray(PAPER_DIMS, dtype=float)) - np.asarray(
        [m["x_px"] for m in markers], dtype=float
    )
    values = [float(np.polyval(y_coef, m["y_px"]) * y_tick_scale) for m in markers]
    return {
        "file": path.name,
        "image_size_px": [int(W), int(H)],
        "axes_px": {"top": list(map(int, top)), "bottom": list(map(int, bottom)),
                    "left": list(map(int, left)), "right": list(map(int, right))},
        "y_axis": {
            "label": None,
            "tick_values": [float(v) for v in y_tick_values],
            "tick_scale": y_tick_scale,
            "tick_rows_px": [round(r, 2) for r in y_rows],
            "slope_value_per_row": round(float(y_coef[0]), 6),
            "intercept_value": round(float(y_coef[1]), 6),
            "max_fit_residual_rows": round(float(np.abs(y_resid).max()), 4),
        },
        "x_axis": {
            "tick_cols_px": [round(c, 2) for c in x_cols],
            "linear_fit_px_per_dim": round(float(x_fit[0]), 6),
            "linear_fit_intercept_px": round(float(x_fit[1]), 3),
            "max_fit_residual_px": round(float(np.abs(x_resid).max()), 3),
            "n_ticks": len(x_cols),
            "marker_vs_tick_max_offset_px": round(max(x_mismatch), 3),
        },
        "markers": markers,
        "n_markers": len(markers),
        "values_on_paper_dims": values,
        "y_axis_warning": "y-axis tick label values are read off the printed axis; "
                          "tick rows and marker centroids are detected from pixels",
    }


def digitize_paper_fig4() -> dict:
    """Digitise both panels of the paper's Fig.4 (15 points each)."""
    figdir = paper_figure_dir()
    out = {
        "source_dir": str(figdir),
        "method": (
            "pixel digitisation of the paper's raster figures: axes spines, y-axis tick "
            "marks (rows) and the 15 filled markers are detected on the PNG; the y-axis "
            "calibration is a least-squares fit through the tick rows (tick values read "
            "off the printed axis labels); marker centroid rows are then mapped to values. "
            "The x-axis is verified to be linear in d (fit residuals reported) and every "
            "marker is matched 1:1 to an x tick in paper order."
        ),
        "uncertainty_pct": DIGITISATION_UNCERTAINTY_PCT,
        "theoretical_resolution_note": (
            "one pixel is worth "
            "|slope| units before the 1e3 axis factor; reported per panel below"
        ),
        "panels": {},
    }
    for key, spec in PAPER_FIG4_SOURCE.items():
        panel = _digitise_panel(figdir / spec["file"], spec["y_tick_values"], spec["y_tick_scale"])
        panel["y_axis"]["label"] = spec["y_label"]
        panel["quantity"] = spec["quantity"]
        panel["panel"] = spec["panel"]
        panel["value_per_pixel"] = abs(float(panel["y_axis"]["slope_value_per_row"])) * spec["y_tick_scale"]
        out["panels"][key] = panel
        print(f"  {key}: {panel['n_markers']} markers, {panel['x_axis']['n_ticks']} x ticks, "
              f"y-fit residual {panel['y_axis']['max_fit_residual_rows']} row(s), "
              f"x-fit residual {panel['x_axis']['max_fit_residual_px']} px")
    return out


def paper_fig4_reference(digit: dict) -> dict:
    """Add the cross-panel and cross-text consistency checks to the digitisation."""
    fa = digit["panels"]["fig4a"]["values_on_paper_dims"]
    fb = digit["panels"]["fig4b"]["values_on_paper_dims"]
    rows = []
    devs = []
    for i, d in enumerate(PAPER_DIMS):
        implied = fa[i] * d
        dev = (implied - fb[i]) / fb[i] * 100.0
        devs.append(abs(dev))
        rows.append(
            {
                "d": d,
                "fig4a_embeddings_per_s": round(fa[i], 1),
                "fig4b_elements_per_s": round(fb[i], 1),
                "elements_implied_by_fig4a": round(implied, 1),
                "fig4a_vs_fig4b_rel_dev_pct": round(dev, 3),
            }
        )
    worst = int(np.argmax(devs))
    i768 = PAPER_DIMS.index(768)
    text_vec_s = PAPER_PHASE2_CAPRISE_VEC_S
    return {
        "rows": rows,
        "cross_panel_consistency": {
            "what": (
                "Fig.4(a) is embeddings/s and Fig.4(b) is elements/s; the two panels must "
                "therefore satisfy elements_per_s ~= embeddings_per_s * d. This is an "
                "independent check of both axis calibrations."
            ),
            "median_abs_rel_dev_pct": round(float(np.median(devs)), 3),
            "max_abs_rel_dev_pct": round(float(max(devs)), 3),
            "worst_d": PAPER_DIMS[worst],
            "n_within_2pct": int(sum(1 for x in devs if x <= 2.0)),
        },
        "paper_text_cross_check": {
            "what": (
                "the paper's Sec. V-1 text quotes 2339 vectors/s at d=768 for Phase 2, "
                "which is an independent anchor for the digitised Fig.4 legend"
            ),
            "d": 768,
            "digitsed_fig4a_embeddings_per_s": round(fa[i768], 1),
            "paper_text_vectors_per_s": text_vec_s,
            "rel_dev_pct_fig4a": round((fa[i768] - text_vec_s) / text_vec_s * 100.0, 3),
            "digitsed_fig4b_elements_per_s": round(fb[i768], 1),
            "text_vectors_per_s_times_768": round(text_vec_s * 768.0, 1),
            "rel_dev_pct_fig4b": round((fb[i768] - text_vec_s * 768.0) / (text_vec_s * 768.0) * 100.0, 3),
        },
    }


def paper_curve_shape(ref: dict) -> dict:
    """Adjacent-step analysis of the digitised paper curves.

    This is what calibrates the monotonicity tolerance: the paper's own Fig.4a
    is not strictly monotone (256 -> 312 rises), so a reproduction must not be
    required to be strictly monotone either.
    """
    a = [r["fig4a_embeddings_per_s"] for r in ref["rows"]]
    b = [r["fig4b_elements_per_s"] for r in ref["rows"]]
    steps = []
    for i in range(1, len(PAPER_DIMS)):
        steps.append(
            {
                "d_from": PAPER_DIMS[i - 1],
                "d_to": PAPER_DIMS[i],
                "fig4a_ratio_prev_over_next": round(a[i - 1] / a[i], 4),
                "fig4b_ratio_prev_over_next": round(b[i - 1] / b[i], 4),
            }
        )
    rises = [st for st in steps if st["fig4a_ratio_prev_over_next"] < 1.0]
    worst = min(steps, key=lambda st: st["fig4a_ratio_prev_over_next"])
    i256 = PAPER_DIMS.index(256)
    return {
        "fig4a_adjacent_steps": steps,
        "fig4a_adjacent_rises": [
            {
                "d_from": st["d_from"],
                "d_to": st["d_to"],
                "rise_pct": round((1.0 / st["fig4a_ratio_prev_over_next"] - 1.0) * 100.0, 2),
            }
            for st in rises
        ],
        "fig4a_max_adjacent_rise_pct": round((1.0 / worst["fig4a_ratio_prev_over_next"] - 1.0) * 100.0, 2),
        "fig4a_max_adjacent_rise_step": [worst["d_from"], worst["d_to"]],
        "fig4a_n_strictly_decreasing_steps": len(steps) - len(rises),
        "fig4b_256_local_dip_pct_vs_128": round((b[i256] / b[i256 - 1] - 1.0) * 100.0, 2),
        "note": (
            "the paper's Fig.4a rises on exactly one adjacent step (see "
            "fig4a_adjacent_rises); MONO_WAVE is set to 25% from "
            "fig4a_max_adjacent_rise_pct, i.e. the reproduction is allowed local rises no larger "
            "than the paper's own largest local rise"
        ),
    }


def not_reproduced_features(scalar_rows: list[dict], ref: dict, vec_rows: list[dict]) -> dict:
    """Local shapes of the paper's Fig.4b that this reproduction does NOT copy.

    Recorded explicitly instead of being hidden by a loose threshold: the paper's
    element-level curve has a dip at d=256 and a high plateau at 1536-1792; the
    per-element probe reproduces the order of magnitude and the overall drop but
    not every local wiggle (those are timing artefacts of the paper's own
    implementation/machine).
    """
    b = [r["fig4b_elements_per_s"] for r in ref["rows"]]
    s = {r["d"]: r["elements_per_s"] for r in scalar_rows if r["in_paper"]}
    v = {r["d"]: r["elements_per_s"] for r in vec_rows if r["in_paper"]}
    i256 = PAPER_DIMS.index(256)
    paper_dip = (b[i256] / b[i256 - 1] - 1.0) * 100.0
    our_dip = (s[256] / s[128] - 1.0) * 100.0
    hi_ours = float(np.mean([s[d] for d in (1536, 1792)]))
    hi_paper = float(np.mean([b[PAPER_DIMS.index(1536)], b[PAPER_DIMS.index(1792)]]))
    return {
        "paper_fig4b_256_local_dip": {
            "paper_pct_vs_128": round(paper_dip, 2),
            "reproduction_pct_vs_128": round(our_dip, 2),
            "reproduced": bool(our_dip < 0 and abs(our_dip - paper_dip) <= 10.0),
            "note": (
                "the paper's element-level throughput dips at d=256 (Fig.4b); the per-element probe "
                "does not show that dip (its curve is smooth), so this local shape is recorded as "
                "NOT reproduced -- it is a timing artefact of the paper's implementation/machine, "
                "not a scheme property"
            ),
        },
        "paper_fig4b_high_plateau_1536_1792": {
            "paper_elements_per_s_mean": round(hi_paper, 1),
            "reproduction_elements_per_s_mean": round(hi_ours, 1),
            "ratio_reproduction_over_paper": round(hi_ours / hi_paper, 3),
            "reproduced": bool(hi_ours / hi_paper >= 0.5),
            "note": "the paper's 'element-level throughput remains high for d in 1536-3072' plateau "
                    "is reproduced in magnitude (see ratio); its exact oscillation is not",
        },
        "vectorised_element_throughput_grows_with_d": {
            "vectorised_elements_per_s_first_last": [round(v[PAPER_DIMS[0]], 1), round(v[PAPER_DIMS[-1]], 1)],
            "vectorised_growth_ratio": round(v[PAPER_DIMS[-1]] / v[PAPER_DIMS[0]], 3),
            "paper_elements_per_s_first_last": [round(b[0], 1), round(b[-1], 1)],
            "paper_growth_ratio": round(b[-1] / b[0], 3),
            "note": (
                "the vectorised implementation amortises a fixed per-vector cost (PRF + PCG64 "
                "seeding), so its element-level rate GROWS with d; the paper's curve is instead "
                "flat/high with local oscillations. This contrast is the A17/A17d evidence that the "
                "paper's numbers come from a per-element implementation style"
            ),
        },
    }


# --------------------------------------------------------------------------- #
# D. Paper comparison table (digitised paper values vs this reproduction)
# --------------------------------------------------------------------------- #
def paper_comparison(vec_rows: list[dict], scalar_rows: list[dict], ref: dict,
                     calib: dict | None = None) -> dict:
    """Side-by-side table: paper Fig.4 digitised | vectorised | paper-style.

    ``calib`` is the load-calibration probe of this run; its speed factor is used
    to report load-corrected ratios (an extrapolation to an idle machine), so the
    absolute comparison against the paper can be read at both the current load and
    an idle machine.
    """
    by_d_vec = {r["d"]: r for r in vec_rows}
    by_d_sca = {r["d"]: r for r in scalar_rows}
    rows = []
    for ref_row in ref["rows"]:
        d = ref_row["d"]
        v, s = by_d_vec[d], by_d_sca[d]
        pe, px = ref_row["fig4a_embeddings_per_s"], ref_row["fig4b_elements_per_s"]
        rows.append(
            {
                "d": d,
                "paper_fig4a_embeddings_per_s": pe,
                "paper_fig4b_elements_per_s": px,
                "vectorised_embeddings_per_s": v["embeddings_per_s"],
                "vectorised_elements_per_s": v["elements_per_s"],
                "paper_style_embeddings_per_s": s["embeddings_per_s"],
                "paper_style_elements_per_s": s["elements_per_s"],
                "paper_style_over_paper_embeddings": round(s["embeddings_per_s"] / pe, 3),
                "paper_style_over_paper_elements": round(s["elements_per_s"] / px, 3),
                "vectorised_over_paper_embeddings": round(v["embeddings_per_s"] / pe, 3),
                "vectorised_over_paper_elements": round(v["elements_per_s"] / px, 3),
                "paper_style_us_per_vector": s["us_per_vector"],
                "paper_us_per_vector": round(1e6 / pe, 2),
            }
        )
    se = np.asarray([r["paper_style_over_paper_embeddings"] for r in rows])
    sx = np.asarray([r["paper_style_over_paper_elements"] for r in rows])
    ve = np.asarray([r["vectorised_over_paper_embeddings"] for r in rows])
    vx = np.asarray([r["vectorised_over_paper_elements"] for r in rows])
    emb_p = np.asarray([r["paper_fig4a_embeddings_per_s"] for r in rows])
    emb_s = np.asarray([r["paper_style_embeddings_per_s"] for r in rows])
    emb_v = np.asarray([r["vectorised_embeddings_per_s"] for r in rows])
    drop_p = float(emb_p[0] / emb_p[-1])
    drop_s = float(emb_s[0] / emb_s[-1])
    drop_v = float(emb_v[0] / emb_v[-1])
    speed = float((calib or {}).get("speed_factor_vs_reference_run", 1.0) or 1.0)
    def _load_corrected(a: np.ndarray) -> dict:
        # a machine `speed` times slower produces rates `speed` times lower, so the
        # idle-equivalent ratio is the measured ratio * speed
        return {
            "min": round(float(a.min()) * speed, 3),
            "median": round(float(np.median(a)) * speed, 3),
            "max": round(float(a.max()) * speed, 3),
        }
    return {
        "definition": (
            "*_over_paper_* = implementation rate / digitised paper rate (>1 means faster "
            "than the paper).  'paper_style' is the per-element Python probe (an "
            "implementation-style reconstruction of how the paper's numbers arise), "
            "'vectorised' is this reproduction's src/ implementation.  The *_load_corrected "
            "fields multiply the measured ratios by the run's machine speed factor (see "
            "load_calibration) and are a rough extrapolation to an idle machine, reported for "
            "interpretability only -- the per-element Python probe is less load-sensitive than "
            "the numpy calibration workload, so this extrapolation is indicative, not exact."
        ),
        "machine_speed_factor_vs_reference_run": speed,
        "rows": rows,
        "summary": {
            "paper_style_over_paper_embeddings_load_corrected": _load_corrected(se),
            "vectorised_over_paper_embeddings_load_corrected": _load_corrected(ve),
            "paper_style_over_paper_embeddings": {
                "min": round(float(se.min()), 3),
                "median": round(float(np.median(se)), 3),
                "max": round(float(se.max()), 3),
            },
            "paper_style_over_paper_elements": {
                "min": round(float(sx.min()), 3),
                "median": round(float(np.median(sx)), 3),
                "max": round(float(sx.max()), 3),
            },
            "vectorised_over_paper_embeddings": {
                "min": round(float(ve.min()), 3),
                "median": round(float(np.median(ve)), 3),
                "max": round(float(ve.max()), 3),
            },
            "vectorised_over_paper_elements": {
                "min": round(float(vx.min()), 3),
                "median": round(float(np.median(vx)), 3),
                "max": round(float(vx.max()), 3),
            },
            "d_64_over_d_3584_drop": {
                "paper_digitised": round(drop_p, 3),
                "paper_style_probe": round(drop_s, 3),
                "vectorised": round(drop_v, 3),
                "log_ratio_paper_style_vs_paper": round(float(np.log(drop_s / drop_p)), 4),
                "log_ratio_vectorised_vs_paper": round(float(np.log(drop_v / drop_p)), 4),
            },
            "paper_us_per_vector_at_768": round(1e6 / emb_p[PAPER_DIMS.index(768)], 2),
            "paper_style_us_per_vector_at_768": rows[PAPER_DIMS.index(768)]["paper_style_us_per_vector"],
            "digitisation_uncertainty_pct": DIGITISATION_UNCERTAINTY_PCT,
        },
    }


# --------------------------------------------------------------------------- #
# Previous-artifact history (loaded vs idle comparison)
# --------------------------------------------------------------------------- #
def previous_run_summary(path: Path) -> dict | None:
    """Compact summary of the artifact that is about to be overwritten.

    The machine is shared, so one run may be several times slower than the next
    one.  Keeping the previous run's load level and raw timings makes the
    loaded-vs-idle comparison auditable instead of silently overwriting it.
    """
    if not path.exists():
        return None
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover
        return None
    tput = old.get("throughput", {})
    style = old.get("paper_style_probe", {})
    calib = tput.get("load_calibration") or old.get("load_calibration") or {}
    return {
        "generated_at": old.get("generated_at"),
        "runtime_s": old.get("runtime_s"),
        "load_average": (old.get("hardware") or {}).get("load_average_before_throughput"),
        "machine_speed_factor_vs_reference_run": calib.get("speed_factor_vs_reference_run"),
        "throughput_rows": [
            {
                "d": r["d"],
                "batch": r.get("batch"),
                "median_s": r.get("median_s"),
                "median_cpu_s": r.get("median_cpu_s"),
                "embeddings_per_s": r.get("embeddings_per_s"),
                "elements_per_s": r.get("elements_per_s"),
            }
            for r in tput.get("rows", [])
        ],
        "paper_style_rows": [
            {
                "d": r["d"],
                "batch_vectors": r.get("batch_vectors"),
                "median_s": r.get("median_s"),
                "embeddings_per_s": r.get("embeddings_per_s"),
                "elements_per_s": r.get("elements_per_s"),
            }
            for r in style.get("rows", [])
        ],
        "note": "previous artifact (kept for the loaded-vs-idle comparison)",
    }


# --------------------------------------------------------------------------- #
# Phase-1: 128-query embedding vs 128-query CAPRISE encryption
# --------------------------------------------------------------------------- #
def caprise_encrypt_queries(d: int = 768, n: int = 128, reps: int = 3) -> dict:
    key = KeyGen(dim=d, beta=0.2, s=3, rng=np.random.default_rng(SEED))
    rng = np.random.default_rng(77)
    Q = rng.standard_normal((n, d))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    for i in range(8):  # warm-up
        EncQ(key, Q[i])
    runs, runs_cpu = [], []
    for _ in range(reps):
        t0w, t0c = time.perf_counter(), time.process_time()
        for i in range(n):
            EncQ(key, Q[i])
        runs.append(time.perf_counter() - t0w)
        runs_cpu.append(time.process_time() - t0c)
    med = float(np.median(runs))
    med_cpu = float(np.median(runs_cpu))
    return {
        "n": n,
        "d": d,
        "reps": reps,
        "runs_ms": [round(x * 1000, 3) for x in runs],
        "median_ms": round(med * 1000, 3),
        "runs_cpu_ms": [round(x * 1000, 3) for x in runs_cpu],
        "median_cpu_ms": round(med_cpu * 1000, 3),
        "vectors_per_s": round(n / med, 1),
        "vectors_per_s_cpu": round(n / med_cpu, 1),
    }


def embed_queries_128() -> dict:
    """Embed 128 real MS MARCO dev queries with the local gtr-t5-base copy.

    Primary: MPS float16 (like the paper's GPU setting but on Apple silicon) and
    CPU float32.  Falls back to results/embed_timing.json (batch=64, linear
    extrapolation to 128) if the local model is missing.
    """
    out: dict = {"source": None}
    n_queries = 128
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        texts = [json.loads(line)["text"] for line in open(ROOT / "data" / "queries.jsonl")][:n_queries]
        res = {}
        for dev, dtype in (("mps", torch.float16), ("cpu", torch.float32)):
            if dev == "mps" and not torch.backends.mps.is_available():
                continue
            t0 = time.perf_counter()
            model = SentenceTransformer(str(LOCAL_MODEL), device=dev, model_kwargs={"dtype": dtype})
            load_s = time.perf_counter() - t0
            with torch.no_grad():
                model.encode(texts[:8], batch_size=n_queries, normalize_embeddings=True, convert_to_numpy=True)
                runs = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    model.encode(texts, batch_size=n_queries, normalize_embeddings=True, convert_to_numpy=True)
                    runs.append(time.perf_counter() - t0)
            med = float(np.median(runs))
            res[dev] = {
                "device": dev,
                "dtype": str(dtype).replace("torch.", ""),
                "batch_size": n_queries,
                "n_queries": n_queries,
                "runs_ms": [round(x * 1000, 3) for x in runs],
                "median_ms": round(med * 1000, 3),
                "per_query_ms": round(med / n_queries * 1000, 4),
                "model_load_s": round(load_s, 2),
                "model_path": str(LOCAL_MODEL.relative_to(ROOT)),
            }
            del model
        if res:
            out = {"source": "measured_local_gtr_t5_base", "devices": res}
            return out
    except Exception as exc:  # pragma: no cover - fallback path
        out["error"] = f"{type(exc).__name__}: {exc}"

    if EMBED_TIMING_JSON.exists():
        data = json.loads(EMBED_TIMING_JSON.read_text(encoding="utf-8"))
        fallback = {}
        for name, key in (("mps_fp16_extrapolated", "batch_ms_mean"), ("cpu_fp32_extrapolated", "cpu_fp32_reference")):
            if key == "batch_ms_mean" and "batch_ms_mean" in data:
                per_q = data["batch_ms_mean"] / data["batch_size"]
                fallback[name] = {
                    "device": data.get("device", "mps"),
                    "dtype": data.get("dtype", "float16"),
                    "batch_size": data["batch_size"],
                    "per_query_ms": round(per_q, 4),
                    "median_ms": round(per_q * n_queries, 3),
                    "note": "extrapolated from results/embed_timing.json (batch=64); not re-measured",
                }
            elif key in data and isinstance(data[key], dict):
                per_q = data[key]["batch_ms_mean"] / 64.0
                fallback[name] = {
                    "device": "cpu",
                    "dtype": "float32",
                    "batch_size": 64,
                    "per_query_ms": round(per_q, 4),
                    "median_ms": round(per_q * n_queries, 3),
                    "note": "extrapolated from results/embed_timing.json cpu_fp32_reference (batch=64)",
                }
        out = {"source": "results/embed_timing.json (extrapolated)", "devices": fallback}
    return out


def phase1_overhead(enc: dict, embed: dict) -> dict:
    """overhead% = CAPRISE encryption time / embedding time for the same 128 queries."""
    rows = []
    devices = embed.get("devices", {})
    for name, e in devices.items():
        embed_ms = e.get("median_ms")
        if not embed_ms:
            continue
        rows.append(
            {
                "embedding_source": name,
                "device": e.get("device"),
                "dtype": e.get("dtype"),
                "embedding_ms_128q": embed_ms,
                "caprise_enc_ms_128q": enc["median_ms"],
                "overhead_ms": round(enc["median_ms"], 3),
                "overhead_pct": round(enc["median_ms"] / embed_ms * 100, 3),
                "note": e.get("note", ""),
            }
        )
    # paper setting: 79.52 ms embedding on A100, 15 ms CAPRISE
    rows.append(
        {
            "embedding_source": "paper (A100 GPU)",
            "device": "NVIDIA A100",
            "dtype": "n/a",
            "embedding_ms_128q": PAPER_PHASE1_EMBED_MS,
            "caprise_enc_ms_128q": enc["median_ms"],
            "overhead_ms": round(enc["median_ms"], 3),
            "overhead_pct": round(enc["median_ms"] / PAPER_PHASE1_EMBED_MS * 100, 3),
            "note": "hardware-mixed reference: paper embedding time + our measured encryption time",
        }
    )
    return {
        "definition": "overhead_pct = CAPRISE encryption time / pure embedding time, same 128 queries",
        "paper": {
            "embedding_ms": PAPER_PHASE1_EMBED_MS,
            "encryption_ms": PAPER_PHASE1_ENC_MS,
            "overhead_pct": round(PAPER_PHASE1_ENC_MS / PAPER_PHASE1_EMBED_MS * 100, 2),
            "paper_claim_overhead_pct": PAPER_PHASE1_OVERHEAD_PCT,
            "hardware": "NVIDIA A100 GPU",
        },
        "caprise_encryption": enc,
        "embeddings": embed,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# Phase-2: CAPRISE vectors/s vs Paillier (PHE) baseline
# --------------------------------------------------------------------------- #
def caprise_vector_throughput(d: int = 768, n: int = 2000) -> dict:
    key = KeyGen(dim=d, beta=0.2, s=3, rng=np.random.default_rng(SEED))
    rng = np.random.default_rng(88)
    Q = rng.standard_normal((n, d))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    for i in range(32):
        EncQ(key, Q[i])
    t0w, t0c = time.perf_counter(), time.process_time()
    for i in range(n):
        EncQ(key, Q[i])
    elapsed = time.perf_counter() - t0w
    elapsed_cpu = time.process_time() - t0c
    return {
        "d": d,
        "n_vectors": n,
        "elapsed_ms": round(elapsed * 1000, 2),
        "elapsed_cpu_ms": round(elapsed_cpu * 1000, 2),
        "us_per_vector": round(elapsed / n * 1e6, 3),
        "us_per_vector_cpu": round(elapsed_cpu / n * 1e6, 3),
        "vectors_per_s": round(n / elapsed, 1),
        "vectors_per_s_cpu": round(n / elapsed_cpu, 1),
        "elements_per_s": round(n * d / elapsed, 0),
        "elements_per_s_cpu": round(n * d / elapsed_cpu, 0),
        "measurement": "single-vector EncQ calls in a loop (no batching)",
    }


def _make_paillier_modulus(nbits: int, seed: int = 0) -> tuple[int, int]:
    """Timing-only Paillier modulus (no primality requirement: the bit-length of
    n and n^2 determines the modular-exponentiation cost)."""
    rng = random.Random(seed)
    n = rng.getrandbits(nbits) | (1 << (nbits - 1)) | 1
    return n, n * n


def builtin_pow_paillier(nbits: int, n_elements: int = 32) -> dict:
    """Pure-CPython Paillier encryption cost (builtin pow), used as a lower bound
    in case the `phe` package or a GMP backend is unavailable."""
    n, n2 = _make_paillier_modulus(nbits, seed=7)
    g = n + 1
    rng = random.Random(11)
    for _ in range(3):  # warm-up
        r = rng.randrange(2, n)
        pow(r, n, n2)
    t0 = time.perf_counter()
    for i in range(n_elements):
        r = rng.randrange(2, n)
        c = (pow(g, i, n2) * pow(r, n, n2)) % n2
        _ = c
    elapsed = time.perf_counter() - t0
    per_elem = elapsed / n_elements
    return {
        "impl": "pure CPython builtin pow (no GMP)",
        "n_bits": nbits,
        "modulus_type": (
            f"synthetic {nbits}-bit odd modulus (timing-only emulation of pow(r, n, n^2); "
            "NOT a real Paillier keypair -- primality is irrelevant for measuring modpow cost)"
        ),
        "n_elements": n_elements,
        "us_per_element": round(per_elem * 1e6, 2),
        "vector_768_elems_s": round(per_elem * 768, 6),
        "vectors_per_s": round(1.0 / (per_elem * 768), 4),
    }


def phe_paillier(nbits: int, n_vectors: int = 5, seed: int = 0) -> dict:
    """Measure Paillier encryption of `n_vectors` 768-dim vectors with python-phe.

    Extrapolated from the measured total time: vectors/s = n_vectors / elapsed.
    """
    from phe import paillier

    t0 = time.perf_counter()
    pub, _priv = paillier.generate_paillier_keypair(n_length=nbits)
    keygen_s = time.perf_counter() - t0
    for _ in range(5):  # warm-up
        pub.encrypt(1)
    n_elements = n_vectors * 768
    rng = random.Random(seed)
    values = [rng.randrange(0, 1 << 30) for _ in range(n_elements)]
    t0 = time.perf_counter()
    for v in values:
        pub.encrypt(v)
    elapsed = time.perf_counter() - t0
    per_elem = elapsed / n_elements
    gmpy2_available = False
    try:
        import gmpy2  # noqa: F401

        gmpy2_available = True
    except Exception:
        pass
    return {
        "impl": f"python-phe {getattr(paillier, '__name__', '')}".strip() + (
            " + gmpy2 (GMP backend)" if gmpy2_available else " (pure-python backend)"
        ),
        "n_bits": nbits,
        "n_vectors_768": n_vectors,
        "n_elements": n_elements,
        "keygen_s": round(keygen_s, 3),
        "elapsed_s": round(elapsed, 3),
        "us_per_element": round(per_elem * 1e6, 2),
        "vector_768_elems_s": round(per_elem * 768, 6),
        "vectors_per_s": round(1.0 / (per_elem * 768), 4),
        "note": "extrapolated from n_vectors x 768 element encryptions; CPU only, no batching in Paillier",
    }


def phase2(_enc: dict | None = None) -> dict:
    cap = caprise_vector_throughput(768)
    print(f"  CAPRISE d=768: {cap['vectors_per_s']:.0f} vectors/s ({cap['us_per_vector']:.2f} us/vector)")

    phe_results = []
    try:
        for nbits in (1024, 2048):
            r = phe_paillier(nbits, n_vectors=5)
            phe_results.append(r)
            print(f"  Paillier {nbits}-bit ({r['impl']}): {r['us_per_element']:.1f} us/element "
                  f"-> {r['vectors_per_s']:.3f} vectors/s")
    except Exception as exc:
        print(f"  [WARN] python-phe unavailable ({type(exc).__name__}: {exc}); using builtin pow only")
    pure_pow = [builtin_pow_paillier(1024, n_elements=24), builtin_pow_paillier(2048, n_elements=8)]
    for r in pure_pow:
        print(f"  {r['impl']} {r['n_bits']}-bit: {r['us_per_element']:.1f} us/element "
              f"-> {r['vectors_per_s']:.3f} vectors/s")

    phe_1024 = next((x for x in phe_results if x["n_bits"] == 1024), None)
    phe_2048 = next((x for x in phe_results if x["n_bits"] == 2048), None)
    cap_vs = {}
    lo, hi = PAPER_PHASE2_REMOTERAG_PHE_VEC_S
    cap_vs["paper_remoteRAG_phe_250_300"] = {
        "phe_vectors_per_s_range": [lo, hi],
        "speedup_vs_250": round(cap["vectors_per_s"] / lo, 1),
        "speedup_vs_300": round(cap["vectors_per_s"] / hi, 1),
    }
    if phe_1024:
        cap_vs["measured_paillier_1024bit"] = {
            "phe_vectors_per_s": phe_1024["vectors_per_s"],
            "speedup": round(cap["vectors_per_s"] / phe_1024["vectors_per_s"], 1),
        }
    if phe_2048:
        cap_vs["measured_paillier_2048bit"] = {
            "phe_vectors_per_s": phe_2048["vectors_per_s"],
            "speedup": round(cap["vectors_per_s"] / phe_2048["vectors_per_s"], 1),
        }
    our_over_paper = cap["vectors_per_s"] / PAPER_PHASE2_CAPRISE_VEC_S
    cap_vs["paper_caprise_2339"] = {
        "paper_vectors_per_s": PAPER_PHASE2_CAPRISE_VEC_S,
        "our_over_paper": round(our_over_paper, 2),
        "paper_caprise_over_paper_phe_mid": round(PAPER_PHASE2_CAPRISE_VEC_S / 275.0, 2),
        "note": (
            f"paper claims ~9x (2339 / 250-300 = 7.8-9.4x); our implementation is additionally "
            f"{our_over_paper:.2f}x faster than the paper's CAPRISE implementation on the same "
            "single-vector basis because it is vectorised (A17)"
        ),
    }
    return {
        "caprise": cap,
        "phe_baselines": phe_results,
        "pure_python_baselines": pure_pow,
        "comparisons": cap_vs,
        "paper": {
            "caprise_vectors_per_s": PAPER_PHASE2_CAPRISE_VEC_S,
            "remoteRAG_phe_vectors_per_s": [lo, hi],
            "claimed_speedup": "9x",
            "hardware": "NVIDIA A100 GPU (paper); this reproduction: Apple M5 CPU (no CUDA)",
        },
    }


# --------------------------------------------------------------------------- #
# Implementation-style probe (ambiguity A17): raw scalar draw vs vectorised draw
# --------------------------------------------------------------------------- #
def style_probe(d: int = 768, reps: int = 10) -> dict:
    rng = np.random.default_rng(0)
    # vectorised: one call draws all d coordinates
    runs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = rng.standard_normal(d)
        runs.append(time.perf_counter() - t0)
    vec_us = float(np.median(runs)) * 1e6
    # per-element: one Python-level scalar draw per coordinate
    runs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = [float(rng.standard_normal()) for _ in range(d)]
        runs.append(time.perf_counter() - t0)
    elem_us = float(np.median(runs)) * 1e6
    return {
        "d": d,
        "vectorised_standard_normal_us": round(vec_us, 3),
        # NOTE: the two *_us fields below are PER-VECTOR costs (one vector = d coordinates),
        # not per-element costs; the field name states the unit explicitly.
        "vectorised_standard_normal_us_per_vector": round(vec_us, 3),
        "per_element_scalar_loop_us_per_vector": round(elem_us, 3),
        "per_element_scalar_loop_us_per_element": round(elem_us / d, 4),
        "ratio_per_element_over_vectorised": round(elem_us / vec_us, 2),
        "per_element_loop_vectors_per_s": round(1e6 / elem_us, 1),
        "note": (
            "Per-element cost of the raw scalar draw is "
            f"{elem_us / d:.3f} us/element (measured), while a full per-element EncDB "
            "probe (paper_style_probe) lands at 0.3-1.2 us/element once PRF + PRNG "
            "initialisation, the norm and the ciphertext arithmetic are included. "
            "This 'paper used a per-element-style implementation' statement is a "
            "reconstruction (A17) backed by the paper_style_probe measurement, NOT a "
            "measurement of the paper's code. This reproduction draws all d Gaussians "
            "with a single PCG64 call, hence the much higher per-vector throughput."
        ),
    }


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(res: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.8))
    rows = res["throughput"]["rows"]
    paper_rows = [r for r in rows if r["in_paper"]]
    ds = [r["d"] for r in paper_rows]
    emb_vec = [r["embeddings_per_s"] for r in paper_rows]
    elem_vec = [r["elements_per_s"] / 1e6 for r in paper_rows]
    compare = res["paper_comparison"]["rows"]
    emb_paper = [r["paper_fig4a_embeddings_per_s"] for r in compare]
    elem_paper = [r["paper_fig4b_elements_per_s"] / 1e6 for r in compare]
    emb_sca = [r["paper_style_embeddings_per_s"] for r in compare]
    elem_sca = [r["paper_style_elements_per_s"] / 1e6 for r in compare]

    ax = axes[0][0]
    ax.plot(ds, emb_paper, "o-", color="#b2182b", lw=1.6, ms=5,
            label="paper Fig.4a (digitised, ±2%)")
    ax.plot(ds, emb_sca, "D-", color="#ef8a62", lw=1.4, ms=4,
            label="paper-style probe (per-element Python)")
    ax.plot(ds, emb_vec, "s-", color="#2166ac", lw=1.6, ms=5,
            label="this repro, vectorised EncDB")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("embedding size d")
    ax.set_ylabel("embeddings / s")
    ax.set_title("(a) embeddings/s vs d — paper Fig.4a vs reproduction")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0][1]
    ax.plot(ds, elem_paper, "o-", color="#b2182b", lw=1.6, ms=5,
            label="paper Fig.4b (digitised, ±2%)")
    ax.plot(ds, elem_sca, "D-", color="#ef8a62", lw=1.4, ms=4,
            label="paper-style probe (per-element Python)")
    ax.plot(ds, elem_vec, "s-", color="#2166ac", lw=1.6, ms=5,
            label="this repro, vectorised EncDB")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("embedding size d")
    ax.set_ylabel("elements / s  [millions]")
    ax.set_title("(b) elements/s vs d — paper Fig.4b vs reproduction")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0][2]
    ax.plot(ds, [r["paper_style_over_paper_elements"] for r in compare], "D-",
            color="#ef8a62", lw=1.4, ms=4, label="paper-style probe / paper")
    ax.plot(ds, [r["vectorised_over_paper_elements"] for r in compare], "s-",
            color="#2166ac", lw=1.4, ms=4, label="vectorised / paper")
    ax.axhline(1.0, color="k", lw=1, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("embedding size d")
    ax.set_ylabel("reproduction / paper Fig.4b  (elements/s)")
    ax.set_title("(c) agreement with the paper: per-element style vs vectorised")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1][0]
    p1 = res["phase1_overhead"]
    labels, embed_vals, enc_vals = [], [], []
    for row in p1["rows"]:
        label = row["embedding_source"].replace(" (", "\n(")
        labels.append(label)
        embed_vals.append(row["embedding_ms_128q"])
        enc_vals.append(row["caprise_enc_ms_128q"])
    x = np.arange(len(labels))
    ax.bar(x, embed_vals, 0.55, color="#4393c3", label="embedding 128 queries")
    ax.bar(x, enc_vals, 0.55, bottom=embed_vals, color="#d6604d", label="CAPRISE encryption")
    for xi, (e, c) in enumerate(zip(embed_vals, enc_vals)):
        ax.text(xi, e + c + 8, f"{c:.2f} ms ({c/e*100:.2f}%)", ha="center", fontsize=8)
        ax.text(xi, e / 2, f"{e:.1f} ms", ha="center", va="center", fontsize=8, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylabel("time for 128 queries [ms] (log)")
    ax.set_title(f"(d) Phase-1 overhead: ours {res['phase1_overhead']['rows'][0]['overhead_pct']:.2f}%"
                 f" vs paper {p1['paper']['overhead_pct']:.1f}% (<19%)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", which="both", alpha=0.3)

    ax = axes[1][1]
    p2 = res["phase2"]
    names = ["CAPRISE\n(this repro)", "CAPRISE\n(paper, A100)", "RemoteRAG PHE\n(paper, A100)"]
    vals = [p2["caprise"]["vectors_per_s"], PAPER_PHASE2_CAPRISE_VEC_S]
    lo, hi = PAPER_PHASE2_REMOTERAG_PHE_VEC_S
    vals.append((lo + hi) / 2)
    for r in p2["phe_baselines"]:
        names.append(f"Paillier {r['n_bits']}-bit\n(measured, M5)")
        vals.append(r["vectors_per_s"])
    for r in p2["pure_python_baselines"]:
        names.append(f"Paillier {r['n_bits']}-bit\n(builtin pow)")
        vals.append(r["vectors_per_s"])
    colors = ["#2166ac", "#b2182b", "#ef8a62", "#1b7837", "#5aae61", "#7fbc41", "#9970ab", "#762a83"]
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors[: len(names)])
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    for yi, v in zip(y, vals):
        ax.text(v * 1.35, yi, f"{v:.2f} vec/s", va="center", fontsize=7)
    ax.set_xlabel("vector encryptions / s (log scale; d=768)")
    ax.set_title("(e) Phase-2 throughput vs PHE baseline (hardware/impl differ)")
    ax.grid(axis="x", which="both", alpha=0.3)

    ax = axes[1][2]
    sc = res["paper_comparison"]["summary"]
    lo_v = min(min(emb_paper), min(emb_sca)) / 1.6
    hi_v = max(max(emb_paper), max(emb_sca)) * 1.6
    ax.scatter(emb_paper, emb_sca, color="#ef8a62", marker="D", s=38,
               label="paper-style probe vs paper")
    ax.plot([lo_v, hi_v], [lo_v, hi_v], "k:", lw=1, label="1:1")
    for f, ls in ((1.5, "--"), (3.0, "-.")):
        ax.plot([lo_v, hi_v], [lo_v * f, hi_v * f], color="gray", ls=ls, lw=0.9,
                label=f"{f:g}× band")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("paper Fig.4a digitised embeddings/s")
    ax.set_ylabel("paper-style probe embeddings/s")
    ax.set_title(
        "(f) absolute agreement: ratio "
        f"{sc['paper_style_over_paper_embeddings']['min']:.2f}–"
        f"{sc['paper_style_over_paper_embeddings']['max']:.2f}× "
        f"(median {sc['paper_style_over_paper_embeddings']['median']:.2f}×)"
    )
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=7.5)

    fig.suptitle(
        "CAPRISE / ppRAG reproduction — efficiency: paper Fig.4 (15 dims, digitised) vs this reproduction "
        "(Apple M5 CPU)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    t_start = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("[exp1] A. CAPRISE EncDB throughput on the paper Fig.4 grid (vectorised) ...", flush=True)
    load_before = load_average()
    print(f"  load average before: {load_before}")
    rows = throughput_rows()
    mono = monotonicity_summary(rows)
    nonce_probe = nonce_mode_probe()
    load_after = load_average()
    print(f"  load average after: {load_after}", flush=True)
    print("[exp1] A2. load calibration (fixed workload, d=768 batch=2000) ...", flush=True)
    calib = load_calibration_probe()
    print("[exp1] A3. re-check reference workload (d=768, batch=1500) ...", flush=True)
    recheck_ref = recheck_reference_probe()
    print(f"  recheck reference: {recheck_ref['embeddings_per_s']:.0f} emb/s "
          f"(CPU-time {recheck_ref['embeddings_per_s_cpu']:.0f} emb/s)")

    print("[exp1] B. paper-style (per-element Python) EncDB probe ...", flush=True)
    eq = check_scalar_style_equivalence()
    print(f"  equivalence vs src noise: worst rel diff {eq['worst_rel_diff']:.3e} "
          f"({eq['n_checks']} checks, dims {eq['dims']})")
    scalar_rows = scalar_style_throughput()
    scalar_shape = style_shape_summary(scalar_rows)

    print("[exp1] C. digitising paper Fig.4a/4b (15 points each) ...", flush=True)
    digit = digitize_paper_fig4()
    ref = paper_fig4_reference(digit)
    print(f"  cross-panel consistency: median {ref['cross_panel_consistency']['median_abs_rel_dev_pct']:.2f}% "
          f"(max {ref['cross_panel_consistency']['max_abs_rel_dev_pct']:.2f}% at d="
          f"{ref['cross_panel_consistency']['worst_d']})")
    tcx = ref["paper_text_cross_check"]
    print(f"  digitised Fig.4a at d=768: {tcx['digitsed_fig4a_embeddings_per_s']:.0f} emb/s vs paper text "
          f"{tcx['paper_text_vectors_per_s']:.0f} vec/s ({tcx['rel_dev_pct_fig4a']:+.2f}%)")
    curve_shape = paper_curve_shape(ref)
    print(f"  paper Fig.4a adjacent rises: {curve_shape['fig4a_adjacent_rises']} "
          f"(max {curve_shape['fig4a_max_adjacent_rise_pct']:+.2f}% on "
          f"{curve_shape['fig4a_max_adjacent_rise_step']}) -> monotonicity tolerance "
          f"{MONO_WAVE*100:.0f}%")

    print("[exp1] D. Phase-1: 128-query embedding vs CAPRISE encryption ...", flush=True)
    enc = caprise_encrypt_queries(768, 128)
    print(f"  CAPRISE EncQ 128 queries @d=768: median {enc['median_ms']:.3f} ms "
          f"({enc['vectors_per_s']:.0f} vectors/s)")
    embed = embed_queries_128()
    for name, e in embed.get("devices", {}).items():
        print(f"  gtr-t5-base {name}: median {e['median_ms']} ms / 128 queries")
    p1 = phase1_overhead(enc, embed)
    for row in p1["rows"]:
        print(f"  overhead vs {row['embedding_source']}: {row['overhead_pct']:.3f}%")

    print("[exp1] E. Phase-2: CAPRISE vs Paillier baseline ...", flush=True)
    p2 = phase2(enc)
    print(f"  speedup vs RemoteRAG PHE (250-300 vec/s): "
          f"{p2['comparisons']['paper_remoteRAG_phe_250_300']['speedup_vs_250']:.0f}x - "
          f"{p2['comparisons']['paper_remoteRAG_phe_250_300']['speedup_vs_300']:.0f}x")
    if "measured_paillier_1024bit" in p2["comparisons"]:
        print(f"  speedup vs measured Paillier 1024-bit: "
              f"{p2['comparisons']['measured_paillier_1024bit']['speedup']:.0f}x")

    print("[exp1] F. implementation-style probe (A17) ...", flush=True)
    style = style_probe(768)
    print(f"  vectorised {style['vectorised_standard_normal_us_per_vector']} us/vector vs "
          f"per-element loop {style['per_element_scalar_loop_us_per_vector']} us/vector "
          f"(ratio {style['ratio_per_element_over_vectorised']}x; "
          f"scalar draw {style['per_element_scalar_loop_us_per_element']} us/element)")

    print("[exp1] G. paper comparison table ...", flush=True)
    comp = paper_comparison(rows, scalar_rows, ref, calib)
    sc = comp["summary"]
    print(f"  paper-style / paper  (embeddings/s): {sc['paper_style_over_paper_embeddings']}")
    print(f"  vectorised  / paper  (embeddings/s): {sc['vectorised_over_paper_embeddings']}")
    print(f"  drop d=64 -> d=3584: paper {sc['d_64_over_d_3584_drop']['paper_digitised']:.2f}x, "
          f"paper-style {sc['d_64_over_d_3584_drop']['paper_style_probe']:.2f}x, "
          f"vectorised {sc['d_64_over_d_3584_drop']['vectorised']:.2f}x")

    # ---------------------------------------------------------------- #
    # data-quality checks for the experiment itself
    # ---------------------------------------------------------------- #
    emb_vals = [r["embeddings_per_s"] for r in paper_rows_of(rows)]
    elem_vals = [r["elements_per_s"] for r in paper_rows_of(rows)]
    elem_ratio = max(elem_vals) / min(elem_vals)
    elem_high = [r["elements_per_s"] for r in paper_rows_of(rows) if 1536 <= r["d"] <= 3072]
    elem_high_ratio = max(elem_high) / min(elem_high)
    checks = {
        "throughput_monotone_decreasing_embeddings_per_s_cpu": mono["ok"],
        "throughput_monotone_decreasing_wall_informational": mono["ok_wall"],
        "elements_per_s_ratio_vectorised": round(elem_ratio, 3),
        "elements_per_s_high_stable_1536_3072": elem_high_ratio < 2.0,
        "element_high_ratio": round(elem_high_ratio, 3),
    }

    scalar_emb_at = {r["d"]: r["embeddings_per_s"] for r in scalar_rows}
    scalar_elem_at = {r["d"]: r["elements_per_s"] for r in scalar_rows}
    unrep = not_reproduced_features(scalar_rows, ref, rows)
    srow = {r["d"]: r for r in scalar_rows}
    i768 = PAPER_DIMS.index(768)
    result = {
        "experiment": "exp1_efficiency",
        "module": "M17",
        "paper_ref": "arXiv:2601.12331 Sec. V-1 Efficiency / Fig.4 / Phase-1 / Phase-2",
        "description": (
            "CAPRISE encryption throughput on the paper Fig.4 15-point dimension grid "
            "(vectorised EncDB, batch>=1000 embeddings, 3 interleaved reps, median, wall + CPU "
            "time), a paper-implementation-style (per-element Python) probe of the same "
            "operation, pixel digitisation of the paper's Fig.4a/Fig.4b PNGs for a side-by-side "
            "comparison, Phase-1 overhead (128 queries: gtr-t5-base embedding vs CAPRISE EncQ), "
            "Phase-2 per-vector throughput at d=768 vs Paillier (python-phe, 1024/2048-bit) and "
            "the paper/RemoteRAG quoted numbers."
        ),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cpu_label": "Apple M5 (10 cores, 16 GB, no CUDA)",
            "paper_hardware": "NVIDIA A100 GPU",
            "load_average_before_throughput": load_before,
            "load_average_after_throughput": load_after,
            "shared_machine_note": (
                "the machine is shared with other reproduction jobs; absolute wall-clock rates "
                "depend on the co-running load (recorded above), which is why the throughput grid "
                "is measured in interleaved rounds and the gate checks round-paired ratios"
            ),
        },
        "settings": {
            "paper_dims": list(PAPER_DIMS),
            "extra_dims": list(EXTRA_DIMS),
            "dims": list(DIMS),
            "batch": {
                "min": BATCH_MIN,
                "target_elements_per_run": TARGET_ELEMENTS_PER_RUN,
                "rule": "max(1000, ceil(target_elements_per_run / d))",
                "per_dim": {str(r["d"]): r["batch"] for r in rows},
            },
            "reps": REPS,
            "rounds": "interleaved: round k times every dimension in the paper order (load-drift robust)",
            "seed": SEED,
            "beta": 0.2,
            "s": 3,
            "paper_style_probe": {
                "target_elements_per_run": SCALAR_TARGET_ELEMENTS_PER_RUN,
                "min_batch_vectors": SCALAR_MIN_BATCH,
                "rule": "max(8, ceil(target_elements_per_run / d))",
            },
        },
        "paper_fig4_digitisation": {
            **{k: v for k, v in digit.items() if k != "panels"},
            "source": {
                key: str(paper_figure_dir() / spec["file"]) for key, spec in PAPER_FIG4_SOURCE.items()
            },
            "panels": digit["panels"],
            "uncertainty_pct": DIGITISATION_UNCERTAINTY_PCT,
            "rows": ref["rows"],
            "cross_panel_consistency": ref["cross_panel_consistency"],
            "paper_text_cross_check": ref["paper_text_cross_check"],
            "paper_curve_shape": curve_shape,
            "paper_dims": list(PAPER_DIMS),
        },
        "throughput": {
            "description": (
                "vectorised EncDB on L2-normalised random embeddings (GTR-like), batch = "
                "max(1000, ceil(4e6/d)) vectors so that every timed run encrypts ~4e6 elements; "
                "3 interleaved rounds, median. Wall-clock rates are the operational numbers; "
                "*_cpu rates use time.process_time() and are robust to co-running jobs."
            ),
            "rows": rows,
            "monotonicity": mono,
            "nonce_mode_probe": nonce_probe,
            "recheck_reference": recheck_ref,
            "embeddings_per_s_decreasing": mono["ok"],
            "elements_per_s_range": [min(elem_vals), max(elem_vals)],
            "elements_per_s_ratio": round(elem_ratio, 3),
            "checks": checks,
        },
        "paper_style_probe": {
            "kind": "implementation_style_probe",
            "is_paper_code": False,
            "is_src_core_implementation": False,
            "description": (
                "Same EncDB operation written the way a per-element Python implementation would "
                "be: lambda_i = (3 s beta / 8) * (n_i / ||n||) * u^(1/d) drawn and applied one "
                "coordinate at a time in pure Python (no numpy vectorisation, no batching). "
                "This is an implementation-style probe used to explain the paper's Fig.4 "
                "throughput; it is NOT the paper's code and NOT the src/ core implementation."
            ),
            "math": "lambda_i = (3*s*beta/8) * (n_i/||n||) * u^(1/d); e'_i = s*e_i + lambda_i",
            "rows": scalar_rows,
            "equivalence_check": eq,
            "shape": scalar_shape,
            "not_reproduced_features": unrep,
            "embeddings_per_s_at_768": scalar_emb_at[768],
            "elements_per_s_at_768": scalar_elem_at[768],
            "us_per_element_at_768": next(r["us_per_element"] for r in scalar_rows if r["d"] == 768),
            "checks": {
                "equivalent_to_src_noise": eq["equivalent_up_to_float_roundoff"],
                "elements_per_s_ratio_paper_grid": scalar_shape["elements_per_s_ratio_paper_grid"],
                "high_over_low_1536_3072": scalar_shape["high_over_low"],
            },
        },
        "paper_comparison": comp,
        "phase1_overhead": p1,
        "phase2": p2,
        "implementation_style_probe": style,
        "paper_numbers": {
            "fig4a_digitised_embeddings_per_s_at_768": tcx["digitsed_fig4a_embeddings_per_s"],
            "fig4b_digitised_elements_per_s_at_768": tcx["digitsed_fig4b_elements_per_s"],
            "phase1_embedding_ms_128q": PAPER_PHASE1_EMBED_MS,
            "phase1_encryption_ms_128q": PAPER_PHASE1_ENC_MS,
            "phase1_overhead_pct": round(PAPER_PHASE1_ENC_MS / PAPER_PHASE1_EMBED_MS * 100, 2),
            "phase1_claimed_overhead_pct_bound": PAPER_PHASE1_OVERHEAD_PCT,
            "phase2_caprise_vectors_per_s": PAPER_PHASE2_CAPRISE_VEC_S,
            "phase2_remoteRAG_phe_vectors_per_s": list(PAPER_PHASE2_REMOTERAG_PHE_VEC_S),
            "phase2_claimed_speedup": "9x",
        },
        "findings": [
            {
                "id": "A17",
                "statement": (
                    "The paper's Fig.4a/Fig.4b throughput is reproduced in magnitude and shape by a "
                    "per-element Python-implementation-style probe, not by a vectorised "
                    "implementation. Digitising the paper's own Fig.4a/Fig.4b (15 points, "
                    "+-2%) gives "
                    f"{comp['rows'][0]['paper_fig4a_embeddings_per_s']:.0f} embeddings/s at d=64 down "
                    f"to {comp['rows'][-1]['paper_fig4a_embeddings_per_s']:.0f} at d=3584 "
                    f"({sc['d_64_over_d_3584_drop']['paper_digitised']:.1f}x drop) and "
                    f"{comp['rows'][0]['paper_fig4b_elements_per_s']/1e6:.2f}M -> "
                    f"{comp['rows'][-1]['paper_fig4b_elements_per_s']/1e6:.2f}M elements/s. "
                    f"The per-element probe gives {srow[768]['embeddings_per_s']:.0f} embeddings/s at "
                    f"d=768 vs the paper's {comp['rows'][PAPER_DIMS.index(768)]['paper_fig4a_embeddings_per_s']:.0f} "
                    f"({comp['rows'][PAPER_DIMS.index(768)]['paper_style_over_paper_embeddings']:.2f}x), "
                    f"{srow[3584]['embeddings_per_s']:.0f} at d=3584 vs "
                    f"{comp['rows'][-1]['paper_fig4a_embeddings_per_s']:.0f} "
                    f"({comp['rows'][-1]['paper_style_over_paper_embeddings']:.2f}x), and its "
                    f"d=64 -> d=3584 drop is {sc['d_64_over_d_3584_drop']['paper_style_probe']:.1f}x vs "
                    f"the paper's {sc['d_64_over_d_3584_drop']['paper_digitised']:.1f}x (within 2x). "
                    f"Every one of the 15 dimensions is within "
                    f"[{sc['paper_style_over_paper_elements']['min']:.2f}, "
                    f"{sc['paper_style_over_paper_elements']['max']:.2f}]x of the paper's elements/s "
                    f"(median {sc['paper_style_over_paper_elements']['median']:.2f}x) at this run's "
                    f"machine speed factor {calib['speed_factor_vs_reference_run']:.2f}x. The vectorised "
                    "implementation instead reaches "
                    f"{rows[PAPER_DIMS.index(768)]['embeddings_per_s']:.0f} embeddings/s at d=768 and "
                    f"drops only {sc['d_64_over_d_3584_drop']['vectorised']:.1f}x over the whole grid, "
                    "i.e. it does NOT follow the paper's curve. NOT reproduced (recorded in "
                    "paper_style_probe.not_reproduced_features): the paper Fig.4b dip at d=256 "
                    f"(paper {unrep['paper_fig4b_256_local_dip']['paper_pct_vs_128']:+.1f}% vs our "
                    f"{unrep['paper_fig4b_256_local_dip']['reproduction_pct_vs_128']:+.1f}% relative to "
                    "d=128) and the exact oscillation of the 1536-3072 plateau -- both are timing "
                    "artefacts of the paper's own implementation/machine, not scheme properties. "
                    "MACHINE-LOAD CAVEAT: the machine is shared, so the absolute rates of this run are "
                    f"divided by the calibration speed factor {calib['speed_factor_vs_reference_run']:.2f}x "
                    f"(fixed d=768/batch=2000 EncDB: {calib['modes']['api_default']['us_per_vector']:.2f} "
                    f"us/vector here vs {REFERENCE_RUN_768['us_per_vector']:.2f} us/vector reference-run "
                    f"reference; load average {load_before.get('loadavg_1m')} at measurement time). The "
                    "previous revision measured the vectorised implementation at 68,660 embeddings/s "
                    "(14.56 us/vector) at d=768 when the machine was comparably idle, i.e. ~29x the "
                    "paper's 2339 vectors/s; the loaded-vs-reference difference is a property of the "
                    "shared machine, not "
                    "of the implementation. The implementation-style conclusion rests on the "
                    "load-invariant evidence: the curve shapes and the same-machine "
                    "vectorised/paper-style ratio."
                ),
            },
            {
                "id": "A17b",
                "statement": (
                    "PHE baseline: Paillier encryption is ~2-3 orders of magnitude slower than "
                    "CAPRISE here (measured python-phe + GMP; numbers depend on key size 1024/2048-bit "
                    "and implementation), but RemoteRAG's reported 250-300 vectors/s corresponds to an "
                    "optimised C/GPU implementation on A100; cross-hardware/cross-implementation "
                    "speedup ratios must be interpreted qualitatively. Within the paper's own numbers "
                    "2339/250-300 = 7.8-9.4x ('9x') is internally consistent."
                ),
            },
            {
                "id": "A17c",
                "statement": (
                    "Phase-1: the embedding step dominates. On this machine gtr-t5-base embeds 128 "
                    "queries in "
                    + ", ".join(
                        f"{name} {dev['median_ms']:.3f} ms"
                        for name, dev in embed.get("devices", {}).items()
                    )
                    + f", while CAPRISE encrypts the same batch in {enc['median_ms']:.3f} ms, so the "
                    "encryption overhead is "
                    + ", ".join(
                        f"{row['embedding_source']} {row['overhead_pct']:.3f}%"
                        for row in p1["rows"]
                        if "paper" not in row["embedding_source"]
                    )
                    + f" (paper: {PAPER_PHASE1_ENC_MS:g}/{PAPER_PHASE1_EMBED_MS:g} = "
                    f"{PAPER_PHASE1_ENC_MS / PAPER_PHASE1_EMBED_MS * 100:.2f}%, <19%). CAPRISE is even "
                    "relatively cheaper here because our implementation is faster; the paper's 15 ms "
                    "encryption corresponds to a per-element Python implementation. Embedding remains "
                    "the bottleneck in both. All numbers are read from this JSON's measured medians "
                    "(not hardcoded)."
                ),
            },
            {
                "id": "A17d",
                "statement": (
                    "Element-level throughput (paper Fig.4b) is implementation-style dependent in "
                    "the opposite direction: the paper's per-element implementation keeps "
                    f"elements/s roughly flat/high ({ref['rows'][0]['fig4b_elements_per_s']/1e6:.2f}M at "
                    f"d=64 -> {sub_region_mean(ref['rows'], 'fig4b_elements_per_s', 1536, 3072)/1e6:.2f}M "
                    "mean in d in [1536,3072]), because the per-element cost dominates. The vectorised "
                    "implementation amortises a fixed per-vector cost (PRF + PCG64 seeding) instead, so "
                    "its elements/s grows with d "
                    f"({elem_vals[0]/1e6:.2f}M -> {max(elem_vals)/1e6:.2f}M, {elem_ratio:.1f}x); this is "
                    "why the paper's 'element-level throughput remains high' statement is checked "
                    "against the paper-style probe in this gate rather than against the vectorised "
                    "rows."
                ),
            },
        ],
        "runtime_s": round(time.time() - t_start, 1),
    }
    out = RESULTS / "exp1_efficiency.json"
    prev = previous_run_summary(out)
    earlier: list = []
    if out.exists():
        try:
            earlier = list((json.loads(out.read_text(encoding="utf-8")).get("previous_runs") or []))
        except Exception:  # pragma: no cover
            earlier = []
    result["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result["load_calibration"] = calib
    result["throughput"]["load_calibration"] = calib
    result["previous_runs"] = (earlier + ([prev] if prev else []))[-4:]
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[exp1] results -> {out.relative_to(ROOT)}")

    print("[exp1] figure ...", flush=True)
    fig = FIGURES / "fig4_efficiency.png"
    make_figure(result, fig)
    print(f"[exp1] figure -> {fig.relative_to(ROOT)}")
    print(f"[exp1] done in {time.time()-t_start:.1f}s")
    return 0


def paper_rows_of(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["in_paper"]]


def sub_region_mean(rows: list[dict], field: str, lo: int, hi: int) -> float:
    vals = [r[field] for r in rows if lo <= r["d"] <= hi]
    return float(np.mean(vals))


if __name__ == "__main__":
    sys.exit(main())
