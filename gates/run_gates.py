"""Run reproduction module gates in canonical order; stop at first failure.

Usage
-----
    .venv/bin/python -m gates.run_gates --all
    .venv/bin/python -m gates.run_gates --from M1 --to M9
    .venv/bin/python -m gates.run_gates --list

Discipline (user requirement + paper-reproduction skill):
modules run in order P0 -> M0 -> M1 -> ...; the runner stops at the first
failing gate and reports the reason.  A missing gate script is NOT_IMPLEMENTED
and (unless --allow-missing) stops the run: a module that has not been verified
must not be considered passed.

``run_sequence`` is exposed separately so the Phase-0 skeleton gate can verify
the fail-fast behaviour of the real runner with a deliberately failing gate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# canonical order: (module id, gate name, module path)
GATES = [
    ("P0", "skeleton", "gates.gate_p0_skeleton"),
    ("M0", "data", "gates.gate_m0_data"),
    ("M1", "prf", "gates.gate_m1_prf"),
    ("M2", "keygen", "gates.gate_m2_keygen"),
    ("M3", "noise", "gates.gate_m3_noise"),
    ("M4", "encdb", "gates.gate_m4_encdb"),
    ("M5", "decdb", "gates.gate_m5_decdb"),
    ("M6", "encq", "gates.gate_m6_encq"),
    ("M7", "eq2", "gates.gate_m7_eq2"),
    ("M8", "eq3", "gates.gate_m8_eq3"),
    ("M9", "adcpe", "gates.gate_m9_adcpe"),
    ("M10", "vector_analysis", "gates.gate_m10_vector_analysis"),
    ("M11", "vec2text", "gates.gate_m11_vec2text"),
    ("M12", "distance_dp", "gates.gate_m12_distance_dp"),
    ("M13", "theorem1", "gates.gate_m13_theorem1"),
    ("M14", "phase1", "gates.gate_m14_phase1"),
    ("M15", "phase2", "gates.gate_m15_phase2"),
    ("M16", "phase3", "gates.gate_m16_phase3"),
    ("M17", "efficiency", "gates.gate_m17_efficiency"),
    ("M18", "retrieval", "gates.gate_m18_retrieval"),
    ("M20", "asr_protocol", "gates.gate_m20_asr_protocol"),
]

ORDER = [g[0] for g in GATES]


def run_sequence(entries: list[tuple[str, str, str]], allow_missing: bool = False) -> tuple[list[dict], tuple[str, str] | None]:
    """Run ``(module_id, gate_name, module_path)`` entries in order, fail-fast.

    Returns (summary, failed) where failed = (module_id, status) or None.
    """
    summary: list[dict] = []
    failed: tuple[str, str] | None = None
    for mid, name, mod in entries:
        script = ROOT / "gates" / f"gate_{mid.lower()}_{name}.py"
        if mod.startswith("gates._selftest."):
            script = ROOT / (mod.replace(".", "/") + ".py")
        res_path = RESULTS / f"gate_{mid}_{name}.json"
        print(f"\n===== {mid} {name} =====")
        if not script.exists():
            status = "NOT_IMPLEMENTED"
            print(f"  [{status}] {script.relative_to(ROOT)} does not exist")
            summary.append({"module": mid, "name": name, "status": status})
            if not allow_missing:
                failed = (mid, status)
                break
            continue
        proc = subprocess.run([sys.executable, "-m", mod], cwd=ROOT)
        if not res_path.exists():
            status = "FAIL_RUN"  # gate did not produce machine-readable evidence
        else:
            res = json.loads(res_path.read_text(encoding="utf-8"))
            status = "PASS" if res.get("pass") else "FAIL"
            if status == "PASS" and proc.returncode != 0:
                status = "FAIL_RUN"  # gate claimed pass but exited non-zero: inconsistent
        summary.append(
            {
                "module": mid,
                "name": name,
                "status": status,
                "result_file": str(res_path.relative_to(ROOT)) if res_path.exists() else None,
            }
        )
        if status != "PASS":
            failed = (mid, status)
            break
    return summary, failed


def _in_range(mid: str, start: str | None, stop: str | None) -> bool:
    i = ORDER.index(mid)
    if start is not None and i < ORDER.index(start):
        return False
    if stop is not None and i > ORDER.index(stop):
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None, metavar="Mx")
    ap.add_argument("--to", dest="stop", default=None, metavar="My")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--allow-missing", action="store_true", help="do not fail on not-yet-written gates")
    args = ap.parse_args()

    if args.list:
        for mid, name, mod in GATES:
            exists = (ROOT / "gates" / f"gate_{mid.lower()}_{name}.py").exists()
            print(f"{mid:>4} {name:<16} {'ready' if exists else 'not implemented'}")
        return 0

    start = args.start if args.start else ORDER[0]
    stop = args.stop if args.stop else ORDER[-1]
    if start not in ORDER or stop not in ORDER:
        print(f"unknown module range {start}..{stop}")
        return 2
    if ORDER.index(start) > ORDER.index(stop):
        print("invalid range")
        return 2
    if args.all:
        start, stop = ORDER[0], ORDER[-1]

    RESULTS.mkdir(exist_ok=True)
    entries = [e for e in GATES if _in_range(e[0], start, stop)]
    summary, failed = run_sequence(entries, allow_missing=args.allow_missing)

    out = {
        "range": [start, stop],
        "modules": summary,
        "failed_module": None if failed is None else failed[0],
        "pass": failed is None and all(m["status"] == "PASS" for m in summary),
    }
    (RESULTS / "gate_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== SUMMARY =====")
    for m in summary:
        print(f"{m['module']:>4} {m['name']:<16} {m['status']}")
    print("OVERALL:", "PASS" if out["pass"] else f"STOPPED AT {failed[0]} ({failed[1]})")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
