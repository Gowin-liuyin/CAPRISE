"""Collect all module-gate and experiment results into a single summary.

Reads results/gate_*.json and results/exp*.json and writes:
  results/SUMMARY.md          human-readable tables for the report
  results/summary.json        machine-readable aggregate

Run: .venv/bin/python scripts/collect_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

GATE_ORDER = [
    "P0", "M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "M10", "M11", "M12", "M13", "M14", "M15", "M16", "M17", "M18",
]


def load_gates() -> dict:
    gates = {}
    for path in sorted(RESULTS.glob("gate_*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        mid = d.get("module")
        if mid in GATE_ORDER:
            gates[mid] = d
    return gates


def load_experiments() -> dict:
    out = {}
    for name in ["exp1_efficiency", "exp2_retrieval", "exp3_vec2text",
                 "exp4_vector_analysis", "exp5_query_dp"]:
        path = RESULTS / f"{name}.json"
        if path.exists():
            try:
                out[name] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                out[name] = {"load_error": str(exc)}
    return out


def main() -> int:
    gates = load_gates()
    exps = load_experiments()

    lines = ["# 复现结果汇总（自动生成）", ""]
    lines.append("## 模块门禁")
    lines.append("")
    lines.append("| 模块 | 门禁 | 结果 | 检查数 | 失败检查 | 证据 |")
    lines.append("|---|---|---|---|---|---|")
    n_pass = 0
    for mid in GATE_ORDER:
        d = gates.get(mid)
        if d is None:
            lines.append(f"| {mid} | — | NOT_RUN | — | — | — |")
            continue
        checks = d.get("checks", [])
        fails = [c for c in checks if not c.get("pass")]
        status = "PASS" if d.get("pass") else "FAIL"
        n_pass += status == "PASS"
        lines.append(
            f"| {mid} | {d.get('name','')} | {status} | {len(checks)} | "
            f"{len(fails)} | results/gate_{mid}_{d.get('name','')}.json |"
        )
    lines.append("")
    lines.append(f"**门禁 PASS: {n_pass}/{len([m for m in GATE_ORDER if m in gates])}（已运行模块）**")
    lines.append("")

    if exps:
        lines.append("## 实验产物")
        lines.append("")
        lines.append("| 实验 | 关键字段 | 备注 |")
        lines.append("|---|---|---|")
        for name, d in exps.items():
            keys = [k for k in d.keys()][:12]
            lines.append(f"| {name} | {', '.join(keys)} | results/{name}.json |")
        lines.append("")

    (RESULTS / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    (RESULTS / "summary.json").write_text(
        json.dumps(
            {
                "gates": {m: {"pass": d.get("pass"), "n_checks": len(d.get("checks", []))}
                          for m, d in gates.items()},
                "experiments_present": list(exps.keys()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote results/SUMMARY.md ({len(lines)} lines) and results/summary.json")
    print(f"gates present: {sorted(gates.keys(), key=lambda m: GATE_ORDER.index(m))}")
    print(f"experiments present: {list(exps.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
