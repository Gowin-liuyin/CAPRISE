"""P0 gate: skeleton / harness self-check (Phase 0 of the reproduction protocol).

The gate that verifies the verification machinery itself:

1. Required reproduction artifacts exist (SPEC, paper analysis, module plan,
   claims checklist, findings, state table, report, gate framework).
2. Module plan defines executable criteria for every module M0..M18.
3. Claims checklist has the required verdict column (VERIFIED/...).
4. **Fail-fast test**: a deliberately failing gate makes the real runner stop at
   that module, exit non-zero, and persist ``pass: false`` in its JSON.
5. **Normal path test**: a passing gate yields ``pass: true`` and the runner
   continues to completion.

No module implementation may start before this gate passes (skill: Phase 0).
"""

from __future__ import annotations

import json
import re
import sys

from gates.common import Gate, add_src_to_path

add_src_to_path()
from gates.run_gates import GATES, ROOT, RESULTS, run_sequence  # noqa: E402

g = Gate("P0", "skeleton", owner="lead", paper_ref="paper-reproduction skill Phase 0")

# --------------------------------------------------------------------------- #
# 1) required artifacts
# --------------------------------------------------------------------------- #
required = [
    "SPEC.md",
    "docs/PAPER_ANALYSIS.md",
    "docs/MODULE_PLAN.md",
    "docs/CLAIMS_CHECKLIST.md",
    "docs/FINDINGS.md",
    "REPRO_STATE.md",
    "REPRODUCTION_REPORT.md",
    "gates/common.py",
    "gates/run_gates.py",
    "requirements.txt",
]
missing = [p for p in required if not (ROOT / p).exists()]
g.check("required_artifacts_present", [], missing, f"{len(required)} required files")

# --------------------------------------------------------------------------- #
# 2) module plan has executable criteria for every module
# --------------------------------------------------------------------------- #
plan = (ROOT / "docs/MODULE_PLAN.md").read_text(encoding="utf-8")
plan_ids = set(re.findall(r"\*\*(M\d+)\*\*", plan)) | set(re.findall(r"\|\s*\*\*(M\d+)\*\*\s*\|", plan))
plan_ids |= set(re.findall(r"^" + r"\| \*\*(M\d+)\*\*", plan, flags=re.M))
expected_ids = {m[0] for m in GATES if m[0].startswith("M")}
missing_ids = sorted(expected_ids - plan_ids, key=lambda x: int(x[1:]))
g.check("module_plan_covers_all_modules", [], missing_ids, f"{len(plan_ids)} module ids found in plan")
g.check_true("module_plan_has_executable_criteria", "门禁标准" in plan and "门禁脚本" in plan)

# --------------------------------------------------------------------------- #
# 3) claims checklist verdict column
# --------------------------------------------------------------------------- #
claims = (ROOT / "docs/CLAIMS_CHECKLIST.md").read_text(encoding="utf-8")
g.check_true("claims_checklist_has_verdict_column", "判定" in claims and "VERIFIED" in claims)

# --------------------------------------------------------------------------- #
# 4) fail-fast test with the real runner
# --------------------------------------------------------------------------- #
fail_entry = [("S1", "selftest_fail", "gates._selftest.gate_selftest_fail"),
              ("S2", "selftest_pass", "gates._selftest.gate_selftest_pass")]
summary, failed = run_sequence(fail_entry)
g.check("failfast_stops_at_failing_module", ("S1", "FAIL"), failed)
g.check("failfast_does_not_continue", 1, len(summary), "runner must stop after the failing module")
fail_json = RESULTS / "gate_S1_selftest_fail.json"
ok_fail_json = False
if fail_json.exists():
    d = json.loads(fail_json.read_text(encoding="utf-8"))
    ok_fail_json = d.get("pass") is False and any(not c["pass"] for c in d.get("checks", []))
g.check_true("failing_gate_persists_pass_false", ok_fail_json, str(fail_json.relative_to(ROOT)))

# --------------------------------------------------------------------------- #
# 5) normal path test
# --------------------------------------------------------------------------- #
summary2, failed2 = run_sequence([("S2", "selftest_pass", "gates._selftest.gate_selftest_pass")])
g.check("normal_path_completes", None, failed2)
pass_json = RESULTS / "gate_S2_selftest_pass.json"
ok_pass_json = False
if pass_json.exists():
    d = json.loads(pass_json.read_text(encoding="utf-8"))
    ok_pass_json = d.get("pass") is True
g.check_true("passing_gate_persists_pass_true", ok_pass_json, str(pass_json.relative_to(ROOT)))

# --------------------------------------------------------------------------- #
# 6) runner order is canonical
# --------------------------------------------------------------------------- #
order = [m[0] for m in GATES]
g.check_true("runner_order_starts_with_P0", order[0] == "P0", f"first={order[0]}")
# M20 is an ad-hoc supplementary module appended after the canonical P0-M18 chain
expected_order = ["P0"] + [f"M{i}" for i in range(0, 19)] + ["M20"]
g.check_true("runner_order_canonical", order == expected_order, f"{order}")

g.note("Skeleton gate: artefacts, plan/claims coverage, fail-fast runner behaviour verified.")
g.note("Fail-fast demonstration: S1 (deliberately failing) stops the run; S2 (passing) completes.")
res = g.finish()
sys.exit(0 if res["pass"] else 1)
