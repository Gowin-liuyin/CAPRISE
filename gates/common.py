"""Gate framework: every reproduction module must pass its gate before the next
module may start (user requirement: module-by-module reproduction).

Usage in a gate script::

    from gates.common import Gate
    g = Gate("M1", "prf", owner="lead")
    g.check("prf_deterministic", expected=True, actual=check(), detail="...")
    g.finish()

``finish()`` writes ``results/gate_<id>_<name>.json`` (pass = all checks pass)
and returns the exit code compatible result dict.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
    }
    try:
        info["cpu"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["ncpu"] = subprocess.check_output(
            ["sysctl", "-n", "hw.ncpu"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # pragma: no cover - non-macOS
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        info["mps"] = bool(torch.backends.mps.is_available())
    except Exception:  # pragma: no cover
        pass
    return info


class Gate:
    def __init__(self, module_id: str, name: str, owner: str = "lead", paper_ref: str = ""):
        self.module_id = module_id
        self.name = name
        self.owner = owner
        self.paper_ref = paper_ref
        self.checks: list[dict] = []
        self.started = time.time()
        self.notes: list[str] = []

    def check(self, name: str, expected, actual, detail: str = "", passed: bool | None = None) -> bool:
        if passed is None:
            passed = bool(expected == actual) if not isinstance(expected, float) else bool(abs(expected - actual) < 1e-12)
        entry = {
            "check": name,
            "expected": expected,
            "actual": actual,
            "pass": bool(passed),
            "detail": detail,
        }
        self.checks.append(entry)
        status = "PASS" if entry["pass"] else "FAIL"
        print(f"  [{status}] {name}: actual={actual!r}" + (f" ({detail})" if detail else ""))
        return entry["pass"]

    def check_true(self, name: str, cond, detail: str = "") -> bool:
        return self.check(name, expected=True, actual=bool(cond), detail=detail, passed=bool(cond))

    def note(self, text: str) -> None:
        self.notes.append(text)

    def result(self) -> dict:
        passed = all(c["pass"] for c in self.checks) and len(self.checks) > 0
        return {
            "module": self.module_id,
            "name": self.name,
            "owner": self.owner,
            "paper_ref": self.paper_ref,
            "pass": passed,
            "checks": self.checks,
            "notes": self.notes,
            "duration_s": round(time.time() - self.started, 3),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hardware": _hardware(),
        }

    def finish(self) -> dict:
        res = self.result()
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"gate_{self.module_id}_{self.name}.json"
        path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        status = "PASS" if res["pass"] else "FAIL"
        print(f"\n[{self.module_id}] {self.name}: {status}  -> {path.relative_to(ROOT)}")
        return res


def add_src_to_path() -> None:
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
