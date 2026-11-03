"""Deliberately passing gate used by the P0 skeleton gate (normal path test)."""

import sys

from gates.common import Gate, add_src_to_path

add_src_to_path()
g = Gate("S2", "selftest_pass", owner="lead", paper_ref="Phase-0 skeleton gate")
g.check("always_passing_check", expected=1, actual=1, detail="this check must pass")
res = g.finish()
sys.exit(0 if res["pass"] else 1)
