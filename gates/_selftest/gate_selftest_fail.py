"""Deliberately failing gate used by the P0 skeleton gate (fail-fast test).

It must write results/gate_S1_selftest_fail.json with pass=false and exit non-zero.
"""

import sys

from gates.common import Gate, add_src_to_path

add_src_to_path()
g = Gate("S1", "selftest_fail", owner="lead", paper_ref="Phase-0 skeleton gate")
g.check("deliberately_failing_check", expected=1, actual=0, detail="this check must fail")
res = g.finish()
sys.exit(0 if res["pass"] else 1)
