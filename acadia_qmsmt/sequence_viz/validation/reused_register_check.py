"""Model bug 16: a test on a register that is LOADED AGAIN later must read the load in force.

    $ACADIA_ENV/bin/python validation/reused_register_check.py <notebook dir>

THE BUG (2026-10-12). describe_registers keeps one source per register: the LAST load in program
order. Z2HiggsSimulation's multi-prep selector is a stream register (all eight are taken) loaded
from the prep cache word, tested once per prep, and reloaded later by the Trotter train from a
pointer. SequenceTrace.evaluate_condition read the train's load, could not decide the prep
tests, and drew ALL the preps back to back in every sweep point.

THE RULE NOW: the value of a register at a test is its last write before the first triggering
block from the test's body on (SequenceTrace._value_in_force); a register loaded once reads as
before. 42/42 seqviz_regress cases identical before and after.

The check: the notebook's own Z2 cell (three Fock preps by JC climbs), traced; every point must
play its own prep and nothing else, and no prep branch may be assumed.
"""
import contextlib
import io
import json
import os
import sys

import numpy as np

FAILS = []


def check(ok, message):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {message}")
    if not ok:
        FAILS.append(message)


nb_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
os.chdir(nb_dir)
from acadia_qmsmt.utils import load_yaml                                    # noqa: E402
from acadia_qmsmt import sequence_viz as sv                                  # noqa: E402

nb = json.load(open("quantum_simulation_tasks.ipynb"))
cell = next("".join(c["source"]) for c in nb["cells"]
            if "jc_prep" in "".join(c["source"]) and "Z2HiggsSimulation" in "".join(c["source"]))
g = {"np": np, "load_yaml": load_yaml, "YAML_PATH": "config_mced.yaml"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(cell[:cell.index("rt_sim.deploy(")], g)
rt = g["rt_sim"]
rt.iterations, rt.trotter_steps, rt.cooling_config = 1, [1], None
preps = g["PREPS"]
tr = sv.trace_runtime(rt, point=0, envelopes=True)
# The prep selector branches come first; the counting rounds' conditional resets after them
# branch on a MEASUREMENT and are legitimately assumed in any version, so only the prep ones
# are required to be decided.
prep_blocks = set(range(len(preps)))
check(not (prep_blocks & set(tr.assumed_paths)),
      f"the {len(preps)} prep branches are decided from the cache (assumed: "
      f"{sorted(prep_blocks & set(tr.assumed_paths))})")
for p, occ in enumerate(preps):
    tr.select_point(p)
    swaps = [str(c.pulse) for b in (tr.placements or tr.blocks) for c in b.commands
             if c.kind in ("ARB", "CONST") and "CR_" in str(getattr(c, "pulse", "")) and "swap" in str(c.pulse)]
    want = sum(occ)                       # one swap per photon loaded
    check(len(swaps) == want, f"point {p} {occ}: {len(swaps)} prep swaps, want {want}")

if FAILS:
    print(f"\nreused_register_check FAILED ({len(FAILS)})")
    sys.exit(1)
print("\nreused_register_check PASSED")
