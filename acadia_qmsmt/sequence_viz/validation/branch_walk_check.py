"""Model bug 15: the executed-path walk must exit an inner spin loop that sits AFTER its goal.

    $ACADIA_ENV/bin/python validation/branch_walk_check.py              # the rule, no data
    $ACADIA_ENV/bin/python validation/branch_walk_check.py <run folder> # + a real Z2 run with cooling

THE BUG (2026-10-10). tracing._branch_is_taken took EVERY backward branch whose target lay behind
the goal ("the goal is behind us, so the edge is the only way"). That is right for the loop that
CONTAINS the goal and wrong for an inner loop placed after it: a fifo-poll spin loop
(``repeat_until(channel_is_fifo_almost_empty): pass``) inside an outer loop whose next pass
revisits an earlier block. The walk took the spin loop's back edge forever, ran out of steps,
pointer_register_words gave up for the whole plan, and every streamed word fell back to a stale
register. Feedback cooling is exactly that shape (rounds of swap + poll + measure), so on a Z2
simulation run with cooling ON the Trotter train was drawn as ~9.8 ms "pulses" whose lengths were
a cooling DMA word, and every counting swap as a 10 ns stub. With cooling off it drew correctly.

THE RULE NOW: a back edge is taken only when the goal lies inside its loop, target <= goal < at.
"""
import sys

from acadia_qmsmt.sequence_viz import tracing as T

FAILS = []


def check(ok, message):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {message}")
    if not ok:
        FAILS.append(message)


print("the back-edge rule")
# an outer loop 200..600 containing the goal (227), with an inner spin loop 497..502 after it
check(not T._branch_is_taken(502, 497, 227, [227, 520], set(), []),
      "inner spin loop AFTER the goal is not taken (the goal belongs to the outer loop)")
check(T._branch_is_taken(600, 200, 227, [227, 520], set(), []),
      "the outer loop's back edge IS taken (the goal lies inside it)")
check(T._branch_is_taken(502, 497, 499, [499], set(), []),
      "a back edge whose own body holds the goal is taken")
check(not T._branch_is_taken(502, 497, 700, [700], set(), []),
      "a back edge with the goal AHEAD is not taken")

if len(sys.argv) > 1:
    from acadia_qmsmt.sequence_viz.folder import trace_folder
    folder = sys.argv[1]
    print(f"a real run with cooling: {folder}")
    import json
    steps = json.load(open(f"{folder}/kwargs.json")).get("trotter_steps") or []
    for point in (1, 3):
        if point >= len(steps):
            continue
        tr = trace_folder(folder, point=point, envelopes=True)
        ns = tr.ns_per_cycle
        cmds = [(int(c.start) * ns, int(c.length or 0) * ns, getattr(c, "pulse", None))
                for b in (tr.placements or tr.blocks) for c in b.commands]
        train = [c for c in cmds if c[2] and "z2_train" in str(c[2])]
        legs = 6 if len({c[2].split("_")[2] for c in train}) > 4 else 4
        huge = [c for c in cmds if c[1] > 1e5]
        check(len(train) == legs * 4 * int(steps[point]),
              f"point {point} ({steps[point]} steps): {len(train)} train pulses, "
              f"want {legs * 4 * int(steps[point])}")
        check(not huge, f"point {point}: no command longer than 100 us (found {len(huge)})")

if FAILS:
    print(f"\nbranch_walk_check FAILED ({len(FAILS)})")
    sys.exit(1)
print("\nbranch_walk_check PASSED")
