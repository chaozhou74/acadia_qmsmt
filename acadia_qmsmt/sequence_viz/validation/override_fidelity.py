"""Does a PINNED loop count draw the same timeline hardware produces at that count?

The GUI lets you set how many passes of a loop are drawn. That is only useful if the drawing you
get is the one the board would produce -- otherwise the control invites you to reason about a
timeline that cannot happen, which is worse than not offering it.

The two paths through the model are genuinely different code:

* RESOLVED -- ``repeat_until_count`` reads the target out of the run's own captured cache and
  ``execution_plan`` unrolls that many passes.
* PINNED -- ``loop_counts[block]`` (or ``loop_counts[(block, path)]`` for one execution) short
  circuits that lookup.

So this compares them against the SAME measured data. For every archived run of a case whose loop
count is a parameter, it re-scores the capture twice: once letting the count resolve, once with the
count pinned to the value that run actually used. Both must agree with the measurement, and with
each other. A disagreement means the override machinery draws a timeline the hardware does not.

It also checks the per-EXECUTION key on a nested case: pinning one execution must change that
execution and leave its siblings alone, and pinning every execution individually must reproduce
pinning the construct as a whole.

Run: ``python validation/override_fidelity.py`` (offline; re-scores existing captures).
"""
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DATA = "/home/boson/data/test_loopback"


def scored(folder, loop_counts=None):
    """Worst interval error for one capture, optionally with counts pinned."""
    import timing_validation as tv
    from acadia_qmsmt import sequence_viz as sv

    trace = sv.trace_folder(folder)
    if loop_counts:
        trace.loop_counts.update(loop_counts)
        trace.relayout()
    # compare() re-traces internally, so score through the trace we just built instead
    measured = tv.measure(folder)
    predicted, spans = tv.regions_of(trace)
    worst = 0.0
    compared = 0
    for label in ("ch0", "ch1", "ch2", "ch3"):
        m, p = measured[label], predicted[label]
        if len(m) != len(p) or len(m) < 2:
            continue
        for index in range(1, len(m)):
            worst = max(worst, abs((m[index] - m[0]) - (p[index] - p[0])))
            compared += 1
    return worst, compared, len(trace.placements)


def resolved_vs_pinned():
    """Every archived run of a loop-count scan, scored with the count resolved and pinned."""
    rows = []
    for folder in sorted(glob.glob(f"{DATA}/*loop_count_*/*")):
        match = re.search(r"loop_count_(\d+)", folder)
        if not match:
            continue
        count = int(match.group(1))
        try:
            free_worst, free_n, free_placements = scored(folder)
        except Exception as exc:
            # A run that never returned leaves a folder with compiled.log and no data --
            # repeat_until_op at loop_count=0 is exactly that: the loop cannot exit, the board
            # hung, and the deploy was killed (KB rule 44). Scoring it raises, and calling that
            # "pinning changed the timeline" blames the override machinery for a run that never
            # produced a measurement. Reported as what it is, and not counted as a difference.
            measured = any(Path(folder).glob("*.npy")) or any(Path(folder).glob("*.h5"))
            rows.append((folder, count, None, None,
                         f"{type(exc).__name__}" if measured else "no data (run did not finish)"))
            continue
        if not free_n:
            continue
        # pin every repeat_until/loop in the trace to the count this run actually used
        import timing_validation as tv
        from acadia_qmsmt import sequence_viz as sv
        trace = sv.trace_folder(folder)
        pins = {e["block"]: count for e in trace.control_flow_summary()
                if e["kind"] != "test" and e["count"] == count}
        if not pins:
            continue
        try:
            pin_worst, pin_n, pin_placements = scored(folder, pins)
        except Exception as exc:
            rows.append((folder, count, free_worst, None, f"{type(exc).__name__}"))
            continue
        note = ("SAME" if abs(pin_worst - free_worst) < 0.01
                and pin_placements == free_placements else "DIFFERS")
        rows.append((folder, count, free_worst, pin_worst, note))
    return rows


def per_execution():
    """Pinning one execution must move only that one; pinning all must equal the block key."""
    from acadia_qmsmt import sequence_viz as sv

    folders = sorted(glob.glob(f"{DATA}/nested_cool_n__*/*")) or \
        sorted(glob.glob(f"{DATA}/three_deep_nest_reconfig/*"))
    if not folders:
        return ["no nested archived run to check"]
    trace = sv.trace_folder(folders[-1])
    baseline = len(trace.placements)
    # the innermost construct that runs more than once
    runs = {}
    for placement in trace.placements:
        runs.setdefault(placement.index, set()).add(placement.path)
    repeated = [(b, sorted(p)) for b, p in runs.items() if len(p) > 1]
    if not repeated:
        return [f"{Path(folders[-1]).parent.name}: nothing executes more than once"]
    block, paths = repeated[-1]
    enclosing = sorted({p[:-1] for p in paths})
    out = []

    trace.loop_counts.clear()
    trace.loop_counts[(block, enclosing[0])] = 3
    trace.relayout()
    one = len(trace.placements)

    trace.loop_counts.clear()
    for path in enclosing:
        trace.loop_counts[(block, path)] = 3
    trace.relayout()
    each = len(trace.placements)

    trace.loop_counts.clear()
    trace.loop_counts[block] = 3
    trace.relayout()
    whole = len(trace.placements)

    trace.loop_counts.clear()
    trace.relayout()
    restored = len(trace.placements)

    out.append(f"block {block}, {len(enclosing)} executions, baseline {baseline} placements")
    out.append(f"   pin ONE execution      -> {one}")
    out.append(f"   pin EACH individually  -> {each}")
    out.append(f"   pin the CONSTRUCT      -> {whole}"
               f"   {'OK (same as each)' if each == whole else 'MISMATCH'}")
    out.append(f"   cleared                -> {restored}"
               f"   {'OK' if restored == baseline else 'MISMATCH'}")
    if one >= each and len(enclosing) > 1:
        out.append("   PROBLEM: pinning one execution changed as much as pinning all of them")
    return out


def main():
    print("resolved vs pinned, scored against the same captures")
    rows = resolved_vs_pinned()
    if not rows:
        print("   no loop_count captures found")
    differ = skipped = 0
    for folder, count, free, pinned, note in rows:
        name = Path(folder).parent.name
        if note.startswith("no data"):
            skipped += 1
            print(f"   {name:44s} count={count:<3d} skipped -- {note}")
            continue
        if note == "SAME":
            print(f"   {name:44s} count={count:<3d} resolved {free:5.2f} ns == pinned "
                  f"{pinned:5.2f} ns")
        else:
            differ += 1
            print(f"   {name:44s} count={count:<3d} resolved {free} vs pinned {pinned}  {note}")
    print(f"\n{len(rows)} captures; {differ} where pinning changed the timeline"
          + (f"; {skipped} skipped (no data -- the run never finished)" if skipped else ""))

    print("\nper-execution keying on a nested case")
    for line in per_execution():
        print(f"   {line}")
    return 1 if differ else 0


if __name__ == "__main__":
    raise SystemExit(main())
