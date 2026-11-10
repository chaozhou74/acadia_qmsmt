"""Take every case parameter to its degenerate value and see what breaks.

This exists because one such value found two bugs at once. `repeat_until_op` at `loop_count=0`
wedged the board (a loop whose counter can never come back to 0) *and* exposed the viewer drawing
that case as a tidy empty body -- confidently wrong about a sequence the hardware cannot produce.
Nothing in the 1..8 sweep could have found either: the interesting behaviour is at the edge.

So this walks the edges deliberately. For each case and each of its numeric parameters it tries the
degenerate values -- 0, 1, and the parameter's own boundary -- and asks, OFF HARDWARE:

* does tracing raise, or silently produce nothing?
* does the model claim something the hardware cannot do (a non-terminating loop, a zero-length
  command that wraps to 21 s)? Those are reported, not deployed.
* does the drawing still agree with the trace?

Only what survives all three is worth board time, and `timing_validation.unsafe_reason()` is asked
before any deploy anyway. Run:

    python validation/degenerate_sweep.py            # offline sweep, report only
    python validation/degenerate_sweep.py --limit 40 # first N combinations
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

#: Parameters worth taking to their edge, with the degenerate values to try. Chosen because each
#: one changes the SHAPE of the sequence rather than a magnitude: a count of zero removes a
#: construct, a single pulse removes an adjacency, one block removes every boundary.
EDGES = {
    "loop_count": (0, 1),
    "blocks_n": (1, 2),
    "batch_resync_pulses": (1, 2),
    "fuzz_steps": (1, 2),
    "test_register_value": (0, 1),
    "iterations": (1,),
}


def cases_to_sweep():
    """(case, field, value) for every case that has a swept parameter."""
    from loopback_timing_cases import CASES

    for case in CASES:
        for field, values in EDGES.items():
            for value in values:
                yield case, field, value


def probe(case, field, value):
    """Trace one degenerate combination. Returns (status, detail)."""
    import timing_validation as tv
    from acadia_qmsmt import sequence_viz as sv
    from acadia_qmsmt.sequence_viz import plotting

    try:
        runtime = tv.build_runtime(case, iterations=10)
        setattr(runtime, field, value)
    except Exception as exc:                                   # noqa: BLE001
        return "build-failed", f"{type(exc).__name__}: {exc}"
    try:
        trace = sv.trace_runtime(runtime, envelopes=False)
    except Exception:
        return "TRACE RAISED", traceback.format_exc(limit=2).strip().splitlines()[-1]
    if not (trace.placements or trace.blocks):
        return "EMPTY", "traced to no blocks at all"

    # the model's own verdicts on whether this can run at all
    stuck = [e for e in trace.control_flow_summary() if e.get("nonterminating")]
    if stuck:
        return "NON-TERMINATING", (f"{stuck[0]['kind']} @{stuck[0]['block']} cannot exit "
                                   f"(counter target 0) -- the board would hang")
    if getattr(trace, "length_underflows", None):
        first = trace.length_underflows[0]
        return "LENGTH UNDERFLOW", (f"{first['register']} on {first['channel']} wraps to "
                                    f"{first['cycles']} cycles ({first['cycles'] * 5e-9:.1f} s)")

    # and the drawing must still describe the trace
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(figsize=(10, 5))
        plotting.draw(axes, trace)
        regions = plotting.branch_regions(trace)
        plt.close(figure)
    except Exception:
        return "DRAW RAISED", traceback.format_exc(limit=2).strip().splitlines()[-1]

    # every construct must still be reachable -- the property the zero-iteration bug broke
    reachable = {(i["block"], i["depth"]) for _s, _e, _c, i in regions}
    missing = [e["key"] for e in trace.control_flow_summary()
               if (e["block"], e["depth"]) not in reachable]
    if missing:
        return "UNREACHABLE", f"{len(missing)} construct(s) with no tab: {missing[:3]}"
    return "ok", f"{len(trace.placements)} placements, {len(regions)} spans"


def main():
    import logging
    logging.getLogger().setLevel(logging.ERROR)

    argv = sys.argv[1:]
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    counts, findings = {}, []
    for index, (case, field, value) in enumerate(cases_to_sweep(), 1):
        if limit and index > limit:
            break
        status, detail = probe(case, field, value)
        counts[status] = counts.get(status, 0) + 1
        if status not in ("ok", "build-failed"):
            findings.append((case, field, value, status, detail))
            print(f"  {case}.{field}={value}: {status} -- {detail}", flush=True)
        if index % 50 == 0:
            print(f"    ... {index} combinations, {len(findings)} findings", flush=True)

    print(f"\n{sum(counts.values())} degenerate combinations traced")
    for status, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"   {status:18s} {count}")
    print(f"\n{len(findings)} worth reading; 'build-failed' means the case ignores that "
          f"parameter, which is not a defect")
    # a TRACE/DRAW failure is a viewer bug; the others are honest reports about the sequence
    bugs = [f for f in findings if "RAISED" in f[3] or f[3] in ("EMPTY", "UNREACHABLE")]
    return 1 if bugs else 0


if __name__ == "__main__":
    raise SystemExit(main())
