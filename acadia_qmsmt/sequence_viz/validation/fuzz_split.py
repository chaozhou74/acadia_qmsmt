"""Split fuzz results by whether the generated sequence actually contains a `stretch`.

Sequences containing `stretch` inherit the documented mixed-ramp MEASUREMENT systematic (~25 ns)
that the pair sweep showed on all 16 stretch pairs. Reporting the two groups together lets a
known artifact masquerade as a model error; reporting only the clean group hides how much of the
sweep is affected. So they are split.

WHICH sequences contain a stretch is taken from the TRACE -- the pulses the sequence actually
schedules -- and not by replaying `random.Random(seed)` here. Reimplementing the generator's draw
order was tried first and was wrong: the emitter consumes randomness per primitive, so a single
mismatched consumption desynchronises every later step. It classified seeds 12 and 19 as
stretch-free when both schedule `stretch_pulse`, which would have reported a known measurement
artifact as two unexplained model failures. Building the runtime is slower and cannot drift.

Usage: ``python validation/fuzz_split.py [logfile]``
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_LOG = ("/tmp/claude-1000/-home-boson/9c308736-8aeb-41b4-9b0f-e2a2110112e6"
               "/scratchpad/rerun.log")


def has_stretch(seed, steps, cache={}):
    """True when the generated sequence for (seed, steps) schedules a stretchable pulse."""
    key = (int(seed), int(steps))
    if key in cache:
        return cache[key]
    import timing_validation as tv
    from acadia_qmsmt import sequence_viz as sv

    runtime = tv.build_runtime("random_seq", iterations=10)
    runtime.fuzz_seed, runtime.fuzz_steps = int(seed), int(steps)
    trace = sv.trace_runtime(runtime, envelopes=False)
    cache[key] = any(c.pulse == "stretch_pulse" for c in trace.commands)
    return cache[key]


def parse(path):
    """``[(steps, seed, worst_ns, n_intervals, status), ...]`` from a scan log."""
    steps, rows = None, []
    for line in Path(path).read_text(errors="ignore").splitlines():
        match = re.search(r"--fuzz-steps (\d+)", line)
        if match:
            steps = int(match.group(1))
        match = re.search(
            r"fuzz_seed=(\d+)\s+worst\s+([\d.]+) ns\s+\((\d+) intervals\)\s+(\w+)", line)
        if match and steps:
            rows.append((steps, int(match.group(1)), float(match.group(2)),
                         int(match.group(3)), match.group(4)))
    return rows


def main():
    rows = parse(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG)
    clean, stretchy = [], []
    for row in rows:
        (stretchy if has_stretch(row[1], row[0]) else clean).append(row)

    print(f"{len(rows)} generated sequences deployed and measured\n")
    for label, group in (
            ("WITHOUT stretch (clean model result)", clean),
            ("WITH stretch (inherits the ~25 ns mixed-ramp measurement systematic)", stretchy)):
        if not group:
            continue
        # "no interval" is not a failure: the generated sequence simply produced nothing
        # measurable on a within-channel interval (0 intervals compared), which the scan reports
        # with the same non-OK status as a real miss. Counting it as a failure overstates the
        # error rate.
        failed = [row for row in group if row[4] not in ("OK", "no") and row[3] > 0]
        vacuous = [row for row in group if row[3] == 0]
        print(f"{label}:")
        scored = [r for r in group if r[3] > 0]
        worst = max((r[2] for r in scored), default=0.0)
        print(f"   {len(group)} sequences ({len(vacuous)} with no measurable interval), "
              f"{sum(r[3] for r in group)} measured intervals, "
              f"worst {worst:.2f} ns, {len(failed)} failing")
        for row in failed[:8]:
            print(f"      steps={row[0]} seed={row[1]:2d}  {row[2]:7.2f} ns")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
