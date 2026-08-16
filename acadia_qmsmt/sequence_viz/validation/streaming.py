"""Streamed gate trains: RB and XEB, the two idioms whose gates come from the CACHE.

Everything else the viewer draws is read out of the compiled program. These two are not:

* **RB** builds a cache-pointer stream -- one DMA command word per gate, walked by a pointer -- so
  the gate COUNT is whatever that run's cache holds. Across a sweep it ranges from 3 to 1791.
* **XEB** latches a gate word into a register (``regs[n].load(bus_read(pointers[n]))``), so the gate
  IDENTITIES come from the cache while the program says only "play the register's gate".

That makes them the cases where the viewer is most able to be confidently wrong: a stale decode
would draw a plausible train of real pulses that simply is not the one this run played. The check
that matters is therefore not "does it draw gates" but "does it draw THIS point's gates" -- XEB
randomises them per sweep point, so a viewer that decoded once and reused the result would show an
identical train at every point, which is exactly what this asserts against.

A previous version of this check counted only placements flagged ``stream``. XEB's gates are
register-latched and carry no such flag, so it reported zero gates for every XEB run and passed --
a vacuous green over half its cases. Both idioms are counted here.

Run: ``python validation/streaming.py`` (offline; the board scans live in timing_validation).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def gate_train(trace):
    """Every drawn gate, streamed OR register-latched, in execution order."""
    train = []
    for placement in trace.placements or ():
        for command in placement.commands:
            if not command.pulse:
                continue
            streamed = getattr(placement, "stream", False)
            latched = str(command.symbolic or "").startswith("REG")
            if streamed or latched:
                train.append(command.pulse)
    return train


def check_folder(folder, experiment, points_to_sample=5):
    """Trace one streamed experiment at several sweep points. Returns (problems, samples)."""
    from acadia_qmsmt import sequence_viz as sv
    from acadia_qmsmt.sequence_viz import plotting

    problems, trains = [], {}
    try:
        base = sv.trace_folder(folder, envelopes=False)
    except Exception as exc:
        return [f"{experiment}: trace raised {type(exc).__name__}"], {}
    points = getattr(base, "n_points", 1) or 1
    wanted = sorted({0, 1, points // 3, 2 * points // 3, points - 1})[:points_to_sample]
    for point in wanted:
        if point >= points:
            continue
        try:
            trace = sv.trace_folder(folder, point=point, envelopes=False)
        except Exception as exc:
            problems.append(f"{experiment} point {point}: {type(exc).__name__}")
            continue
        train = gate_train(trace)
        trains[point] = tuple(train)
        if not train:
            problems.append(f"{experiment} point {point}: no gates decoded at all")
        # a gate drawn with no pulse is a decode that produced nothing recognisable
        blanks = sum(1 for placement in trace.placements
                     for command in placement.commands
                     if not command.pulse and command.kind == "ARB"
                     and getattr(placement, "stream", False))
        if blanks:
            problems.append(f"{experiment} point {point}: {blanks} streamed command(s) with "
                            f"no pulse")
        # the panel and the diagram must still agree on a streamed sequence
        rows = set()
        for entry in trace.control_flow_summary():
            rows.add(plotting.flow_label(entry))
            for run in entry.get("executions") or ():
                rows.add(plotting.flow_label(
                    dict(entry, execution=plotting.execution_tag(run["path"]))))
        for _start, _stop, _context, info in plotting.branch_regions(trace):
            text = plotting.tab_label(info)
            if not any(row.endswith(text) for row in rows):
                problems.append(f"{experiment} point {point}: tab {text!r} has no row")
    # XEB randomises its gates per point. Identical trains everywhere means the drawing is not
    # following the per-point cache -- a plausible picture of the wrong run.
    if "XEB" in experiment and len(trains) > 1 and len(set(trains.values())) == 1:
        problems.append(f"{experiment}: every sampled point decoded an IDENTICAL gate train")
    return problems, trains


def pinning_never_invents_gates(folder, experiment):
    """Pinning a gate loop may leave passes UNIDENTIFIED; it must never invent a gate.

    A run's cache holds exactly the words its own circuit played, so drawing more passes than it ran
    leaves nothing to decode. Two things must hold, and they pull in opposite directions:

    * the decoded gates must not multiply -- an identity copied into a pass the run never played is
      a plausible picture of a circuit that never existed;
    * the extra commands must still be DRAWN, as `indeterminate (register)`, so the reader sees
      "unknown gate here" instead of a silent gap that reads as "nothing plays here".
    """
    from acadia_qmsmt import sequence_viz as sv

    problems, pins = [], 0
    try:
        trace = sv.trace_folder(folder, envelopes=False)
    except Exception:
        return problems, pins
    baseline = len(gate_train(trace))
    holders = []
    gate_blocks = {p.index for p in trace.placements for c in p.commands
                   if c.pulse and str(c.symbolic or "").startswith("REG")}
    for entry in trace.control_flow_summary():
        if entry["kind"] == "test" or entry["block"] not in gate_blocks:
            continue
        holders.append(entry)
    for entry in holders[:2]:
        for count in (2, 5):
            pins += 1
            trace.loop_counts.clear()
            trace.loop_counts[entry["key"]] = count
            trace.relayout()
            decoded = len(gate_train(trace))
            unidentified = [c for p in trace.placements for c in p.commands
                            if str(c.symbolic or "").startswith("REG") and not c.pulse]
            if decoded > baseline:
                problems.append(f"{experiment}: pinning {entry['key']}={count} decoded "
                                f"{decoded} gates from a cache that holds {baseline} -- "
                                f"an identity was invented")
            if count > 1 and not unidentified and decoded == baseline:
                problems.append(f"{experiment}: pinning {entry['key']}={count} drew no extra "
                                f"gate commands at all -- the extra passes are invisible")
    trace.loop_counts.clear()
    trace.relayout()
    return problems, pins


def main():
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    import stress_campaign as sc

    problems, checked, experiments, pinned = [], 0, 0, 0
    seen = set()
    for folder in sc.folders():
        parts = Path(folder).parts
        experiment = parts[-4] if len(parts) >= 4 else parts[-1]
        if not any(tag in experiment for tag in ("RB", "XEB", "rb_stream")):
            continue
        if experiment in seen or not Path(folder, "compiled.log").exists():
            continue
        seen.add(experiment)
        found, trains = check_folder(str(folder), experiment)
        problems += found
        pin_problems, pins = pinning_never_invents_gates(str(folder), experiment)
        problems += pin_problems
        pinned += pins
        checked += len(trains)
        experiments += 1
        distinct = len(set(trains.values()))
        lengths = sorted({len(train) for train in trains.values()})
        print(f"   {experiment:24s} {len(trains)} points sampled | {distinct} distinct train(s) | "
              f"gate counts {lengths}", flush=True)

    print(f"\n{checked} (streamed experiment, sweep point) combinations over {experiments} "
          f"experiments; {pinned} pinned-count combinations; {len(problems)} problems")
    if experiments and not pinned:
        print("   NOTE: no gate loop was pinned -- the 'never invents a gate' property "
              "checked nothing")
    for line in problems[:12]:
        print(f"   {line}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
