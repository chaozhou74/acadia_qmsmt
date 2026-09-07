"""The FPGA-looped counting round, measured on the loopback board.

resonator_number_measurement wants each counting round to play a ladder swap whose LENGTH comes
from a register (design A) or from a streamed DMA word (design B), with the readout waiting out
that swap inside the SAME barrier-free block. That replaces a Python `for r in range(rounds)`,
whose instruction count grows with the schedule, with a fixed loop body.

Five things have to be true for it to work, and each is asked separately here so a failure names
its own half rather than "the round is broken":

  Q1  does a BARRIER survive a register-driven length, on hardware?
  Q2  the register -> duration law: slope (expect exactly 1 cycle per cycle) and intercept.
  Q3  does a manual `dwell(register)` track a STRETCHED pulse, independently of the register?
  Q4  the same for a STREAMED command, whose length the sequencer cannot see.
  Q5  the zero-length wrap -- checked offline, because it is a 21 s hang, not a short pulse.
  Q6  the whole round body -- swap, readout, quadrant, cache write, conditional pi -- in a loop.
  Q7  ONE pointer advanced K times per pass -- interleaved ladder-descent cooling of K modes,
      where the pointer's word range and the loop's pass count stop being the same number.

Run one phase at a time; each deploys a handful of runs:

    python counting_round_timing.py --phase q1
    python counting_round_timing.py --phase q2
    ...
    python counting_round_timing.py --phase all
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import prep_selector_timing as ps                                    # noqa: E402
import timing_validation as tv                                       # noqa: E402

#: register values to sweep, in CYCLES. 1 and 2 are the floor, 115 is longer than the pulse's
#: own flat -- a law fitted only over the middle would not show a saturation at either end.
CYCLES = (1, 2, 5, 10, 40, 115)


def interval_rows(result, channel):
    for row in result.get("rows", ()):
        if row["channel"] == channel:
            return row
    return None


def report(title, rows, unit="cycles"):
    print(f"\n{title}")
    print(f"  {unit:>8} {'measured':>10} {'model':>10} {'err':>8}")
    for value, meas, pred in rows:
        if meas is None:
            print(f"  {value:>8} {'--':>10} {'--':>10}   no interval")
            continue
        print(f"  {value:>8} {meas:>10.1f} {pred:>10.1f} {meas - pred:>+8.1f}   [ns]")
    good = [(v, m, p) for v, m, p in rows if m is not None]
    if len(good) >= 2:
        (v0, m0, _), (v1, m1, _) = good[0], good[-1]
        if v1 != v0:
            slope = (m1 - m0) / (v1 - v0)
            print(f"  measured slope {slope:+.3f} ns per cycle "
                  f"({slope / 5.0:+.3f} cycles per cycle);"
                  f"  intercept at 0 -> {m0 - slope * v0:.1f} ns")


def phase_q1(_args):
    print("\nQ1 -- a BARRIER in a block that also carries a register-driven length.")
    print("  The KB has two rules that disagree: one says calculate_subschedule_dwells raises")
    print("  Operation(builtins.max, ...), the other says that error is a client-compile")
    print("  artifact and the barrier works on hardware. The compile already happened (see the")
    print("  dry run); this is whether the board plays it.")
    result = ps.deploy({"count_cycles": 40}, tag="q1_barrier",
                       case="stretch_barrier_twochan")
    if "skipped" in result:
        print(f"  REFUSED: {result['skipped']}")
        return
    for row in result.get("rows", ()):
        if row["n_measured"]:
            print(f"    {row['channel']}: {row['n_measured']} pulse(s) measured vs "
                  f"{row['n_predicted']} predicted, widths {row['measured_widths_ns']} "
                  f"(model {row['predicted_widths_ns']})")
    print(f"    worst interval error {result['worst_error_ns']} ns")
    print(f"    {result['folder']}")


def phase_q2(args):
    rows = []
    for cycles in args.cycles:
        result = ps.deploy({"count_cycles": cycles}, tag=f"q2_law_c{cycles}",
                           case="stretch_duration_law")
        row = interval_rows(result, "ch0")
        meas = row["measured_intervals_ns"][0] if row and row["n_compared"] else None
        pred = row["predicted_intervals_ns"][0] if row and row["n_compared"] else None
        rows.append((cycles, meas, pred))
        print(f"    reg={cycles}: measured {meas}", flush=True)
    report("Q2 -- marker -> marker across ONE register-driven stretch", rows)


def phase_q3(args):
    """ch0 stretch(reg) and a hand-built ramp+reg wait on ch1, in one barrier-free block."""
    for grid in (True, False):
        rows = []
        for cycles in args.cycles:
            result = ps.deploy({"count_cycles": cycles, "count_ramp_grid": grid},
                               tag=f"q3_align_c{cycles}_{'grid' if grid else 'raw'}",
                               case="stretch_dwell_align")
            row = interval_rows(result, "ch1")
            # ch1 carries the marker that stands in for the readout; ch0 carries the swap.
            # What matters is the OFFSET between them, so both first edges are reported.
            ch0 = interval_rows(result, "ch0")
            offset = None
            if row and ch0 and row["n_measured"] and ch0["n_measured"]:
                swap_end = ch0["first_edge_ns"] + sum(ch0["measured_widths_ns"])
                offset = row["first_edge_ns"] - swap_end
            rows.append((cycles, offset, 0.0))
            print(f"    reg={cycles} grid={grid}: readout starts {offset} ns after the swap ends",
                  flush=True)
        report(f"Q3 -- swap-end -> readout-start, ramp {'snapped' if grid else 'RAW'}", rows)


def phase_q4(args):
    rows = []
    for cycles in args.cycles[:4]:
        result = ps.deploy({"count_cycles": cycles}, tag=f"q4_direct_c{cycles}",
                           case="direct_dwell_align")
        ch0, ch1 = interval_rows(result, "ch0"), interval_rows(result, "ch1")
        offset = None
        if ch0 and ch1 and ch0["n_measured"] and ch1["n_measured"]:
            offset = ch1["first_edge_ns"] - (ch0["first_edge_ns"]
                                             + sum(ch0["measured_widths_ns"]))
        rows.append((cycles, offset, 0.0))
        print(f"    reg={cycles}: streamed pulse widths {ch0['measured_widths_ns'] if ch0 else '--'}"
              f", readout offset {offset}", flush=True)
    report("Q4 -- streamed command: pulse end -> readout start", rows)


def phase_q5(_args):
    """Offline only. A zero-length register is a 21 s command, not a short pulse."""
    from acadia_qmsmt import sequence_viz as sv
    print("\nQ5 -- the zero-length wrap. Checked OFFLINE: deploying it would hang the board for")
    print("  21 s and return a timeout, which is a slow way to learn what the compiled command")
    print("  word already says.")
    for cycles in (0, 1):
        runtime = tv.build_runtime("stretch_zero", iterations=1)
        runtime.count_cycles = cycles
        probe = tv.build_runtime("stretch_zero", iterations=10)
        probe.count_cycles = cycles
        trace = sv.trace_runtime(runtime, capture_points=True, envelopes=False)
        under = getattr(trace, "length_underflows", None) or []
        print(f"\n  cache[0] = {cycles} cycles")
        print(f"    traced sequence length : {trace.length_ns / 1e6:.3f} ms")
        print(f"    length_underflows      : {len(under)}"
              + (f"  {under[0]['register']} on {under[0]['channel']} -> "
                 f"{under[0]['cycles']} cycles" if under else ""))
        print(f"    refused to deploy      : {tv.unsafe_reason(probe) or 'no -- safe'}")


def phase_q6(args):
    print("\nQ6 -- the whole counting round in an FPGA loop, both designs.")
    for case in ("loop_measure_feedback", "loop_stream_feedback"):
        result = ps.deploy({"count_rounds": args.rounds, "count_cycles": args.cycles[-2]},
                           tag=f"q6_{args.rounds}rounds", case=case)
        if "skipped" in result:
            print(f"  {case}: REFUSED -- {result['skipped']}")
            continue
        print(f"\n  {case}: {result['blocks']} blocks, {result['executed_blocks']} executed, "
              f"gaps {result['gaps_ns']}")
        for row in result.get("rows", ()):
            if row["n_measured"]:
                print(f"    {row['channel']}: {row['n_measured']} pulses measured vs "
                      f"{row['n_predicted']} predicted   intervals "
                      f"{row['measured_intervals_ns']}")
        print(f"    worst error {result['worst_error_ns']} ns")
        print(f"    {result['folder']}")


def interleaved_model(steps, rounds, cycles):
    """Trace ``interleaved_stretch`` offline and return (drawn widths per channel, the cache).

    Widths, not starts: what a stretched pulse's LENGTH resolved to is the whole question, and
    on the board it is the pulse's own width -- a quantity the recording gives directly, with the
    cable latency cancelling out of it entirely.
    """
    from acadia_qmsmt import sequence_viz as sv

    runtime = tv.build_runtime("interleaved_stretch", iterations=1)
    runtime.interleave_channels = steps
    runtime.count_rounds = rounds
    runtime.count_cycles = cycles
    trace = sv.trace_runtime(runtime, capture_points=True, envelopes=False)
    drawn = {}
    for command in trace.commands:
        if command.kind != "CONST_CONT" or not command.symbolic:
            continue
        drawn.setdefault(command.channel, []).append((command.length, command.resolution))
    want = [max(cycles * (k + 1) - 5 * r, 1) for r in range(rounds) for k in range(steps)]
    return trace, drawn, want


def phase_q7(args):
    """Q7 -- ONE pointer advanced several times per pass (interleaved ladder-descent cooling).

    Offline first, because this is a claim about the MODEL and the cache settles it exactly: the
    pointer's word range is rounds x steps while the loop runs `rounds` passes, and channel k's
    round r is word r*steps + k. Then on the board, where the widths are physical.
    """
    print("\nQ7 -- interleaved ladder descent: one pointer, several advances per pass.")
    for steps in (1, 2, 3):
        rounds, cycles = args.rounds, args.cycles[-2]
        trace, drawn, want = interleaved_model(steps, rounds, cycles)
        n_stretch = sum(len(v) for v in drawn.values())
        print(f"\n  steps={steps} rounds={rounds}: cache (round-major) {want}")
        count_ok = n_stretch == rounds * steps
        print(f"    stretched commands drawn {n_stretch}, expected {rounds * steps}   "
              + ("OK" if count_ok else
                 "<-- FAIL: the pass count was taken as the pointer's WORD range"))
        ok = count_ok
        channels = tv.channel_of(trace)
        for k in range(steps):
            channel = channels[f"ch{k}"]
            got = [n for n, _ in drawn.get(channel, ())]
            mine = want[k::steps]
            flag = "OK" if got == mine else "<-- FAIL"
            if got != mine:
                ok = False
            print(f"    ch{k} ({channel}): drawn {got}  cache {mine}   {flag}")
            unresolved = [r for _, r in drawn.get(channel, ()) if r != "cache"]
            if unresolved:
                ok = False
                print(f"    ch{k}: {len(unresolved)} length(s) not resolved from the cache "
                      f"({set(unresolved)}) <-- FAIL")
        # The failure this exists for is not "wrong number" but "same number on every channel",
        # which reads as a plausible schedule. Say so explicitly.
        if steps > 1:
            per_channel = [[n for n, _ in drawn.get(channels[f"ch{k}"], ())]
                           for k in range(steps)]
            if len({tuple(v) for v in per_channel}) == 1:
                ok = False
                print("    every channel drew the SAME lengths <-- FAIL: a shared per-round "
                      "schedule is exactly the bug, and it looks correct")
        print(f"    offline verdict: {'OK' if ok else 'FAIL'}")

    if args.offline:
        print("\n  --offline: skipping the board run.")
        return
    result = ps.deploy({"interleave_channels": 3, "count_rounds": args.rounds,
                        "count_cycles": args.cycles[-2]},
                       tag=f"q7_{args.rounds}rounds", case="interleaved_stretch")
    if "skipped" in result:
        print(f"\n  board: REFUSED -- {result['skipped']}")
        return
    print(f"\n  board: {result['blocks']} blocks, {result['executed_blocks']} executed, "
          f"gaps {result['gaps_ns']}")
    for row in result.get("rows", ()):
        if not row["n_measured"]:
            continue
        print(f"    {row['channel']}: {row['n_measured']} pulses measured vs "
              f"{row['n_predicted']} predicted")
        print(f"      widths  measured {row.get('measured_widths_ns')}")
        print(f"              model    {row.get('predicted_widths_ns')}")
    print(f"    worst error {result['worst_error_ns']} ns")
    print(f"    {result['folder']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                        choices=("q1", "q2", "q3", "q4", "q5", "q6", "q7", "all"))
    parser.add_argument("--cycles", type=int, nargs="+", default=CYCLES)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--offline", action="store_true",
                        help="q7 only: run the model half and skip the board deploy")
    args = parser.parse_args()
    for name, fn in (("q1", phase_q1), ("q2", phase_q2), ("q3", phase_q3),
                     ("q4", phase_q4), ("q5", phase_q5), ("q6", phase_q6),
                     ("q7", phase_q7)):
        if args.phase in (name, "all"):
            fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
