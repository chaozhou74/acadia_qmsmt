"""What a SKIPPED `test()` arm costs, measured on the loopback board.

The shape is the register-selected prep from resonator_number_measurement: one
`a.sequencer().test(sel == i)` per prep state, at most one taken, the rest skipped.

    prep_selector.load(prep_select_cache[0])
    for i, prep in enumerate(self.all_preps):
        if not prep:
            continue
        with a.sequencer().test(prep_selector == i):
            with a.channel_synchronizer(...):
                self.schedule_pulse_list(a, prep, ...)

The intuition that needs checking is that a skipped arm is nearly free -- no pulse plays and
no DMA is touched, so what is there to pay for? But the sequencer still evaluates the
condition and takes a branch for every arm it skips, and those are instruction fetches at
5 ns each. On a real run (data/tunmay/20260816/number_measurement/C1/260819/163543) the gap
between the last cooling pulse and the first counting round is 1850 ns, of which the model
accounts 361 cycles as INSTRUCTIONS -- against 21 cycles for the same edge on the passes
where the chain is not there.

So the question is quantitative, and it needs a controlled sequence to answer: the
`test_chain` loopback case puts a marker pulse before and after the chain ON ONE CHANNEL, so
the marker interval IS the chain's cost. Sweeping the number of arms gives the per-skipped-arm
slope; sweeping the arm's body size answers whether a skip is O(1) or O(body).

Every number here is measured off the loopback capture and compared against what sequence_viz
predicts, so it says two things at once: what the hardware does, and whether the viewer
(and therefore the SeeQuence tab) is telling the truth about it.

    python prep_selector_timing.py --phase count     # arms swept, both synchronizer modes
    python prep_selector_timing.py --phase body      # body size swept
    python prep_selector_timing.py --phase join      # is the 1-cycle join dwell load-bearing?
    python prep_selector_timing.py --phase all
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import paths_local                                                   # noqa: E402
import timing_validation as tv                                       # noqa: E402


def deploy(fields, iterations=5000, tag=None, case="test_chain"):
    """Deploy one loopback case with these field overrides. Returns compare()'s result.

    Shared with counting_round_timing.py, which drives a different case through the same
    path -- probe, safety check, capture window, deploy, compare.
    """
    board_ip = paths_local.require("board_ip")
    save_root = paths_local.require("loopback_data_root")

    probe = tv.build_runtime(case, iterations=10)
    for name, value in fields.items():
        setattr(probe, name, value)
    unsafe = tv.unsafe_reason(probe)
    if unsafe:
        return {"skipped": unsafe}
    window = tv.capture_window_for(probe)

    runtime = tv.build_runtime(case, iterations=iterations)
    for name, value in fields.items():
        setattr(runtime, name, value)
    if not runtime.capture_length_override:
        runtime.capture_length_override = window
    label = tag or "_".join(f"{k}{v}" for k, v in fields.items())
    runtime.deploy(board_ip,
                   local_directory=f"{save_root}/{case}__{label}/%y%m%d_%H%M%S")
    runtime.wait_for_deploy_completion()
    result = tv.compare(runtime.local_directory, verbose=False)
    result["folder"] = str(runtime.local_directory)
    return result


def marker_interval(result, channel="ch0"):
    """(measured, predicted) marker interval in ns, or (None, None)."""
    for row in result.get("rows", ()):
        if row["channel"] == channel and row["n_compared"]:
            return row["measured_intervals_ns"][0], row["predicted_intervals_ns"][0]
    return None, None


def show(title, rows, swept):
    """rows: [(swept value, measured, predicted)]"""
    print(f"\n{title}")
    print(f"  {swept:>12} {'measured':>10} {'predicted':>10} {'model err':>10}")
    for value, meas, pred in rows:
        if meas is None:
            print(f"  {value:>12} {'--':>10} {'--':>10}   no interval")
            continue
        print(f"  {value:>12} {meas:>10.1f} {pred:>10.1f} {meas - pred:>+10.1f}   [ns]")
    usable = [(v, m, p) for v, m, p in rows if m is not None and isinstance(v, (int, float))]
    if len(usable) >= 2:
        (v0, m0, p0), (v1, m1, p1) = usable[0], usable[-1]
        if v1 != v0:
            print(f"  slope: measured {(m1 - m0) / (v1 - v0):+.1f} ns per unit, "
                  f"model {(p1 - p0) / (v1 - v0):+.1f} ns per unit")


def phase_count(args):
    """Cost per skipped arm, for each synchronizer mode. Arm 0 is always the taken one."""
    for mode in ("blocking", "trigger_false"):
        rows = []
        for n in args.counts:
            fields = {"chain_tests": n, "chain_sync": mode, "chain_body": args.body,
                      "test_register_value": 0}          # arm 0 taken, the other n-1 skipped
            result = deploy(fields, tag=f"{mode}_n{n}_b{args.body}")
            rows.append((n, *marker_interval(result)))
            print(f"    {mode} n={n}: {rows[-1][1]} ns measured", flush=True)
        show(f"MARKER INTERVAL vs number of arms  [{mode}, {args.body} pulse(s) per arm, "
             f"arm 0 taken so n-1 are SKIPPED]", rows, "arms")


def phase_body(args):
    """Is a skip O(1) or O(body)? Fix the arm count, grow what each arm contains."""
    for mode in ("blocking", "trigger_false"):
        rows = []
        for body in args.bodies:
            fields = {"chain_tests": args.tests, "chain_sync": mode, "chain_body": body,
                      "test_register_value": 0}
            result = deploy(fields, tag=f"{mode}_body{body}_n{args.tests}")
            rows.append((body, *marker_interval(result)))
            print(f"    {mode} body={body}: {rows[-1][1]} ns measured", flush=True)
        show(f"MARKER INTERVAL vs pulses per arm  [{mode}, {args.tests} arms, arm 0 taken]",
             rows, "pulses/arm")


def phase_join(args):
    """Is the 1-cycle join dwell load-bearing after channel_trigger?

    With `trigger=False` the arms only QUEUE their commands; `channel_trigger` fires them and
    returns immediately. Marker B is on ANOTHER channel, so the next blocking synchronizer
    waits only on ITS channel -- nothing holds marker B back while the arm is still playing.
    If the join dwell matters, dropping it should let marker B move EARLIER, and by enough to
    overlap the arm's pulse.
    """
    print("\nJOIN: does dropping the 1-cycle join dwell let the next block start early?")
    print("  'other' = the next block is on a DIFFERENT channel from the arms (nothing but the")
    print("  join can hold it back);  'same' = it is on the arms' own channel, where the DMA")
    print("  FIFO orders the commands whether or not the sequencer waited.")
    for join, after in [(j, a) for a in ("other", "same") for j in ("dwell", "none")]:
        fields = {"chain_tests": args.tests, "chain_sync": "trigger_false",
                  "chain_body": args.body, "chain_join": join,
                  "chain_after": after, "test_register_value": 0}
        result = deploy(fields, tag=f"join_{join}_{after}_n{args.tests}")
        meas, pred = marker_interval(result, "ch0" if after == "other" else "ch1")
        # where each pulse actually lands, so an overlap is visible rather than inferred
        edges = {row["channel"]: (row.get("first_edge_ns"), row.get("measured_widths_ns"))
                 for row in result.get("rows", ()) if row.get("n_measured")}
        print(f"\n  after={after:<6} join={join:<6} marker interval measured {meas} ns, "
              f"model {pred} ns")
        for ch, (first, widths) in sorted(edges.items()):
            print(f"      {ch}: first edge {first} ns, pulse widths {widths}")
        print(f"      {result['folder']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                        choices=("count", "body", "join", "all"))
    parser.add_argument("--counts", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--bodies", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--tests", type=int, default=4)
    parser.add_argument("--body", type=int, default=1)
    args = parser.parse_args()

    phases = (("count", phase_count), ("body", phase_body), ("join", phase_join))
    for name, fn in phases:
        if args.phase in (name, "all"):
            fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
