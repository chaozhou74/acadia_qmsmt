"""
Validate sequence_viz's timing model against the 4-channel loopback.

Each DAC is cabled to its own ADC, so a capture is a direct recording of that DAC's
output. That makes the loopback a physical oracle for the parts of the visualizer that
are *derived* rather than measured -- above all the inter-block dead time.

Method
------
Compare INTERVALS BETWEEN EDGES ON ONE CHANNEL, never absolute arrival times. The
constant DAC->cable->ADC latency (~283 ns) is unknown and unmeasurable, but it cancels
in a within-channel interval, as does any residual capture-trigger skew. Edges are taken
at 50% of the pulse plateau with linear interpolation; averaging 5000 iterations gets the
edge to ~0.05 ns even though the samples are 5 ns apart.

Usage
-----
    $ACADIA_ENV/bin/python timing_validation.py --case two_blocks
    $ACADIA_ENV/bin/python timing_validation.py --all
    $ACADIA_ENV/bin/python timing_validation.py --analyse /path/to/test_loopback/two_blocks/<run>

`--analyse` re-reads an existing folder without deploying.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # loopback_timing_cases, paths_local

import paths_local

# Board IP and data root are machine/board-specific -- read from the gitignored
# paths.local.yaml (or SEQVIZ_* env vars), never committed. May be None when only
# an off-hardware --dryrun is being run; the deploy/analyse paths require() them.
_LOCAL = paths_local.load()
BOARD_IP = _LOCAL.get("board_ip")
SAVE_ROOT = _LOCAL.get("loopback_data_root")
YAML_PATH = _LOCAL.get("yaml_path") or str(Path(__file__).parent / "validation_board_config.yaml")
REPORT = Path(__file__).parent / "timing_validation_results.json"

CHANNEL_OF = {"ch0": "DAC0", "ch1": "DAC2", "ch2": "DAC4", "ch3": "DAC6"}

# Cases whose measured error is a known MEASUREMENT systematic, not a model error.
KNOWN_SYSTEMATIC = {
    "stretch_two_blocks":
        "mixed ramp shapes (50 ns stretch vs 10 ns test pulse): a 50%-of-power crossing "
        "sits at a different point on each ramp, costing ~20-25 ns of apparent error. "
        "stretch_two_blocks_same uses identical pulses and agrees to 0.05 ns.",
    "phase_pair":
        "edge-estimation limited, not a model error: the stage-2 pulses have 50 ns ramps, "
        "5x slower than the 10 ns-ramp cases, so the same amplitude noise buys 5x the "
        "timing jitter. Errors are MIXED SIGN and sub-cycle (-1.14/+0.70/+0.82/+0.26 ns); "
        "a real model error shows one sign and one magnitude on all four channels, as the "
        "+1 cycle boundary offset did (+4.96..+5.00 everywhere).",
    "test_true_nospec":
        "UNSUPPORTED, see KI_004: test(speculation=False) places the body out of line so "
        "address order stops being execution order. The taken arm mistimes by ~25 ns and "
        "the skipped arm (test_false_nospec) hangs the sequencer. The default "
        "speculation=True is correct in both arms (0.04 / 0.10 ns) -- use that.",
    "test_false_nospec":
        "UNSUPPORTED, see KI_004: this variant HANGS the board. Do not re-deploy it.",
    "detune_pair":
        "same slow-ramp edge limit as phase_pair, plus SSB imbalance ripple that differs "
        "between the 10 MHz and 25 MHz pulses and perturbs the plateau peak the half level "
        "is taken from. Timing conclusions rest on the fast-ramp cases.",
    "rb_stream":
        "the final '8 basic gates' block plays lo/mid/hi amplitudes back-to-back, merging into "
        "one region; the 50%-of-region-peak rising edge then latches onto the first HIGH gate, "
        "reading the block ~70 ns (2 gates) late. Measurement systematic, not a model error -- "
        "the region START matches the tracer to ~5 ns. The amplitude variation is for gate "
        "IDENTITY; for timing of that block, set rb_final_gate to one shape (rb_stream_uniform / "
        "rb_final_gate='rb_gate_hi' validates to ~0-5 ns). The stream unroll itself is exact.",
}


# ---------------------------------------------------------------- measurement

def pulse_regions_ns(runtime, label, threshold_sigmas=10.0):
    """(start, stop, peak) of every above-threshold region, in ns.

    Width matters for the seamless-join test: two pulses butted together with no gap
    merge into ONE region of twice the width, which is the signature of a correct
    zero-gap prediction rather than a missing pulse.
    """
    t = np.asarray(runtime.t_data) * 1e9
    y = np.asarray(runtime.avg_trace_pwr[label])
    # Baseline from a LOW percentile, not the first 30 samples: a capture_start_delay can
    # slide a pulse into those early samples (a first-samples baseline would then inflate the
    # threshold and hide every region), while a median baseline sits ON the pulse when a
    # pulse fills most of the trace (the stretch cases). The 20th percentile tracks the quiet
    # level as long as some of the trace is quiet; the noise comes from the lower band so a
    # plateau cannot inflate it.
    level = float(np.percentile(y, 20))
    band = y[y <= np.percentile(y, 40)]
    noise = float(band.std()) if band.size > 1 else float(y.std())
    if noise == 0.0:                                        # degenerate flat trace
        noise = 1.0
    above = np.where(y > level + threshold_sigmas * noise)[0]
    if not len(above):
        return []
    regions, start = [], above[0]
    for i in range(1, len(above)):
        if above[i] - above[i - 1] > 1:
            regions.append((start, above[i - 1]))
            start = above[i]
    regions.append((start, above[-1]))

    # A baseline-sigma threshold alone lets single-sample noise through: on a quiet
    # channel it flagged a 1-sample "pulse" at peak 0.77 against a real peak of 4.8e4.
    # Require a real pulse to be at least 2 samples wide and 5% of the channel maximum.
    floor = 0.05 * y.max()
    return [(float(t[a]), float(t[b]), float(y[a:b + 1].max()))
            for a, b in regions if b > a and y[a:b + 1].max() >= floor]


def pulse_edges_ns(runtime, label, threshold_sigmas=10.0):
    """Rising-edge times of every pulse on one channel, in ns, sub-sample resolved."""
    t = np.asarray(runtime.t_data) * 1e9
    y = np.asarray(runtime.avg_trace_pwr[label])
    edges = []
    for start_ns, stop_ns, peak in pulse_regions_ns(runtime, label, threshold_sigmas):
        # Take the half level from THIS region's own peak. A fixed-width window does not
        # reach the plateau of a long pulse, which biases the half level -- and therefore
        # the crossing -- by pulse length, costing ~1-2 ns on 300 ns pulses.
        a = int(np.searchsorted(t, start_ns))
        b = int(np.searchsorted(t, stop_ns))
        if b <= a:
            continue
        half = peak / 2.0
        rising = np.where(y[max(a - 4, 1):b + 1] > half)[0]
        if not len(rising):
            continue
        j = max(a - 4, 1) + int(rising[0])
        if j < 1:
            continue
        edges.append(float(np.interp(half, [y[j - 1], y[j]], [t[j - 1], t[j]])))
    return edges


def measure(folder):
    """{label: [edge_ns, ...]} for a finished loopback folder."""
    from acadia_qmsmt.utils.saved_runtime_loader import load_runtime_from_data_dir

    # str(): --revalidate/--all pass Path objects, and some DataManager.load
    # builds reject a Path outright ("Load location must be a string"). The loader
    # coerces too, but don't depend on that -- folder.py does the same for its own
    # load_runtime_from_data_dir call.
    runtime = load_runtime_from_data_dir(str(folder))
    runtime.process_current_data()
    return {label: pulse_edges_ns(runtime, label) for label in
            ("ch0", "ch1", "ch2", "ch3")}


# ---------------------------------------------------------------- prediction

def predict(folder):
    """{label: [region start ns, ...]} from sequence_viz's trace of the same folder.

    Pulses that are contiguous in the prediction are MERGED into one region, because that
    is what the measurement sees: two pulses butted together with no gap cross the threshold
    once and come back as a single wide region. Without merging, a case like
    ``two_same_block`` compares nothing (fewer measured regions than predicted pulses) and
    ``loop_2_double`` compares the wrong pairs.
    """
    from acadia_qmsmt import sequence_viz as sv

    trace = sv.trace_folder(folder)
    starts = {}
    for label, channel in CHANNEL_OF.items():
        pulses = sorted((c for c in trace.commands
                         if c.pulse and c.channel == channel),
                        key=lambda c: c.start)
        merged = []
        for command in pulses:
            if merged and command.start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], command.stop)
            else:
                merged.append([command.start, command.stop])
        starts[label] = [region[0] * trace.ns_per_cycle for region in merged]
    return starts, trace


# ---------------------------------------------------------------- comparison

def compare(folder, verbose=True):
    """Predicted vs measured intervals for one run. Returns a result dict."""
    measured = measure(folder)
    predicted, trace = predict(folder)

    rows, worst = [], 0.0
    for label in ("ch0", "ch1", "ch2", "ch3"):
        m, p = measured[label], predicted[label]
        # intervals relative to the channel's own first pulse -- the latency cancels
        m_int = [x - m[0] for x in m[1:]] if len(m) > 1 else []
        p_int = [x - p[0] for x in p[1:]] if len(p) > 1 else []
        n = min(len(m_int), len(p_int))
        errors = [m_int[i] - p_int[i] for i in range(n)]
        if errors:
            worst = max(worst, max(abs(e) for e in errors))
        # Count and span, which the interval metric alone cannot see: a dropped or collapsed
        # train (e.g. an un-unrolled cache-pointer stream) merges to one region with no
        # intervals, so the interval error is a misleading 0. The tell-tale is the hardware
        # showing MORE pulses than the trace predicted -- `dropped` -- which means the tracer
        # is missing pulses. (Measured FEWER than predicted is the opposite, benign case of
        # two pulses merging into one region, e.g. stretch_then_pulse.)
        span_m = (m[-1] - m[0]) if len(m) > 1 else 0.0
        span_p = (p[-1] - p[0]) if len(p) > 1 else 0.0
        rows.append({"channel": label, "n_measured": len(m), "n_predicted": len(p),
                     "count_ok": len(m) == len(p),
                     # only where the tracer drew SOME pulses: a channel it (correctly) left
                     # empty can still pick up a stray noise region, which is not a drop
                     "dropped": (max(0, len(m) - len(p)) if len(p) > 0 else 0),
                     "span_measured_ns": round(span_m, 1),
                     "span_predicted_ns": round(span_p, 1),
                     "span_error_ns": round(abs(span_m - span_p), 1),
                     # absolute first edge = the constant DAC->cable->ADC latency;
                     # only meaningful as a per-channel calibration constant
                     "first_edge_ns": round(m[0], 2) if m else None,
                     "measured_intervals_ns": [round(x, 2) for x in m_int],
                     "predicted_intervals_ns": [round(x, 2) for x in p_int],
                     "error_ns": [round(e, 2) for e in errors],
                     "error_cycles": [round(e / trace.ns_per_cycle, 3) for e in errors]})

    result = {
        "folder": str(folder),
        "runtime": trace.runtime_class,
        "case": getattr(trace, "case", None),
        "blocks": len(trace.blocks),
        # gaps live on placements (what executes, loops unrolled), not on blocks
        "gaps_ns": [round(p.gap_after * trace.ns_per_cycle, 1)
                    for p in (trace.placements or trace.blocks) if p.gap_after],
        "gap_breakdowns": [p.gap_breakdown
                           for p in (trace.placements or trace.blocks) if p.gap_after],
        "executed_blocks": len(trace.placements or trace.blocks),
        "worst_error_ns": round(worst, 2),
        "worst_error_cycles": round(worst / trace.ns_per_cycle, 3),
        # channels where hardware shows more pulses than the trace drew: a structural miss
        # (the tracer dropped pulses), separate from and invisible to the interval error
        "pulses_dropped": [r["channel"] for r in rows if r["dropped"] > 0],
        "rows": rows,
    }

    # Cache-pointer stream (rb_stream / randomized benchmarking) guard: the loop MUST be
    # unrolled from the cache to the count the cache holds, and the post-loop blocks MUST sit
    # after the train. This is exactly the regression the tracer fix prevents (a collapsed
    # stream would leave one symbolic placeholder and drag the readout/tail up to the front).
    if getattr(trace, "stream", None):
        streamed = [p for p in trace.placements if getattr(p, "stream", False)]
        gates = sum(len(p.commands) for p in streamed)
        expected = int(trace.point_cache.get(trace.stream["count_word"], 0))
        train_stop = max((p.stop for p in streamed), default=0)
        after = any(not getattr(p, "stream", False) and p.start >= train_stop
                    and any(c.pulse for c in p.commands) for p in trace.placements)
        result["stream_gates"] = gates
        result["stream_expected"] = expected
        result["stream_ok"] = bool(gates == expected and expected > 0
                                    and not result["pulses_dropped"] and after)

    if verbose:
        print(f"\n=== {folder}")
        print(f"    {trace.runtime_class}: {len(trace.blocks)} blocks, "
              f"gaps {result['gaps_ns']} ns")
        for row in rows:
            if row["dropped"] > 0:
                flag = (f"  <-- DROPPED {row['dropped']} pulses "
                        f"(meas {row['n_measured']} > pred {row['n_predicted']}, "
                        f"span meas/pred {row['span_measured_ns']}/{row['span_predicted_ns']} ns)")
            elif row["error_ns"] and max(abs(e) for e in row["error_ns"]) >= 1.0:
                flag = "  <-- MISMATCH"
            else:
                flag = "  OK" if row["error_ns"] else ""
            print(f"    {row['channel']}  pulses meas/pred {row['n_measured']}/{row['n_predicted']}"
                  f"  measured {row['measured_intervals_ns']}"
                  f"  predicted {row['predicted_intervals_ns']}"
                  f"  error {row['error_ns']} ns "
                  f"({row['error_cycles']} cyc){flag}")
        print(f"    worst error: {result['worst_error_ns']} ns "
              f"= {result['worst_error_cycles']} cycles")
        if "stream_ok" in result:
            print(f"    cache-pointer stream: {result['stream_gates']} gates unrolled "
                  f"(cache says {result['stream_expected']}), tail after train: "
                  f"{'OK' if result['stream_ok'] else '<-- FAIL (stream not unrolled)'}")
    return result


# ---------------------------------------------------------------- running

def build_runtime(case, iterations=5000, dwell_length=200e-9):
    from loopback_timing_cases import (
        LoopbackTimingCaseRuntime)

    return LoopbackTimingCaseRuntime(
        stimulus0="stimulus0", stimulus1="stimulus1",
        stimulus2="stimulus2", stimulus3="stimulus3",
        capture0="capture0", capture1="capture1",
        capture2="capture2", capture3="capture3",
        capture_dummy="capture_dummy",
        case=case,
        dwell_length=dwell_length,
        iterations=iterations,
        # the test_* cases compare a cached register against 0, so 0 forces the
        # condition true and 1 forces it false
        test_register_value=0 if "true" in case else 1,
        yaml_path=YAML_PATH,
    )


def dry_run(case, **kwargs):
    """Trace a case off-hardware. Catches compile errors before any deploy.

    The dry run calls the real ``acadia.compile()``, so anything the board would
    reject at compile time is rejected here too -- barrier_uneven's multi-argument
    max() is the example.
    """
    from acadia_qmsmt import sequence_viz as sv

    trace = sv.trace_runtime(build_runtime(case, **kwargs), capture_points=False,
                             envelopes=False)
    # gaps live on placements (what executes), not on blocks (what was compiled)
    executed = trace.placements or trace.blocks
    gaps = [round(p.gap_after * trace.ns_per_cycle, 1) for p in executed if p.gap_after]
    unrolled = ("" if len(executed) == len(trace.blocks)
                else f" -> {len(executed)} executed")
    print(f"  {case:20s} OK   {len(trace.blocks)} blocks{unrolled}, "
          f"{trace.length_ns:.0f} ns, gaps {gaps} ns")
    return trace


def run_case(case, iterations=5000, dwell_length=200e-9):
    """Deploy one timing case to the board. Returns the local data directory."""
    board_ip = paths_local.require("board_ip")
    save_root = paths_local.require("loopback_data_root")
    runtime = build_runtime(case, iterations=iterations, dwell_length=dwell_length)
    runtime.deploy(board_ip, local_directory=f"{save_root}/{case}/%y%m%d_%H%M%S")
    runtime.wait_for_deploy_completion()
    return runtime.local_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--cases", type=str, default=None,
                        help="comma-separated list, run in one invocation")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--analyse", type=str, default=None,
                        help="analyse an existing folder without deploying")
    parser.add_argument("--regions", type=str, default=None,
                        help="dump above-threshold regions per channel for a folder")
    parser.add_argument("--logs", type=str, default=None,
                        help="print the remote logs of the newest run of a case")
    parser.add_argument("--dryrun", type=str, default=None,
                        help="trace cases off-hardware ('all' for every case)")
    parser.add_argument("--revalidate", action="store_true",
                        help="re-compare the newest run of every case, compactly")
    parser.add_argument("--program", type=str, default=None,
                        help="dump the compiled sequencer program for a case (dry run)")
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()

    if args.program:
        # Tagged dump of the compiled program, so control-flow edges can be read off.
        dry_run(args.program)   # prints a summary + reproduces any compile error
        from acadia_qmsmt import sequence_viz as sv
        runtime = build_runtime(args.program)
        # dry_run already compiled a runtime, but it is not returned -- recompile a fresh
        # one purely to hold the program (a runtime can only be traced once, by design)
        sv.trace_runtime(runtime, capture_points=False, envelopes=False)
        program = [instruction.pprint() for sequencer
                   in runtime.acadia._sequencer_type.instances
                   for instruction in sequencer._compiled_program]
        for index, line in enumerate(program):
            # Tag on the instruction's OWN destination, which is the tail of the line --
            # a Symbol repr earlier in the line quotes some *other* instruction and must
            # not decide the tag. This is what hid the `test` skip branch before.
            tail = line[-90:]
            tag = ""
            if "; Trigger DMAs" in line:
                tag = "TRIGGER"
            elif "PC (absolute hold) if BUS_DATA AND MASK != 0" in tail:
                tag = "DMA-POLL"
            elif "PC (absolute branch)" in tail:
                tag = "BRANCH"
            elif "PC (absolute hold)" in tail:
                tag = "hold"
            elif "Command DMA" in line:
                tag = "push"
            elif "FIFO latency" in line:
                tag = "nop"
            elif "DSP" in line:
                tag = "dsp"
            if tag:
                print(f"  {index:04d} {tag:9s} ...{tail}")
        print(f"  ({len(program)} instructions total)")
        return

    if args.revalidate:
        paths_local.require("loopback_data_root")   # SAVE_ROOT must be set
        # regression check: newest run of each case, one line each
        print(f"{'case':22s} {'blocks':>6s} {'gaps (ns)':>22s} {'worst err':>10s}")
        worst_overall = 0.0
        for case_dir in sorted(p for p in Path(SAVE_ROOT).glob("*") if p.is_dir()):
            # a case dir holds run dirs; a run dir is one with kwargs.json in it.
            # SAVE_ROOT also holds legacy flat runs from before cases existed.
            runs = sorted((p for p in case_dir.glob("*")
                           if p.is_dir() and (p / "kwargs.json").is_file()),
                          key=lambda p: p.stat().st_mtime)
            if not runs:
                continue
            try:
                result = compare(runs[-1], verbose=False)
            except KeyError:
                # no t_data group: main() raised on the board before writing any.
                # Expected for the KI_002 cases, which cannot compile at all.
                print(f"{case_dir.name:22s} {'-':>6s} {'-':>22s}   no data "
                      f"(run failed; see --logs {case_dir.name})")
                continue
            except Exception as exc:
                print(f"{case_dir.name:22s} {'-':>6s} {'-':>22s}  "
                      f"{type(exc).__name__}")
                continue
            note = KNOWN_SYSTEMATIC.get(case_dir.name)
            if not note:
                worst_overall = max(worst_overall, result["worst_error_ns"])
            # dropped pulses / a collapsed cache-pointer stream are structural misses the
            # interval error is blind to -- flag first, even when the interval error is small
            if result.get("stream_ok") is False:
                flag = (f"   <-- STREAM NOT UNROLLED "
                        f"({result['stream_gates']}/{result['stream_expected']} gates)")
            elif result["pulses_dropped"]:
                flag = f"   <-- DROPPED PULSES on {','.join(result['pulses_dropped'])}"
            elif result["worst_error_ns"] < 1.0:
                flag = ""
            elif note:
                flag = "   <-- measurement systematic, see note"
            else:
                flag = "   <-- MISMATCH"
            print(f"{case_dir.name:22s} {result['blocks']:6d} "
                  f"{str(result['gaps_ns']):>22s} "
                  f"{result['worst_error_ns']:8.2f} ns{flag}")
        print(f"\nworst across all cases (excluding known systematics): "
              f"{worst_overall:.2f} ns")
        for case, note in KNOWN_SYSTEMATIC.items():
            print(f"\nnote on {case}: {note}")
        return

    if args.dryrun:
        from loopback_timing_cases import CASES
        cases = list(CASES) if args.dryrun == "all" else [
            c.strip() for c in args.dryrun.split(",") if c.strip()]
        print("dry run (no hardware):")
        for case in cases:
            try:
                dry_run(case)
            except Exception as exc:
                print(f"  {case:20s} FAIL {type(exc).__name__}: "
                      f"{str(exc)[:110]}")
        return

    if args.logs:
        paths_local.require("loopback_data_root")   # SAVE_ROOT must be set
        # newest run folder for a case, plus the tail of its remote logs -- kept here
        # rather than done in the shell so every invocation stays a single command
        root = Path(SAVE_ROOT) / args.logs
        folders = sorted((p for p in root.glob("*") if p.is_dir()),
                         key=lambda p: p.stat().st_mtime)
        if not folders:
            print(f"no runs under {root}")
            return
        folder = folders[-1]
        print(f"--- {folder}")
        for name in ("remote_stderr.log", "remote_main.log", "runtime.log"):
            path = folder / name
            if not path.is_file() or not path.stat().st_size:
                continue
            lines = path.read_text(errors="replace").splitlines()
            print(f"  [{name}] last {min(len(lines), 20)} of {len(lines)} lines")
            for line in lines[-20:]:
                print("    " + line)
        return

    if args.regions:
        from acadia_qmsmt.utils.saved_runtime_loader import load_runtime_from_data_dir
        for folder in args.regions.split(","):
            runtime = load_runtime_from_data_dir(folder.strip())
            runtime.process_current_data()
            print(f"--- {folder.strip()}")
            for label in ("ch0", "ch1", "ch2", "ch3"):
                regions = pulse_regions_ns(runtime, label)
                print("  %-4s %s" % (label, " | ".join(
                    f"{a:.0f}-{b:.0f} (w={b - a:.0f}, pk={p:.3g})" for a, b, p in regions)))
        return

    if args.analyse:
        compare(args.analyse)
        return

    from loopback_timing_cases import CASES
    if args.all:
        cases = list(CASES)
    elif args.cases:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        cases = [args.case or "two_blocks"]

    results = []
    for case in cases:
        print(f"\n>>> deploying case {case!r} ...")
        try:
            folder = run_case(case, iterations=args.iterations)
            results.append(compare(folder))
        except Exception as exc:              # keep the batch going unattended
            print(f"    FAILED {type(exc).__name__}: {exc}")
            results.append({"case": case, "error": f"{type(exc).__name__}: {exc}"})

    previous = json.loads(REPORT.read_text()) if REPORT.exists() else []
    REPORT.write_text(json.dumps(previous + results, indent=2))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
