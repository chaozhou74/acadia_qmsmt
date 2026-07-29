"""
Stage 2: check the envelope, detune and phase the visualizer reports against the loopback.

Stage 1 established that the *timing* is right. This checks what the pulse actually looks
like: is `trace.envelope(io, pulse, "memory")` -- the samples read back out of DAC memory --
the waveform that physically came out?

What can and cannot be compared
-------------------------------
* SHAPE, yes. Normalise both to unit peak and compare after aligning in time. Amplitude is
  per-channel relative only (the user's call), because the DAC gain, cable loss and ADC
  scaling are all unknown constants.
* DETUNE, yes, as a phase slope. The captured IQ phase advances at the pulse's detune
  because the DAC and ADC NCOs are set to the same frequency, so the residual carrier is the
  SSB detune alone. Capture spacing is 5 ns, so anything beyond ~100 MHz aliases.
* PHASE, only as a DIFFERENCE between two pulses in the same trace. The absolute captured
  phase also contains the propagation delay and the DAC/ADC NCO phase offset, neither known.
* Edges between pulses of DIFFERENT ramp shape: no. A 50%-of-power crossing sits at a
  different point on a fast ramp than a slow one -- this cost ~25 ns of apparent error in
  stage 1. Compare like with like.

Usage
-----
    $ACADIA_ENV/bin/python shape_validation.py --case shape
    $ACADIA_ENV/bin/python shape_validation.py --cases shape,detune_pair,phase_pair
    $ACADIA_ENV/bin/python shape_validation.py --analyse /path/to/test_loopback/shape/<run>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from timing_validation import CHANNEL_OF, pulse_regions_ns, run_case

IO_OF = {"ch0": "stimulus0", "ch1": "stimulus1", "ch2": "stimulus2", "ch3": "stimulus3"}


def measured_iq(runtime, label, start_ns, stop_ns):
    """Complex captured trace over one pulse region, and its time base in ns."""
    t = np.asarray(runtime.t_data) * 1e9
    iq = np.asarray(runtime.avg_trace_iq[label])
    mask = (t >= start_ns) & (t <= stop_ns)
    return t[mask], iq[mask, 0] + 1j * iq[mask, 1]


def fit_detune_hz(t_ns, samples, magnitude_floor=0.5):
    """Least-squares slope of the unwrapped phase, over the strong part of the pulse.

    Restricted to samples above ``magnitude_floor`` of the peak: on the ramps the amplitude
    is small and the phase is noise-dominated, which would bias the slope.
    """
    magnitude = np.abs(samples)
    keep = magnitude > magnitude_floor * magnitude.max()
    if keep.sum() < 4:
        return None
    phase = np.unwrap(np.angle(samples[keep]))
    slope = np.polyfit(t_ns[keep], phase, 1)[0]      # rad/ns
    return slope / (2 * np.pi) * 1e9                 # Hz


def mean_phase(t_ns, samples, detune_hz, magnitude_floor=0.5):
    """Phase at the pulse start with the detune ramp removed, in radians."""
    magnitude = np.abs(samples)
    keep = magnitude > magnitude_floor * magnitude.max()
    if keep.sum() < 4:
        return None
    rotation = np.exp(-2j * np.pi * (detune_hz or 0.0) * (t_ns[keep] - t_ns[0]) * 1e-9)
    return float(np.angle(np.mean(samples[keep] * rotation)))


def shape_error(predicted, t_ns, samples):
    """RMS difference between normalised |predicted| and |measured|, after time alignment.

    The predicted envelope is resampled onto the measured grid by matching the pulse's own
    duration; both are peak-normalised so the unknown gain drops out.
    """
    predicted_magnitude = np.abs(predicted)
    if predicted_magnitude.max() <= 0:
        return None
    predicted_magnitude = predicted_magnitude / predicted_magnitude.max()
    measured_magnitude = np.abs(samples)
    if measured_magnitude.max() <= 0:
        return None
    measured_magnitude = measured_magnitude / measured_magnitude.max()

    # align by cross-correlation on the measured grid, then compare on the overlap
    grid = np.linspace(0, 1, len(measured_magnitude))
    resampled = np.interp(grid, np.linspace(0, 1, len(predicted_magnitude)),
                          predicted_magnitude)
    shift = int(np.argmax(np.correlate(measured_magnitude, resampled, mode="same"))
                - len(resampled) // 2)
    rolled = np.roll(resampled, shift)
    return float(np.sqrt(np.mean((rolled - measured_magnitude) ** 2)))


def analyse(folder, verbose=True):
    import sequence_viz as sv
    from acadia_qmsmt.utils.saved_runtime_loader import load_runtime_from_data_dir

    trace = sv.trace_folder(folder)
    runtime = load_runtime_from_data_dir(folder)
    runtime.process_current_data()

    rows = []
    for label, channel in CHANNEL_OF.items():
        regions = pulse_regions_ns(runtime, label)
        # pulses the visualizer says this channel plays, in time order
        scheduled = [c for c in sorted(trace.commands, key=lambda c: c.start)
                     if c.pulse and c.channel == channel]
        for index, (region, command) in enumerate(zip(regions, scheduled)):
            start_ns, stop_ns, _peak = region
            t_ns, samples = measured_iq(runtime, label, start_ns, stop_ns)
            predicted = trace.envelope(command.io_name or IO_OF[label],
                                       command.pulse, "memory")
            configured_detune = 0.0
            try:
                configured_detune = float(runtime.io(
                    command.io_name or IO_OF[label]).get_pulse_config(
                        command.pulse).get("detune") or 0.0)
            except Exception:
                pass
            detune = fit_detune_hz(t_ns, samples)
            rows.append({
                "channel": label, "index": index, "pulse": command.pulse,
                "configured_detune_hz": configured_detune,
                "measured_detune_hz": detune,
                "phase_rad": mean_phase(t_ns, samples, detune),
                "shape_rms": (shape_error(predicted, t_ns, samples)
                              if predicted is not None else None),
            })

    if verbose:
        print(f"\n=== {folder}")
        print(f"    {trace.runtime_class}, case pulses: "
              f"{sorted({r['pulse'] for r in rows})}")
        print(f"    {'ch':4s} {'#':>2s} {'pulse':16s} {'detune cfg':>11s} "
              f"{'detune meas':>12s} {'err':>9s} {'phase':>8s} {'shape rms':>10s}")
        for row in rows:
            measured = row["measured_detune_hz"]
            configured = row["configured_detune_hz"]
            measured_text = "n/a" if measured is None else f"{measured / 1e6:11.3f}M"
            error_text = "" if measured is None else f"{(measured - configured) / 1e6:8.3f}M"
            phase = row["phase_rad"]
            phase_text = "n/a" if phase is None else f"{phase:8.3f}"
            rms = row["shape_rms"]
            rms_text = "n/a" if rms is None else f"{rms:10.4f}"
            print(f"    {row['channel']:4s} {row['index']:2d} {row['pulse']:16s} "
                  f"{configured / 1e6:10.3f}M {measured_text} {error_text:>9s} "
                  f"{phase_text} {rms_text}")
        # Relative phase between consecutive pulses on one channel. Only meaningful when
        # the two pulses share a detune -- each pulse's phase is referenced to its own
        # start, so comparing pulses with different detunes compares nothing.
        detunes = {r["configured_detune_hz"] for r in rows}
        comparable = len(detunes) == 1
        for label in CHANNEL_OF:
            phases = [r["phase_rad"] for r in rows
                      if r["channel"] == label and r["phase_rad"] is not None]
            if len(phases) > 1:
                delta = np.angle(np.exp(1j * (phases[1] - phases[0])))
                note = "" if comparable else "   (NOT meaningful: detunes differ)"
                print(f"    {label}: pulse[1] - pulse[0] phase = {delta:+.4f} rad "
                      f"({np.degrees(delta):+.2f} deg){note}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--cases", type=str, default=None)
    parser.add_argument("--analyse", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()

    if args.analyse:
        analyse(args.analyse)
        return

    cases = ([c.strip() for c in args.cases.split(",")] if args.cases
             else [args.case or "shape"])
    for case in cases:
        print(f"\n>>> deploying case {case!r} ...")
        try:
            analyse(run_case(case, iterations=args.iterations))
        except Exception as exc:
            print(f"    FAILED {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
