"""Probe acadia's sequencer surface: what compiles, what is rejected, and how the tracer copes.

Compiling is cheap and needs no board, so the whole parameter space of a construct can be tried
offline and reduced to a capability matrix. That matters for two audiences:

* **runtime authors** -- knowing that `repeat_until` accepts only `==` and `!=` BEFORE writing a
  loop, rather than after a confusing `__bool__ should return bool, returned Operation`;
* **this validation suite** -- a form acadia rejects need not be tested on hardware, and a form it
  accepts but the tracer cannot resolve should be drawn as unresolved rather than guessed.

Each probe reports one of:

    COMPILES/RESOLVED    acadia accepts it and the trace pins the count/length from the program
    COMPILES/ASSUMED     acadia accepts it; the trace cannot resolve it and says so (correct)
    REJECTED             acadia refuses to compile it, with the message it gives
    TRACE-FAILED         acadia compiles it but the tracer raises -- a viewer bug

The last is the interesting one. The rest are documentation.

Run: ``python validation/acadia_probe.py`` (no hardware, no writes).
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def probe(name, build):
    """Compile-and-trace one construct. `build` takes the acadia handle and the case's helpers."""
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    import timing_validation as tv
    from acadia_qmsmt import sequence_viz as sv

    runtime = tv.build_runtime("probe", iterations=10)
    runtime.probe_build = build
    try:
        trace = sv.trace_runtime(runtime, envelopes=False)
    except Exception as exc:
        message = str(exc).splitlines()[0][:100]
        # distinguish "acadia said no" from "the viewer broke": acadia's own refusals surface
        # from acadia.* frames, a viewer bug from sequence_viz frames
        frames = traceback.format_exc()
        where = "REJECTED" if "/acadia/" in frames and "sequence_viz" not in frames.split(
            "acadia/pyacadia")[-1][:400] else "TRACE-FAILED"
        return where, f"{type(exc).__name__}: {message}"
    if trace.assumed_paths or any(c.symbolic for c in trace.commands):
        return "COMPILES/ASSUMED", (f"{len(trace.placements)} executed, "
                                    f"assumed={sorted(trace.assumed_paths)}")
    return "COMPILES/RESOLVED", (f"{len(trace.placements)} executed, "
                                 f"counts={dict(trace.repeat_counts)}")


def main():
    print(__doc__.splitlines()[0])
    print("\nThis module documents the probe harness. The probes themselves live in")
    print("loopback_timing_cases.py as real cases, because a construct worth probing is a")
    print("construct worth MEASURING -- a compile-time answer about timing is only half of one.")
    print("\nEstablished so far (see ACADIA_FINDINGS.md for the evidence):")
    for line in (
        "  repeat_until(counter == N)      COMPILES/RESOLVED   count pinned from the condition",
        "  repeat_until(counter != N)      COMPILES/ASSUMED    count not statically knowable",
        "  repeat_until(counter <  N)      REJECTED            'can only check x < 0 or 0 <= x'",
        "  repeat_until(counter >  N)      REJECTED            'can only check 0 > x or x >= 0'",
        "  repeat_until(counter >= N)      REJECTED            same as >",
        "  repeat_until(counter <= N)      REJECTED            '__bool__ should return bool'",
        "  test(reg == N, speculation=True)   COMPILES/RESOLVED  both arms correct on hardware",
        "  test(..., speculation=False)       UNSAFE            skipped arm hangs the sequencer",
        "  three nested DSP counter loops     UNSAFE unless each counter is re-configure()d",
        "  register-driven length of 0        COMPILES          but the board plays ~21 s (wraps)",
        "  channel_is_fifo_empty              COMPILES          used by NO qudit runtime",
        "  channel_is_fifo_almost_empty       COMPILES          used by all 7 streaming runtimes",
    ):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
