"""The viewport at every scale, and at inputs no sensible caller would send.

A window is the one control every reader uses, on every sequence, constantly -- and it is the one
that has already produced two wrong pictures rather than an error: stale tab text left over from a
wider frame, and a window kept across a length change so that 78 us of a 319 us run looked like the
whole run. Both were silent. Neither showed up at the handful of window sizes the other checks use.

So this walks the width axis end to end -- the full extent down to below ``MIN_SPAN_NS``, a span of
twelve decades -- and then feeds the inputs a dragged scrollbar, a text box or a wheel gesture can
actually produce: inverted, zero-width, negative, past the end, absurdly large, and non-finite.

The properties are the ones that must hold at EVERY scale, because a reader has no way to tell a
lie at 5 ns from a lie at 5 ms:

* the window stays finite, ordered, no narrower than the floor, and inside the sequence -- the
  scrollbars assume it is a slice of the sequence and go degenerate if it is not;
* no more ``@`` labels than there are tabs, which is the stale-text signature;
* asking for the same window twice gives the same window (an idempotent control);
* ``reset()`` restores the full extent from anywhere, including from a broken input.

The LANE axis gets the same treatment. It is the second viewport -- its own scrollbar, its own
degenerate cases -- and it had none of the time axis's protection: ``set_lanes`` sorted its
arguments and stored them, so a NaN reached matplotlib exactly as it did on the time axis. Both now
go through one rule (``SequenceView.slice_of``), and both are checked here, because a rule written
once but exercised on one axis is a rule that holds on one axis.

Run: ``python validation/zoom_extremes.py [folders]`` (offline, no Qt, no board).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

#: Inputs a real gesture can produce. A scrollbar at its rail sends the end exactly; a text box
#: sends whatever was typed; a wheel at a chart edge sends a span wider than the sequence.
BROKEN = [
    ("inverted", lambda lo, hi: (hi, lo)),
    ("zero width", lambda lo, hi: (0.5 * (lo + hi), 0.5 * (lo + hi))),
    ("negative start", lambda lo, hi: (-abs(hi) * 3.0, hi * 0.5)),
    ("past the end", lambda lo, hi: (hi * 0.9, hi * 50.0)),
    ("astronomically wide", lambda lo, hi: (-1e18, 1e18)),
    ("tiny at the very end", lambda lo, hi: (hi - 1e-9, hi)),
    ("tiny at the very start", lambda lo, hi: (lo, lo + 1e-9)),
    ("not a number", lambda lo, hi: (float("nan"), hi)),
    ("infinite", lambda lo, hi: (lo, float("inf"))),
]


def window_problems(view, name, what):
    """The invariants that must hold whatever the window is."""
    problems = []
    lo, hi = view.xlim_ns
    full_lo, full_hi = view.full_xlim
    if not (math.isfinite(lo) and math.isfinite(hi)):
        problems.append(f"{name}: {what} left a non-finite window {lo}..{hi}")
        return problems
    if hi <= lo:
        problems.append(f"{name}: {what} left an empty window {lo:.3f}..{hi:.3f}")
    elif hi - lo < view.MIN_SPAN_NS - 1e-9:
        problems.append(f"{name}: {what} left {hi - lo:.4f} ns, under the "
                        f"{view.MIN_SPAN_NS} ns floor")
    if lo < full_lo - 1e-6 or hi > full_hi + 1e-6:
        problems.append(f"{name}: {what} left {lo:.1f}..{hi:.1f} outside the sequence "
                        f"{full_lo:.1f}..{full_hi:.1f} -- the scrollbars treat the window as a "
                        f"slice of it")
    axes = view.ax
    labels = [text for text in axes.texts if text.get_text().startswith("@")]
    tabs = [info for _frame, info in (getattr(axes, "_seqviz_flow_frames", None) or [])
            if info.get("tab_rect")]
    if len(labels) > len(tabs) + 1:
        problems.append(f"{name}: {what} shows {len(labels)} labels for {len(tabs)} tabs -- "
                        f"text left over from an earlier frame")
    return problems


def lane_problems(view, name, what):
    """The lane viewport must be a finite, ordered slice of the lane stack, like the time one."""
    problems = []
    lo, hi = view.ylim or view.full_ylim
    full_lo, full_hi = view.full_ylim
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return [f"{name}: {what} left a non-finite lane range {lo}..{hi}"]
    if hi <= lo:
        problems.append(f"{name}: {what} left an empty lane range {lo:.3f}..{hi:.3f}")
    elif hi - lo < view.MIN_LANE_SPAN - 1e-9:
        problems.append(f"{name}: {what} left {hi - lo:.4f} lanes, under the "
                        f"{view.MIN_LANE_SPAN} floor")
    if lo < full_lo - 1e-6 or hi > full_hi + 1e-6:
        problems.append(f"{name}: {what} left lanes {lo:.2f}..{hi:.2f} outside the stack "
                        f"{full_lo:.2f}..{full_hi:.2f} -- the lane scrollbar treats the range as "
                        f"a slice of it")
    return problems


def exercise_lanes(view, name):
    """Every lane window, and the malformed ones a drag or a wheel can produce."""
    problems, tried = [], 0
    full_lo, full_hi = view.full_ylim
    lanes = full_hi - full_lo
    height = lanes
    while height > 0.05:
        for anchor in (0.0, 0.5, 1.0):
            lo = full_lo + (lanes - height) * anchor
            view.set_lanes(lo, lo + height)
            tried += 1
            problems += lane_problems(view, name, f"{height:.3g} lanes at {anchor:.0%}")
        height /= 4.0
    for what, make in BROKEN:
        try:
            view.set_lanes(*make(full_lo, full_hi))
            tried += 1
        except Exception as exc:                          # noqa: BLE001
            problems.append(f"{name}: lane {what} raised {type(exc).__name__}: {exc}")
            view.reset()
            continue
        problems += lane_problems(view, name, f"lane {what}")
        # and through the other door: a window carrying a lane range with it. The TIME window is
        # the sequence's own extent -- passing lane coordinates here would zoom the time axis to a
        # few ns, which is a different test wearing this one's name.
        time_lo, time_hi = view.full_xlim
        try:
            view.set_window(time_lo, time_hi, ylim=make(full_lo, full_hi))
            tried += 1
        except Exception as exc:                          # noqa: BLE001
            problems.append(f"{name}: set_window(ylim={what}) raised {type(exc).__name__}: {exc}")
            view.reset()
            continue
        problems += lane_problems(view, name, f"set_window ylim {what}")
    view.reset()
    return problems, tried


def exercise(trace, name):
    """Returns (problems, windows_tried)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from acadia_qmsmt import sequence_viz as sv

    problems, tried = [], 0
    figure, axes = plt.subplots(figsize=(12, 6))
    view = sv.SequenceView(trace, axes)
    full_lo, full_hi = view.full_xlim
    length = max(full_hi - full_lo, 1.0)
    try:
        # every width from the whole sequence down past the floor, at three anchor points
        width = length
        while width > 1e-4:
            for anchor in (0.0, 0.5, 1.0):
                lo = full_lo + (length - width) * anchor
                view.set_window(lo, lo + width)
                tried += 1
                problems += window_problems(view, name, f"width {width:.4g} ns at {anchor:.0%}")
                seen = view.xlim_ns
                view.set_window(lo, lo + width)          # idempotent: same ask, same window
                if view.xlim_ns != seen:
                    problems.append(f"{name}: asking for width {width:.4g} ns twice gave "
                                    f"{seen} then {view.xlim_ns}")
            width /= 10.0

        for what, make in BROKEN:
            try:
                view.set_window(*make(full_lo, full_hi))
                tried += 1
            except Exception as exc:                      # noqa: BLE001
                problems.append(f"{name}: {what} raised {type(exc).__name__}: {exc}")
                view.reset()
                continue
            problems += window_problems(view, name, what)
            view.reset()
            lo, hi = view.xlim_ns
            if abs(lo - full_lo) > 1e-6 or abs(hi - full_hi) > 1e-6:
                problems.append(f"{name}: reset after {what} left {lo:.1f}..{hi:.1f} of "
                                f"{full_lo:.1f}..{full_hi:.1f}")
        found, lane_tried = exercise_lanes(view, name)
        problems += found
        tried += lane_tried
    except Exception:
        import traceback
        problems.append(f"{name}: {traceback.format_exc(limit=3)}")
    finally:
        plt.close(figure)
    return problems, tried


def main():
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    from gui_validation import broad_folders
    from acadia_qmsmt import sequence_viz as sv

    limit = next((int(a) for a in sys.argv[1:] if a.isdigit()), 12)
    problems, windows, folders = [], 0, 0
    for folder in broad_folders(limit):
        name = folder.split("/")[-4]
        try:
            trace = sv.trace_folder(folder, envelopes=False)
        except Exception:
            continue
        folders += 1
        found, tried = exercise(trace, name)
        problems += found
        windows += tried
        print(f"   {name:28s} {tried:3d} windows -- "
              f"{'ok' if not found else f'{len(found)} PROBLEM(S)'}", flush=True)

    print(f"\n{windows} viewport states over {folders} folders, on BOTH axes "
          f"({len(BROKEN)} malformed inputs each, and again through set_window's ylim); "
          f"{len(problems)} problems")
    for line in problems[:12]:
        print(f"   {line}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
