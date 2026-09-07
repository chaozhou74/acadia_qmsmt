"""The tooltip must describe the thing under the cursor -- at every zoom.

"What is this bar?" is the most direct question a reader asks of the picture, and the tooltip is
the only thing that answers it. Nothing had ever checked its ANSWER. The GUI gate fires hover
events at the control-flow tabs and asserts they do not raise; the tooltip's text was never read.

That is the wrong half to check, because the failure mode here is not an exception. The hover index
is rebuilt for every window and holds each bar's extent in the PLOTTED unit -- nanoseconds when
zoomed in, microseconds when zoomed out -- so it carries a unit that changes under it. The same
divisor already caused one bug of exactly this shape: an event's ``xdata`` in ns scaled by the µs
divisor threw the view out by 1000x. An index built in one unit and hit-tested in another names a
neighbouring pulse, or one three bars away, with total confidence and no error anywhere.

So the properties are about content, not survival:

* **Zoom invariance.** A pulse hovered at ns zoom and at µs zoom must report the SAME length. The
  physical pulse did not change; only the unit the picture is drawn in did.
* **Index fidelity.** Every entry in the hover index must correspond to something the trace
  actually contains -- same lane, same duration -- so a tooltip cannot describe a bar that is not
  there.
* **Emptiness.** A cursor on a lane at a time where that lane plays nothing must not report a BAR
  there. (It may still report the control-flow box it is inside -- that box really is there, and
  answering "you are inside repeat_until @11" is the point of it. Only a claimed pulse is wrong.)

The bottom-left readout is held to the same standard, and it matters more: it is the only NUMBER
the viewer offers about the data itself. It reports the cursor's time and, over a pulse, the
complex envelope sampled at that instant -- which is what someone reads when asking "what amplitude
was actually playing here". Both halves can be wrong quietly:

* the time is ``xdata * divisor``, so it carries the same unit hazard as the index;
* the envelope is sampled by FRACTION across the bar, so an off-by-one at the ends, a reversal, or
  a fraction taken against the wrong span reports a real amplitude from the wrong instant.

So the readout is walked across each bar: the ends must anchor to the envelope's first and last
sample, the interior must follow it in order, and the same physical instant must report the same
value zoomed in as zoomed out.

Run: ``python validation/hover_truth.py [folders]`` (offline, no Qt, no board).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))


def motion(axes, x, y):
    """A mouse-motion event as matplotlib delivers it, in the frame's plotted unit."""
    return SimpleNamespace(inaxes=axes, xdata=x, ydata=y, x=0, y=0, button=None, key=None)


def tooltip_at(view, x, y):
    """Text the tooltip would show at this point, or None if it shows nothing."""
    view._compute_tooltip(motion(view.ax, x, y))
    return view._hover.get_text() if view._hover.get_visible() else None


def index_fidelity(view, trace, name):
    """Every hover entry must match something the trace contains."""
    problems = []
    lengths = {round((c.stop - c.start) * trace.ns_per_cycle, 3) for c in trace.commands}
    # a use_stretch pulse is hovered as one span of three commands, so sums of adjacent
    # commands are legitimate too; the check is that the entry is not longer than the run
    longest = max(lengths, default=0.0) * 3 + 1.0
    lanes = len(trace.channels)
    for lane_y, x0, x1, entry_name, length_ns, _kind in view._hover_index:
        if lane_y is not None and not (0 <= lane_y <= lanes - 1):
            problems.append(f"{name}: hover entry {entry_name!r} sits on lane {lane_y}, "
                            f"outside the {lanes} lanes")
            break
        if x1 < x0:
            problems.append(f"{name}: hover entry {entry_name!r} spans backwards "
                            f"{x0:.3f}..{x1:.3f}")
            break
        if length_ns <= 0 or length_ns > longest:
            problems.append(f"{name}: hover entry {entry_name!r} claims {length_ns:.1f} ns, "
                            f"which no command in this run is close to")
            break
    return problems


def zoom_invariance(view, trace, name, samples=12):
    """The same pulse must report the same length however the picture is scaled."""
    problems, checked = [], 0
    full_lo, full_hi = view.full_xlim
    span = max(full_hi - full_lo, 1.0)
    view.reset()
    # sample bars from the index of the FULL view, in the units of that view
    wide = list(view._hover_index)
    picked = wide[:: max(len(wide) // samples, 1)][:samples]
    for lane_y, x0, x1, entry_name, length_ns, kind in picked:
        if lane_y is None or kind != "toggle":
            continue
        view.reset()
        divisor = view.divisor
        mid_ns = (x0 + x1) / 2 * divisor                 # back to nanoseconds
        wide_text = tooltip_at(view, mid_ns / view.divisor, lane_y)
        # now zoom in around that pulse, which changes the plotted unit
        width = max((x1 - x0) * divisor * 6, span * 1e-4)
        view.set_window(mid_ns - width / 2, mid_ns + width / 2)
        near_text = tooltip_at(view, mid_ns / view.divisor, lane_y)
        checked += 1
        if wide_text is None or near_text is None:
            problems.append(f"{name}: {entry_name!r} tooltips "
                            f"{'zoomed out' if wide_text is None else 'zoomed in'} to nothing "
                            f"at the middle of its own bar")
        elif wide_text != near_text:
            problems.append(f"{name}: {entry_name!r} reads {wide_text!r} zoomed out and "
                            f"{near_text!r} zoomed in -- the same pulse, two answers")
        elif wide_text != f"{length_ns:.0f} ns" and wide_text != entry_name:
            problems.append(f"{name}: {entry_name!r} ({length_ns:.0f} ns) tooltips "
                            f"{wide_text!r}, which is neither its length nor its name")
    view.reset()
    return problems, checked


def emptiness(view, trace, name):
    """A lane that plays nothing at this time must not claim a BAR is there.

    Asked of the bar hit-test rather than of the tooltip, because a control-flow caption at the
    same point is a true answer to a different question -- the reader IS inside that construct.
    """
    problems = []
    lanes = len(trace.channels)
    occupied, full_height = {}, []
    for lane_y, x0, x1, *_rest in view._hover_index:
        if lane_y is None:
            # the inter-block gap spans every lane, so a point inside it is not empty on ANY
            # lane -- the whole sequencer is between blocks there, which is what it reports
            full_height.append((x0, x1))
        else:
            occupied.setdefault(lane_y, []).append((x0, x1))
    lo, hi = view.xlim_ns
    divisor = view.divisor
    for lane_y in range(lanes):
        spans = sorted(occupied.get(lane_y, []))
        # the first gap wide enough to aim at, in plotted units
        cursor = lo / divisor
        for x0, x1 in spans:
            if x0 - cursor > (hi - lo) / divisor * 0.01:
                where = (cursor + x0) / 2
                if any(band0 <= where <= band1 for band0, band1 in full_height):
                    cursor = max(cursor, x1)
                    continue
                hit = view._hit_pulse(motion(view.ax, where, lane_y))
                if hit is not None:
                    problems.append(f"{name}: lane {lane_y} plays nothing at {where:.3f} but the "
                                    f"hit-test claims {hit[3]!r} ({hit[1]:.3f}..{hit[2]:.3f})")
                break
            cursor = max(cursor, x1)
    return problems


def _closest(wave, value):
    """Which sample of ``wave`` a reported value is -- the comparison that survives rounding."""
    best, best_d = 0, None
    for index, sample in enumerate(wave):
        d = abs(complex(sample) - value)
        if best_d is None or d < best_d:
            best, best_d = index, d
    return best


def readout_at(view, x, y):
    """The bottom-left readout's text at this point, or None when it shows nothing."""
    view._compute_readout(motion(view.ax, x, y))
    return view._readout_text


def readout_truth(view, trace, name, samples=8):
    """The time and the envelope value the readout claims must both be true.

    Returns (problems, readings).
    """
    problems, readings = [], 0
    view.reset()
    entries = [e for e in view._pulse_iq if trace.envelope(e[3], e[4]) is not None]
    picked = entries[:: max(len(entries) // samples, 1)][:samples]
    for lane_y, x0, x1, io_name, pulse in picked:
        wave = trace.envelope(io_name, pulse)
        if wave is None or len(wave) < 2 or x1 <= x0:
            continue
        view.reset()
        divisor = view.divisor
        seen = []
        # Strictly INSIDE the bar, half a SAMPLE in from each end. Bars that touch share an
        # endpoint exactly and the hit-test is inclusive at both ends, so a probe at a shared edge
        # is answered by the earlier bar -- a deterministic tie, not a wrong answer. Half a sample
        # is the smallest step that avoids the tie while still landing on the end sample; a fixed
        # 0.1% would be ten samples into a long envelope and would not be an END probe at all.
        edge = 0.5 / (len(wave) - 1)
        for fraction in (edge, 0.25, 0.5, 0.75, 1.0 - edge):
            x = x0 + (x1 - x0) * fraction
            text = readout_at(view, x, lane_y)
            readings += 1
            value = view._iq_at(motion(view.ax, x, lane_y))
            if value is None:
                problems.append(f"{name}: {pulse!r} reports no envelope value {fraction:.0%} "
                                f"into its own bar")
                break
            seen.append(value)
            # ...and to within ONE sample, because the fraction is recomputed from a float x: at
            # an exact half the position lands between two samples and either is the middle.
            exact = fraction * (len(wave) - 1)
            near = {complex(wave[max(0, min(len(wave) - 1, int(exact) + step))])
                    for step in (0, 1, -1)}
            want = complex(wave[int(round(exact))])
            if not any(abs(value - candidate) <= 1e-9 for candidate in near):
                problems.append(f"{name}: {pulse!r} on {io_name} ({x0:.4f}..{x1:.4f}, "
                                f"{len(wave)} samples) at {fraction:.0%} reads {value:.4f} but "
                                f"the envelope holds {want:.4f} there")
                break
            # the time must be the time: parse it back out of what the reader sees
            t_ns = x * divisor
            shown = float(text.split("=")[1].split()[0]) * (1000.0 if "µs" in text else 1.0)
            if abs(shown - t_ns) > max(abs(t_ns) * 1e-4, 0.2):
                problems.append(f"{name}: cursor at {t_ns:.1f} ns but the readout says "
                                f"{shown:.1f} ns")
                break
        else:
            ends_ok = (abs(seen[0] - complex(wave[0])) <= 1e-9
                       or abs(seen[0] - complex(wave[1])) <= 1e-9)
            ends_ok = ends_ok and (abs(seen[-1] - complex(wave[-1])) <= 1e-9
                                   or abs(seen[-1] - complex(wave[-2])) <= 1e-9)
            if not ends_ok:
                problems.append(f"{name}: {pulse!r} does not anchor to its envelope's ends "
                                f"({seen[0]:.4f}/{seen[-1]:.4f} vs "
                                f"{complex(wave[0]):.4f}/{complex(wave[-1]):.4f})")
            # ...and the same instant must read the same when the picture is scaled differently
            middle_ns = (x0 + x1) / 2 * divisor
            width = max((x1 - x0) * divisor * 4, 4.0)
            view.set_window(middle_ns - width / 2, middle_ns + width / 2)
            near_value = view._iq_at(motion(view.ax, middle_ns / view.divisor, lane_y))
            readings += 1
            # ...to within one sample: the probe position itself is quantised by the plotted unit,
            # so an instant exactly between two samples can land either side when the unit changes.
            # A real unit error is not off by one sample, it is off by a thousand.
            wide_at = _closest(wave, seen[2])
            near_at = None if near_value is None else _closest(wave, near_value)
            if near_at is None or abs(near_at - wide_at) > 1:
                problems.append(f"{name}: {pulse!r} reads sample {wide_at} zoomed out and "
                                f"{near_at} zoomed in, at the same instant")
    view.reset()
    return problems, readings


def exercise(trace, name):
    """Returns (problems, hovers)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from acadia_qmsmt import sequence_viz as sv

    figure, axes = plt.subplots(figsize=(12, 6))
    problems, hovers = [], 0
    try:
        view = sv.SequenceView(trace, axes)
        problems += index_fidelity(view, trace, name)
        found, checked = zoom_invariance(view, trace, name)
        problems += found
        hovers += checked * 2
        problems += emptiness(view, trace, name)
        found, readings = readout_truth(view, trace, name)
        problems += found
        hovers += readings
    except Exception:
        import traceback
        problems.append(f"{name}: {traceback.format_exc(limit=3)}")
    finally:
        plt.close(figure)
    return problems, hovers


def main():
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    from gui_validation import broad_folders
    from acadia_qmsmt import sequence_viz as sv

    limit = next((int(a) for a in sys.argv[1:] if a.isdigit()), 12)
    problems, hovers, folders = [], 0, 0
    for folder in broad_folders(limit):
        name = folder.split("/")[-4]
        try:
            trace = sv.trace_folder(folder, envelopes=True)
        except Exception:
            continue
        folders += 1
        found, count = exercise(trace, name)
        problems += found
        hovers += count
        print(f"   {name:28s} {count:3d} hovers -- "
              f"{'ok' if not found else f'{len(found)} PROBLEM(S)'}", flush=True)

    problems += zero_amplitude_reads_as_zero()

    print(f"\n{hovers} tooltip readings over {folders} folders; {len(problems)} problems")
    if folders and not hovers:
        print("   NOTE: no pulse was hovered -- the zoom-invariance property checked nothing")
    for line in problems[:12]:
        print(f"   {line}")
    return 1 if problems else 0


def zero_amplitude_reads_as_zero():
    """A pulse LOADED with amplitude 0 must read as 0, not as "no data" or as its nominal shape.

    An idle gate written as ``{"scale": "0.0"}`` still plays for its full duration, so the
    readout owes the reader a number -- ``|A| = 0.000`` -- and must never substitute the config's
    nominal amplitude, which is one the board did not play. The trap is that a memory nobody
    loaded holds zeros too, and for THAT the config fallback is the useful answer; only the record
    of which pulses were loaded separates them. Deterministic, so it does not depend on an
    archived run happening to contain a zero-amplitude pulse.
    """
    import numpy as np
    from acadia_qmsmt.sequence_viz.tracing import SequenceTrace

    zeros = np.zeros(64, dtype=complex)
    nominal = np.full(64, 0.2 + 0j)
    key = ("rf1_stimulus", "CR_idle")
    problems = []

    loaded = SequenceTrace()
    loaded.loaded_envelopes = {key: zeros}
    loaded.envelopes = {key: nominal}
    loaded.loaded_pulses = {key}
    got = loaded.envelope(*key)
    if got is None:
        problems.append("a pulse loaded with amplitude 0 reads as no data (readout goes blank)")
    elif np.abs(got).max() != 0:
        problems.append(f"a pulse loaded with amplitude 0 reads as {np.abs(got).max():.3f} -- the "
                        f"config's nominal amplitude, which the board never played")

    never = SequenceTrace()
    never.loaded_envelopes = {key: zeros}
    never.envelopes = {key: nominal}
    got = never.envelope(*key)
    if got is None or np.abs(got).max() == 0:
        problems.append("a memory that was NEVER loaded no longer falls back to the config shape")

    print(f"   zero-amplitude readout      2 cases -- "
          f"{'ok' if not problems else f'{len(problems)} PROBLEM(S)'}")
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
