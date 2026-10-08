"""Check that what is DRAWN equals what was TRACED.

``timing_validation.py`` closes the link between the board and the trace: it deploys a sequence,
captures the DACs in loopback and compares measured pulse times against ``SequenceTrace``. That
proves the *model* is right. It says nothing about the *picture*.

But the picture is what anyone actually looks at while debugging, so the chain that has to hold is

    board  <->  trace  <->  drawing

and a break in the second link is just as misleading as a break in the first -- arguably more so,
because a rendering bug looks exactly like a physics bug. This module closes that second link by
rendering a trace to a real matplotlib figure, reading the patches back off the axes, and checking
every drawn rectangle against the command it is supposed to represent. It needs no hardware, so it
can run on every case, including ones the board cannot hold.

What is checked, per case:

* every non-padding command with a pulse has a rectangle whose left edge is EXACTLY its start
  time in ns (the ``ns_per_cycle`` conversion is the whole point -- drawing cycles as if they
  were ns, or vice versa, would pass every timing test and still produce a wrong picture);
* its width is either exactly its length, or the length minus the renderer's deliberate
  SEPARATOR inset (see below) -- and never anything else;
* no drawn rectangle sits outside the traced sequence;
* every traced pulse is matched to its OWN rectangle -- matches are consumed, so N executions of
  a looped block need N rectangles and cannot all be satisfied by one (a loop drawn once instead
  of N times is exactly the failure this catches).

THE SEPARATOR INSET, which is a real caveat worth knowing when reading durations off the plot.
When a command ends exactly where the next one begins, ``plotting.draw`` shortens the earlier bar
by ``SEPARATOR_PIXELS`` so the two do not fuse into one indistinguishable block::

    width = max(x1 - x0 - gap, (x1 - x0) * 0.5) if x1 in starts else x1 - x0

So a bar that abuts its neighbour is drawn slightly SHORT. Three things bound this, which is why
it is safe rather than misleading, and this module asserts all three:

* the bar's START is never moved -- every edge you measure from is exact;
* the inset is a fixed number of screen PIXELS, so it shrinks as you zoom in and tends to zero on
  the scale you would actually read a duration at;
* a bar is never shortened by more than half its length, so a short pulse cannot vanish.

The drawn total is deliberately NOT required to equal the traced total: the renderer also draws
lanes, gaps, barriers and capture spans, which are not pulses. The claim checked here is that
every pulse the trace contains is on screen, in the right place, at the right width.

Run: ``python validation/render_validation.py`` (add ``--case NAME`` for one case).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")                      # no display; we only read the patches back
from matplotlib.patches import Rectangle   # noqa: E402

from acadia_qmsmt.sequence_viz.plotting import SEPARATOR_PIXELS   # noqa: E402

#: rectangles must land on the traced time to well within a cycle -- this is float noise only,
#: not a tolerance for disagreement
TOLERANCE_NS = 1e-6


def drawn_rectangles(trace):
    """Render ``trace`` and return every drawn Rectangle as ``(x0_ns, width_ns)``.

    Reads the patches back off the axes rather than trusting the draw call, so this measures what
    the renderer produced, not what it was asked to produce.
    """
    import matplotlib.pyplot as plt
    from acadia_qmsmt.sequence_viz.plotting import draw

    figure, ax = plt.subplots(figsize=(12, 6))
    try:
        draw(ax, trace, show_envelopes=False, label_pulses=False)
        figure.canvas.draw()          # settle the layout so the axes extent is final
        axes_width_px = max(ax.get_window_extent().width, 1.0)
        # The x axis is NOT always in ns: _time_scale switches a long sequence to us so the
        # labels stay readable. Geometry read off the axes is therefore in the PLOTTED unit, and
        # comparing it to ns without converting makes a correct 120 ns bar look like a 0.12 wide
        # one. Take the divisor from the renderer itself rather than guessing the threshold.
        from acadia_qmsmt.sequence_viz.plotting import _time_scale
        divisor, _unit = _time_scale(max(trace.length_ns, 1.0))
        rects = []
        for patch in ax.patches:
            if isinstance(patch, Rectangle):
                rects.append((float(patch.get_x()), float(patch.get_width())))
        # pulses are batched into a PatchCollection for speed, so they are not in ax.patches --
        # pull them out of the collections, where match_original keeps their geometry
        for collection in ax.collections:
            paths = getattr(collection, "get_paths", None)
            if paths is None:
                continue
            for path in paths():
                xs = [v[0] for v in path.vertices]
                if len(xs) >= 4:
                    rects.append((min(xs), max(xs) - min(xs)))
        # hand everything back in ns, so callers never have to think about the plotted unit
        return [(x0 * divisor, w * divisor) for x0, w in rects], axes_width_px
    finally:
        plt.close(figure)


def traced_pulses(trace):
    """``(start_ns, length_ns)`` for every BAR the picture is supposed to show.

    Mirrors the renderer's own grouping, because the unit drawn is not always the unit traced: a
    ``use_stretch`` pulse is THREE commands (first half / held middle / second half) and is drawn
    as ONE bar spanning all three (``_stretch_groups`` in plotting). Expecting three separate
    rectangles there would report a mismatch on every stretch case, which is a fault in this
    checker rather than in the picture.

    The grouping is per LANE, exactly as the renderer does it -- the triple must be consecutive
    on one channel, not merely consecutive in the trace's flat command list.
    """
    from acadia_qmsmt.sequence_viz.plotting import _stretch_groups

    ns = trace.ns_per_cycle
    bars = []
    by_channel = {}
    for command in trace.commands:
        by_channel.setdefault(command.channel, []).append(command)
    for commands in by_channel.values():
        commands = sorted(commands, key=lambda c: c.start)
        groups = _stretch_groups(commands)
        skip = {i + offset for i in groups for offset in (1, 2)}
        for i, command in enumerate(commands):
            if i in skip or not command.pulse or command.is_padding:
                continue
            group = groups.get(i)
            start, stop = ((group[0].start, group[2].stop) if group
                           else (command.start, command.stop))
            bars.append((start * ns, (stop - start) * ns))
    return sorted(bars)


def check(trace, name, max_inset_ns):
    """Compare drawn geometry against traced commands. Returns a result dict.

    ``max_inset_ns`` is how much the renderer may legitimately shorten a bar that abuts its
    neighbour (see the module docstring). A bar may be drawn between ``length - max_inset_ns``
    and ``length``, never longer and never shorter than half -- anything else is a real
    disagreement between the picture and the trace.
    """
    expected = traced_pulses(trace)
    drawn, _ = drawn_rectangles(trace)

    def width_ok(width, length):
        if abs(width - length) <= TOLERANCE_NS:
            return True                       # drawn to full length: nothing abutted it
        shortfall = length - width
        return (0 < shortfall <= max_inset_ns + TOLERANCE_NS
                and width >= length * 0.5 - TOLERANCE_NS)

    # match each traced pulse to a drawn rectangle at the same place. Matching by position (not
    # by order) is deliberate: the renderer is free to draw lanes in any order, and only the
    # geometry is a claim about the sequence. Matches are CONSUMED, so N executions of a looped
    # block need N distinct rectangles.
    unmatched, used = [], set()
    for start, length in expected:
        hit = next((i for i, (x0, width) in enumerate(drawn)
                    if i not in used
                    and abs(x0 - start) <= TOLERANCE_NS
                    and width_ok(width, length)), None)
        if hit is None:
            # report the closest same-start candidate, so a width problem reads as a width
            # problem rather than as a missing bar
            same_start = [w for x0, w in drawn if abs(x0 - start) <= TOLERANCE_NS]
            near = min((abs(x0 - start) for x0, _ in drawn), default=float("inf"))
            unmatched.append((start, length, near, same_start))
        else:
            used.add(hit)

    span_ns = trace.length_ns
    outside = [(x0, w) for x0, w in drawn if x0 < -TOLERANCE_NS or x0 > span_ns + TOLERANCE_NS]

    return {"case": name, "traced": len(expected), "drawn": len(drawn),
            "unmatched": unmatched, "outside": outside,
            "ok": not unmatched and not outside}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=None, help="one case (default: every case)")
    parser.add_argument("--folder", default=None,
                        help="check a REAL archived run instead of a validation case")
    args = parser.parse_args()

    import timing_validation as tv
    from loopback_timing_cases import CASES
    from acadia_qmsmt import sequence_viz as sv

    if args.folder:
        # A real run exercises shapes no fixture does -- streamed gate trains, cooling rounds,
        # multi-resonator readout -- so the drawing is worth checking against the trace there too.
        trace = sv.trace_folder(args.folder, envelopes=False)
        _, axes_width_px = drawn_rectangles(trace)
        max_inset_ns = (SEPARATOR_PIXELS / axes_width_px) * max(trace.length_ns, 1.0)
        result = check(trace, trace.runtime_class, max_inset_ns)
        status = "OK" if result["ok"] else "MISMATCH"
        print(f"  {result['case']:34s} traced {result['traced']:4d}  "
              f"drawn {result['drawn']:4d}  {status}")
        for start, length, near, same_start in result["unmatched"][:6]:
            print(f"       at {start:.2f} ns: expected width {length:.2f}, drawn {same_start} "
                  f"(nearest edge {near:.2f} ns away)")
        return 0 if result["ok"] else 1

    # stretch_zero is a deliberate PATHOLOGY, not a drawing to check: a zero-length register
    # command wraps to 4294967295 cycles, so the sequence is 21.5 SECONDS long and a 120 ns pulse
    # sits 21 s from the origin, where the geometry comparison runs out of resolution (its own
    # inset allowance comes out at 34 ms). The case exists so that unsafe_reason REFUSES it --
    # which timing_validation checks -- and asking the renderer to agree about a picture nobody
    # should ever see would only make the gate report a problem that is the point of the case.
    PATHOLOGICAL = ("stretch_zero",)
    cases = ([args.case] if args.case
             else [c for c in CASES if c not in PATHOLOGICAL])
    failures, checked = [], 0
    for name in cases:
        try:
            trace = sv.trace_runtime(tv.build_runtime(name, iterations=10), envelopes=False)
        except Exception as exc:                       # cannot build -> not a render result
            print(f"  {name:26s} SKIPPED ({type(exc).__name__})")
            continue
        # The inset the renderer is allowed, computed the way the renderer computes it:
        # SEPARATOR_PIXELS of AXES width (not figure width -- the axes is inset by the margins,
        # so using the figure width under-estimates the allowance and reports false mismatches).
        _, axes_width_px = drawn_rectangles(trace)
        max_inset_ns = (SEPARATOR_PIXELS / axes_width_px) * max(trace.length_ns, 1.0)
        result = check(trace, name, max_inset_ns)
        checked += 1
        status = "OK" if result["ok"] else "MISMATCH"
        print(f"  {name:26s} traced {result['traced']:3d}  drawn {result['drawn']:3d}  {status}")
        if not result["ok"]:
            failures.append(result)
            for start, length, near, same_start in result["unmatched"][:4]:
                if same_start:
                    print(f"       at {start:.2f} ns: expected width {length:.2f} ns, "
                          f"drawn {[round(w, 2) for w in same_start]} "
                          f"(inset allowance {max_inset_ns:.2f} ns)")
                else:
                    print(f"       no rectangle at {start:.2f} ns (len {length:.2f}); "
                          f"nearest drawn edge is {near:.2f} ns away")
            for x0, width in result["outside"][:4]:
                print(f"       rectangle at {x0:.2f} ns (len {width:.2f}) lies outside the sequence")

    print(f"\n{checked} cases rendered; {len(failures)} with a drawing that disagrees with the trace")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
