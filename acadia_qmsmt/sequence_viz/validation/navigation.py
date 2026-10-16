"""The controls that MOVE you must land where they say.

Three of them, none ever checked: the block dropdown, the time scrollbar, the lane scrollbar. They
share a hazard, which is why they are checked together -- each one converts between two coordinate
systems, and a conversion that is wrong is not an error, it is a picture of somewhere else:

* the dropdown holds a span in CYCLES and the viewport takes NANOSECONDS (and the entries come from
  placements, not blocks, because a looped block runs many times and each pass is its own
  destination);
* the scrollbar is integer nanoseconds with a page step, while the window is float ns -- and the
  bar's own maximum is derived from the window width, so the two define each other.

The properties:

* **Arrival.** Choosing "block N (pass k)" must leave that pass's whole span inside the window. A
  destination you cannot see is not a destination.
* **Round trip.** A window written to the scrollbar and read back must be the same window. The bar
  reports where you are; if the value it shows would not take you back there, it is lying about
  the position in a way that only shows up when someone drags it.
* **Monotonicity.** Dragging the bar one way must move the window that way, at a fixed zoom.
* **Reachability.** Every part of the sequence must be reachable at every zoom: the bar's maximum
  plus its page must cover the end, or the tail cannot be scrolled to at all.

Run: ``python validation/navigation.py [folders]`` (needs Qt, offscreen).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def arrival(widget, app, name):
    """Every entry in the jump list must actually take you to what it names."""
    problems, jumps = [], 0
    trace = widget.trace
    ns = trace.ns_per_cycle
    for index in range(widget.jump.count()):
        span = widget.jump.itemData(index)
        label = widget.jump.itemText(index)
        widget.jump.setCurrentIndex(index)
        app.processEvents()
        jumps += 1
        lo, hi = widget.view.xlim_ns
        if span is None:                                  # "whole seq"
            full_lo, full_hi = widget.view.full_xlim
            if lo > full_lo + 1e-6 or hi < full_hi - 1e-6:
                problems.append(f"{name}: {label!r} left {lo:.0f}..{hi:.0f} of "
                                f"{full_lo:.0f}..{full_hi:.0f}")
            continue
        start, stop = span[0] * ns, span[1] * ns
        if start < lo - 1e-6 or stop > hi + 1e-6:
            problems.append(f"{name}: {label!r} spans {start:.0f}..{stop:.0f} ns but the window "
                            f"landed on {lo:.0f}..{hi:.0f}")
    return problems, jumps


def round_trip(widget, app, name):
    """What the scrollbar reports must take you back to where you are."""
    problems, trips = [], 0
    view = widget.view
    full_lo, full_hi = view.full_xlim
    length = max(full_hi - full_lo, 1.0)
    for fraction in (0.0, 0.2, 0.5, 0.8):
        for width in (length / 3, length / 20, length / 200):
            lo = full_lo + (length - width) * fraction
            view.set_window(lo, lo + width)
            app.processEvents()
            was = view.xlim_ns
            trips += 1
            bar = widget.time_scroll
            if not bar.isEnabled():
                continue
            # the bar now claims a position; feeding that claim back must not move the view
            widget._scroll_to(bar.value())
            app.processEvents()
            now = view.xlim_ns
            if abs(now[0] - was[0]) > max(1.5, abs(was[0]) * 1e-6):
                problems.append(f"{name}: the bar said {bar.value()} for a window at "
                                f"{was[0]:.1f} ns, and going back there landed at {now[0]:.1f}")
            # ...and the end of the sequence must be reachable at this zoom
            if bar.maximum() + bar.pageStep() < int(full_hi) - 1:
                problems.append(f"{name}: at width {width:.0f} ns the bar reaches "
                                f"{bar.maximum() + bar.pageStep()} ns of {full_hi:.0f} -- the "
                                f"tail cannot be scrolled to")
    return problems, trips


def monotonic(widget, app, name):
    """Dragging one way must move the window that way."""
    problems, drags = [], 0
    view = widget.view
    full_lo, full_hi = view.full_xlim
    width = max((full_hi - full_lo) / 12, 4.0)
    view.set_window(full_lo, full_lo + width)
    app.processEvents()
    bar = widget.time_scroll
    if not bar.isEnabled():
        return problems, drags
    previous = view.xlim_ns[0]
    steps = [bar.minimum() + (bar.maximum() - bar.minimum()) * n // 6 for n in range(7)]
    for value in steps:
        widget._scroll_to(value)
        app.processEvents()
        drags += 1
        now = view.xlim_ns[0]
        if now < previous - 1.5:
            problems.append(f"{name}: dragging to {value} moved the window backwards "
                            f"({previous:.0f} -> {now:.0f} ns)")
            break
        previous = now
    return problems, drags


def lanes(widget, app, name):
    """The lane bar answers to the same rules as the time bar."""
    problems, moves = [], 0
    view = widget.view
    bar = widget.lane_scroll
    full_lo, full_hi = view.full_ylim
    height = max((full_hi - full_lo) / 3, 1.0)
    for fraction in (0.0, 0.5, 1.0):
        low = full_lo + (full_hi - full_lo - height) * fraction
        view.set_lanes(low, low + height)
        app.processEvents()
        moves += 1
        seen = view.ylim or view.full_ylim
        if seen[0] < full_lo - 1e-6 or seen[1] > full_hi + 1e-6:
            problems.append(f"{name}: lane range {seen[0]:.2f}..{seen[1]:.2f} outside the stack "
                            f"{full_lo:.2f}..{full_hi:.2f}")
        if bar.isEnabled() and bar.maximum() < bar.minimum():
            problems.append(f"{name}: the lane bar's range inverted "
                            f"({bar.minimum()}..{bar.maximum()})")
    return problems, moves


def check_folder(folder, name):
    from PyQt5.QtWidgets import QApplication
    from gui_validation import panel_class
    from acadia_qmsmt import sequence_viz as sv

    app = QApplication.instance() or QApplication([])
    widget = panel_class()()
    widget.resize(1400, 850)
    widget.show()
    problems, moves = [], 0
    try:
        widget.adopt_trace(sv.trace_folder(folder, envelopes=False))
        for check in (arrival, round_trip, monotonic, lanes):
            found, count = check(widget, app, name)
            problems += found
            moves += count
        if widget.faults:
            problems.append(f"{name}: panel recorded {len(widget.faults)} fault(s): "
                            f"{widget.faults[0][0]} {widget.faults[0][1]}")
    except Exception:
        import traceback
        problems.append(f"{name}: {traceback.format_exc(limit=3)}")
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
    return problems, moves


def main():
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    from gui_validation import broad_folders

    limit = next((int(a) for a in sys.argv[1:] if a.isdigit()), 10)
    problems, moves, folders = [], 0, 0
    for folder in broad_folders(limit):
        name = folder.split("/")[-4]
        found, count = check_folder(folder, name)
        problems += found
        moves += count
        folders += 1
        print(f"   {name:28s} {count:4d} moves -- "
              f"{'ok' if not found else f'{len(found)} PROBLEM(S)'}", flush=True)

    print(f"\n{moves} navigation moves over {folders} folders; {len(problems)} problems")
    for line in problems[:12]:
        print(f"   {line}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
