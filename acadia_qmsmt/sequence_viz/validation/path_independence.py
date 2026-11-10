"""What is on screen must depend on WHERE you are, never on how you got there.

Every other check in this suite examines a single state: load a folder, set something, assert the
picture is right. None of them ask whether two ways of reaching the SAME state agree. That gap
matters here more than it would in most panels, because the sweep point and the control-flow pins
are not independent knobs:

* a sweep point change does not re-trace -- ``select_point`` swaps in that point's pulse data and
  register values over a shared schedule;
* those register values are what DECIDE a test's arm and a ``repeat_until``'s count, so a point
  change can add or remove whole constructs (Readout_Fidelity grows one at point 1 that point 0
  does not have);
* a pin is keyed by ``(block, depth[, path])``, and that key can name a construct at one point and
  nothing at all at another.

So "pin at point 0, then move to point 5" and "move to point 5, then pin" run different code and
could easily disagree -- and if they do, one of them is showing a hypothesis dressed as a
measurement, with nothing on screen to say which. This asserts they cannot:

1. **Route independence.** A target state ``(point, pins)`` reached by three routes -- pins last,
   pins first, and via a decoy state with a different point and different pins -- must fingerprint
   identically.
2. **Purity.** That fingerprint must also equal a trace built FRESH at that point with those pins,
   which is the definition of the panel being a pure function of its inputs rather than a machine
   with memory.
3. **Construction agreement.** ``trace_folder(point=N)`` and ``trace_folder(point=0).select_point(N)``
   must agree -- the panel only ever does the second, so if they differ the panel has been showing
   something no fresh trace of that point would produce.

The fingerprint covers what a reader would actually use to draw a conclusion: total length,
placement count, every construct's resolved count/source/indeterminacy, and the tab text.

Run: ``python validation/path_independence.py [folders]`` (offline, needs Qt only for the panel
routes; the purity and construction checks are pure ``sequence_viz``).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def fingerprint(trace):
    """What a reader would use to draw a conclusion, reduced to a comparable value."""
    from acadia_qmsmt.sequence_viz import plotting

    constructs = tuple(sorted(
        (str(entry["key"]), entry["kind"], str(entry.get("count")), entry["source"],
         bool(entry.get("indeterminate")), bool(entry.get("nonterminating")))
        for entry in trace.control_flow_summary()))
    tabs = tuple(sorted(plotting.tab_label(info)
                        for _s, _e, _c, info in plotting.branch_regions(trace)))
    return (round(trace.length_ns, 6), len(trace.placements or ()), constructs, tabs)


def differences(one, other):
    """Human-readable account of why two fingerprints differ."""
    names = ("length_ns", "placements", "constructs", "tabs")
    out = []
    for name, a, b in zip(names, one, other):
        if a == b:
            continue
        if isinstance(a, tuple):
            only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
            out.append(f"{name}: {len(only_a)} only-here {only_a[:2]}, "
                       f"{len(only_b)} only-there {only_b[:2]}")
        else:
            out.append(f"{name}: {a} vs {b}")
    return "; ".join(out) or "identical fields but unequal fingerprints"


def apply_pins(trace, pins):
    """Set exactly ``pins`` on a trace and nothing else."""
    trace.loop_counts.clear()
    trace.path_choices.clear()
    for key, (kind, value) in pins.items():
        if kind == "test":
            trace.path_choices[key] = value
        else:
            trace.loop_counts[key] = value
    trace.relayout()


def target_states(trace, points):
    """A handful of (point, pins) states worth reaching more than one way.

    Chosen to cross the two knobs that interact: constructs whose value comes from a REGISTER (so
    the sweep point can change them under the pin) are preferred over ones the program fixes.
    """
    entries = trace.control_flow_summary()
    volatile = [e for e in entries if e["source"] in ("assumed", "register")] or entries
    states = []
    for point in points:
        pins = {}
        for entry in volatile[:2]:
            pins[entry["key"]] = ("test", 1) if entry["kind"] == "test" else ("loop", 2)
        states.append((point, pins))
    return states


def check_folder(folder, name):
    """Returns (problems, states_checked, routes_checked)."""
    from acadia_qmsmt import sequence_viz as sv

    problems, states, routes = [], 0, 0
    try:
        base = sv.trace_folder(folder, envelopes=False)
    except Exception as exc:                                      # noqa: BLE001
        return [f"{name}: trace raised {type(exc).__name__}: {exc}"], 0, 0
    n_points = max(getattr(base, "n_points", 1) or 1, 1)
    points = sorted({0, min(1, n_points - 1), n_points // 2, n_points - 1})

    # (3) construction agreement: building AT a point vs selecting it must not differ
    for point in points:
        if point == 0:
            continue
        built = sv.trace_folder(folder, point=point, envelopes=False)
        walked = sv.trace_folder(folder, envelopes=False)
        walked.select_point(point)
        walked.relayout()
        if fingerprint(built) != fingerprint(walked):
            problems.append(f"{name}: point {point} built != selected -- "
                            f"{differences(fingerprint(built), fingerprint(walked))}")

    for point, pins in target_states(base, points):
        states += 1
        # (2) purity: the reference is a fresh trace at this point with these pins
        reference = sv.trace_folder(folder, point=point, envelopes=False)
        apply_pins(reference, pins)
        want = fingerprint(reference)

        # route A -- move to the point, then pin
        a = sv.trace_folder(folder, envelopes=False)
        a.select_point(point)
        apply_pins(a, pins)
        routes += 1

        # route B -- pin first, then move to the point
        b = sv.trace_folder(folder, envelopes=False)
        apply_pins(b, pins)
        b.select_point(point)
        b.relayout()
        routes += 1

        # route C -- via a decoy: a different point and different pins, then correct to the target
        c = sv.trace_folder(folder, envelopes=False)
        decoy_point = points[-1] if point != points[-1] else points[0]
        c.select_point(decoy_point)
        apply_pins(c, {k: (kind, 3 if kind != "test" else 0) for k, (kind, _v) in pins.items()})
        c.select_point(point)
        apply_pins(c, pins)
        routes += 1

        for label, trace in (("point-then-pin", a), ("pin-then-point", b), ("via-decoy", c)):
            got = fingerprint(trace)
            if got != want:
                problems.append(f"{name}: point {point} via {label} != a fresh trace of the "
                                f"same state -- {differences(got, want)}")
    return problems, states, routes


def check_panel(folder, name):
    """The same property through the real widget, whose knobs are the ones a user turns."""
    from PyQt5.QtWidgets import QApplication
    from gui_validation import panel_class
    from acadia_qmsmt import sequence_viz as sv

    app = QApplication.instance() or QApplication([])
    problems, routes = [], 0
    widget = panel_class()()
    widget.resize(1400, 850)
    widget.show()
    try:
        widget.adopt_trace(sv.trace_folder(folder, envelopes=False))
        entries = widget.trace.control_flow_summary()
        volatile = [e for e in entries if e["source"] in ("assumed", "register")] or entries
        if not volatile:
            return problems, routes
        entry = volatile[0]
        n_points = max(getattr(widget.trace, "n_points", 1) or 1, 1)
        point = n_points - 1
        pin = ("test", 1) if entry["kind"] == "test" else ("loop", 2)

        def drive(point_first):
            if point_first:
                widget.point.setValue(point)
                app.processEvents()
            if pin[0] == "test":
                widget._set_path_choice(entry["key"], pin[1])
            else:
                widget._set_loop_count(entry["key"], pin[1])
            app.processEvents()
            if not point_first:
                widget.point.setValue(point)
                app.processEvents()
            return fingerprint(widget.trace)

        widget.point.setValue(0)
        app.processEvents()
        first = drive(True)
        routes += 1
        widget.adopt_trace(sv.trace_folder(folder, envelopes=False))
        second = drive(False)
        routes += 1
        if first != second:
            problems.append(f"{name}: PANEL point {point} + pin {entry['key']} depends on the "
                            f"order you turned the knobs -- {differences(first, second)}")
        if widget.faults:
            problems.append(f"{name}: panel recorded {len(widget.faults)} fault(s): "
                            f"{widget.faults[0][0]} {widget.faults[0][1]}")
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
    return problems, routes


def main():
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from gui_validation import broad_folders

    limit = next((int(a) for a in sys.argv[1:] if a.isdigit()), 12)
    problems, states, routes, folders = [], 0, 0, 0
    for folder in broad_folders(limit):
        name = folder.split("/")[-4]
        found, seen, walked = check_folder(folder, name)
        panel_found, panel_routes = check_panel(folder, name)
        problems += found + panel_found
        states += seen
        routes += walked + panel_routes
        folders += 1
        note = "ok" if not (found or panel_found) else f"{len(found) + len(panel_found)} PROBLEM(S)"
        print(f"   {name:28s} {seen} states, {walked + panel_routes} routes -- {note}", flush=True)

    print(f"\n{routes} routes to {states} (sweep point, pin) states over {folders} folders; "
          f"{len(problems)} problems")
    for line in problems[:12]:
        print(f"   {line}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
