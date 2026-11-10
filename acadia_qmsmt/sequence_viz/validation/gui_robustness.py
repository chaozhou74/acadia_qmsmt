"""Can the sequence panel be broken? Hostile inputs, degenerate traces, and random event storms.

The three existing gates check that the GUI is CORRECT on real data. This one checks that it cannot
be made to die, which is a different property and needs different pressure:

* **Every callback is guarded** -- a static invariant, read out of the source. PyQt5 turns an
  exception that escapes a slot into ``qFatal()``: the whole data browser aborts, not just this
  panel. So any bound method handed to Qt or matplotlib must carry the guard. Checking this by
  AST rather than by list means a callback added next month is covered too.

* **Degenerate and hostile inputs** -- counts of 0 and 100000, a pinned arm on a test that never
  ran, a window of zero width, a widget resized to 1x1 pixels, a folder that is not data at all, a
  trace replaced underneath a built panel. Each of these is a thing a user can actually do.

* **Random event storms** -- thousands of clicks, drags, scrolls, hovers, key presses and resizes
  at random positions, from a seed. Fixed scripts only ever find the bugs someone thought of; the
  interleavings are where a GUI actually breaks.

A fired guard counts as a FAILURE here, exactly as in gui_validation.py: the guard exists so the
user never sees a crash, not so the tests can pass with broken code. The storm runs in a SUBPROCESS
so that a hard abort (qFatal, segfault) is detectable as a signal rather than taking the harness
down with it -- and the seed is printed, so any crash is reproducible.

**How long it takes: about a minute per seed** (380 events on a three-level nest measured at 48 s).

It was believed to be a ~1100 s soak, and that was wrong in an instructive way: the time was the
panel sitting in a MODAL DIALOG. A random press that lands on a construct tab opens the edit box --
correctly -- and QInputDialog waits for an answer nobody was giving, so every storm stopped dead at
its first tab click while still reporting "OK" for the events it never ran. The dialogs are now
answered (see `storm`), and the true cost is a minute.

Stuck is therefore judged by LACK OF PROGRESS, not by a deadline: the child reports the event it has
reached and the parent kills it only when that number stops moving. A deadline cannot tell a slow
machine from a hung panel, and picking one either cries wolf under load or waits far too long.

Run:
    python validation/gui_robustness.py              # invariant + degenerate + 6 storm seeds
    python validation/gui_robustness.py --seeds 40   # longer soak
    python validation/gui_robustness.py --seeds 8 --from 20   # interleavings 20..27, not 0..7
    python validation/gui_robustness.py --storm 7    # replay one seed in-process, verbosely
"""
import ast
import os
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

GUI_SOURCE = Path("/home/boson/acadia_gui/acadia_gui/gui/sequence_view.py")
EVENTS = 400          # per storm seed
DATA = "/home/boson/data/test_loopback"


# --------------------------------------------------------------------------- static invariant

def unguarded_callbacks(source=GUI_SOURCE):
    """Bound methods handed to Qt/matplotlib that are NOT wrapped in the guard.

    A bound method referenced as a VALUE (``x.connect(self._foo)``, ``on_viewport=self._bar``,
    ``claim_press = self._baz``) is a callback: something else will call it, from inside the event
    loop, where an exception is fatal. A method CALLED (``self._foo()``) is not -- its caller is
    Python code that can handle the failure. The distinction is exactly ``ast.Attribute`` in a load
    context that is not the function of a ``Call``, which is what this reads.
    """
    tree = ast.parse(source.read_text())
    guarded, called_as_value = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names = [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
            if "_guarded" in names:
                guarded.add(node.name)
        if isinstance(node, ast.Call):
            # mark the callee so `self._foo()` is not mistaken for passing `self._foo`
            if isinstance(node.func, ast.Attribute):
                node.func._is_callee = True
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "self" and not getattr(node, "_is_callee", False)
                and isinstance(node.ctx, ast.Load)):
            called_as_value.add(node.attr)
    # only methods count; attributes like self.trace are data
    methods = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    return sorted((called_as_value & methods) - guarded)


# --------------------------------------------------------------------------- helpers

def _app():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _panel(folder=None, app=None):
    """A built panel, reusing gui_validation's discovery so nothing is duplicated here."""
    from gui_validation import panel_class
    from acadia_qmsmt import sequence_viz as sv

    widget = panel_class()()
    widget.resize(1200, 700)
    if folder:
        # adopt_trace does everything a real load does, the sweep-point RANGE included. Assigning
        # widget.trace by hand left that range at 0..0, so a test could set the point all it liked
        # and nothing moved -- a whole axis that looked covered and was not.
        trace = sv.trace_folder(folder, envelopes=False)
        trace.relayout()
        widget.adopt_trace(trace)
    if app:
        app.processEvents()
    return widget


def folders(limit=4):
    """A few archived runs that actually contain control flow -- the interesting ones to break.

    Deliberately cheap: the loopback archive is named after its cases, so the nesting cases can be
    picked by name without tracing anything. Walking every run first (as the box-count property
    does) costs minutes over NFS and buys nothing here -- this harness needs a handful of panels to
    attack, not coverage of the archive.
    """
    import glob

    picked = []
    for pattern in ("*three_deep*", "*nest*", "*counter_loop_in_test*", "*test_in_counter*",
                    "*repeat_until*", "*loop*"):
        for folder in sorted(glob.glob(f"{DATA}/{pattern}/*")):
            if Path(folder, "compiled.log").exists() and folder not in picked:
                picked.append(folder)
                break                       # one run per case is enough
        if len(picked) >= limit:
            break
    return picked[:limit]


def _faults(widget):
    return [f"{w}: {m}" for w, m, _d in getattr(widget, "faults", ())]


# --------------------------------------------------------------------------- degenerate inputs

def degenerate(app):
    """Things a user can do that the panel must survive. Returns a list of problems."""
    problems = []
    from acadia_qmsmt import sequence_viz as sv

    # a folder that is not data at all, and one that does not exist
    for bad in ("/home/boson", "/nonexistent/nowhere", "", None):
        widget = _panel(app=app)
        widget.load_folder(bad)
        app.processEvents()
        if widget.trace is not None:
            problems.append(f"load_folder({bad!r}) left a trace behind")
        problems += [f"load_folder({bad!r}): {f}" for f in _faults(widget)]

    folder = folders(1)[0]
    # Counts at both extremes, on every construct, through the real control -- and each edit is
    # TIMED. An unbounded spin box (it used to accept 100000) expands the plan to that many copies
    # of the body, and the redraw then never returns: the GUI looks hung, which is exactly the
    # failure this file exists to catch. The control's range is now derived from what the body costs
    # to draw, so asking for its maximum must stay inside the budget.
    import time
    widget = _panel(folder, app)
    for key in list(widget._flow_widgets):
        control = widget._flow_widgets.get(key)
        if control is None:
            continue
        if hasattr(control, "setValue"):
            values = [0, 1, control.maximum(), control.maximum() + 1_000, 7]
        else:
            values = [0, 1, 2, 0]
        for value in values:
            # Re-fetch before every touch. The panel adds and removes rows when the number of
            # EXECUTIONS changes -- pinning an outer loop gives the constructs inside it more -- so a
            # widget captured before that edit is a deleted C++ object, and touching it raises
            # "wrapped C/C++ object has been deleted". The harness must follow the panel, not a
            # snapshot of it.
            control = widget._flow_widgets.get(key)
            if control is None:
                break
            started = time.perf_counter()
            if hasattr(control, "setValue"):
                control.setValue(value)
            else:
                control.setCurrentIndex(value % 3)
            app.processEvents()
            spent = time.perf_counter() - started
            # The measured worst case across the nesting cases is 1.3 s at the panel's own maximum
            # (it was 475 s before the execution-tag lookup and body-cost fixes). 8 s leaves a wide
            # margin for a loaded machine while still failing on a real regression -- the earlier
            # 30 s threshold would have sailed past a 20x slowdown.
            if spent > 8.0:
                problems.append(f"setting {key} to {value} took {spent:.1f} s -- an edit should be "
                                f"about a second; the panel offers a value that stalls the GUI")
    problems += [f"extreme counts: {f}" for f in _faults(widget)]

    # a construct set to zero, or a test to its skipped arm, must keep a clickable handle
    widget5 = _panel(folder, app)
    from acadia_qmsmt.sequence_viz import plotting
    for entry in list(widget5.trace.control_flow_summary()):
        control = widget5._flow_widgets.get(entry["key"])
        if control is None:
            continue
        if hasattr(control, "setValue"):
            control.setValue(0)
        else:
            control.setCurrentIndex(2)               # skipped
        app.processEvents()
        reachable = {(i["block"], i["depth"])
                     for _s, _e, _c, i in plotting.branch_regions(widget5.trace)}
        if (entry["block"], entry["depth"]) not in reachable:
            problems.append(f"{entry['kind']} {entry['key']} lost its tab when set to "
                            f"zero/skipped -- unreachable from the diagram")
        control = widget5._flow_widgets.get(entry["key"])   # rows may have been rebuilt
        if control is None:
            continue
        if hasattr(control, "setValue"):
            control.setValue(1)
        else:
            control.setCurrentIndex(0)
        app.processEvents()
    problems += [f"zero/skip reachability: {f}" for f in _faults(widget5)]

    # a degenerate viewport: zero width, inverted, beyond the sequence, and a 1x1 widget
    widget2 = _panel(folder, app)
    length = widget2.trace.length_ns
    for lo, hi in ((0.0, 0.0), (500.0, 500.0), (length, 0.0), (-1e9, 1e9),
                   (length * 2, length * 3), (0.0, 1e-12)):
        widget2.view.set_window(lo, hi)
        app.processEvents()
    for size in ((1, 1), (4, 3), (2000, 40), (60, 1400)):
        widget2.resize(*size)
        widget2._redraw()
        app.processEvents()
    problems += [f"degenerate viewport: {f}" for f in _faults(widget2)]

    # the trace swapped underneath a built panel, and removed entirely
    widget3 = _panel(folder, app)
    widget3.trace = sv.trace_folder(folders(2)[-1], envelopes=False)
    widget3._refresh_control_flow()
    widget3._redraw()
    app.processEvents()
    widget3.trace = None
    for method in ("_redraw", "_refresh_control_flow", "_refresh_registers",
                   "_populate_blocks", "_reset", "reload"):
        getattr(widget3, method)()
        app.processEvents()
    problems += [f"trace swapped/removed: {f}" for f in _faults(widget3)]

    # overrides pinned to keys that do not exist must not break the layout
    widget4 = _panel(folder, app)
    widget4.trace.loop_counts.update({(9999, 9): 4, 9999: 3, (0, 1, (7,)): 2})
    widget4.trace.path_choices.update({(9999, 1): True, 4242: False})
    widget4.trace.relayout()
    widget4._redraw()
    app.processEvents()
    problems += [f"stale override keys: {f}" for f in _faults(widget4)]
    return problems


# --------------------------------------------------------------------------- random storm

def storm(folder, seed, events=EVENTS, verbose=False, progress=None):
    """Fire `events` random interactions at a built panel. Returns a list of problems."""
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
    from PyQt5.QtWidgets import QApplication, QComboBox, QSpinBox

    from PyQt5.QtCore import QTimer

    app = _app()
    rng = random.Random(seed)
    widget = _panel(folder, app)
    canvas = widget.canvas
    problems = []

    # ANSWER MODAL DIALOGS. A random press that lands on a construct tab opens the edit dialog --
    # correctly, that is what a tab is for -- and QInputDialog runs its own event loop waiting for
    # an answer that never comes offscreen. Every storm therefore stopped dead at its first tab
    # click and reported nothing, so the events after it were never run at all: a fuzzer that hangs
    # itself looks exactly like a fuzzer that found nothing.
    #
    # Dismissed rather than stubbed out, so the click path stays under test: the dialog really
    # opens, really closes, and the panel has to survive that.
    dismissed = [0]

    def answer_modal():
        dialog = app.activeModalWidget()
        if dialog is not None:
            dismissed[0] += 1
            dialog.close()

    closer = QTimer()
    closer.timeout.connect(answer_modal)
    closer.start(120)

    def point():
        return QPoint(rng.randrange(1, max(canvas.width(), 2)),
                      rng.randrange(1, max(canvas.height(), 2)))

    buttons = [Qt.LeftButton, Qt.RightButton, Qt.MiddleButton]
    mods = [Qt.NoModifier, Qt.ShiftModifier, Qt.ControlModifier, Qt.AltModifier]
    keys = [Qt.Key_R, Qt.Key_Escape, Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down,
            Qt.Key_Plus, Qt.Key_Minus, Qt.Key_Home, Qt.Key_A]

    for step in range(events):
        if progress and step % 5 == 0:
            # A heartbeat, so the parent can tell SLOW from STUCK. Elapsed time alone cannot: this
            # storm takes ~1100 s alone and over 1800 s when the machine is busy, so any fixed
            # deadline either cries wolf under load or waits far too long for a real hang.
            try:
                Path(progress).write_text(str(step))
            except OSError:
                pass
        what = rng.choice(["press", "release", "move", "drag", "wheel", "key", "resize",
                           "control", "window", "reset", "theme", "point", "redraw"])
        try:
            if what in ("press", "release", "move"):
                kind = {"press": QMouseEvent.MouseButtonPress,
                        "release": QMouseEvent.MouseButtonRelease,
                        "move": QMouseEvent.MouseMove}[what]
                event = QMouseEvent(kind, point(), rng.choice(buttons), Qt.NoButton,
                                    rng.choice(mods))
                QApplication.sendEvent(canvas, event)
            elif what == "drag":
                start, end = point(), point()
                button = rng.choice(buttons)
                QApplication.sendEvent(canvas, QMouseEvent(
                    QMouseEvent.MouseButtonPress, start, button, button, rng.choice(mods)))
                for fraction in (0.25, 0.5, 0.75, 1.0):
                    mid = QPoint(int(start.x() + (end.x() - start.x()) * fraction),
                                 int(start.y() + (end.y() - start.y()) * fraction))
                    QApplication.sendEvent(canvas, QMouseEvent(
                        QMouseEvent.MouseMove, mid, Qt.NoButton, button, Qt.NoModifier))
                QApplication.sendEvent(canvas, QMouseEvent(
                    QMouseEvent.MouseButtonRelease, end, button, Qt.NoButton, Qt.NoModifier))
            elif what == "wheel":
                delta = rng.choice([-720, -240, -120, 120, 240, 720])
                QApplication.sendEvent(canvas, QWheelEvent(
                    point(), canvas.mapToGlobal(point()), QPoint(0, delta), QPoint(0, delta),
                    Qt.NoButton, rng.choice(mods), Qt.NoScrollPhase, False))
            elif what == "key":
                QApplication.sendEvent(canvas, QKeyEvent(
                    QKeyEvent.KeyPress, rng.choice(keys), rng.choice(mods)))
            elif what == "resize":
                widget.resize(rng.randrange(1, 1600), rng.randrange(1, 900))
            elif what == "control":
                controls = list(widget._flow_widgets.values()) + [
                    widget.point, widget.time_scroll, widget.lane_scroll]
                control = rng.choice(controls) if controls else None
                if isinstance(control, QSpinBox):
                    control.setValue(rng.randrange(0, 12))
                elif isinstance(control, QComboBox):
                    control.setCurrentIndex(rng.randrange(0, max(control.count(), 1)))
                elif control is not None:
                    control.setValue(rng.randrange(control.minimum(), control.maximum() + 1)
                                     if control.maximum() > control.minimum() else 0)
            elif what == "window" and widget.view is not None:
                length = max(widget.trace.length_ns, 1.0)
                lo = rng.uniform(-length, length * 1.5)
                widget.view.set_window(lo, lo + rng.uniform(0.0, length * 1.2))
            elif what == "reset":
                widget._reset()
            elif what == "theme":
                widget.set_theme(rng.choice(["dark", "light", "", None]))
            elif what == "point":
                widget.point.setValue(rng.randrange(0, max(widget.trace.n_points, 1)))
            elif what == "redraw":
                widget._redraw()
            app.processEvents()
        except Exception as exc:                      # noqa: BLE001
            problems.append(f"seed {seed} step {step} ({what}) RAISED OUT of the panel: "
                            f"{type(exc).__name__}: {exc}")
            if verbose:
                import traceback
                traceback.print_exc()
            break
    closer.stop()
    problems += [f"seed {seed}: GUARD FIRED in {f}" for f in _faults(widget)]
    if verbose:
        print(f"   seed {seed}: {events} events, {dismissed[0]} dialogs answered, "
              f"{len(problems)} problems")
    return problems


# --------------------------------------------------------------------------- driver

def _storm_child(folder, seed, events, progress=None):
    """Run one storm in this process and print problems; used as a subprocess entry point."""
    problems = storm(folder, seed, events, progress=progress)
    for line in problems:
        print(f"PROBLEM {line}")
    return 1 if problems else 0


def main():
    argv = sys.argv[1:]
    if "--child" in argv:
        index = argv.index("--child")
        folder, seed, events = argv[index + 1], int(argv[index + 2]), int(argv[index + 3])
        progress = argv[index + 4] if len(argv) > index + 4 else None
        return _child_exit(folder, seed, events, progress)
    if "--storm" in argv:
        seed = int(argv[argv.index("--storm") + 1])
        return 1 if storm(folders(1)[0], seed, EVENTS, verbose=True) else 0

    seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else 6
    # Each seed IS a distinct interleaving, so a longer soak that always starts at 0 re-runs every
    # interleaving already known to pass to reach a few new ones. --from moves the window instead.
    start = int(argv[argv.index("--from") + 1]) if "--from" in argv else 0
    problems = []

    print("every callback guarded (static, from the source)")
    unguarded = unguarded_callbacks()
    if unguarded:
        problems.append(f"{len(unguarded)} callback(s) reach the event loop unguarded: "
                        f"{', '.join(unguarded)}")
        print(f"   UNGUARDED: {', '.join(unguarded)}")
    else:
        print("   all bound methods passed as callbacks carry the guard")

    print("\ndegenerate and hostile inputs")
    app = _app()
    bad = degenerate(app)
    problems += bad
    print(f"   {len(bad)} problem(s)")
    for line in bad[:8]:
        print(f"      {line}")

    print(f"\nrandom event storms (seeds {start}..{start + seeds - 1} x {EVENTS} events, each in "
          f"its own process so an abort is caught)")
    targets = folders(3)
    for seed in range(start, start + seeds):
        folder = targets[seed % len(targets)]
        # A seed that hangs must be REPORTED, not raise. subprocess.run re-raises TimeoutExpired,
        # which aborted the whole suite and threw away the other seeds' results -- so one slow seed
        # hid five good ones, and the run looked like a crash rather than a finding.
        stdout, returncode, hung = run_storm_subprocess(folder, seed, EVENTS)
        if hung:
            problems.append(f"seed {seed}: STUCK -- {hung}; reproduce with --storm {seed}")
            print(f"   seed {seed:<3d} {Path(folder).parent.name:34s} STUCK ({hung})")
            continue
        found = [line[8:] for line in stdout.splitlines() if line.startswith("PROBLEM ")]
        if returncode is not None and returncode < 0:
            found.append(f"seed {seed}: the process DIED on signal {-returncode} "
                         f"-- a hard crash, reproduce with --storm {seed}")
        problems += found
        print(f"   seed {seed:<3d} {Path(folder).parent.name:34s} "
              f"{'OK' if not found else f'{len(found)} PROBLEM(S)'}")
        for line in found[:4]:
            print(f"      {line}")
        continue

    print(f"\n{len(problems)} problem(s) total")
    return 1 if problems else 0


def _child_exit(folder, seed, events, progress=None):
    return _storm_child(folder, seed, events, progress)


def run_storm_subprocess(folder, seed, events, stall_seconds=300, ceiling_seconds=7200):
    """Run one storm in its own process, judging STUCK by lack of progress rather than by elapsed.

    Returns ``(stdout, returncode, hung)``. A storm is a soak -- ~1100 s for 400 events on a
    three-level nest -- and how long it takes depends on what else the machine is doing, so a fixed
    deadline reports a healthy seed as hung the moment the box is busy. That is the worst kind of
    test result: it looks like a finding, and it teaches you to ignore the suite. The child writes
    the event it has reached every few events; if that number stops moving for `stall_seconds` the
    panel really is stuck, whatever the clock says.
    """
    import tempfile
    import time

    with tempfile.TemporaryDirectory() as workspace:
        beat = str(Path(workspace) / "progress")
        Path(beat).write_text("0")
        child = subprocess.Popen(
            [sys.executable, __file__, "--child", folder, str(seed), str(events), beat],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
        started = last_change = time.monotonic()
        seen = "0"
        while child.poll() is None:
            time.sleep(2.0)
            try:
                now = Path(beat).read_text()
            except OSError:
                now = seen
            if now != seen:
                seen, last_change = now, time.monotonic()
            stalled = time.monotonic() - last_change
            if stalled > stall_seconds or time.monotonic() - started > ceiling_seconds:
                child.kill()
                child.communicate()
                return "", child.returncode, (f"no progress for {stalled:.0f} s at event {seen} "
                                              f"of {events}")
        out, _err = child.communicate()
        return out, child.returncode, None


if __name__ == "__main__":
    raise SystemExit(main())
