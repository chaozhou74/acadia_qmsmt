"""Drive the SeeQuence GUI headlessly and assert it survives real interaction.

Written because a control-flow panel shipped that CRASHED on its first edit. The bug was not in
the model or the drawing -- both were fine -- it was that the panel rebuilt itself from inside a
spin box's own ``valueChanged`` handler, which deletes the widget currently emitting the signal.
Nothing that inspects the trace, the layout or the figure can see that. Only pushing the widget
can.

So this exercises the GUI the way a person does: build the panel from a real trace, change values
through the WIDGETS (not the backing dicts), let the event loop run, and check both that the
process is still alive and that the timeline actually changed. Every control-flow row is driven,
not a sample -- a crash in the one row nobody tried is still a crash.

Run: ``QT_QPA_PLATFORM=offscreen python validation/gui_validation.py [folder ...]``
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

#: Folders covering the shapes that stress the panel: deep nesting with assumed branches
#: (tomography), a streamed gate train, and cooling loops.
#:
#: These three are the START, not the whole gate. Coverage by SEQUENCE SHAPE is what this suite was
#: missing: three deeply-nested folders never changed how many EXECUTIONS a construct had, so the
#: panel adding rows for new executions was never exercised and tabs appeared on the diagram with no
#: row beside them -- in 7 of 24 experiment types. Run `broad_folders()` (below) to sweep one run per
#: experiment type; it is slower, and it is where that class of bug lives.
DEFAULT_FOLDERS = (
    "/home/boson/data/tunmay/20260806/DualRail_gate_tomo/Q1C2_e1f0_CZ/Q1C1_Q2C2/260807/180839",
    "/home/boson/data/ziqian/20260806/DualRail_XEB_2DR/reference/260807/200632",
    "/home/boson/data/tunmay/20260718/DualRail_RB/Q1C1/260724/154854",
)


def assumed_alternatives(widget, app):
    """Every alternative of every ASSUMED construct must draw a coherent, reversible picture.

    An assumed value is one the BOARD decides at runtime, so correctness cannot mean "matches the
    hardware" -- a static trace cannot know what it chose. What can be guaranteed is that every
    possibility you are able to ask for is drawn coherently, still marked as a possibility, and
    fully reversible. The last part is the one that matters in practice: exploring alternatives is
    only safe if you can always get back to what the run actually did, or you end up reading a
    hybrid of your hypothesis and the measurement without knowing it.
    """
    problems, checked = [], 0
    trace = widget.trace
    assumed = [e for e in trace.control_flow_summary() if e["source"] == "assumed"]
    if not assumed:
        return problems, checked
    baseline = (round(trace.length_ns), len(trace.placements))
    for entry in assumed:
        options = [True, False] if entry["kind"] == "test" else [0, 1, 2]
        for option in options:
            if entry["kind"] == "test":
                trace.path_choices[entry["key"]] = option
            else:
                trace.loop_counts[entry["key"]] = option
            trace.relayout()
            app.processEvents()
            checked += 1
            from acadia_qmsmt.sequence_viz import plotting as _plotting
            reachable = {(info["block"], info["depth"])
                         for _s, _e, _c, info in _plotting.branch_regions(trace)}
            for other in trace.control_flow_summary():
                if (other["block"], other["depth"]) not in reachable:
                    problems.append(f"{entry['key']}={option} left {other['key']} with no handle")
            now = next((e for e in trace.control_flow_summary()
                        if e["key"] == entry["key"]), None)
            if now and now["source"] == "pinned" and not now.get("indeterminate"):
                problems.append(f"pinning {entry['key']}={option} dropped the "
                                f"'decided at runtime' marker")
        trace.path_choices.pop(entry["key"], None)
        trace.loop_counts.pop(entry["key"], None)
    trace.relayout()
    app.processEvents()
    if (round(trace.length_ns), len(trace.placements)) != baseline:
        problems.append(f"clearing every assumed pin did not restore the run "
                        f"({trace.length_ns:.0f} ns vs {baseline[0]} ns)")
    return problems, checked


def tab_text_is_legible(minimum=4.5):   # WCAG AA for small text; the labels are 6 pt
    # Was 4.0, which is a bar set to the result rather than to the requirement: depth 1
    # measured 4.19:1 and passed. Raised once readable_on was made to guarantee the target.
    """Every tab label must read against whatever it sits on, in EVERY theme.

    Checked against the palette rather than against a rendered figure, because that is what the
    drawing uses and it cannot go stale. The tab inks lighten with nesting depth on purpose; a label
    fixed at white was 2.5:1 at depth 3 and 1.7:1 at depth 4, and a hollow tab (a construct drawn
    zero times) inked onto the dark theme's page was 1.7:1 -- invisible exactly where someone would
    look for the handle of something that is not drawn.
    """
    from acadia_qmsmt.sequence_viz import plotting

    problems = []
    for depth, ink in enumerate(plotting.TAB_DEPTH_INK, 1):
        # the PAIR, as drawn: a mid-tone fill can be out of reach of both ink candidates at once,
        # so the fill moves the last few percent rather than the label being left at "least bad"
        fill, label = plotting.legible_tab(ink)
        ratio = plotting.contrast_ratio(label, fill)
        if ratio < minimum - 1e-6:
            problems.append(f"solid tab at depth {depth}: label {label} on {fill} is "
                            f"{ratio:.2f}:1, below {minimum}:1")
    for theme_name in ("LIGHT_THEME", "DARK_THEME"):
        page = getattr(plotting, theme_name)["axes_bg"]
        for depth, ink in enumerate(plotting.TAB_DEPTH_INK, 1):
            chosen = plotting.legible_ink(ink, page)
            ratio = plotting.contrast_ratio(chosen, page)
            if ratio < minimum:
                problems.append(f"hollow tab at depth {depth} on {theme_name}: {chosen} on "
                                f"{page} is {ratio:.2f}:1, below {minimum}:1")
    return problems


def sweep_points(widget, app, folder, app_points=6):
    """Step the SWEEP POINT and check the panel keeps up. Returns (problems, combinations).

    A swept delay re-times the sequence per point -- DualRail_RB runs 78 us at point 0 and 319 us
    at point 279 -- so this is the axis where "the view still covers the whole sequence" and "every
    tab still has a row" are worth asking. It found the viewport staying at the old length: the
    reader saw the first quarter of the sequence with nothing to say the rest existed.

    Requires the widget to have adopted its trace (adopt_trace), which is what sets the point range.
    Assigning ``widget.trace`` by hand leaves it 0..0, and then every point set here is clamped to 0
    and this function silently checks point 0 six times.
    """
    from acadia_qmsmt.sequence_viz import plotting

    problems, checked = [], 0
    points = getattr(widget.trace, "n_points", 1) or 1
    if points < 2:
        return problems, checked
    if widget.point.maximum() < points - 1:
        problems.append(f"sweep range is {widget.point.minimum()}..{widget.point.maximum()} for "
                        f"{points} points -- point changes will be clamped and prove nothing")
        return problems, checked
    for point in sorted({0, 1, points // 4, points // 2, 3 * points // 4, points - 1}):
        if point >= points:
            continue
        widget.point.setValue(point)
        app.processEvents()
        checked += 1
        rows = {label.text().strip() for label, _indent in widget._flow_labels.values()}
        regions = plotting.branch_regions(widget.trace)
        ambiguous = plotting.ambiguous_blocks(regions)
        for _s, _e, _c, info in regions:
            text = plotting.tab_label(info, info.get("block") in ambiguous)
            if not any(row.endswith(text) for row in rows):
                problems.append(f"point {point}: tab {text!r} has no control-flow row")
        lo, hi = widget.view.xlim_ns
        full_lo, full_hi = widget.view.full_xlim
        if hi < full_hi - 1e-6 or lo > full_lo + 1e-6:
            problems.append(f"point {point}: the view shows {lo:.0f}..{hi:.0f} of a sequence that "
                            f"runs {full_lo:.0f}..{full_hi:.0f} -- part of it is off screen with "
                            f"nothing to say so")
    return problems, checked


def broad_folders(limit=24):
    """One archived run per EXPERIMENT TYPE, newest first -- breadth by sequence shape.

    The default three folders are all deep-nesting cases. Real runtimes look nothing like them: a
    Rabi or a resonator spec has no control flow at all (the panel must hide the group rather than
    draw an empty one), an RB run is a streamed gate train with 30+ constructs, a chevron sweeps a
    register. Each exercises branches the nesting cases cannot reach.
    """
    import stress_campaign as sc

    seen, picked = set(), []
    for folder in sc.folders():
        parts = Path(folder).parts
        experiment = parts[-4] if len(parts) >= 4 else parts[-1]
        if experiment in seen or not Path(folder, "compiled.log").exists():
            continue
        seen.add(experiment)
        picked.append(str(folder))
        if len(picked) >= limit:
            break
    return picked


def panel_class():
    import acadia_gui.gui.sequence_view as module
    return next(obj for obj in vars(module).values()
                if isinstance(obj, type) and hasattr(obj, "_build_control_flow"))


def drive(folder, app):
    """Build the panel for one folder and push every control. Returns a list of problems."""
    from PyQt5.QtWidgets import QComboBox, QSpinBox
    from acadia_qmsmt import sequence_viz as sv

    problems = []
    widget = panel_class()()
    # adopt_trace, not a bare assignment: it is the same bookkeeping a real load does, including
    # the sweep-point RANGE. Setting widget.trace by hand left that range at 0..0, so every point
    # change in a test was clamped to 0 and the sweep-point axis was never actually exercised.
    trace = sv.trace_folder(folder, envelopes=False)
    trace.relayout()
    widget.adopt_trace(trace)
    widget._build_control_flow()
    app.processEvents()

    rows = dict(widget._flow_widgets)
    if not rows:
        return problems, (0, 0, 0, 0, 0, 0, 0)
    baseline = len(widget.trace.placements)

    for key in list(rows):
        # Re-read the widget each time instead of holding the one captured above. The panel adds
        # and removes rows when the number of EXECUTIONS changes (pinning an outer loop gives every
        # construct inside it more), and a reference captured before that is a deleted C++ object:
        # touching it raises "wrapped C/C++ object has been deleted". A test that holds stale
        # widgets is testing a panel that no longer exists.
        control = widget._flow_widgets.get(key)
        if control is None:
            continue
        try:
            if isinstance(control, QSpinBox):
                # a value the sequence would not otherwise have, so a change is detectable
                control.setValue(max(2, control.value() + 1))
                app.processEvents()
                if len(widget.trace.placements) == baseline:
                    problems.append(f"construct {key}: loop count changed but the timeline "
                                    f"did not")
            elif isinstance(control, QComboBox):
                for index in (1, 2, 0):          # taken, skipped, back to auto
                    # re-fetch between touches: pinning an arm can change how many executions
                    # exist, which rebuilds the rows and deletes the widget mid-loop
                    control = widget._flow_widgets.get(key)
                    if control is None or not isinstance(control, QComboBox):
                        break
                    control.setCurrentIndex(index)
                    app.processEvents()
        except Exception:
            problems.append(f"construct {key}: {traceback.format_exc(limit=3)}")
        # the panel is rebuilt on a timer after each edit; make sure that lands too
        app.processEvents()

    # ---- the TAB is the click target, and it must not trigger a zoom ----
    # This is the check that matters for the reported bug: clicking a construct to edit it used
    # to ALSO drag out a box-zoom rectangle, because the whole plot area belongs to that gesture.
    # A press inside a tab must be CLAIMED (the viewport must not act on it), and a press outside
    # every tab must NOT be claimed, or zooming would stop working entirely.
    from types import SimpleNamespace
    from acadia_qmsmt.sequence_viz.plotting import (branch_regions, control_flow_tab,
                                                    TAB_PIXELS_W, _time_scale)

    if getattr(widget, "view", None) is None:
        widget._redraw()
        app.processEvents()
    axes = widget.view.ax if getattr(widget, "view", None) is not None else None
    if axes is None:
        problems.append("no axes after redraw: tab interaction was not exercised at all")

    frames = [(f, i) for f, i in (getattr(axes, "_seqviz_flow_frames", None) or [])
              if i.get("tab_rect")]
    if not frames:
        problems.append("no control-flow tabs published on the axes")

    claimed = unclaimed = 0
    original_edit = widget._edit_construct
    # accept the same signature the real one has (block, path) -- a stub that silently has the
    # wrong arity turns a passing test into a TypeError the moment the caller gains an argument
    widget._edit_construct = lambda block, path=(), depth=1: None
    try:
        for frame, info in frames:
            x0, y0, width, height = info["tab_rect"]
            centre = SimpleNamespace(inaxes=axes, xdata=x0 + width / 2,
                                     ydata=y0 + height / 2, button=1, key=None,
                                     x=0, y=0, dblclick=False)
            if not widget._claim_flow_tab(centre):
                problems.append(f"tab for block {info['block']} did not claim its own press "
                                f"-- clicking it would drag a zoom rectangle")
            else:
                claimed += 1
            # a point well below every tab must be left to the zoom gesture
            # well inside the LANES, below the whole tab strip: the zoom gesture must keep this
            below = SimpleNamespace(inaxes=axes, xdata=x0 + width / 2, ydata=0.5,
                                    button=1, key=None, x=0, y=0, dblclick=False)
            if widget._claim_flow_tab(below):
                problems.append("a press outside every tab was claimed -- zooming would break")
            else:
                unclaimed += 1
    finally:
        widget._edit_construct = original_edit

    # ---- hover highlighting on the tabs ----
    # Synthesised motion events at the middle of every control-flow span. The claim being
    # checked is not just "does not crash" but that the INNERMOST construct is the one found:
    # control flow nests, so a point inside a cooling round is also inside the mode loop around
    # it, and offering the outer one would edit the wrong thing.
    from types import SimpleNamespace
    from acadia_qmsmt.sequence_viz.plotting import branch_regions, _time_scale

    hovers = 0
    for frame, info in frames:
        x0, y0, width, height = info["tab_rect"]
        event = SimpleNamespace(inaxes=axes, xdata=x0 + width / 2, ydata=y0 + height / 2,
                                button=1, key=None, x=0, y=0, dblclick=False)
        try:
            hit = widget._flow_tab_at(event)
            widget._on_flow_hover(event)
        except Exception:
            problems.append(f"hover on tab {info['block']}: {traceback.format_exc(limit=3)}")
            continue
        if hit is None:
            problems.append(f"tab for block {info['block']} is not hit-testable at its centre")
        else:
            hovers += 1

    # every control back to auto/unset must restore the original timeline exactly
    widget.trace.loop_counts.clear()
    widget.trace.path_choices.clear()
    widget.trace.relayout()
    if len(widget.trace.placements) != baseline:
        problems.append(f"clearing the overrides did not restore the timeline "
                        f"({len(widget.trace.placements)} vs {baseline})")
    # Distinct constructs must have DISTINCT override keys. Nested constructs often begin at the
    # same block, and keying by block alone made three tabs edit the same thing -- the drawing
    # said three constructs, the controls behaved as one.
    keys = [widget.trace.construct_key(info["block"], info["depth"],
                                       tuple(info.get("path", ())) or None)
            for _f, info in frames]
    if len(set(keys)) != len(keys):
        problems.append(f"{len(keys) - len(set(keys))} tab(s) share an override key with another "
                        f"-- editing one would move the other")

    zooms = 0
    # ---- ACROSS ZOOM LEVELS ----
    # The reported glitch only appears at some zooms: a tab is a fixed number of PIXELS wide, so
    # how much SEQUENCE it covers changes as you zoom, and constructs that are comfortably apart
    # in one view crowd in another. Checking a single view is why this shipped twice. Every window
    # below is checked for tab/label overlap.
    # A SWEEP of zoom levels, not four hand-picked ones. The glitch is zoom dependent and the
    # widths change continuously with the window, so any fixed sample can sit either side of a
    # collision. Every width from the whole sequence down to 2% of it, at several offsets.
    length = widget.trace.length_ns
    windows = [(0.0, length)]
    for fraction in (0.6, 0.4, 0.24, 0.12, 0.06, 0.02):
        for offset in (0.0, 0.3, 0.55):
            lo = length * offset
            windows.append((lo, min(lo + length * fraction, length)))
    for lo, hi in windows:
        widget.view.set_window(lo, hi)
        app.processEvents()
        zoom_axes = widget.view.ax
        zoomed = [i["tab_rect"] for _f, i in
                  (getattr(zoom_axes, "_seqviz_flow_frames", None) or [])
                  if i.get("tab_rect")]
        bad = [(a, b) for i, a in enumerate(zoomed) for b in zoomed[i + 1:]
               if abs(a[1] - b[1]) < 0.01 and a[0] < b[0] + b[2] and b[0] < a[0] + a[2]]
        if bad:
            problems.append(f"at zoom [{lo/1000:.1f}, {hi/1000:.1f}] us: {len(bad)} tab(s) "
                            f"overlap, e.g. x={bad[0][0][0]:.2f}(w{bad[0][0][2]:.2f}) and "
                            f"x={bad[0][1][0]:.2f}")
        # the scrollbar must be able to reach the tabs: a control above the pannable range is a
        # control you cannot click
        top = widget.view.full_ylim[1]
        if zoomed:
            highest = max(r[1] + r[3] for r in zoomed)
            if top < highest:
                problems.append(f"at zoom [{lo/1000:.1f}, {hi/1000:.1f}] us the scroll range "
                                f"tops out at {top:.2f} but the highest tab reaches {highest:.2f}"
                                f" -- unreachable")
        zooms += 1

    # No two tabs may overlap. Being hit-testable at its own centre is not enough -- two tabs
    # can both pass that while sitting on top of each other, which is what made a pair of `test`
    # labels render as `tdstst` and left a click landing on whichever was drawn last.
    placed = [info["tab_rect"] for _f, info in frames if info.get("tab_rect")]
    clashes = [(a, b) for i, a in enumerate(placed) for b in placed[i + 1:]
               if abs(a[1] - b[1]) < 0.01 and abs(a[0] - b[0]) < min(a[2], b[2])]
    if clashes:
        problems.append(f"{len(clashes)} pair(s) of tabs overlap on the same row, e.g. "
                        f"x={clashes[0][0][0]:.2f} and x={clashes[0][1][0]:.2f}")

    # Every construct must have its OWN tab, at every nesting level -- that is what makes the
    # levels independently editable rather than only the innermost one reachable.
    depths = {info["depth"] for _f, info in frames}
    blocks = {info["block"] for _f, info in frames}
    if len(frames) != len(blocks) and len(depths) < 2:
        problems.append(f"tabs do not cover the nesting: {len(frames)} tabs, "
                        f"{len(blocks)} distinct constructs, depths {sorted(depths)}")

    # ---- every control, and the viewport ----
    control_problems, controls_touched = exercise_every_control(widget, app)
    problems.extend(control_problems)
    viewport_problems, viewport_steps = exercise_viewport(widget, app)
    problems.extend(viewport_problems)
    if not controls_touched:
        problems.append("no controls were discovered -- the introspection found nothing to drive")
    if not viewport_steps:
        problems.append("the viewport was not exercised")

    if not hovers:
        problems.append("tab hover exercised 0 tabs -- the check proved nothing")
    if zooms < 15:
        problems.append(f"only {zooms} zoom level(s) checked -- the overlap glitch is zoom "
                        f"dependent, so a thin sample proves little")
    if not claimed or not unclaimed:
        problems.append(f"press claiming proved nothing (claimed={claimed}, "
                        f"left-alone={unclaimed})")
    # THE TAB AND THE PANEL ROW MUST SAY THE SAME THING. They are the same construct, and the user
    # is asked to tie them together by eye, so the tab's text has to be the tail of its row's text
    # ("repeat_until @11 ?" in the panel, "@11 ?" on the diagram). They came from two different
    # facts once -- the row read the summary's count, the tab read repeat_counts -- and disagreed on
    # every construct whose count was only assumed: the row claimed "x1", the tab said nothing.
    #
    # A row with no tab is legitimate in exactly one case: a construct that is not in the drawn
    # timeline at all, which is what pinning a test to "skipped" does. A tab with no row never is.
    from acadia_qmsmt.sequence_viz import plotting as _plotting
    summary = widget.trace.control_flow_summary()
    # the same ambiguity rule the panel applies: a block that keys several constructs needs its
    # depth in the label, and the expected text must be computed the same way or the check is wrong
    shared = {e["block"] for e in summary
              if sum(1 for other in summary if other["block"] == e["block"]) > 1}
    # Read the text that is ACTUALLY ON SCREEN, not a fresh call to the labelling function. Row
    # labels are written when the panel is built; if an edit does not re-write them the row keeps
    # saying "x1" while its own spin box says 3 and the tab says "x3". Recomputing the label here
    # would compare the function against itself and pass with a stale panel on display.
    row_texts = {label.text().strip() for label, _indent in
                 getattr(widget, "_flow_labels", {}).values()}
    if not row_texts:
        row_texts = {_plotting.flow_label(e) for e in summary}
    regions = _plotting.branch_regions(widget.trace)
    ambiguous = _plotting.ambiguous_blocks(regions)
    for _s, _e, _c, info in regions:
        text = _plotting.tab_label(info, info.get("block") in ambiguous)
        if not any(row.endswith(text) for row in row_texts):
            problems.append(f"tab {text!r} matches no control-flow row -- the diagram and the "
                            f"panel disagree about the same construct")
    drawn = {(info["block"], info["depth"]) for _s, _e, _c, info in regions}
    for entry in summary:
        if (entry["block"], entry["depth"]) in drawn:
            continue
        if entry["kind"] == "test" and entry["source"] == "pinned" and not entry["taken"]:
            continue                      # pinned skipped: correctly absent from the drawing
        problems.append(f"row {_plotting.flow_label(entry)!r} has no span in the drawing")
    # ...and every row must DISPLAY its construct's current value. Compared against the label
    # computed the same way the panel computes it -- with the same ambiguity flag. Leaving that out
    # made the check itself wrong: it expected "test @19" where the panel correctly showed
    # "test @19.1", because block 19 keys two constructs and the depth is what separates them.
    for entry in summary:
        pair = getattr(widget, "_flow_labels", {}).get(entry["key"])
        if pair is None:
            continue
        expected = _plotting.flow_label(entry, entry["block"] in shared)
        if pair[0].text().strip() != expected:
            problems.append(f"row label {pair[0].text().strip()!r} is stale -- the construct "
                            f"now reads {expected!r}")
        for number, run in enumerate(entry.get("executions") or (), 1):
            child = getattr(widget, "_flow_labels", {}).get(run["key"])
            if child is None:
                continue
            pinned = run.get("pinned")
            shown_entry = dict(
                entry, count=(pinned if pinned is not None and entry["kind"] != "test"
                              else entry["count"]),
                taken=(pinned if pinned is not None and entry["kind"] == "test"
                       else entry["taken"]),
                source="pinned" if pinned is not None else entry["source"])
            want = _plotting.flow_label(
                dict(shown_entry, execution=_plotting.execution_tag(run["path"])),
                entry["block"] in shared)
            if child[0].text().strip() != want:
                problems.append(f"execution row {child[0].text().strip()!r} is stale -- "
                                f"it now reads {want!r}")

    # A REGISTER OVERRIDE must actually re-time the command it drives. Nothing else in this suite
    # touches that path: every archived run resolves its registers from its own captured cache, so
    # the Registers panel's settable spin boxes never appear on real data and the override code is
    # reachable only by setting one explicitly -- which is what the panel does when you type in it.
    for entry in widget.trace.register_summary():
        # Only what the PANEL exposes. A cache-resolved register has no spin box, and forcing an
        # override on one tests a control the user does not have: XEB's REG0/REG1 name commands
        # whose length is decoded per GATE from the cache word, so a register-value override cannot
        # reach them and never could. `settable` is exactly the panel's own rule.
        if not entry.get("settable") or not entry.get("is_length"):
            continue
        name = entry["name"]
        before = [c.length for c in widget.trace.commands if c.symbolic == name]
        if not before:
            continue
        widget._set_register(name, max(int(before[0]) * 2, 2))
        app.processEvents()
        after = [c.length for c in widget.trace.commands if c.symbolic == name]
        if after == before:
            problems.append(f"register {name}: an override left its command length at {before} "
                            f"-- the control does nothing")
        widget.trace.register_overrides.pop(name, None)
        widget.trace.relayout()
        widget._redraw()
        app.processEvents()
        restored = [c.length for c in widget.trace.commands if c.symbolic == name]
        if restored != before:
            problems.append(f"register {name}: clearing the override left {restored}, "
                            f"not the run's own {before}")

    # HIDING THE TABS must leave no invisible click targets. The tabs are a click surface layered
    # over the plot area, and the viewport yields presses to them -- so if they stop being drawn but
    # their hit rectangles survive, a press in the strip is swallowed and the zoom gesture dies
    # somewhere the user can see nothing to explain it.
    if hasattr(widget, "show_flow_tabs"):
        widget.show_flow_tabs.setChecked(False)
        app.processEvents()
        axes_now = widget.view.ax
        live = [i for _f, i in (getattr(axes_now, "_seqviz_flow_frames", None) or [])
                if i.get("tab_rect")]
        if live:
            problems.append(f"{len(live)} tab hit-rectangles survive with the tabs hidden -- "
                            f"presses there would be swallowed by handles nobody can see")
        labels = [text for text in axes_now.texts if text.get_text().startswith("@")]
        if labels:
            problems.append(f"{len(labels)} tab label(s) still drawn with the tabs hidden")
        probe = SimpleNamespace(inaxes=axes_now, xdata=sum(axes_now.get_xlim()) / 2,
                                ydata=axes_now.get_ylim()[1] * 0.98, button=1, key=None,
                                x=0, y=0, dblclick=False)
        if widget._claim_flow_tab(probe):
            problems.append("a press was claimed with the tabs hidden -- zooming would break there")
        widget.show_flow_tabs.setChecked(True)
        app.processEvents()
        if not [i for _f, i in (getattr(widget.view.ax, "_seqviz_flow_frames", None) or [])
                if i.get("tab_rect")]:
            problems.append("the tabs did not come back when re-enabled")

    # ASSUMED constructs: every alternative must be coherent, marked, and reversible.
    assumed_problems, assumed_checks = assumed_alternatives(widget, app)
    problems += assumed_problems

    # THE SWEEP POINT is its own axis: a swept delay changes the sequence LENGTH per point, and
    # both the viewport and the panel have to keep up with that.
    point_problems, point_checks = sweep_points(widget, app, folder)
    problems += point_problems
    # reported in the coverage line: a folder with one sweep point checks nothing here, and that
    # has to be visible rather than hidden behind a pass

    # A GUARD THAT FIRED IS A BUG. The panel wraps its callbacks so a failure cannot abort the
    # process (PyQt5 turns an escaped exception into qFatal), but that safety net must never become
    # a way of passing tests with broken code -- so every recorded fault fails the suite here, with
    # the traceback that caused it.
    for where, message, detail in getattr(widget, "faults", ()):
        problems.append(f"GUARD FIRED in {where}: {message}\n"
                        + "\n".join(detail.strip().splitlines()[-4:]))
    return problems, (len(rows), hovers, claimed, controls_touched, viewport_steps, point_checks,
                      assumed_checks)


def exercise_every_control(widget, app):
    """Toggle, cycle and press every control the widget has, and check nothing breaks.

    Discovered by INTROSPECTION rather than from a hand-written list, so a control added later is
    exercised automatically and cannot quietly go untested. The bar is deliberately low per
    control -- no exception, and the figure still renders -- because the failure mode being hunted
    is the one that took down the control-flow panel: a widget interaction that is fine in the
    model and fatal in Qt.

    Modal dialogs are stubbed out. A test that opens one blocks forever, and a blocked test is
    indistinguishable from a hung GUI.
    """
    from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                                 QInputDialog, QPushButton, QSpinBox, QToolButton)

    problems, touched = [], 0
    saved = (QInputDialog.getInt, QInputDialog.getItem, QFileDialog.getExistingDirectory)
    QInputDialog.getInt = staticmethod(lambda *a, **k: (0, False))
    QInputDialog.getItem = staticmethod(lambda *a, **k: ("", False))
    QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: "")

    def alive(thing):
        """Is this widget still a live C++ object?

        findChildren() hands back a snapshot, and the panel legitimately rebuilds rows while this
        loop runs (the number of executions changes when a loop count is pinned). Touching an
        entry from before that rebuild raises "wrapped C/C++ object has been deleted" -- which
        looks like a GUI crash and is really the test holding a widget that no longer exists.
        """
        from PyQt5 import sip

        return not sip.isdeleted(thing)

    def settle(what):
        try:
            app.processEvents()
            widget.canvas.figure.canvas.draw()
            app.processEvents()
        except Exception:
            problems.append(f"{what}: {traceback.format_exc(limit=3)}")
            return False
        return True

    try:
        for box in widget.findChildren(QCheckBox):
            if not alive(box):
                continue
            label = box.text() or box.objectName() or "checkbox"
            before = box.isChecked()
            for state in (not before, before):
                if not alive(box):
                    break                    # a rebuild landed between two touches of this widget
                box.setChecked(state)
                if not settle(f"checkbox {label!r} -> {state}"):
                    break
            touched += 1
        for combo in widget.findChildren(QComboBox):
            if not alive(combo):
                continue
            label = combo.objectName() or "combo"
            start = combo.currentIndex()
            for index in range(combo.count()):
                if not alive(combo):
                    break
                combo.setCurrentIndex(index)
                if not settle(f"combo {label!r} item {combo.itemText(index)!r}"):
                    break
            if alive(combo):
                combo.setCurrentIndex(start)
                settle(f"combo {label!r} restored")
            touched += 1
        for spin in widget.findChildren(QSpinBox) + widget.findChildren(QDoubleSpinBox):
            if not alive(spin):
                continue
            label = spin.objectName() or "spin"
            start = spin.value()
            for value in (spin.minimum(), start, min(start + 1, spin.maximum())):
                if not alive(spin):
                    break
                spin.setValue(value)
                if not settle(f"spin {label!r} -> {value}"):
                    break
            if alive(spin):
                spin.setValue(start)
                settle(f"spin {label!r} restored")
            touched += 1
        for button in widget.findChildren(QPushButton) + widget.findChildren(QToolButton):
            if not alive(button):
                continue
            label = (button.text() or button.toolTip() or "button").strip()
            # a folder picker or a re-trace is not what this test is for; both are covered
            # elsewhere and one of them is slow enough to dominate the run
            if any(word in label.lower() for word in ("root", "reload", "open", "browse")):
                continue
            button.click()
            settle(f"button {label!r}")
            touched += 1
    finally:
        QInputDialog.getInt, QInputDialog.getItem, QFileDialog.getExistingDirectory = saved
    return problems, touched


def exercise_viewport(widget, app):
    """Zoom, pan, reset and scroll, through the same paths a person uses."""
    problems = []
    view, trace = widget.view, widget.trace
    if view is None:
        return ["no view to exercise"], 0
    length = trace.length_ns
    steps = 0

    def check(what):
        nonlocal steps
        try:
            app.processEvents()
            lo, hi = view.xlim_ns
            if not (hi > lo):
                problems.append(f"{what}: degenerate window {lo}..{hi}")
            steps += 1
        except Exception:
            problems.append(f"{what}: {traceback.format_exc(limit=3)}")

    for lo, hi in ((0.0, length), (length * 0.1, length * 0.2),
                   (length * 0.45, length * 0.46), (0.0, length * 0.02)):
        view.set_window(lo, hi)
        check(f"set_window {lo:.0f}..{hi:.0f}")
    view.reset()
    check("reset")
    # a window wider than the sequence must be clamped, not accepted blindly
    view.set_window(-length, length * 3)
    lo, hi = view.xlim_ns
    if hi - lo > length * 2.5:
        problems.append(f"an over-wide window was not clamped: {lo:.0f}..{hi:.0f}")
    view.reset()
    check("reset after over-wide")
    # lane panning, including into the tab strip
    full_lo, full_hi = view.full_ylim
    view.set_lanes(full_hi - 2.0, full_hi)
    check("set_lanes at the top of the strip")
    view.set_lanes(full_lo, full_hi)
    check("set_lanes full")
    return problems, steps


def main():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    # a palette-level invariant, checked once: it does not depend on any folder
    legibility = tab_text_is_legible()
    for line in legibility:
        print(f"  LEGIBILITY: {line}")
    if "--broad" in sys.argv:
        folders = broad_folders()
        print(f"broad sweep: {len(folders)} folders, one per experiment type")
    else:
        folders = [a for a in sys.argv[1:] if not a.startswith("--")] or list(DEFAULT_FOLDERS)
    failures = 0
    for folder in folders:
        name = Path(folder).parent.parent.name
        try:
            problems, rows = drive(folder, app)
        except Exception:
            print(f"  {name:28s} RAISED\n{traceback.format_exc(limit=4)}")
            failures += 1
            continue
        # exactly what drive() returns, named once: unpacking it as a 3-tuple silently printed the
        # whole tuple as "controls" and zero for everything else, and the success branch referred to
        # names that did not exist -- so a folder with no problems crashed the summary instead of
        # reporting the coverage that is the point of printing it
        coverage = f"{rows[0]} flow-rows, {rows[1]} hovers, {rows[2]} claims, " \
                   f"{rows[3]} widgets, {rows[4]} viewport ops, {rows[5]} sweep points, " \
                   f"{rows[6]} assumed alternatives"
        if problems:
            failures += 1
            print(f"  {name:28s} {coverage}, {len(problems)} PROBLEM(S)")
            for problem in problems[:6]:
                print(f"       {problem}")
        else:
            print(f"  {name:28s} {coverage} -- OK")
    print(f"\n{len(folders)} folders; {failures} with problems"
          + (f"; {len(legibility)} legibility problem(s)" if legibility else ""))
    return 1 if failures or legibility else 0


if __name__ == "__main__":
    raise SystemExit(main())
