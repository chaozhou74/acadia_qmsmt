"""One dashed box per ENTRY of a construct -- checked as a property, over every archived run.

A vertical dashed edge in the sequence view is a claim: "the construct was entered or left here."
That claim was wrong. Spans were grouped by ``block.index`` adjacency, and a loop replays its body,
so the placement indices run 11,12,11,12... and adjacency broke at every pass boundary. One
execution of a construct came out as one rectangle *per pass*, each with its own vertical edges --
so raising an inner ``repeat_until`` count grew extra edges on the OUTER construct, which is
entered exactly once.

The rule that should hold instead is derivable from the plan alone, with nothing measured and no
case-specific knowledge:

    rectangles(construct, depth) == distinct enclosing-pass prefixes path[:depth]

An outermost construct has one prefix -- the empty one -- so it gets exactly ONE box no matter how
many passes it runs or what is nested inside it. A construct one level in runs once per enclosing
pass, so it gets one box per enclosing pass: those edges are real re-entries.

Two consequences are checked separately because they are what a reader actually relies on:

* raising an inner count must not change the number of boxes of anything ENCLOSING it;
* raising a construct's own count must not change ITS OWN number of boxes either (more passes,
  same single entry) -- and the inner constructs it repeats must scale with it.

Run: ``python validation/nesting_boxes.py`` (offline, no board, no deploy).
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DATA = "/home/boson/data/test_loopback"


def boxes_by_construct(trace):
    """``{(block, depth): [region, ...]}`` -- the drawn rectangles, grouped per construct."""
    from acadia_qmsmt.sequence_viz import plotting

    out = defaultdict(list)
    for start, stop, _context, info in plotting.branch_regions(trace):
        if info.get("elided"):
            # a construct drawn zero times contributes a clickable HANDLE and no rectangle, so it
            # must not be counted as a box -- but it must still exist, which handles() checks
            continue
        out[(info["block"], info["depth"])].append((start, stop, info))
    return out


def handles(trace):
    """``{(block, depth)}`` -- every construct that has a clickable tab, drawn or not.

    Reachability is the property here: a construct you cannot point at can only be changed from the
    panel, so setting a loop to 0 passes used to remove its own handle from the diagram.
    """
    from acadia_qmsmt.sequence_viz import plotting

    return {(info["block"], info["depth"])
            for _s, _e, _c, info in plotting.branch_regions(trace)}


def execution_handles(trace):
    """``{(block, depth, path)}`` -- every EXECUTION that has a clickable tab, drawn or not."""
    from acadia_qmsmt.sequence_viz import plotting

    return {(info["block"], info["depth"], tuple(info.get("path", ()) or ()))
            for _s, _e, _c, info in plotting.branch_regions(trace)}


def boxes_nest_inside_their_parents(lane_counts=(1, 2, 3, 4, 5, 6, 8, 9, 13, 17, 24), depths=13):
    """An inner construct's box must fit inside its parent's, at every channel count.

    Pure geometry, so it is checked directly rather than per trace. The height is floored so a deep
    nest cannot collapse the innermost box -- and on a SHORT plot that floor used to break the very
    nesting it protects: at one lane every box past depth 2 clamped to the same height while the
    insets kept pushing them up, so a depth-3 box drew outside its depth-1 parent and the picture
    said the inner construct contained the outer one. Nothing shipped is that shallow and that
    deeply nested, which is exactly why it needs a check rather than a reader noticing.
    """
    from acadia_qmsmt.sequence_viz import plotting

    problems = []
    for lanes in lane_counts:
        previous = None
        for depth in range(1, depths):
            # NOTE the sweep runs far past any nesting the archive contains. The inset and the
            # palette both CLAMP once the nest outruns them, and a clamp is exactly where "each
            # box sits inside the last" stops being automatic -- which is how the one-lane bug
            # got in. Depth is cheap to sweep and expensive to meet for the first time on screen.
            low, height = plotting.control_flow_box(depth, lanes)
            high = low + height
            if previous is not None:
                parent_low, parent_high = previous
                if high > parent_high + 1e-9 or low < parent_low - 1e-9:
                    problems.append(f"{lanes} lane(s): the depth-{depth} box {low:.2f}..{high:.2f} "
                                    f"escapes its parent {parent_low:.2f}..{parent_high:.2f}")
            previous = (low, high)
    return problems


def labels_read_at_every_depth(depths=13, target=4.5):
    """A tab's text must be readable on the tab, at any nesting depth, in either theme.

    The palette lightens with depth and then clamps, and both ends of that have already produced an
    unreadable label: white text on pale lilac at depth 3 (2.48:1) and depth 4 (1.65:1), and a
    hollow tab's near-black ink on the dark theme's near-black page (1.71:1) -- the handle for a
    construct that is NOT drawn, invisible exactly where a reader would look for it. Both were
    fixed by choosing colours by contrast -- and a third came out of asking the question at every
    depth rather than the four the palette names: at depth 1 NEITHER ink candidate could reach the
    threshold on that fill (4.19:1 light, 3.72:1 dark), so "the better of the two" shipped an
    unreadable 6 pt label. Contrast is a property of the fill-and-ink PAIR, which is what
    ``legible_tab`` returns and what this asserts, at depths the archive does not reach.

    The pair is checked with the production option set AND with each theme's own, because the
    tab's fill is its own colour rather than the page's: the answer must not depend on the theme.
    """
    from acadia_qmsmt.sequence_viz import plotting

    problems = []
    for theme_name, theme in (("light", plotting.LIGHT_THEME), ("dark", plotting.DARK_THEME)):
        page = theme["axes_bg"]
        options = (theme["surface"], theme["ink_primary"])
        for depth in range(0, depths):
            ink = plotting.TAB_DEPTH_INK[min(depth, len(plotting.TAB_DEPTH_INK) - 1)]
            for which, chosen in (("as drawn", plotting.legible_tab(ink)),
                                  (f"{theme_name} inks", plotting.legible_tab(ink, options))):
                fill, text = chosen
                solid = plotting.contrast_ratio(text, fill)
                if solid < target - 1e-6:
                    problems.append(f"{theme_name} depth {depth}: label on a solid tab ({which}) "
                                    f"is {solid:.2f}:1, under {target}:1")
            hollow_ink = plotting.legible_ink(ink, page, target=target)
            hollow = plotting.contrast_ratio(hollow_ink, page)
            if hollow < target - 1e-6:
                problems.append(f"{theme_name} depth {depth}: a hollow tab's ink is "
                                f"{hollow:.2f}:1 on the page, under {target}:1")
    return problems


def markers_sit_in_execution_order(trace):
    """A handle for something not drawn must still sit WHERE it would have run.

    Reported by eye: skipping a test put its hollow tab to the right of a construct that comes after
    it. The marker was placed by sorting placements on ``(path, index)``, which orders a nested
    placement after every top-level one regardless of when it runs. Two checks hold the fix:

    * executions of one construct are in ascending order -- execution #1 cannot sit after #2;
    * an elided construct's marker is not past the start of the next construct that IS drawn and
      begins at a later block.
    """
    from acadia_qmsmt.sequence_viz import plotting

    problems = []
    regions = plotting.branch_regions(trace)
    per_construct = defaultdict(list)
    for start, _stop, _context, info in regions:
        per_construct[(info["block"], info["depth"])].append(
            (tuple(info.get("path", ()) or ()), start, bool(info.get("elided"))))
    for (block, depth), spans in per_construct.items():
        ordered = sorted(spans, key=lambda item: item[0])       # by enclosing pass path
        positions = [start for _path, start, _elided in ordered]
        if positions != sorted(positions):
            problems.append(f"block {block} depth {depth}: executions out of order "
                            f"{[round(p) for p in positions]}")
    drawn_starts = {(info["block"], info["depth"]): start
                    for start, _stop, _c, info in regions if not info.get("elided")}
    for start, _stop, _context, info in regions:
        if not info.get("elided"):
            continue
        for (other_block, _other_depth), other_start in drawn_starts.items():
            if other_block > info["block"] and start > other_start + 1e-9:
                problems.append(f"block {info['block']} (not drawn) is marked at {start:.0f}, "
                                f"after block {other_block} which starts at {other_start:.0f}")
                break
    return problems


def zeroing_keeps_every_handle(trace):
    """Set each construct AND each of its executions to 0 in turn; the handle must survive.

    This is the reported bug, as a property: "the tab disappears when repeat until iteration set to
    zero". A construct drawn zero times has no span, and the handle used to be derived from the span,
    so the setting removed its own control -- at the construct level first, and then one level down
    for a single execution of a construct whose siblings were still drawn.
    """
    problems = []
    for entry in trace.control_flow_summary():
        if entry["kind"] == "test":
            continue
        targets = [(entry["key"], (entry["block"], entry["depth"], ()))]
        for run in entry.get("executions") or ():
            targets.append((run["key"],
                            (entry["block"], entry["depth"], tuple(run["path"]))))
        for key, expected in targets:
            trace.loop_counts.clear()
            trace.loop_counts[key] = 0
            trace.relayout()
            reachable = execution_handles(trace)
            here = (expected[0], expected[1])
            if expected not in reachable and not any(
                    (b, d) == here for b, d, _p in reachable):
                problems.append(f"pinning {key}=0 left no tab for {expected} -- "
                                f"unreachable from the diagram")
    trace.loop_counts.clear()
    trace.relayout()
    return problems


def entries_by_construct(trace):
    """``{(block, depth): {path_prefix, ...}}`` -- how many times each construct is ENTERED.

    Computed straight from the placements, independently of the drawing code, so this is a real
    cross-check rather than the renderer agreeing with itself.
    """
    out = defaultdict(set)
    for placement in (trace.placements or trace.blocks):
        context = getattr(placement, "conditional", ()) or ()
        path = tuple(getattr(placement, "path", ()) or ())
        for depth in range(len(context)):
            # the construct's key is the first block of its body, which is what branch_regions
            # reports and what loop_counts / path_choices are keyed on
            out[(_first_block(trace, context, depth), depth + 1)].add(path[:depth])
    return out


def _first_block(trace, context, depth):
    """The lowest placement index sharing this construct instance at this level."""
    best = None
    for placement in (trace.placements or trace.blocks):
        other = getattr(placement, "conditional", ()) or ()
        if len(other) > depth and trace._same_context(other[depth], context[depth]):
            best = placement.index if best is None else min(best, placement.index)
    return best


def check(trace):
    """Every construct: rectangles == entries. Returns a list of complaint strings."""
    drawn = boxes_by_construct(trace)
    entries = entries_by_construct(trace)
    problems = []
    for key, paths in entries.items():
        block, depth = key
        got = len(drawn.get(key, ()))
        want = len(paths)
        if got != want:
            problems.append(f"block {block} depth {depth}: {got} boxes, entered {want}x")
    for key in drawn:
        if key not in entries:
            problems.append(f"block {key[0]} depth {key[1]}: drawn but never entered")
    # EVERY construct must be reachable from the diagram, drawn or not
    reachable = handles(trace)
    for entry in trace.control_flow_summary():
        if (entry["block"], entry["depth"]) not in reachable:
            problems.append(f"block {entry['block']} depth {entry['depth']}: no tab at all -- "
                            f"not editable from the diagram")
    # a box must not overlap another box of the SAME construct -- that would be two claims about
    # one region, which is how the split showed up visually
    for key, regions in drawn.items():
        spans = sorted((r[0], r[1]) for r in regions)
        for (a_start, a_stop), (b_start, _b_stop) in zip(spans, spans[1:]):
            if b_start < a_stop:
                problems.append(f"block {key[0]} depth {key[1]}: boxes overlap "
                                f"({a_start:.0f}-{a_stop:.0f} vs {b_start:.0f})")
    return problems


def independence(trace):
    """Raise each construct's count in turn; its own and its enclosing boxes must not multiply.

    Pinned through the CANONICAL key ``(block, depth)``, which is what the tabs and the control-flow
    panel now write. A bare block index is the pre-depth fallback and it matches every construct
    whose body starts at that block -- and nested constructs frequently share a first block, so
    pinning by block moved two constructs at once. Testing through the ambiguous key measured the
    ambiguity, not the layout: it reported the inner construct's boxes multiplying when what had
    actually happened was that the OUTER loop it shares a block with got pinned too, which really
    does re-enter the inner one.
    """
    problems = []
    summary = [e for e in trace.control_flow_summary() if e["kind"] != "test"]
    base = {k: len(v) for k, v in boxes_by_construct(trace).items()}
    for entry in summary:
        key, depth = entry["key"], int(entry["depth"])
        # zero passes: the box goes away, the handle must not
        trace.loop_counts.clear()
        trace.loop_counts[key] = 0
        trace.relayout()
        if (entry["block"], depth) not in handles(trace):
            problems.append(f"pinning {key}=0 removed its own tab -- unreachable from the diagram")
        for count in (2, 3, 5):
            trace.loop_counts.clear()
            trace.loop_counts[key] = count
            trace.relayout()
            now = {k: len(v) for k, v in boxes_by_construct(trace).items()}
            for other, was in base.items():
                if other == (entry["block"], depth):
                    # its own box count must not change: more passes, still one entry
                    if now.get(other, 0) != was:
                        problems.append(f"pinning {key}={count} changed its OWN box count: "
                                        f"{was} -> {now.get(other, 0)}")
                elif other[1] < depth:
                    # anything ENCLOSING it is entered the same number of times as before
                    if now.get(other, 0) != was:
                        problems.append(f"pinning {key}={count} changed ENCLOSING construct "
                                        f"{other}: {was} -> {now.get(other, 0)}")
        trace.loop_counts.clear()
        trace.relayout()
    return problems


def folders():
    """Every archived run worth checking, one per experiment group, breadth first.

    Reuses ``stress_campaign.folders()`` -- the campaign already maintains a cached, ``find``-based,
    round-robin index of the archive, and rebuilding that here with globs was both a duplicate and
    minutes slower on the NFS mount. Only the ``compiled.log`` filter is local, because this check
    needs a run whose own compiled program is on disk to trace against.
    """
    import stress_campaign as sc

    # the loopback archive first: it holds the deliberately-nested cases (three_deep_nest_reconfig,
    # counter_loop_in_test) that exist precisely to exercise this rule
    seen, out = set(), []
    for folder in sc.folders(roots=(DATA,) + tuple(sc.DATA_ROOTS),
                             cache_name="nesting_index.txt"):
        folder = str(folder)
        group = Path(folder).parent.name
        if group in seen or not Path(folder, "compiled.log").exists():
            continue
        seen.add(group)
        out.append(folder)
    return out


def main():
    from acadia_qmsmt import sequence_viz as sv
    import logging
    logging.getLogger().setLevel(logging.ERROR)

    checked = nested = bad = 0
    geometry = boxes_nest_inside_their_parents() + labels_read_at_every_depth()
    for line in geometry:
        print(f"  GEOMETRY: {line}")
    for folder in folders():
        try:
            trace = sv.trace_folder(folder)
        except Exception:
            continue
        checked += 1
        depth = max((len(getattr(b, "conditional", ()) or ())
                     for b in (trace.placements or trace.blocks)), default=0)
        problems = check(trace)
        problems += markers_sit_in_execution_order(trace)
        problems += zeroing_keeps_every_handle(trace)
        if depth >= 2:
            nested += 1
            problems += independence(trace)
        if problems:
            bad += 1
            print(f"\n{Path(folder).parent.name}  (max depth {depth})")
            for line in problems[:6]:
                print(f"   {line}")
    print(f"\n{checked} runs checked ({nested} nested deeply enough to test independence); "
          f"{bad} with wrong box counts"
          + (f"; {len(geometry)} geometry problem(s)" if geometry else ""))
    return 1 if bad or geometry else 0


if __name__ == "__main__":
    raise SystemExit(main())
