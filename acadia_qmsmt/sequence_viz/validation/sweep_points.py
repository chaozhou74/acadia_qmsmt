"""Every drawing invariant, at EVERY captured sweep point -- not a handful of them.

The suite samples sweep points sparsely: six in the GUI gate, five in the streaming gate, four in
path_independence. A run captures up to 280. That sampling was never justified by anything -- it
was just enough to notice a point change did something -- and the points are not interchangeable:

* a point's cache is what supplies register-driven lengths, so the sequence's LENGTH is a function
  of the point (a swept delay takes DualRail_RB from 78 us to 319 us);
* those same registers decide a test's arm and a ``repeat_until``'s count, so the set of constructs
  that exist at all changes point to point (Readout_Fidelity grows one at point 1);
* an RB cache holds a different gate train at every point.

So "the picture describes the run" is a claim that has to hold 280 times per folder, and had been
checked six. This walks a dense sample of points and re-applies the properties that already exist
in ``nesting_boxes`` -- box count equals entry count, handles sit in execution order, zeroing any
construct keeps its handle -- rather than restating them here, plus the sanity a length must have
at every point.

The expensive property (zeroing every construct in turn, which relayouts once per construct) runs
at a few points per folder; the cheap ones run at all of them.

Run: ``python validation/sweep_points.py [folders] [--points N]`` (offline, no Qt, no board).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def sample(n_points, cap):
    """A dense, evenly spread sample of point indices, always including both ends."""
    if n_points <= cap:
        return list(range(n_points))
    step = (n_points - 1) / (cap - 1)
    return sorted({int(round(i * step)) for i in range(cap)} | {0, n_points - 1})


def walk(trace, name, points, deep_at):
    """Apply the drawing properties at each point. Returns (problems, checked, lengths seen)."""
    import nesting_boxes as nb

    problems, checked, lengths = [], 0, set()
    for point in points:
        try:
            trace.select_point(point)
        except Exception as exc:                                   # noqa: BLE001
            problems.append(f"{name} point {point}: select_point raised "
                            f"{type(exc).__name__}: {exc}")
            continue
        checked += 1
        length = trace.length_ns
        if not math.isfinite(length) or length <= 0:
            problems.append(f"{name} point {point}: length is {length}")
        else:
            lengths.add(round(length, 3))
        if not (trace.placements or trace.blocks):
            problems.append(f"{name} point {point}: nothing laid out at all")
        problems += [f"{name} point {point}: {line}" for line in nb.check(trace)]
        problems += [f"{name} point {point}: {line}"
                     for line in nb.markers_sit_in_execution_order(trace)]
        if point in deep_at:
            problems += [f"{name} point {point}: {line}"
                         for line in nb.zeroing_keeps_every_handle(trace)]
    return problems, checked, lengths


def main():
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    from gui_validation import broad_folders
    from acadia_qmsmt import sequence_viz as sv

    argv = sys.argv[1:]
    cap = int(argv[argv.index("--points") + 1]) if "--points" in argv else 40
    limit = next((int(a) for a in argv if a.isdigit()), 12)

    problems, points_seen, folders, lengths_moved = [], 0, 0, 0
    for folder in broad_folders(limit):
        name = folder.split("/")[-4]
        try:
            trace = sv.trace_folder(folder, envelopes=True)
        except Exception:
            continue
        folders += 1
        n_points = max(getattr(trace, "n_points", 1) or 1, 1)
        points = sample(n_points, cap)
        deep_at = {points[0], points[len(points) // 2], points[-1]}
        # how far the sequence itself moves across the sample is reported, not asserted: a folder
        # whose length never changes exercises the point axis far less than one that swings by 4x,
        # and that is worth seeing in the output rather than inferring from a pass
        found, checked, seen_lengths = walk(trace, name, points, deep_at)
        if len(seen_lengths) > 1:
            lengths_moved += 1
        problems += found
        points_seen += checked
        span = (f"{min(seen_lengths) / 1000:.1f}-{max(seen_lengths) / 1000:.1f} us"
                if seen_lengths else "?")
        print(f"   {name:28s} {checked:3d}/{n_points:<4d} points, {len(seen_lengths):3d} distinct "
              f"length(s) {span:22s} "
              f"{'ok' if not found else f'{len(found)} PROBLEM(S)'}", flush=True)

    print(f"\n{points_seen} sweep points over {folders} folders "
          f"({lengths_moved} whose length changes across the sweep); {len(problems)} problems")
    for line in problems[:12]:
        print(f"   {line}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
