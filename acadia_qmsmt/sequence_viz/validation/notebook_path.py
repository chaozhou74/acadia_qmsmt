"""The viewer WITHOUT Qt: `SequenceView` on a bare axes, as a notebook uses it.

Every other GUI check drives the Qt panel, but `sequence_viz` is also used straight from a notebook
-- `plot_trace` for a static figure, `SequenceView` on an ipympl canvas for zoom and pan. That path
has no panel in between, so nothing papers over a fault in `interactive.py`; and `interactive.py` is
exactly where two of the 2026-08-14 bugs lived:

* stale tab text, because `render()` did not clear the labels it had drawn;
* a viewport that kept the old window after the sequence changed length, because `set_point()`
  called `render()` where `relayout()` called `reset()`.

Both would have been visible here first. So this exercises what a notebook user actually has --
set a window, scroll-zoom about a point, reset -- and asserts the invariants the Qt gate asserts:

* the window never collapses to nothing;
* labels never outnumber the tabs that exist (the stale-text signature);
* `reset()` restores the full extent.

Run: ``python validation/notebook_path.py`` (offline, no Qt, no board).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))


def exercise(trace, name):
    """Drive one trace through the notebook gestures. Returns (problems, operations)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from acadia_qmsmt import sequence_viz as sv

    problems, operations = [], 0
    figure, axes = plt.subplots(figsize=(12, 6))
    view = sv.SequenceView(trace, axes)
    length = max(trace.length_ns, 1.0)
    try:
        for low, high in ((0.0, length), (length * 0.2, length * 0.4),
                          (length * 0.45, length * 0.46), (length * 0.9, length)):
            view.set_window(low, high)
            operations += 1
            seen_low, seen_high = view.xlim_ns
            if seen_high <= seen_low:
                problems.append(f"{name}: window collapsed to {seen_low}..{seen_high}")
            labels = [text for text in axes.texts if text.get_text().startswith("@")]
            tabs = [info for _frame, info in (getattr(axes, "_seqviz_flow_frames", None) or [])
                    if info.get("tab_rect")]
            if len(labels) > len(tabs) + 1:
                problems.append(f"{name}: {len(labels)} labels for {len(tabs)} tabs at "
                                f"{low:.0f}..{high:.0f} -- text left over from an earlier frame")
        for step in (1, -1, 1, -1):
            event = SimpleNamespace(inaxes=axes, xdata=length * 0.3, ydata=1.0, step=step,
                                    button=None, key=None, x=0, y=0)
            view._on_scroll(event)
            operations += 1
        view.reset()
        operations += 1
        low, high = view.xlim_ns
        full_low, full_high = view.full_xlim
        if high < full_high - 1e-6 or low > full_low + 1e-6:
            problems.append(f"{name}: reset left {low:.0f}..{high:.0f} of "
                            f"{full_low:.0f}..{full_high:.0f}")
    except Exception:
        import traceback
        problems.append(f"{name}: {traceback.format_exc(limit=3)}")
    finally:
        plt.close(figure)
    return problems, operations


def main():
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    from gui_validation import broad_folders
    from acadia_qmsmt import sequence_viz as sv

    limit = next((int(a) for a in sys.argv[1:] if a.isdigit()), 12)
    problems, operations, folders = [], 0, 0
    for folder in broad_folders(limit):
        name = folder.split("/")[-4]
        try:
            trace = sv.trace_folder(folder, envelopes=False)
        except Exception:
            continue
        folders += 1
        found, count = exercise(trace, name)
        problems += found
        operations += count
        print(f"   {name:28s} {count:3d} operations, "
              f"{'ok' if not found else f'{len(found)} PROBLEM(S)'}", flush=True)

    # static one-shot path too: a notebook's first call is usually plot_trace
    try:
        trace = sv.trace_folder(broad_folders(1)[0], envelopes=False)
        figure, axes = sv.plot_trace(trace)
        if axes is None or not axes.get_children():
            problems.append("plot_trace produced an empty axes")
        operations += 1
    except Exception:
        import traceback
        problems.append(f"plot_trace: {traceback.format_exc(limit=3)}")

    print(f"\n{operations} notebook-path operations over {folders} folders; "
          f"{len(problems)} problems")
    for line in problems[:10]:
        print(f"   {line}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
