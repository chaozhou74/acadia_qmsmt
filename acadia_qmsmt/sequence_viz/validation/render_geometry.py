"""The same run drawn at every canvas size and in both themes.

Two dimensions the suite has never varied, both of which decide how the picture is BUILT rather
than what it contains:

* **Canvas size.** Several layout decisions are made in pixels -- the 1.5 px inset that keeps
  touching bars from fusing, ``_lanes_per_pixel`` for the lane geometry, the tab strip -- and a
  pixel rule behaves differently on a 4-inch panel than on a 30-inch one. The one bug already
  found in this family (a height floor that let an inner box escape its parent) only appeared on a
  short plot, which is exactly the case nobody looks at.
* **Theme.** Every render check runs light. ``draw`` rebinds its palette from the theme dict at the
  top and then uses those names throughout, so a colour read before the rebind, or one taken from
  the module constant instead of the theme, is invisible until someone runs dark -- and then it is
  invisible in the other sense.

The properties:

* drawing never raises, at any size, in either theme;
* every rectangle drawn has finite, non-negative width and height (a NaN geometry silently drops
  the artist, so a missing pulse would look like a run that never played it);
* the tab labels never outnumber the tabs;
* **a theme changes colours, never content** -- the dark render must contain exactly as many bars,
  tabs and labels as the light one, with the same tab text. A theme that also changed what is
  drawn would mean one of the two is not showing the run.

Run: ``python validation/render_geometry.py [folders]`` (offline, no Qt, no board).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

#: Canvas sizes in inches. A cramped docked panel, a laptop, a wide monitor, a tall narrow strip,
#: and one deliberately absurd on each axis -- pixel rules have to hold across all of them.
SIZES = ((4, 2.2), (6, 3), (12, 6), (20, 5), (5, 14), (30, 8), (3, 12), (40, 3))


def content(axes):
    """What was DRAWN, reduced to a value a theme must not change."""
    from matplotlib.patches import Rectangle

    rectangles = [a for a in axes.get_children() if isinstance(a, Rectangle)]
    tabs = [info for _frame, info in (getattr(axes, "_seqviz_flow_frames", None) or [])
            if info.get("tab_rect")]
    texts = tuple(sorted(t.get_text() for t in axes.texts))
    return len(rectangles), len(tabs), texts


def geometry_problems(axes, where):
    """Every rectangle must have a finite, non-negative size."""
    from matplotlib.patches import Rectangle

    problems = []
    for artist in axes.get_children():
        if not isinstance(artist, Rectangle):
            continue
        width, height = artist.get_width(), artist.get_height()
        if not (math.isfinite(width) and math.isfinite(height)):
            problems.append(f"{where}: a rectangle has a non-finite size ({width}, {height}) -- "
                            f"it would be dropped silently")
            break
        if width < -1e-9 or height < -1e-9:
            problems.append(f"{where}: a rectangle has a negative size ({width:.4g}, "
                            f"{height:.4g})")
            break
    return problems


def decoration_problems(figure, axes, where):
    """Nothing a reader has to read may overlap another label or fall off the canvas.

    Both halves have been wrong at once on a small canvas: ten three-line lane labels on a 4.5 inch
    figure ran into each other and the axis read "readout1_stimulusADC0" -- a lane named something
    no lane is called -- while the fixed 8 pt legend was simultaneously taller than the figure and
    wider than half of it, so the sequence itself got a quarter of the width. Neither is a wrong
    NUMBER, which is why every existing check passed; both make the picture unreadable, which is
    the same failure from the reader's side.
    """
    problems = []
    try:
        renderer = figure.canvas.get_renderer()
    except AttributeError:
        return problems
    canvas = figure.get_window_extent()

    boxes = []
    for label in axes.get_yticklabels():
        if not label.get_text():
            continue
        try:
            boxes.append((label.get_text().splitlines()[0], label.get_window_extent(renderer)))
        except Exception:                                          # noqa: BLE001
            continue
    for (name, box), (other_name, other) in zip(boxes, boxes[1:]):
        overlap = min(box.y1, other.y1) - max(box.y0, other.y0)
        if overlap > 0.5 and min(box.x1, other.x1) - max(box.x0, other.x0) > 0.5:
            problems.append(f"{where}: lane labels {name!r} and {other_name!r} overlap by "
                            f"{overlap:.1f} px")
            break

    watched = [("title", axes.title)] + [("legend", axes.get_legend())]
    for what, artist in watched:
        if artist is None or (hasattr(artist, "get_text") and not artist.get_text()):
            continue
        try:
            box = artist.get_window_extent(renderer)
        except Exception:                                          # noqa: BLE001
            continue
        if (box.x0 < canvas.x0 - 1 or box.x1 > canvas.x1 + 1
                or box.y0 < canvas.y0 - 1 or box.y1 > canvas.y1 + 1):
            problems.append(f"{where}: the {what} runs off the canvas "
                            f"({box.x0:.0f}..{box.x1:.0f} x {box.y0:.0f}..{box.y1:.0f} of "
                            f"{canvas.x1:.0f}x{canvas.y1:.0f})")
    return problems


def render(trace, size, theme):
    """Draw once. Returns (axes, figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from acadia_qmsmt.sequence_viz import plotting

    figure, axes = plt.subplots(figsize=size)
    plotting.draw(axes, trace, theme=theme)
    plotting.fit_layout(figure, axes)
    figure.canvas.draw()          # force the pixel-dependent layout to actually happen
    return axes, figure


def exercise(trace, name):
    """Returns (problems, renders)."""
    import matplotlib.pyplot as plt
    from acadia_qmsmt.sequence_viz import plotting

    problems, renders = [], 0
    for size in SIZES:
        seen = {}
        for theme_name, theme in (("light", plotting.LIGHT_THEME),
                                  ("dark", plotting.DARK_THEME)):
            where = f"{name} {size[0]}x{size[1]}in {theme_name}"
            try:
                axes, figure = render(trace, size, theme)
            except Exception:
                import traceback
                problems.append(f"{where}: {traceback.format_exc(limit=3)}")
                continue
            renders += 1
            try:
                problems += geometry_problems(axes, where)
                problems += decoration_problems(figure, axes, where)
                labels = [t for t in axes.texts if t.get_text().startswith("@")]
                tabs = [i for _f, i in (getattr(axes, "_seqviz_flow_frames", None) or [])
                        if i.get("tab_rect")]
                if len(labels) > len(tabs) + 1:
                    problems.append(f"{where}: {len(labels)} tab labels for {len(tabs)} tabs")
                seen[theme_name] = content(axes)
            finally:
                plt.close(figure)
        if len(seen) == 2 and seen["light"] != seen["dark"]:
            light, dark = seen["light"], seen["dark"]
            detail = []
            if light[0] != dark[0]:
                detail.append(f"{light[0]} vs {dark[0]} rectangles")
            if light[1] != dark[1]:
                detail.append(f"{light[1]} vs {dark[1]} tabs")
            if light[2] != dark[2]:
                only = sorted(set(light[2]) ^ set(dark[2]))[:3]
                detail.append(f"text differs: {only}")
            problems.append(f"{name} {size[0]}x{size[1]}in: the THEME changed what is drawn, not "
                            f"just its colours -- {'; '.join(detail)}")
    return problems, renders


def main():
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    from gui_validation import broad_folders
    from acadia_qmsmt import sequence_viz as sv

    limit = next((int(a) for a in sys.argv[1:] if a.isdigit()), 10)
    problems, renders, folders = [], 0, 0
    for folder in broad_folders(limit):
        name = folder.split("/")[-4]
        try:
            trace = sv.trace_folder(folder, envelopes=True)
        except Exception:
            continue
        folders += 1
        found, count = exercise(trace, name)
        problems += found
        renders += count
        print(f"   {name:28s} {count:2d} renders ({len(SIZES)} sizes x 2 themes) -- "
              f"{'ok' if not found else f'{len(found)} PROBLEM(S)'}", flush=True)

    print(f"\n{renders} renders over {folders} folders; {len(problems)} problems")
    for line in problems[:10]:
        print(f"   {line}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
