"""
Render a :class:`~.tracing.SequenceTrace` as a per-channel timeline.

One lane per physical channel, time running left to right. Pulses are filled
bars carrying their waveform envelope; alignment dwells acadia inserted to
reconcile a barrier are muted and hatched, so padding never reads as a pulse.

:func:`draw` is the zoom-aware core: it takes the visible time window and picks
label density, time unit and mark spacing to suit it. :func:`plot_trace` wraps it
for a one-shot static figure; :class:`~.interactive.SequenceView` calls it on
every viewport change. Both therefore look identical, in a notebook or in the GUI.
"""
import re

import numpy as np

COPY_SUFFIX = re.compile(r"_copy\d*$")

# Fixed categorical order. Hues are assigned to pulse names in first-appearance
# order and never cycled -- past the eighth distinct pulse everything folds into
# a single neutral, and identity is carried by the direct label instead.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

INK_PRIMARY = "#1a1a19"
INK_SECONDARY = "#5c5b55"
INK_MUTED = "#8c8b83"
SURFACE = "#ffffff"
NEUTRAL_FILL = "#c9c8c0"
GAP_FILL = "#f2c94c"        # inter-block gap -- a light yellow band
BRANCH_INK = "#4a3aa7"      # control flow is a caveat, not a series

LANE_HEIGHT = 0.62
SEPARATOR_PIXELS = 1.5      # visual gap between fills that butt up against each other

#: Size of the clickable TAB drawn at each control-flow block's top-left corner, in screen
#: pixels. A tab is the handle you point at to read or change that construct's iteration count;
#: sized in pixels so it stays the same on screen at any zoom.
TAB_PIXELS_W = 34.0
#: The tab strip is sized in PIXELS, not lane units. Lane units shrink with the plot: at
#: 0.42 lanes of pitch a vertically compressed view squeezed the rows to a few pixels apart and
#: the labels of different rows ran into each other ("x3x1", "x1x1x1"). A pixel pitch stays
#: readable at any window size, which is the same reason the tab WIDTH is in pixels.
TAB_HEIGHT_PX = 15.0
TAB_ROW_PX = 19.0
TAB_FONTSIZE = 6.0

#: One colour per nesting level, darkest outermost. Three constructs can legitimately begin at
#: the same instant -- the outer loop, the round inside it and the reset inside that all start
#: together -- so their tabs stack, and identical tabs stacked three high are unreadable. Colour
#: says which level each one governs; the connector below says which box.
TAB_DEPTH_INK = ("#3b3f9e", "#6f74c9", "#9aa0e0", "#c3c7ef")


# Patches are attached with ``ax.add_artist``, not ``ax.add_patch``. Both set the transform and the
# clip path identically; the difference is that add_patch also folds every patch into the axes'
# data limits, for an autoscale this function never uses -- ``draw`` sets both xlim and ylim
# explicitly at the end. That bookkeeping is quadratic in practice (matplotlib rescans every artist
# for sticky edges), and it dominated everything: drawing 1013 placements took over a minute, so
# pinning a loop to a few hundred passes looked like the GUI had hung. It is the drawing, not the
# timing model, that made large counts unusable.

def _relative_luminance(colour):
    """WCAG relative luminance of a matplotlib colour."""
    from matplotlib.colors import to_rgb

    def channel(value):
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (channel(c) for c in to_rgb(colour))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(one, other):
    """WCAG contrast between two colours, 1:1 (identical) to 21:1 (black on white)."""
    first, second = _relative_luminance(one), _relative_luminance(other)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


def readable_on(background, options=(SURFACE, INK_PRIMARY)):
    """Whichever candidate reads best on ``background``.

    The tab palette lightens with nesting depth on purpose -- an outer construct should look
    stronger than the ones inside it -- but the label colour was fixed at white, so a depth-3 tab
    put white text on pale lilac (2.5:1) and depth 4 was 1.65:1, which is not readable at 6 pt.
    Choosing by contrast keeps the palette and fixes the text, and it keeps working if either the
    palette or the theme changes. It answers only "which of these reads better", which is not the
    same as "this reads well" -- see :func:`legible_tab` for the pair that has to be readable.
    """
    return max(options, key=lambda option: contrast_ratio(option, background))


def legible_ink(colour, background, target=4.5):
    """``colour`` moved toward the background's opposite until it is readable on it.

    Used for a HOLLOW tab, whose text and outline are the depth ink drawn straight onto the axes:
    in the dark theme the depth-1 ink is near-black on near-black (1.7:1) and the handle for a
    construct that is not drawn simply vanished. Blending preserves the hue that says which nesting
    level it is, rather than replacing it with a flat colour that says nothing.
    """
    from matplotlib.colors import to_hex, to_rgb

    if contrast_ratio(colour, background) >= target:
        return colour
    toward = (1.0, 1.0, 1.0) if _relative_luminance(background) < 0.5 else (0.0, 0.0, 0.0)
    base = to_rgb(colour)
    for fraction in (0.25, 0.4, 0.55, 0.7, 0.85):
        blended = tuple(base[i] + (toward[i] - base[i]) * fraction for i in range(3))
        if contrast_ratio(blended, background) >= target:
            return to_hex(blended)
    return to_hex(toward)


def legible_tab(fill, options=(SURFACE, INK_PRIMARY), target=4.5):
    """A solid tab as a readable (fill, ink) PAIR, nudging the fill only if the ink cannot reach.

    Choosing the better ink is not enough on its own. A mid-tone fill is far from both ends of the
    ink range at once, so neither candidate clears the threshold and the best of two is still
    unreadable: the depth-1 ink tops out at 4.19:1 with white text, and 3.72:1 against the dark
    theme's. Contrast is a property of the PAIR, and once the text is already at the extreme the
    only thing left to move is the fill.

    So the fill is blended away from the ink in the smallest step that clears the target -- ~8% of
    the way for depth 1, which keeps the ladder that makes nesting depth readable at a glance while
    making the label legible at 6 pt. Returns ``(fill, ink)``.
    """
    from matplotlib.colors import to_hex, to_rgb

    ink = readable_on(fill, options)
    if contrast_ratio(ink, fill) >= target:
        return fill, ink
    toward = (0.0, 0.0, 0.0) if _relative_luminance(ink) > 0.5 else (1.0, 1.0, 1.0)
    base = to_rgb(fill)
    for fraction in (0.04, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7):
        blended = tuple(base[i] + (toward[i] - base[i]) * fraction for i in range(3))
        if contrast_ratio(ink, blended) >= target:
            return to_hex(blended), ink
    return to_hex(toward), ink


def _lanes_per_pixel(ax, lanes):
    """How many lane units one vertical pixel is worth, for the current axes."""
    try:
        height_px = max(ax.get_window_extent().height, 1.0)
    except Exception:
        height_px = 400.0
    span = max(lanes + 0.4, 1.0)          # the y range the lanes plus margins occupy
    return span / height_px


def control_flow_box(depth, lanes):
    """``(y0, height)`` of the dashed frame drawn for a construct at ``depth``.

    Shared with the GUI so a hit test cannot drift from what was drawn. Nesting is shown by
    insetting each level, so an outer construct visibly encloses the ones inside it.
    """
    level = min(max(int(depth) - 1, 0), 4)
    # A generous step per level: at 0.09 the nesting was technically visible and practically
    # invisible -- three frames within 0.18 of a lane on a nine-lane plot read as one thick line.
    # 0.30 makes containment obvious at a glance, which is the entire point of drawing it.
    #
    # But the step has to FIT. The height is floored at MIN_BOX so a deep nest cannot collapse the
    # innermost box, and on a short plot that floor breaks the nesting it is protecting: at one lane
    # a depth-3 box came out taller than its depth-1 parent and drew OUTSIDE it, so the picture said
    # the inner construct contained the outer one. Scaling the step to the lanes available keeps
    # every box inside its parent at any depth and any channel count. Nothing shipped is that
    # shallow AND that deeply nested -- the narrowest real sequence has two lanes and no nesting --
    # so this is a latent case, fixed as a rule rather than left to be discovered.
    MIN_BOX = 0.6
    room = max(lanes - 0.05 - MIN_BOX, 0.0)
    step = min(0.30, room / 8.0)          # 4 levels, inset applied top and bottom
    inset = level * step
    return -0.62 + inset, max(lanes - 0.05 - 2 * inset, MIN_BOX)


def control_flow_tab(start_ns, depth, lanes, tab_width_ns, row_h=None, tab_h=None):
    """``(x0, y0, width, height)`` of the tab handle for a construct, in data coordinates.

    ABOVE the lanes, in a strip of its own, one row per nesting level. Inside the plot the tabs
    sat on top of whatever the first channel was playing -- on a tomography run they landed
    squarely on the readout pulses, so the handle and the data fought for the same pixels and the
    label was unreadable. A construct's tab is also the thing you go looking for, so it should be
    somewhere predictable rather than wherever its first block happens to be.

    Deeper levels sit higher, which keeps them clear of each other and matches the frames below:
    the outermost construct's tab is nearest the sequence, its children stack above it.
    """
    level = max(int(depth) - 1, 0)
    return control_flow_tab_at_row(start_ns, level, lanes, tab_width_ns, row_h, tab_h)


def control_flow_tab_at_row(start_ns, row, lanes, tab_width_ns, row_h=None, tab_h=None):
    """The tab rect for an explicitly chosen strip row (see :func:`assign_tab_rows`).

    ``row_h``/``tab_h`` are the pitch and height in LANE units, converted from pixels by the
    caller so the strip reads the same at any window size. They default to a sensible fixed
    value for callers that have no axes to measure.
    """
    pitch = TAB_ROW_PX * (row_h if row_h is not None else 0.022)
    height = TAB_HEIGHT_PX * (tab_h if tab_h is not None else 0.022)
    return start_ns, lanes - 0.45 + int(row) * pitch, tab_width_ns, height


def construct_tag(block, depth=None, ambiguous=False, execution=None):
    """A construct's identity as the reader sees it: ``@11``, or ``@11.2`` when it needs the depth.

    A block can key several constructs -- a cooling round and the active reset inside it begin at
    the same block -- and ``@11`` alone then names all of them. Wherever that happens the nesting
    depth is appended, so every label on screen identifies exactly one construct. It is added only
    where it disambiguates: an unconditional suffix would put noise on every tab in every sequence.
    """
    tag = f"@{block}" if not (ambiguous and depth is not None) else f"@{block}.{int(depth)}"
    # ...and which EXECUTION of it, when there is more than one. A construct inside a loop runs once
    # per enclosing pass and each run is separately settable, so three tabs reading "@11" named three
    # different things you could edit independently. Added only where there IS more than one.
    return tag if execution in (None, "") else f"{tag}#{execution}"


def construct_suffix(kind, count=None, source=None, taken=None, indeterminate=False,
                     nonterminating=False):
    """What follows the identity, with one meaning per glyph.

    Three independent facts have to be readable off a tab a few characters wide, and conflating any
    two of them makes the picture lie:

    * **the value being drawn** -- ``x3`` passes, or ``skip`` for a test arm that is not drawn;
    * ``*`` -- **you pinned this**, so the drawing is your hypothesis and not what the run did.
      Without it a pinned 5 was indistinguishable from a measured 5;
    * ``?`` -- **the board decides this at runtime**: the count or arm cannot be established from
      this run's data. It is a property of the CONSTRUCT, so it stays whatever you pin. It used to
      disappear the moment you set a value, which is exactly backwards -- choosing a value does not
      make the hardware deterministic, it just picks which possibility is on screen.

    So ``@11 x4*?`` reads: four passes drawn, pinned by you, and the real number is decided at
    runtime. ``@8 x3`` reads: three passes, read out of this run's own cache.
    """
    if nonterminating and source != "pinned":
        # This loop cannot exit: its counter starts at 0 and only reaches the exit target after
        # wrapping. Drawing a pass count for it would be a number where there is none -- measured
        # on the board, the run simply never returns.
        body = " x\u221e"
    elif kind == "test":
        # a test pinned to its skipped arm is not "unknown", it is deliberately not drawn
        body = " skip" if taken is False else ""
    elif count == 0:
        body = " x0"                 # drawn zero times -- still a construct, still editable
    else:
        body = f" x{count}" if count else ""
    # The flags go in their own token, separated by a space. Run together with the count they were
    # genuinely misread: at the tab's 6 pt, "x1?" looks like "x17" -- a legibility failure that turns
    # an honest marker into a wrong number, which is worse than no marker at all.
    # `x∞` already says the count is not a number, so the "runtime-decided" marker would be
    # redundant next to it -- one glyph, one meaning.
    flags = ("*" if source == "pinned" else "") + (
        "?" if (indeterminate or source == "assumed") and not nonterminating else "")
    if not flags:
        return body
    return f"{body} {flags}" if body else f" {flags}"


def depth_is_visually_distinct(depth):
    """Can a reader tell this nesting level apart by the DRAWING alone?

    The box inset and the tab ink both stop changing once the nesting runs past the palette
    (``TAB_DEPTH_INK``) and the inset cap -- deliberately, so a deep nest cannot collapse its
    innermost box to nothing. Past that point two different levels look identical, so the label has
    to carry the depth instead. Tied to the palette rather than to a number, so widening the palette
    widens this automatically.
    """
    return int(depth) <= len(TAB_DEPTH_INK)


def tab_label(info, ambiguous=False):
    """The text on a construct's tab.

    Shared with the GUI's control-flow panel through :func:`flow_label` so the two cannot drift:
    the panel row and the tab in the diagram are the same construct, and the user was asked to
    match them by eye.
    """
    depth = info.get("depth")
    return (construct_tag(info.get("block", "?"), depth,
                          ambiguous or not depth_is_visually_distinct(depth or 1),
                          info.get("execution"))
            + construct_suffix(info.get("kind"), info.get("count"), info.get("source"),
                               info.get("taken"), info.get("indeterminate", False),
                               info.get("nonterminating", False)))


def flow_label(entry, ambiguous=False, execution=None):
    """Row text for a ``control_flow_summary`` entry: the tab's text, plus the kind.

    The tab is a few characters wide so it carries only the identity and count; a panel row has
    space for the kind as well. The shared part is byte-identical, which is what lets a reader tie
    ``repeat_until @11.2 x3`` in the panel to ``@11.2 x3`` on the diagram.
    """
    depth = entry.get("depth")
    return (f"{entry.get('kind', 'loop')} "
            + construct_tag(entry.get("block", "?"), depth,
                            ambiguous or not depth_is_visually_distinct(depth or 1),
                            execution if execution is not None else entry.get("execution"))
            + construct_suffix(entry.get("kind"), entry.get("count"), entry.get("source"),
                               entry.get("taken"), entry.get("indeterminate", False),
                               entry.get("nonterminating", False)))


def ambiguous_blocks(regions):
    """Blocks that key more than one construct in this trace -- these tabs need their depth."""
    depths = {}
    for _s, _e, _c, info in regions:
        depths.setdefault(info.get("block"), set()).add(info.get("depth"))
    return {block for block, levels in depths.items() if len(levels) > 1}


def tab_widths(regions, min_width, per_char):
    """``{id(info): width}`` -- each tab as wide as its own LABEL needs, never narrower than
    ``min_width``.

    A fixed width was wrong: the label is centred on the tab and overflows a narrow one, so tabs
    that do not overlap can still have labels that do. Zoomed out, `@14 x1` and `@16` sat a few
    pixels apart and their text ran together. Sizing each tab to its content makes "tabs do not
    overlap" and "labels do not overlap" the same statement, which is the property worth
    enforcing.
    """
    shared = ambiguous_blocks(regions)
    return {id(info): max(min_width,
                          per_char * (len(tab_label(info, info.get("block") in shared)) + 1.4))
            for _s, _e, _c, info in regions}


def assign_tab_rows(regions, widths, ns_per_cycle):
    """``{id(info): row}`` -- which strip row each construct's tab sits on.

    A tab starts on its own nesting level's row, and is bumped up only when it would land on top
    of one already there. Two constructs beginning within a tab's width of each other used to
    draw their handles in the same place, which made both labels unreadable (a run of tomography
    prep blocks produced `tdstst` where three labels overlapped) and left the hit test picking
    whichever happened to be last.

    Rows are grouped into a BAND PER DEPTH, and a collision adds a sub-row inside that construct's
    own band -- never in another depth's row. Bumping across bands was tried first and reads wrong:
    a depth-2 tab bumped up landed on the row belonging to depth 3, which pushed a depth-3 tab to a
    row above everything, so a sequence nested three deep drew a four-row strip with one tab alone
    on top. The row then no longer meant the nesting level, which is the one thing the strip is
    supposed to show. Banding keeps "deeper sits higher" exactly true: every tab of a given depth is
    in that depth's band, and a band is only as tall as its own worst overlap.
    """
    # `regions` carries starts in CYCLES while the tab width is in PLOTTED units (ns or us,
    # whichever the axis chose). Comparing them directly compares microseconds against cycles, so
    # the collision test silently never fired -- two `test` tabs 0.22 us apart stayed on the same
    # row and drew their labels through each other. Convert first.
    placed, bands = {}, {}
    for start, _stop, _context, info in sorted(regions, key=lambda r: r[0]):
        depth = max(int(info.get("depth", 1)), 1)
        left = start * ns_per_cycle
        band = bands.setdefault(depth, [])       # right edge of the last tab on each sub-row
        sub = 0
        while sub < len(band) and band[sub] > left:
            sub += 1
        if sub == len(band):
            band.append(float("-inf"))
        band[sub] = left + widths.get(id(info), 0.0)
        placed[id(info)] = (depth, sub)
    # stack the bands bottom-up in depth order, each as tall as the sub-rows it actually needed
    base, offset = {}, 0
    for depth in sorted(bands):
        base[depth] = offset
        offset += len(bands[depth])
    return {key: base[depth] + sub for key, (depth, sub) in placed.items()}


def tab_strip_top(max_depth, lanes, row_h=None):
    """Top of the y range needed to hold the tab strip for ``max_depth`` nesting levels."""
    _x, y0, _w, height = control_flow_tab(0.0, max(max_depth, 1), lanes, 0.0,
                                          row_h, row_h)
    # A small margin only. The strip steals y range from the lanes, and on a short window that
    # squeezes the channel labels into each other -- a fix for the tabs must not degrade the part
    # of the plot the tabs are there to annotate.
    return y0 + height + 0.5 * TAB_ROW_PX * (row_h if row_h is not None else 0.022)

# Pulse fills are drawn softened so the envelope stroke on top reads as the shape.
PULSE_FILL_ALPHA = 0.7
ENVELOPE_LINEWIDTH = 1.7    # magnitude stroke; iq uses 0.75x (two overlapping lines)
ENVELOPE_MIN_PX = 6         # skip a pulse's envelope below this on-screen width --
                            # its shape is unreadable there and each envelope is a
                            # separate plot call (the dominant zoom-render cost)

# Color themes. draw() takes one (default LIGHT_THEME); the Qt widget swaps to DARK
# when the app is in a dark theme. Envelope + all labels + title use ink_primary, so a
# near-white ink turns them white on dark; surface (capture/register fill) is kept off
# pure white; figure/axes bg is opaque so there is no white frame in dark mode.
LIGHT_THEME = {
    "series": SERIES_LIGHT,
    "ink_primary": INK_PRIMARY, "ink_secondary": INK_SECONDARY, "ink_muted": INK_MUTED,
    "surface": SURFACE, "neutral_fill": NEUTRAL_FILL,
    "gap_fill": GAP_FILL, "branch_ink": BRANCH_INK,
    "figure_bg": "#ffffff", "axes_bg": "#ffffff",
    "fill_alpha": PULSE_FILL_ALPHA,
}
DARK_THEME = {
    "series": SERIES_LIGHT,
    "ink_primary": "#f2f2ec", "ink_secondary": "#b6b5ae", "ink_muted": "#7f7e77",
    "surface": "#3a3a42", "neutral_fill": "#4c4c54",
    "gap_fill": "#d9c24e", "branch_ink": "#9d90ff",
    "figure_bg": "#26262b", "axes_bg": "#26262b",
    "fill_alpha": 0.82,
}

# Above this many bars in view, drop the per-bar extras -- they are unreadable at
# that density anyway and redrawing them makes dragging feel sticky.
DETAIL_BUDGET = 400


def base_pulse(name, group_copies=True):
    """``swap_copy3`` -> ``swap``. Duplicates exist to hold different phases or
    amplitudes of one logical pulse, so they share a hue and differ by label."""
    return COPY_SUFFIX.sub("", name) if group_copies else name


def color_key(cmd, color_by="memory", group_copies=True):
    """Identity a hue is assigned to, for one command."""
    if color_by == "memory":
        # a waveform memory belongs to exactly one DAC, so the "is this the same
        # pulse memory?" question is inherently per-channel
        return (cmd.channel, cmd.address)
    if color_by == "channel":
        return cmd.channel
    if color_by == "name":
        return base_pulse(cmd.pulse, group_copies) if cmd.pulse else None
    raise ValueError(f"unknown color_by: {color_by!r}")


def assign_styles(trace, color_by="memory", group_copies=True, series=None):
    """Identity -> ``{"color", "generation"}``, assigned in first-appearance order.

    ``color_by="memory"`` (default) gives one style per **waveform memory**, keyed
    globally on ``(channel, address)``. Two ``schedule_pulse`` calls sharing a
    memory necessarily play the same samples, so they share a style; duplicates
    made to hold different phases or amplitudes get their own, and two pulses that
    merely share a *name* on different channels no longer collide.

    A sequence can hold more distinct memories than the palette has slots (12 in
    one of the beam-splitting runs). Rather than invent hues, the palette wraps and
    ``generation`` counts the wrap -- the renderer marks later generations with an
    outline, so identity stays unambiguous.

    ``color_by="name"`` keys on the pulse name instead, so a name means one color
    everywhere at the cost of merging same-named pulses across channels.
    ``color_by="channel"`` gives one style per lane.
    """
    series = series or SERIES_LIGHT
    styles = {}
    for cmd in trace.commands:
        if cmd.pulse is None:
            continue
        key = color_key(cmd, color_by, group_copies)
        if key in styles:
            continue
        slot = len(styles)
        styles[key] = {"color": series[slot % len(series)],
                       "generation": slot // len(series)}
    return styles


def assign_colors(trace, color_by="memory", group_copies=True):
    """Identity -> hex. See :func:`assign_styles` for the full style."""
    return {k: v["color"]
            for k, v in assign_styles(trace, color_by, group_copies).items()}


def _time_scale(span_ns):
    """(divisor, unit label) so long sequences read in us rather than ns."""
    return (1000.0, "µs") if span_ns >= 5000 else (1.0, "ns")


def _pixels_in_cycles(ax, pixels, span, ns):
    """``pixels`` of screen width expressed in sequencer cycles."""
    try:
        width_px = max(ax.get_window_extent().width, 1.0)
    except Exception:
        width_px = 800.0
    return (pixels / width_px) * span / ns


#: The plot is what the figure is FOR: the key beside it may take at most this share of the width,
#: and must fit inside the canvas vertically. Beyond that the key is shrunk, then dropped.
LEGEND_MAX_WIDTH_SHARE = 0.42
LEGEND_FONT_STEPS = (8.0, 7.0, 6.0, 5.0)


def legend_fontsize(fig, labels):
    """Point size for the key beside the plot, or ``None`` if it cannot fit at any size.

    The legend was a fixed 8 pt regardless of canvas. On a 5x3 inch figure -- a narrowly docked
    panel, which is a size a reader produces by dragging a splitter -- twenty entries are taller
    than the whole figure and wider than half of it: the key ran off the bottom AND squeezed the
    sequence into a quarter of the width, so the one thing the figure exists to show became
    unreadable in order to display a colour key that was itself clipped.

    Estimated from the canvas rather than measured because it decides how the legend is BUILT;
    :func:`fit_layout` then measures the real thing and enforces the same share. Same idea as
    dropping the per-bar extras when the bars get too dense to read.
    """
    width_in, height_in = fig.get_size_inches()
    longest = max((len(label) for label in labels), default=0)
    for size in LEGEND_FONT_STEPS:
        rows_in = len(labels) * size * 1.6 / 72.0          # a row is ~1.6x the point size
        text_in = longest * size * 0.62 / 72.0 + 0.32      # plus the handle patch
        if rows_in <= height_in * 0.98 and text_in <= width_in * LEGEND_MAX_WIDTH_SHARE:
            return size
    return None


def fit_layout(fig, ax):
    """Make room for the axis labels and for the legend outside the axes.

    ``tight_layout`` does not account for a legend anchored outside the axes, and
    an interactive canvas gets no ``bbox_inches="tight"`` expansion to paper over
    it -- so the legend and the lane labels are clipped unless the margin is
    reserved explicitly.

    This is also where the measured version of :func:`legend_fontsize`'s promise is kept: a key
    that still would not fit after being shrunk is removed rather than allowed to crowd out the
    sequence.
    """
    try:
        fig.tight_layout()
    except Exception:
        pass
    legend = ax.get_legend()
    if legend is None:
        return
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        return                      # backend cannot measure before a draw
    try:
        box = legend.get_window_extent(renderer)
        canvas = fig.get_window_extent()
    except Exception:
        return
    needed = box.width / max(canvas.width, 1.0)
    # CONTAINMENT, not just size: the legend hangs from the axes' top edge, so one shorter than the
    # figure can still run off the bottom -- which is what a 5x3 inch canvas did, showing a clipped
    # key at the cost of half the sequence's width. A key that does not fit is worth less than the
    # room it takes: the colours it names are also on the bars themselves.
    fits = (needed <= LEGEND_MAX_WIDTH_SHARE
            and box.y0 >= canvas.y0 - 1.0 and box.y1 <= canvas.y1 + 1.0)
    if not fits:
        legend.remove()
        return
    right = 1.0 - needed - 0.02
    if right < fig.subplotpars.right:
        fig.subplots_adjust(right=right)


def _stream_gates_of(trace, block_index):
    """Gates the cache-pointer stream drew inside the construct whose body is ``block_index``.

    The stream loop's PASS count is deliberately unresolved -- `_expand_stream` unrolls the whole
    train inside one pass, and resolving the count as well would draw it once per pass (measured:
    1791 gates became 3207681). So the tab has no number to show and falls back to "the board
    decides this at runtime", which is false: the count is read from this run's own cache. This
    gives the caption the real figure to print instead.
    """
    stream = getattr(trace, "stream", None)
    if not stream or not getattr(trace, "stream_gates", 0):
        return None
    blocks = getattr(trace, "blocks", None) or ()
    if not (0 <= block_index < len(blocks)):
        return None
    if not any(trace._is_stream_command(c) for c in blocks[block_index].commands):
        return None
    return int(trace.stream_gates)


def branch_regions(trace):
    """``[(start, stop, context, info), ...]`` -- one span per CONSTRUCT, at every nesting level.

    One span per LEVEL, not one per block-run at its full depth. A block sitting inside three
    nested constructs used to produce a single depth-3 box, so the loops enclosing it had no box
    of their own anywhere near it: the drawing showed the innermost construct and left the reader
    to infer the rest, and a UI hit-testing the picture could only ever reach the innermost one.

    Now each enclosing construct also gets a span covering its whole body, so an outer loop
    visibly contains the inner ones and can be pointed at. Combined with the depth inset in
    :func:`draw`, that makes the nesting readable and selectable: the bands above and below an
    inner box belong to its parent.

    :return: ``info`` carries the iteration, the block that KEYS this construct's overrides
        (``loop_counts`` / ``path_choices``), its depth, and caption facts.
    """
    blocks = list(trace.placements or trace.blocks)
    regions = []
    # The count and where it came from are established ONCE, by control_flow_summary, and merged in
    # below. Recomputing them here is how the tab came to say "@2" while the panel row for the same
    # construct said "repeat_until @2 x1": two implementations of "what is this construct's count"
    # drifting apart. Keyed by (block, one-based depth), which is the canonical construct key.
    summary = {(e["block"], e["depth"]): e for e in trace.control_flow_summary()}
    tags = _execution_tags(summary)
    max_depth = max((len(getattr(b, "conditional", ()) or ()) for b in blocks), default=0)

    for depth in range(max_depth):
        run = None
        for position, block in enumerate(blocks):
            context = getattr(block, "conditional", ()) or ()
            here = context[depth] if len(context) > depth else None
            iteration = getattr(block, "iteration", 0)
            # The enclosing pass indices for THIS execution of the construct. A nested loop is
            # compiled once but runs once per outer pass, and each execution is independently
            # settable, so a span must say which execution it is or a tab would edit all at once.
            full_path = tuple(getattr(block, "path", ()) or ())
            path = full_path[:depth]
            # This construct's OWN pass index, which is its slot in the path. `iteration` is a
            # single scalar per placement -- the innermost loop's index -- so counting it at an
            # outer level counted the wrong loop's passes: a 3-pass outer loop containing a
            # 4-pass inner one was captioned "4 passes of 3".
            own = full_path[depth] if len(full_path) > depth else iteration
            # One run is ONE EXECUTION of this construct: same context instance at this level,
            # same enclosing passes, and adjacent IN THE EXECUTION PLAN. Deeper levels are free
            # to change inside it, which is exactly what makes this span enclose them.
            #
            # Plan position, not block.index. A loop replays its body, so placement indices go
            # 11,12,11,12 and index-adjacency breaks at every pass boundary -- which split one
            # execution into one rectangle per pass, and each rectangle drew its own vertical
            # dashed edge. Raising an inner count therefore grew extra edges on the OUTER
            # construct, which is entered exactly once. A vertical edge now means what it looks
            # like it means: the construct was entered or left there.
            #
            # The path test is what still separates genuine re-entries. An inner construct runs
            # once per enclosing pass and those executions differ in path[:depth], so they stay
            # separate rectangles -- the nesting you can see is real re-entry, not replay.
            same = (run is not None and here is not None
                    and trace._same_context(run["context"], here)
                    and path == run["path"]
                    and position == run["last_pos"] + 1)
            if same:
                run["stop"] = max(run["stop"], block.stop)
                run["last_pos"] = position
                run["passes"].add(own)
                continue
            if run is not None:
                regions.append(run)
            run = ({"start": block.start, "stop": block.stop, "context": here,
                    "full": context[:depth + 1], "block": block.index,
                    "last_pos": position, "kind": here.get("kind"), "iteration": iteration,
                    # how many passes of the construct this one rectangle covers, so the caption
                    # can say "3 passes" instead of naming a single pass it no longer bounds
                    "passes": {own}, "path": path}
                   if here is not None else None)
        if run is not None:
            regions.append(run)

    out = []
    for run in regions:
        first = run["block"]
        out.append((run["start"], run["stop"], run["full"], {
            "iteration": run["iteration"],
            "passes": len(run.get("passes") or (0,)),
            # the block that KEYS this construct's overrides -- the first block of the body,
            # which is what loop_counts / path_choices use and what control_flow_summary reports
            "block": first,
            # from the context, which is where the kind lives; the run only carries a copy of it,
            # and having both here meant two entries for one key with the later one silently winning
            "kind": (run["context"].get("kind")
                     if isinstance(run["context"], dict) else run.get("kind")),
            "depth": len(run["full"]),
            "assumed": first in trace.assumed_paths,
            "unsupported": first in getattr(trace, "unsupported_paths", ()),
            "stream_count": _stream_gates_of(trace, first),
            # which execution this is, for a per-execution override key
            "path": run.get("path", ()),
            "repeat_count": getattr(trace, "repeat_counts", {}).get(first),
            # this EXECUTION's pinned count wins over the construct-wide one, so a tab shows what
            # its own span was drawn with rather than what its siblings were
            # which execution this span is, numbered the way control_flow_summary orders them so a
            # tab and its panel row carry the same number
            "execution": tags.get((first, len(run["full"]), tuple(run.get("path", ())))),
            **_count_and_source(trace, summary, first, len(run["full"]), run.get("path", ()))}))
    # A CONSTRUCT THAT IS NOT DRAWN STILL NEEDS A HANDLE -- and so does each of its EXECUTIONS.
    # Setting a loop to 0 passes, or pinning a test to its skipped arm, removes the body from the
    # plan: no span, so no tab, so the only control left was the panel. The tab is how the diagram is
    # edited, and a setting that hides its own control is a trap. This covers both levels: a whole
    # construct that draws nothing, and a single execution of one that draws nothing while its
    # siblings still do.
    #
    # The marker goes where the plan NEXT DRAWS SOMETHING after that point, in execution order. That
    # is not a timing claim -- nothing is drawn -- it is exactly where the sequencer carried on, so a
    # zero-pass execution sits at the instant it would have occupied and took no time.
    drawn = {(info["block"], info["depth"], tuple(info.get("path", ()) or ()))
             for _s, _e, _c, info in out}
    for (block, depth), entry in sorted(summary.items()):
        static = next((b for b in trace.blocks if b.index == block), None)
        stack = tuple(getattr(static, "conditional", ()) or ())[:depth]
        if len(stack) < depth:
            continue                     # cannot describe it honestly, so do not draw a handle
        runs = entry.get("executions") or ()
        wanted = [tuple(run["path"]) for run in runs] or [()]
        for path in wanted:
            if (block, depth, path) in drawn:
                continue
            at = _elided_position(blocks, block, path)
            pinned = trace._count_override(block, depth, path if path else None)
            out.append((at, at, stack, {
                "iteration": 0, "passes": 0, "block": block, "kind": entry["kind"],
                "depth": depth, "assumed": entry["source"] == "assumed",
                "unsupported": False, "stream_count": None, "path": path,
                "repeat_count": None,
                "count": int(pinned) if pinned is not None else entry["count"],
                "source": "pinned" if pinned is not None else entry["source"],
                "taken": entry["taken"], "indeterminate": bool(entry.get("indeterminate")),
                "execution": tags.get((block, depth, path)),
                # no rectangle for this one: there is no span to enclose, only a handle
                "elided": True}))
    # innermost last, so the deepest box is drawn on top of the ones enclosing it
    return sorted(out, key=lambda r: r[3]["depth"])

def execution_tag(path):
    """Which execution this is, as a stable dotted 1-based pass path: ``(1, 0)`` -> ``"2.1"``.

    Derived from the PATH, never from a position in a list. Numbering by list position renumbered
    every sibling the moment one execution stopped being drawn -- skip the first of two and the
    second became "#1", so two rows in the panel read the same and the one you had just changed
    appeared to be a different construct. The path is the enclosing loop's pass index, so "2.1"
    means "the 1st pass of this construct during the 2nd pass of the loop around it" and it does not
    move when a sibling changes.
    """
    return ".".join(str(int(step) + 1) for step in (path or ()))


def _elided_position(blocks, block, path):
    """Where a construct that draws nothing belongs on the timeline.

    Read off the PLAN, in the order the sequencer runs it -- which is the order ``blocks`` is already
    in. Sorting placements by ``(path, index)`` was tried as a proxy and is wrong: a nested placement
    sorts after every top-level one no matter when it runs, so a skipped test at block 20 was marked
    after block 22 when block 21 (nested, and 345 ns earlier) was what actually came next, and the
    tab sat to the right of a construct it precedes.

    Anchored INSIDE the construct's own enclosing pass, in this order:

    1. the start of the first placement in that pass whose block comes after this one -- the instant
       the sequencer moved on, which is where something that took no time belongs;
    2. otherwise the END of the last placement in that pass before it, for a construct that would
       have run at the tail of its pass;
    3. otherwise the next thing the plan draws at all.

    Falling back past the pass boundary is what put execution #1 of a loop *after* execution #2:
    its own pass was empty, so the search ran on into the next pass and marked it there.
    """
    prefix = tuple(path or ())
    after = before = None
    entered = False
    for placement in blocks:
        here = tuple(getattr(placement, "path", ()) or ())
        in_pass = here[:len(prefix)] == prefix
        if in_pass:
            entered = True
            if placement.index > block:
                after = placement.start
                break
            before = placement.stop
        elif entered:
            break                        # the pass ended without reaching this block
    if after is not None:
        return after
    if before is not None:
        return before
    for placement in blocks:             # the pass drew nothing at all
        if placement.index > block:
            return placement.start
    return max((placement.stop for placement in blocks), default=0)


def _execution_tags(summary):
    """``{(block, depth, path): tag}`` for every execution of a multi-execution construct.

    Built once per call instead of scanning each construct's execution list per span. That scan was
    O(spans x executions): pinning an outer loop to N passes gives every inner construct N
    executions AND N spans, so an edit that should cost milliseconds took 475 seconds at N=250.
    """
    out = {}
    for (block, depth), entry in summary.items():
        runs = entry.get("executions") or ()
        if len(runs) < 2:
            continue
        for run in runs:
            path = tuple(run["path"])
            out[(block, depth, path)] = execution_tag(path) or None
    return out


def _count_and_source(trace, summary, block, depth, path):
    """``{"count": n|None, "source": "resolved"|"pinned"|"assumed"}`` for one execution."""
    entry = summary.get((block, depth))
    taken = entry.get("taken") if entry else None
    pinned = trace._count_override(block, depth, tuple(path) if path else None)
    indeterminate = bool(entry.get("indeterminate")) if entry else True
    stuck = bool(entry.get("nonterminating")) if entry else False
    if pinned is not None:
        return {"count": int(pinned), "source": "pinned", "taken": taken,
                "indeterminate": indeterminate, "nonterminating": stuck}
    if entry is None:
        return {"count": None, "source": "assumed", "taken": taken, "indeterminate": True,
                "nonterminating": stuck}
    return {"count": entry["count"], "source": entry["source"], "taken": taken,
            "indeterminate": indeterminate, "nonterminating": stuck}


def branch_caption(trace, context, info):
    """Label for a control-flow region, honest about what is known.

    A ``loop`` has a deterministic count, so the region says which pass it is. A
    ``repeat_until`` does not -- its count depends on a measurement, so one pass is drawn
    and the label says so rather than implying the drawn timeline is complete. A ``test``
    says whether the shown arm was decided from the cache or merely assumed.
    """
    inner = context[-1]
    kind, condition = inner["kind"], inner["condition"]

    # A box spans one EXECUTION of the construct, which for a loop is all of its passes, so it
    # must not claim to bound a single pass -- it says how many it encloses instead. A box that
    # really does cover one pass (count 1, or one execution of a nested construct) still names it.
    passes = info.get("passes", 1)
    if kind == "loop":
        count = inner.get("count")
        total = count if count else "unbounded"
        if passes > 1:
            return f"loop {condition} — {passes} passes of {total}"
        return f"loop {condition} — pass {info['iteration'] + 1} of {total}"
    if kind == "repeat_until":
        # A cache-pointer pulse stream is fully unrolled -- its count is read from the
        # per-point cache -- so say how many gates, not that the count is unknown.
        stream = info.get("stream_count")
        if stream is not None:
            # One pass per cache word, and `bs_repeats` copies of that word per pass -- the copies
            # abut, so they are one logical gate played several times, not several gates. Saying
            # only the gate total would hide exactly the structure that matters here: dualrail_rb
            # issues five copies of a short half-swap so the loop can keep up, and whether those
            # five carry the same word is the thing worth checking on hardware.
            words = int(getattr(trace, "stream_words", 0) or 0)
            copies = int(getattr(trace, "stream_repeats", 1) or 1)
            how = (f"{words} cache words x {copies} copies = {stream} pulses"
                   if copies > 1 else f"{stream} gates")
            return (f"repeat_until({condition}) — {how} from cache (this sweep point); "
                    f"the loop runs {words} passes, all drawn unrolled inside one")
        if info.get("nonterminating") and info.get("source") != "pinned":
            return (f"repeat_until({condition}) — NEVER EXITS: the counter starts at 0 and is "
                    f"incremented before this test is next evaluated, so it cannot reach 0. "
                    f"The board hangs (measured). One pass is drawn.")
        resolved = info.get("repeat_count")
        drawn = int(info.get("passes", 1) or 1)
        plural = "" if drawn == 1 else "es"
        # PINNED: say it is drawn, say it is yours, and say what the run itself implies. The old
        # text fell through to "1 pass shown" no matter what had been pinned, so a box drawn with
        # four passes was captioned as one -- the caption contradicted the picture around it.
        if info.get("source") == "pinned":
            tail = (f"this run's cache says {resolved}" if resolved is not None
                    else "the real count is decided at runtime")
            return (f"repeat_until({condition}) — {drawn} pass{plural} drawn (pinned); {tail}")
        if resolved is not None:
            if drawn > 1:
                return (f"repeat_until({condition}) — {drawn} passes of {resolved} "
                        f"(this sweep point)")
            return (f"repeat_until({condition}) — pass {info['iteration'] + 1} "
                    f"of {resolved} (this sweep point)")
        return (f"repeat_until({condition}) — {drawn} pass{plural} shown; "
                f"real count is data-dependent")
    if kind == "test":
        if info.get("unsupported"):
            # KI_004: body is out of line, so the drawn positions are not trustworthy
            return (f"test({condition}), speculation=False — TIMING NOT MODELLED "
                    f"(see KI_004)")
        decided = "assumed taken" if info.get("assumed") else "condition true"
        speculation = inner.get("speculation")
        suffix = "" if speculation is None else f", speculation={speculation}"
        return f"test({condition}) — {decided}{suffix}"
    return f"{kind}({condition})"


def _stretch_groups(commands):
    """Indices of ARB/CONST_CONT/ARB_CONT triples -- one stretched pulse each."""
    groups = {}
    for i in range(len(commands) - 2):
        a, b, c = commands[i:i + 3]
        if (a.kind == "ARB" and b.kind == "CONST_CONT" and c.kind == "ARB_CONT"
                and a.pulse and a.stop == b.start and b.stop == c.start):
            groups[i] = (a, b, c)
    return groups


def _stretched_time(group, count):
    """Sample times for a stretched pulse: first half, held middle, second half."""
    first, hold, second = group
    half = count // 2
    return np.concatenate([
        np.linspace(first.start, first.stop, max(half, 1)),
        np.linspace(hold.start, hold.stop, 2),
        np.linspace(second.start, second.stop, max(count - half, 1)),
    ]), half


def _stretched_values(values, half):
    """Insert the held middle sample into a value array."""
    held = values[half] if half < len(values) else values[-1]
    return np.concatenate([values[:half], np.full(2, held), values[half:]])


def _reference_peaks(trace, envelope_scale, envelope_source):
    """Peak each envelope is divided by, per ``(io_name, pulse)``.

    ``per-pulse`` fills every bar (best for reading shape, tells you nothing about
    amplitude). ``channel`` normalises to the loudest pulse on that DAC, ``shared``
    to the loudest anywhere, ``absolute`` to DAC full scale -- those three make bar
    heights comparable.
    """
    keys = {(c.io_name, c.pulse) for c in trace.commands if c.pulse}
    peaks, by_channel = {}, {}
    channel_of = {(c.io_name, c.pulse): c.channel for c in trace.commands if c.pulse}
    for key in keys:
        samples = trace.envelope(key[0], key[1], envelope_source)
        peak = float(np.abs(samples).max()) if samples is not None and len(samples) else 0.0
        peaks[key] = peak
        channel = channel_of.get(key)
        by_channel[channel] = max(by_channel.get(channel, 0.0), peak)

    if envelope_scale == "per-pulse":
        return {k: v for k, v in peaks.items()}
    if envelope_scale == "absolute":
        return {k: 1.0 for k in peaks}
    if envelope_scale == "channel":
        return {k: by_channel.get(channel_of.get(k), 0.0) for k in peaks}
    if envelope_scale == "shared":
        overall = max(peaks.values(), default=0.0)
        return {k: overall for k in peaks}
    raise ValueError(f"unknown envelope_scale: {envelope_scale!r}")


def _legend_labels(trace, colors, color_by, group_copies):
    """Human label per color key. Memory keys are named by their pulse, prefixed
    with the channel only when that name is ambiguous across lanes."""
    if color_by != "memory":
        return {k: str(k) for k in colors}

    names, channels = {}, {}
    for cmd in trace.commands:
        if cmd.pulse is None:
            continue
        key = color_key(cmd, color_by, group_copies)
        names.setdefault(key, cmd.pulse)
        channels.setdefault(cmd.pulse, set()).add(cmd.channel)
    return {key: (name if len(channels.get(name, ())) < 2 else f"{key[0]}/{name}")
            for key, name in names.items()}


#: Point sizes the lane labels may take, largest first, before lines start being dropped.
LANE_FONT_STEPS = (9.0, 8.0, 7.0, 6.0)


def lane_label(trace, channel, max_lines=None):
    # One io per line: several ios sharing a physical channel joined with '/' ran wide and
    # ate horizontal plot space, so stack them under the channel name instead.
    ios = trace.channel_ios.get(channel, [])
    lines = [channel, *ios] if ios else [channel]
    if max_lines is not None and len(lines) > max_lines:
        # keep the channel -- it is what the lane IS -- and say that names were dropped rather
        # than silently showing a lane as though it carried fewer ios than it does
        lines = lines[:max(max_lines, 1)]
        lines[-1] = (lines[-1] + " …") if len(lines) > 1 else lines[-1]
    return "\n".join(lines)


def lane_text_layout(ax, span_units, most_lines):
    """Size, line budget and STRIDE for the lane labels, from the room each lane actually has.

    The labels were a fixed 9 pt however many lanes were stacked in however tall a canvas. Ten
    lanes on a 4.5 inch figure gives each about 21 px, and a three-line label needs 36 -- so a
    lane's name ran into its neighbour's and the axis read "readout1_stimulusADC0". Two labels
    overlapping is worse than either being shortened: it is not merely ugly, it names a lane
    something no lane is called.

    ``span_units`` is the axes' whole y range in lane units, which is NOT the lane count: the tab
    strip adds headroom above the lanes, so a nine-lane plot can span seventeen units and each lane
    gets barely half the pixels the lane count implies. Sizing against the count instead put three
    lines where one fits.

    Shrink first (the names stay complete), then drop lines, and past that label only every Nth
    lane. Thinning out is the last step because it is the only one that leaves a lane anonymous --
    but an unlabelled lane is honest about there being no room, while two names drawn on top of
    each other is not. Uses the same pixels-per-lane measure the lane geometry already works in,
    so both degrade together. Returns ``(fontsize, lines, stride)``.
    """
    from math import ceil
    try:
        height_px = max(ax.get_window_extent().height, 1.0)
    except Exception:                                              # noqa: BLE001
        height_px = 400.0
    per_lane = height_px / max(span_units, 1e-9)
    # points -> pixels, or the comparison is between two different units and always says yes: a
    # 3-line 9 pt label is 34 POINTS but 47 pixels at 100 dpi, which is where the overlap came from
    line_px = 1.3 * ax.figure.dpi / 72.0
    # a label is CENTRED on its lane, so one exactly a lane tall already touches both neighbours
    room = per_lane * 0.9
    for size in LANE_FONT_STEPS:
        if room >= most_lines * size * line_px:
            return size, most_lines, 1
    size = LANE_FONT_STEPS[-1]
    lines = int(room // (size * line_px))
    if lines >= 1:
        return size, lines, 1
    return size, 1, max(2, ceil(size * line_px / max(room, 1e-9)))


def _discard(artists):
    """Take a batch of artists off their axes and empty the list."""
    for artist in artists:
        try:
            artist.remove()
        except (ValueError, NotImplementedError):
            pass                        # already gone (a stray ax.clear())
    artists.clear()


def _hide_overflowing_labels(marks, ax):
    """Hide any bar label wider than its bar.

    Called after ``xlim`` is set, so ``transData`` gives the bars' real pixel
    width. A hidden label reads once you zoom in; hovering shows it meanwhile
    (see :class:`~.interactive.SequenceView`). ``marks`` is
    ``[(text_artist, bar_left, bar_right), ...]`` in plotted units.
    """
    if not marks:
        return
    try:
        renderer = ax.figure.canvas.get_renderer()
    except Exception:
        renderer = None                 # estimate from the font metrics instead
    transform = ax.transData
    for artist, left, right in marks:
        try:
            bar_px = abs(transform.transform((right, 0))[0]
                         - transform.transform((left, 0))[0])
            if renderer is not None:
                text_px = artist.get_window_extent(renderer).width
            else:
                text_px = (len(artist.get_text()) * artist.get_fontsize()
                           * 0.6 * ax.figure.dpi / 72.0)
        except Exception:
            continue
        if text_px > bar_px:
            artist.set_visible(False)


def draw(ax, trace, xlim_ns=None, show_envelopes=True, label_pulses=True,
         pulse_label="name",
         show_barriers=True, show_blocks=True, show_gaps=True, show_branches=True,
         show_flow_tabs=True,
         group_copies=True,
         color_by="memory", envelope_source="memory", envelope_scale="per-pulse",
         envelope_mode="magnitude", legend=True, title=None, theme=None,
         fast=False):
    """Draw ``trace`` into ``ax`` for the visible window. Returns the x limits
    actually applied, in the plotted time unit.

    :param color_by: ``"memory"`` (default, one hue per waveform memory),
        ``"name"`` or ``"channel"`` -- see :func:`assign_colors`.
    :param envelope_source: ``"memory"`` (default) draws what was actually loaded
        at this sweep point; ``"config"`` draws the nominal pulse from the yaml.
    :param envelope_scale: ``"per-pulse"`` (default) fills every bar;
        ``"channel"``, ``"shared"`` and ``"absolute"`` make heights comparable so
        amplitude is readable.
    :param envelope_mode: ``"magnitude"`` (default) draws ``|s|``;
        ``"iq"`` draws I and Q about the lane centre, which is the only way to see
        detune, phase and the DRAG quadrature.
    :param pulse_label: what a pulse bar is labelled with -- ``"name"`` (default,
        the pulse name) or ``"length"`` (its duration in ns, matching the gap and
        register labels, so every mark on the timeline reads in time). Set
        ``label_pulses=False`` to drop pulse labels entirely.
    :param fast: draw a *gesture* frame -- the bars and envelopes you are aiming
        with, but no text and no legend. Every label is laid out and rasterised
        individually, so dropping them takes a fifth to a quarter off a frame while
        you scroll or pan; :class:`~.interactive.SequenceView` draws the full frame
        the moment the gesture stops. What is left is matplotlib's own floor --
        rebuilding the axes and rastering it.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Patch, Rectangle

    # Theme colors, bound as locals so the body below reads the same in either mode.
    th = theme or LIGHT_THEME
    INK_PRIMARY = th["ink_primary"]
    INK_SECONDARY = th["ink_secondary"]
    INK_MUTED = th["ink_muted"]
    SURFACE = th["surface"]
    NEUTRAL_FILL = th["neutral_fill"]
    GAP_FILL = th["gap_fill"]
    BRANCH_INK = th["branch_ink"]
    fill_alpha = th["fill_alpha"]

    channels = list(trace.channels)
    if not channels:
        raise ValueError("trace contains no channels")
    if pulse_label not in ("name", "length"):
        raise ValueError(f"unknown pulse_label: {pulse_label!r} "
                         f"(expected 'name' or 'length')")

    limits_ns = tuple(xlim_ns) if xlim_ns else (0.0, max(trace.length_ns, 1.0))
    span_ns = max(limits_ns[1] - limits_ns[0], 1e-6)
    divisor, unit = _time_scale(span_ns)
    ns = trace.ns_per_cycle / divisor          # cycles -> plotted time unit
    span = span_ns / divisor
    # separator between fills that actually touch: a true pixel width, so it does
    # not eat a percentage of the pulse and does not change as you zoom
    gap = _pixels_in_cycles(ax, SEPARATOR_PIXELS, span, ns)
    try:
        axes_px = max(ax.get_window_extent().width, 1.0)   # for envelope downsampling
    except Exception:
        axes_px = 800.0
    min_label = 0.022 * span
    t0, t1 = limits_ns[0] / divisor, limits_ns[1] / divisor

    lane = {ch: len(channels) - 1 - i for i, ch in enumerate(channels)}
    styles = assign_styles(trace, color_by, group_copies, series=th["series"])
    peaks = (_reference_peaks(trace, envelope_scale, envelope_source)
             if show_envelopes else {})

    visible = [c for c in trace.commands
               if c.stop * ns >= t0 and c.start * ns <= t1]
    detailed = len(visible) <= DETAIL_BUDGET
    # every label on the timeline, gated together: dropped when the window is too
    # dense to read them, and while a gesture is in flight (see ``fast``)
    labelled = detailed and not fast

    # Replace what the last call drew, rather than clearing the axes.
    # ``ax.clear()`` throws the ticks, their labels and the spines away and
    # rebuilds them from rcParams every frame -- 75 new artists and thousands of
    # deepcopies for a window that owns nine bars, which measured at three
    # quarters of the cost of a gesture frame. So only the marks this function
    # draws are replaced; the chrome is rebuilt when it would actually change.
    # A gesture frame drops the labels, as it always did -- they are per-window and
    # go stale the moment you move -- but leaves the legend up. Measured on a
    # 9-lane RB sequence: 48.5 ms for a frame with neither, 53.1 with the legend,
    # 62.4 with both. The legend is the most conspicuous thing on the plot, so 4.6
    # ms is a good price for it not to blink on and off through a drag.
    state = getattr(ax, "_sequence_viz", None)
    if state is None:
        ax.clear()                     # first draw on this axes, or someone else's
        state = ax._sequence_viz = {"marks": [], "labels": [], "chrome": None}
    _discard(state["marks"])
    _discard(state["labels"])

    def mark(artist):
        state["marks"].append(artist)
        return artist

    def label(artist):
        state["labels"].append(artist)
        return artist

    used_pulses = set()
    drew_capture = drew_padding = drew_symbolic = drew_gap = drew_branch = False
    drew_dwell = False
    label_marks = []       # (text artist, bar left, bar right) in plotted units
    # envelope polylines are collected and drawn as one LineCollection each,
    # rather than one ax.plot() per pulse -- the dominant zoom-render cost
    env_mag, env_iq_real, env_iq_imag = [], [], []
    pulse_rects = []       # solid pulse fills, batched into one PatchCollection
    flow_frames = []       # (Rectangle, info) per control-flow span; published on the axes below
    tabbed = set()         # construct executions that already have a handle drawn

    # one outline per control-flow region, not per block: a repeat_until wraps
    # every block in its body, and per-block captions collide
    # ``show_branches`` decides whether control flow is drawn at all; ``show_flow_tabs`` decides
    # only whether its TABS are. They are separate because they answer different questions: the
    # dashed boxes say what the sequence's structure is, while the tabs are the handles for editing
    # it, and on a sequence with many constructs the strip of handles can be more clutter than help
    # when you are reading pulses rather than changing counts. Hiding the tabs also gives their
    # vertical strip back to the lanes.
    flow_regions = branch_regions(trace) if show_branches else []
    # which blocks key several constructs, so their tabs carry the depth that tells them apart
    _ambiguous = ambiguous_blocks(flow_regions)
    # one plotted unit per pixel, so both the minimum tab width and the per-character width are
    # expressed in pixels and stay constant on screen at any zoom
    per_px = _pixels_in_cycles(ax, 1.0, span, ns) * ns
    # Character width DERIVED from the font size and the figure's dpi, the same way the caption
    # width is computed a few lines below (fontsize * 0.6 * dpi / 72). A hand-picked 4.3 px was
    # tried and was an under-estimate at the GUI's dpi, which let labels overflow their tabs and
    # collide again -- the exact failure the label-driven width exists to prevent. Deriving it
    # means the tab fits its text at any font size or display scaling.
    char_px = TAB_FONTSIZE * 0.6 * ax.figure.dpi / 72.0
    tab_sizes = tab_widths(flow_regions, TAB_PIXELS_W * per_px, char_px * per_px)
    tab_rows = assign_tab_rows(flow_regions, tab_sizes, ns)
    lane_px = _lanes_per_pixel(ax, len(channels))
    for start, stop, context, info in flow_regions:
        r0, r1 = start * ns, stop * ns
        if r1 < t0 or r0 > t1:
            continue
        drew_branch = True
        # NESTING IS DRAWN, not just implied. Every control-flow box used to be the same full
        # height, so a cooling round inside a mode loop inside a sweep drew three boxes on top of
        # one another and you could not see which contained which -- exactly the question you ask
        # when a nested sequence looks wrong. Each level is now inset vertically, so an outer
        # construct visibly encloses the ones inside it, like nested brackets. The inset is small
        # and capped so a deep nest cannot collapse the innermost box to nothing.
        level = max(int(info.get("depth", 1)) - 1, 0)
        box_y, box_h = control_flow_box(info.get("depth", 1), len(channels))
        elided = bool(info.get("elided"))
        if elided:
            # Nothing is drawn for this construct -- 0 passes, or a test on its skipped arm -- so
            # there is no span to enclose. A zero-width invisible frame keeps the hover and
            # hit-test paths byte-identical to every other construct's; only the tab is visible.
            # Drawing the usual rectangle here would invent a box a gap wide out of an empty span.
            frame = Rectangle((r0, box_y), 0.0, box_h, facecolor="none", edgecolor="none",
                              zorder=1.5)
            frame.set_visible(False)
        else:
            frame = Rectangle(
                (r0, box_y), max(r1 - r0, gap * ns), box_h,
                facecolor="none", edgecolor=BRANCH_INK, linestyle=(0, (4, 3)),
                # deeper levels draw a little lighter, so the outermost frame stays the strongest
                linewidth=max(1.6 - 0.18 * level, 0.8),
                alpha=max(1.0 - 0.13 * level, 0.45), zorder=1.5)
        mark(ax.add_artist(frame))
        # Publish the frame together with the construct it belongs to. A UI needs both: the
        # geometry, to hit-test the tab and to highlight the span, and the block index, to know
        # which construct that tab edits. Recomputing either would duplicate the depth/inset
        # arithmetic above and drift from it.
        flow_frames.append((frame, info))

        # THE TAB: a small solid handle at the block's top-left. Pointing at it highlights the
        # block; clicking it edits that construct's iteration count or test arm. It exists
        # because the whole span is not a usable target -- the plot area belongs to the
        # box-zoom gesture, so clicking a span to edit it also dragged out a zoom rectangle.
        # A tab is a small, unambiguous hit area that the viewport interaction can yield to.
        tab_span = tab_sizes.get(id(info), TAB_PIXELS_W * per_px)
        tab_x, tab_y, tab_w, tab_h = control_flow_tab_at_row(
            r0, tab_rows.get(id(info), level), len(channels), tab_span, lane_px, lane_px)
        # ONE tab per construct execution. A body can be split into several spans (blocks that
        # are not consecutive), and drawing a tab on each gave the same construct two or three
        # handles that all did the same thing -- indistinguishable from three different
        # constructs, which is precisely the confusion being reported. The frames still draw for
        # every span; only the handle is deduplicated, on the earliest one.
        tab_key = (info.get("block"), info.get("depth"), tuple(info.get("path", ()) or ()))
        if tab_key in tabbed or not show_flow_tabs:
            # tab_rect is what a UI hit-tests, so leaving it None when the tabs are hidden means a
            # press lands on the viewport as it would anywhere else -- no invisible click targets
            info["tab_rect"] = None
            continue
        tabbed.add(tab_key)
        tab_ink = TAB_DEPTH_INK[min(level, len(TAB_DEPTH_INK) - 1)]
        # HOLLOW when the construct is not drawn, so a handle for something absent cannot be
        # mistaken for a handle on something present -- it is still the same size and still
        # clickable, which is the whole point of keeping it.
        # A hollow tab is drawn straight onto the axes, so its ink has to read against the
        # background rather than against a fill of its own.
        page = ax.get_facecolor()
        # A hollow tab is drawn straight onto the axes, so its ink reads against the page; a solid
        # one carries its own fill, and fill and label have to be legible as a pair.
        if elided:
            tab_ink, tab_text_ink = legible_ink(tab_ink, page), None
        else:
            tab_ink, tab_text_ink = legible_tab(tab_ink)
        mark(ax.add_artist(Rectangle(
            (tab_x, tab_y), tab_w, tab_h,
            facecolor="none" if elided else tab_ink, edgecolor=tab_ink,
            linewidth=0.9 if elided else 0.0, linestyle=(0, (2, 1.4)) if elided else "solid",
            alpha=0.95, zorder=3.5)))
        # A faint leader from the tab down to the box it governs. Without it a stack of tabs is
        # just a stack: the reader can see there are three constructs but not which frame each
        # one belongs to, which is the only thing they actually need to know before editing one.
        if not elided:                       # nothing below to point at
            mark(ax.plot([tab_x + tab_w / 2, tab_x + tab_w / 2],
                         [tab_y, box_y + box_h], color=tab_ink, linewidth=0.8,
                         alpha=0.45, zorder=3.4, solid_capstyle="butt")[0])
        # The drawn rect, not a recomputation: the GUI hit-tests exactly what is on screen, so a
        # row bump or a geometry change cannot leave the clickable area somewhere else.
        info["tab_rect"] = (tab_x, tab_y, tab_w, tab_h)
        if labelled:
            # Labelled with the SAME identifier the control-flow panel lists ("@8"), so a tab on
            # the plot and a row in the panel are obviously the same construct. Without it the
            # two views shared no vocabulary: the panel said "repeat_until @8" and the tab said
            # "x3", leaving the reader to match them by position.
            tag = tab_label(info, info.get("block") in _ambiguous)
            # mark() so this text is REMOVED on the next render, like every other artist here.
            # Unmarked, each zoom or pan left its labels behind and the next frame drew over them:
            # zoom in, zoom out, and the tabs read "@5x1x1" -- several frames' labels stacked in
            # one place. The rectangles were always marked, which is why only the text garbled.
            mark(ax.text(tab_x + tab_w / 2, tab_y + tab_h / 2, tag, ha="center", va="center",
                         fontsize=TAB_FONTSIZE,
                         # chosen for contrast against whatever the text sits on: the tab's own
                         # fill when solid, the page when hollow
                         color=tab_ink if elided else tab_text_ink,
                         zorder=3.6, clip_on=True))
        if labelled and not elided:      # a zero-width span has nothing to caption
            caption = branch_caption(trace, context, info)
            vis0, vis1 = max(r0, t0), min(r1, t1)      # visible part of the region
            char_px = 7 * 0.6 * ax.figure.dpi / 72.0
            # label the box only where the *visible* part is wide enough to hold the
            # caption (it reads on hover otherwise); anchor to the visible left edge
            # so it stays on screen when zoomed/panned inside the region, just above
            # the top stroke (top is len(channels) - 0.67)
            if (vis1 - vis0) / span * axes_px >= len(caption) * char_px:
                label(ax.text(vis0 + 0.004 * span, len(channels) - 0.65, caption,
                              ha="left", va="bottom", fontsize=7, color=BRANCH_INK,
                              zorder=6, clip_on=True))

    for blk in (trace.placements or trace.blocks):
        if blk.stop * ns < t0 or blk.start * ns > t1:
            continue

        if show_gaps and blk.gap_after:
            g0, g1 = blk.stop * ns, (blk.stop + blk.gap_after) * ns
            if g1 >= t0 and g0 <= t1:
                mark(ax.axvspan(g0, g1, facecolor=GAP_FILL, alpha=0.22,
                                linewidth=0, zorder=1))
                drew_gap = True
                if labelled and (g1 - g0) >= min_label * 1.5:
                    # sit above the control-flow caption (at len(channels) - 0.65)
                    # so the two never crowd where a gap meets a branch edge
                    label(ax.text((g0 + g1) / 2, len(channels) - 0.45,
                                  f"{blk.gap_after * trace.ns_per_cycle:.0f} ns",
                                  ha="center", va="bottom", fontsize=7,
                                  color=INK_SECONDARY, zorder=5, clip_on=True))

        by_channel = {}
        for c in blk.commands:
            by_channel.setdefault(c.channel, []).append(c)

        for ch, cmds in by_channel.items():
            y = lane[ch]
            stretched = _stretch_groups(cmds)
            starts = {c.start for c in cmds}
            skip = set()
            for i, cmd in enumerate(cmds):
                if i in skip:
                    continue

                group = stretched.get(i)
                if group:
                    skip.update({i + 1, i + 2})
                    head, x0, x1 = group[0], group[0].start, group[2].stop
                    kind = "ARB"
                else:
                    head, x0, x1 = cmd, cmd.start, cmd.stop
                    kind = cmd.kind
                if x1 * ns < t0 or x0 * ns > t1:
                    continue

                pulse = head.pulse
                key = color_key(head, color_by, group_copies) if pulse else None
                is_capture = kind == "CONST_CONT" and ch.startswith("ADC")
                # only inset where something actually butts up against this bar,
                # so a lone pulse is drawn to its full length
                width = (max(x1 - x0 - gap, (x1 - x0) * 0.5) if x1 in starts
                         else x1 - x0)

                if pulse:
                    style = styles[key]
                    face, hatch, alpha = style["color"], None, fill_alpha
                    # a wrapped palette slot is marked, not recolored
                    edge = INK_PRIMARY if style["generation"] else "none"
                    used_pulses.add(key)
                elif is_capture:
                    face, edge, hatch, alpha = SURFACE, INK_SECONDARY, None, 1.0
                    drew_capture = True
                elif head.symbolic and head.resolution == "fallback":
                    # a register/DSP length that could not be recovered from the cache:
                    # drawn indeterminate -- cross-hatched, clamped to a visible width
                    face, edge, hatch, alpha = SURFACE, INK_MUTED, "xx", 1.0
                    width = max(width, 0.02 * span / ns)
                    drew_symbolic = True
                elif head.symbolic:
                    # a register/DSP length resolved from the per-point cache: its width
                    # is known, so it reads as an ordinary dwell -- the register label on
                    # top says where it came from
                    face, edge, hatch, alpha = NEUTRAL_FILL, "none", None, 0.75
                    drew_dwell = True
                elif head.is_padding:
                    # hatch renders in the edge color, so padding needs one
                    face, edge, hatch, alpha = NEUTRAL_FILL, INK_MUTED, "///", 0.35
                    drew_padding = True
                else:
                    face, edge, hatch, alpha = NEUTRAL_FILL, "none", None, 0.75
                    drew_dwell = True      # a dwell you scheduled (plain grey)

                lw = 0.0 if edge == "none" else (0.5 if hatch else 1.0)
                rect = Rectangle(
                    (x0 * ns, y - LANE_HEIGHT / 2), width * ns, LANE_HEIGHT,
                    facecolor=face, edgecolor=edge, hatch=hatch, alpha=alpha,
                    linewidth=lw, zorder=2)
                if pulse and hatch is None:   # plain pulse fill -> batch it
                    pulse_rects.append(rect)
                else:                          # capture/padding/symbolic/dwell
                    mark(ax.add_artist(rect))

                if show_envelopes and detailed and pulse and not head.symbolic:
                    bar_px = (x1 - x0) * ns / span * axes_px
                    samples = trace.envelope(head.io_name, pulse, envelope_source)
                    reference = peaks.get((head.io_name, pulse), 0.0)
                    if (bar_px >= ENVELOPE_MIN_PX and samples is not None
                            and len(samples) and reference > 0):
                        # downsample to ~2 points per on-screen pixel (cheap, keeps
                        # detail when zoomed in)
                        target = int(min(len(samples), max(2.0 * bar_px, 8)))
                        if target < len(samples):
                            keep = np.linspace(0, len(samples) - 1, target).astype(int)
                            samples = samples[keep]
                        if group:
                            times, half = _stretched_time(group, len(samples))
                            pick = lambda v: _stretched_values(v, half)
                        else:
                            # match the fill, so the envelope never overhangs it
                            times = np.linspace(x0, x0 + width, len(samples))
                            pick = lambda v: v

                        xs = times * ns
                        if envelope_mode == "iq":
                            # about the lane centre, so sign and phase are visible
                            env_iq_real.append(np.column_stack(
                                [xs, y + pick(samples.real) / reference
                                 * LANE_HEIGHT * 0.46]))
                            env_iq_imag.append(np.column_stack(
                                [xs, y + pick(samples.imag) / reference
                                 * LANE_HEIGHT * 0.46]))
                        elif envelope_mode == "magnitude":
                            env_mag.append(np.column_stack(
                                [xs, y - LANE_HEIGHT / 2
                                 + pick(np.abs(samples)) / reference
                                 * LANE_HEIGHT * 0.92]))
                        else:
                            raise ValueError(
                                f"unknown envelope_mode: {envelope_mode!r}")

                if labelled:
                    if head.symbolic:
                        text = trace.register_label(head.symbolic)
                    elif label_pulses and pulse:
                        # "length" makes pulse labels read in time like the gap and
                        # register labels; "name" keeps the pulse name
                        text = (f"{(x1 - x0) * trace.ns_per_cycle:.0f} ns"
                                if pulse_label == "length" else pulse)
                    elif is_capture:
                        text = (f"{(x1 - x0) * trace.ns_per_cycle:.0f} ns"
                                if pulse_label == "length" else "capture")
                    elif head.kind == "DWELL" and not head.is_padding:
                        # a scheduled dwell has only a duration -- always label it in ns
                        text = f"{(x1 - x0) * trace.ns_per_cycle:.0f} ns"
                    else:
                        text = None
                    if text and (x1 - x0) * ns >= min_label:
                        # a register label sits over an "xx" hatch -- put it on a
                        # small surface box so it stays readable
                        bbox = (dict(facecolor=SURFACE, edgecolor="none", pad=1.5,
                                     alpha=0.9) if head.symbolic else None)
                        artist = label(ax.text(
                            (x0 + (x1 - x0) / 2) * ns, y, text,
                            ha="center", va="center", fontsize=7.5,
                            color=INK_PRIMARY, zorder=4, clip_on=True, bbox=bbox))
                        # remember the drawn bar extent so a label wider than its
                        # bar can be hidden once xlim is set (hover shows it then)
                        label_marks.append((artist, x0 * ns, (x0 + width) * ns))

        if show_barriers:
            for t in blk.barriers:
                mark(ax.axvline(t * ns, color=INK_MUTED, linestyle=":",
                                linewidth=1.0, zorder=1))
        if show_blocks:
            mark(ax.axvline(blk.start * ns, color=INK_MUTED, linewidth=0.6,
                            alpha=0.35, zorder=0))

    # batch the solid pulse fills into one collection (per-patch colour/alpha
    # preserved via match_original) instead of one add_patch() per pulse
    if pulse_rects:
        from matplotlib.collections import PatchCollection
        mark(ax.add_collection(
            PatchCollection(pulse_rects, match_original=True, zorder=2)))

    # one LineCollection per envelope style instead of one plot() per pulse
    if env_mag:
        mark(ax.add_collection(LineCollection(
            env_mag, colors=INK_PRIMARY, linewidths=ENVELOPE_LINEWIDTH,
            alpha=0.7, zorder=3)))
    if env_iq_real:
        mark(ax.add_collection(LineCollection(
            env_iq_real, colors=INK_PRIMARY, linewidths=ENVELOPE_LINEWIDTH * 0.75,
            alpha=0.75, zorder=3)))
    if env_iq_imag:
        mark(ax.add_collection(LineCollection(
            env_iq_imag, colors=INK_PRIMARY, linewidths=ENVELOPE_LINEWIDTH * 0.75,
            alpha=0.45, zorder=3)))

    # the lane labels, spines, grid and colours only change when the channels or the
    # theme do -- and re-setting them is what makes matplotlib rebuild every tick
    chrome = (tuple(channels), INK_PRIMARY, INK_SECONDARY, INK_MUTED,
              th["axes_bg"], th["figure_bg"])
    if state["chrome"] != chrome:
        state["chrome"] = chrome
        ax.set_yticks(range(len(channels)))
        ax.grid(axis="x", color=INK_MUTED, alpha=0.18, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK_MUTED)
        # Per axis, because a blanket tick_params overrode the lane labels set two lines above:
        # they read as INK_PRIMARY at the chosen size in the source and drew as INK_SECONDARY at 8
        # pt on screen, so the size chosen for the room each lane has had no effect at all.
        ax.tick_params(axis="x", colors=INK_SECONDARY, labelsize=8)
        ax.tick_params(axis="y", color=INK_SECONDARY, labelcolor=INK_PRIMARY)
        ax.set_facecolor(th["axes_bg"])
        ax.figure.set_facecolor(th["figure_bg"])
    # Headroom for the tab strip above the lanes (see control_flow_tab). Taken from the frames
    # already computed for this draw, so the deepest nesting decides how tall the strip is.
    nesting = max((info.get("depth", 1) for _frame, info in flow_frames), default=0)
    max_row = max(tab_rows.values(), default=0) + 1 if tab_rows else 0
    strip_top = (tab_strip_top(max(nesting, max_row), len(channels), lane_px)
                 if nesting and show_flow_tabs else len(channels) - 0.3)
    # Published so the viewport's scroll range can reach the tabs without recomputing the row
    # assignment. A depth-based estimate is not enough: tabs are bumped to a free row when they
    # would collide, so the highest row can sit above what the nesting alone implies -- and a
    # scrollbar that stops short of a control is a control you cannot reach.
    ax._seqviz_strip_top = strip_top
    ax.set_ylim(-0.7, strip_top)
    # Only now is the y span known, and the span is what decides how much room a lane label has --
    # so the labels are written here rather than with the rest of the chrome above.
    most_lines = max((len(trace.channel_ios.get(ch, [])) + 1 for ch in channels), default=1)
    lanes = lane_text_layout(ax, strip_top + 0.7, most_lines)
    if state.get("lanes") != (tuple(channels), INK_PRIMARY, lanes):
        state["lanes"] = (tuple(channels), INK_PRIMARY, lanes)
        lane_size, lane_lines, lane_stride = lanes
        ax.set_yticklabels([lane_label(trace, ch, lane_lines) if index % lane_stride == 0 else ""
                            for index, ch in enumerate(reversed(channels))],
                           fontsize=lane_size, color=INK_PRIMARY)
    # Attached to the axes rather than returned, so every existing caller of draw() is unaffected
    # and a UI can find the spans after the fact.
    ax._seqviz_flow_frames = flow_frames
    ax.set_xlim(t0, t1)
    if ax.get_xlabel() != f"time [{unit}]":         # ns <-> µs as you zoom
        ax.set_xlabel(f"time [{unit}]", fontsize=9, color=INK_SECONDARY)

    _hide_overflowing_labels(label_marks, ax)

    if legend and not fast:
        labels = _legend_labels(trace, styles, color_by, group_copies)
        handles = [Patch(facecolor=styles[k]["color"], label=labels[k],
                         alpha=fill_alpha,
                         edgecolor=INK_PRIMARY if styles[k]["generation"] else "none",
                         linewidth=1.0 if styles[k]["generation"] else 0)
                   for k in styles if k in used_pulses]
        if drew_capture:
            handles.append(Patch(facecolor=SURFACE, edgecolor=INK_SECONDARY,
                                 label="ADC capture"))
        if drew_padding:
            handles.append(Patch(facecolor=NEUTRAL_FILL, edgecolor=INK_MUTED,
                                 alpha=0.35, hatch="///", label="barrier padding"))
        if drew_dwell:
            handles.append(Patch(facecolor=NEUTRAL_FILL, alpha=0.75,
                                 label="dwell (scheduled)"))
        if drew_symbolic:
            handles.append(Patch(facecolor=SURFACE, edgecolor=INK_MUTED, hatch="xx",
                                 label="indeterminate (register)"))
        if drew_gap:
            handles.append(Patch(facecolor=GAP_FILL, alpha=0.22,
                                 label="inter-block gap"))
        if drew_branch:
            handles.append(Patch(facecolor="none", edgecolor=BRANCH_INK,
                                 linestyle=(0, (4, 3)),
                                 label="control flow (see label)"))
        size = (legend_fontsize(ax.figure, [h.get_label() for h in handles])
                if len(handles) >= 2 else None)
        if size is not None:
            ax.legend(handles=handles, loc="upper left",
                      bbox_to_anchor=(1.005, 1.0), frameon=False, fontsize=size,
                      labelcolor=INK_PRIMARY)
        elif ax.get_legend() is not None:
            ax.get_legend().remove()        # nothing left to key: no stale legend
    elif not legend and ax.get_legend() is not None:
        ax.get_legend().remove()            # legend switched off (a fast frame keeps it)

    text = (None if title is False else
            title or f"{trace.runtime_class} - sweep point {trace.point}")
    if text is None:
        ax.set_title("", loc="left")
    elif ax.get_title(loc="left") != text:
        ax.set_title(text, fontsize=10, color=INK_PRIMARY, loc="left")
    return t0, t1


def plot_trace(trace, ax=None, figsize=None, **draw_kwargs):
    """One-shot static figure. Returns ``(fig, ax)``.

    ``draw_kwargs`` are passed to :func:`draw` -- notably ``xlim_ns=(t0, t1)`` to
    render a window, and ``group_copies=False`` to color duplicates separately.
    For drag-box zooming use :func:`~.interactive.interactive_view` instead.
    """
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (13, 1.0 + 0.62 * len(trace.channels))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    draw(ax, trace, **draw_kwargs)
    fit_layout(fig, ax)
    return fig, ax
