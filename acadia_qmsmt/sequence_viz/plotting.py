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


def fit_layout(fig, ax):
    """Make room for the axis labels and for the legend outside the axes.

    ``tight_layout`` does not account for a legend anchored outside the axes, and
    an interactive canvas gets no ``bbox_inches="tight"`` expansion to paper over
    it -- so the legend and the lane labels are clipped unless the margin is
    reserved explicitly.
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
        needed = legend.get_window_extent(renderer).width / fig.get_window_extent().width
    except Exception:
        return
    right = 1.0 - needed - 0.02
    if 0.3 < right < fig.subplotpars.right:
        fig.subplots_adjust(right=right)


def branch_regions(trace):
    """Contiguous spans of blocks sharing one control-flow context.

    One region per *pass*: an unrolled loop's iterations do not merge, because each
    iteration is a separate placement of the same block index rather than a run of
    consecutive indices.

    :return: ``[(start_cycles, stop_cycles, context, info), ...]`` where ``info`` carries
        the iteration number so the caption can say which pass this is.
    """
    regions, previous = [], None
    for block in (trace.placements or trace.blocks):
        context = getattr(block, "conditional", ())
        if not context:
            previous = None
            continue
        iteration = getattr(block, "iteration", 0)
        # adjacency is by block index, not time -- consecutive blocks in one body are
        # separated by the inter-block gap
        if (regions and previous == block.index - 1 and regions[-1][2] == context
                and regions[-1][3]["iteration"] == iteration):
            regions[-1][1] = block.stop
        else:
            regions.append([block.start, block.stop, context,
                            {"iteration": iteration,
                             "assumed": block.index in trace.assumed_paths,
                             "unsupported": block.index in getattr(
                                 trace, "unsupported_paths", ()),
                             # an unrolled cache-pointer stream: gate count is resolved,
                             # so the caption states it instead of "count data-dependent"
                             "stream_count": (len(block.commands)
                                              if getattr(block, "stream", False)
                                              else None),
                             # a repeat_until whose count was resolved from its condition
                             # register (see repeat_until_count) -- caption states it too
                             "repeat_count": getattr(
                                 trace, "repeat_counts", {}).get(block.index)}])
        previous = block.index
    return [(a, b, c, d) for a, b, c, d in regions]


def branch_caption(trace, context, info):
    """Label for a control-flow region, honest about what is known.

    A ``loop`` has a deterministic count, so the region says which pass it is. A
    ``repeat_until`` does not -- its count depends on a measurement, so one pass is drawn
    and the label says so rather than implying the drawn timeline is complete. A ``test``
    says whether the shown arm was decided from the cache or merely assumed.
    """
    inner = context[-1]
    kind, condition = inner["kind"], inner["condition"]

    if kind == "loop":
        count = inner.get("count")
        return (f"loop {condition} — pass {info['iteration'] + 1} of "
                f"{count if count else 'unbounded'}")
    if kind == "repeat_until":
        # A cache-pointer pulse stream is fully unrolled -- its count is read from the
        # per-point cache -- so say how many gates, not that the count is unknown.
        count = info.get("stream_count")
        if count is not None:
            return (f"repeat_until({condition}) — {count} gates from cache "
                    f"(this sweep point)")
        # A counter loop whose target register resolved: draw and label the real count.
        count = info.get("repeat_count")
        if count is not None:
            return (f"repeat_until({condition}) — pass {info['iteration'] + 1} "
                    f"of {count} (this sweep point)")
        return (f"repeat_until({condition}) — 1 pass shown; "
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


def lane_label(trace, channel):
    # One io per line: several ios sharing a physical channel joined with '/' ran wide and
    # ate horizontal plot space, so stack them under the channel name instead.
    ios = trace.channel_ios.get(channel, [])
    return "\n".join([channel, *ios]) if ios else channel


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
         group_copies=True,
         color_by="memory", envelope_source="memory", envelope_scale="per-pulse",
         envelope_mode="magnitude", legend=True, title=None, theme=None):
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

    ax.clear()
    used_pulses = set()
    drew_capture = drew_padding = drew_symbolic = drew_gap = drew_branch = False
    drew_dwell = False
    label_marks = []       # (text artist, bar left, bar right) in plotted units
    # envelope polylines are collected and drawn as one LineCollection each,
    # rather than one ax.plot() per pulse -- the dominant zoom-render cost
    env_mag, env_iq_real, env_iq_imag = [], [], []
    pulse_rects = []       # solid pulse fills, batched into one PatchCollection

    # one outline per control-flow region, not per block: a repeat_until wraps
    # every block in its body, and per-block captions collide
    for start, stop, context, info in (branch_regions(trace) if show_branches else []):
        r0, r1 = start * ns, stop * ns
        if r1 < t0 or r0 > t1:
            continue
        drew_branch = True
        ax.add_patch(Rectangle(
            (r0, -0.62), max(r1 - r0, gap * ns), len(channels) - 0.05,
            facecolor="none", edgecolor=BRANCH_INK, linestyle=(0, (4, 3)),
            linewidth=1.6, zorder=1.5))
        if detailed:
            caption = branch_caption(trace, context, info)
            vis0, vis1 = max(r0, t0), min(r1, t1)      # visible part of the region
            char_px = 7 * 0.6 * ax.figure.dpi / 72.0
            # label the box only where the *visible* part is wide enough to hold the
            # caption (it reads on hover otherwise); anchor to the visible left edge
            # so it stays on screen when zoomed/panned inside the region, just above
            # the top stroke (top is len(channels) - 0.67)
            if (vis1 - vis0) / span * axes_px >= len(caption) * char_px:
                ax.text(vis0 + 0.004 * span, len(channels) - 0.65, caption,
                        ha="left", va="bottom", fontsize=7, color=BRANCH_INK,
                        zorder=6, clip_on=True)

    for blk in (trace.placements or trace.blocks):
        if blk.stop * ns < t0 or blk.start * ns > t1:
            continue

        if show_gaps and blk.gap_after:
            g0, g1 = blk.stop * ns, (blk.stop + blk.gap_after) * ns
            if g1 >= t0 and g0 <= t1:
                ax.axvspan(g0, g1, facecolor=GAP_FILL, alpha=0.22, linewidth=0,
                           zorder=1)
                drew_gap = True
                if detailed and (g1 - g0) >= min_label * 1.5:
                    # sit above the control-flow caption (at len(channels) - 0.65)
                    # so the two never crowd where a gap meets a branch edge
                    ax.text((g0 + g1) / 2, len(channels) - 0.45,
                            f"{blk.gap_after * trace.ns_per_cycle:.0f} ns",
                            ha="center", va="bottom", fontsize=7,
                            color=INK_SECONDARY, zorder=5, clip_on=True)

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
                    ax.add_patch(rect)

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

                if detailed:
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
                        artist = ax.text((x0 + (x1 - x0) / 2) * ns, y, text,
                                         ha="center", va="center", fontsize=7.5,
                                         color=INK_PRIMARY, zorder=4, clip_on=True,
                                         bbox=bbox)
                        # remember the drawn bar extent so a label wider than its
                        # bar can be hidden once xlim is set (hover shows it then)
                        label_marks.append((artist, x0 * ns, (x0 + width) * ns))

        if show_barriers:
            for t in blk.barriers:
                ax.axvline(t * ns, color=INK_MUTED, linestyle=":", linewidth=1.0,
                           zorder=1)
        if show_blocks:
            ax.axvline(blk.start * ns, color=INK_MUTED, linewidth=0.6, alpha=0.35,
                       zorder=0)

    # batch the solid pulse fills into one collection (per-patch colour/alpha
    # preserved via match_original) instead of one add_patch() per pulse
    if pulse_rects:
        from matplotlib.collections import PatchCollection
        ax.add_collection(PatchCollection(pulse_rects, match_original=True, zorder=2))

    # one LineCollection per envelope style instead of one plot() per pulse
    if env_mag:
        ax.add_collection(LineCollection(
            env_mag, colors=INK_PRIMARY, linewidths=ENVELOPE_LINEWIDTH,
            alpha=0.7, zorder=3))
    if env_iq_real:
        ax.add_collection(LineCollection(
            env_iq_real, colors=INK_PRIMARY, linewidths=ENVELOPE_LINEWIDTH * 0.75,
            alpha=0.75, zorder=3))
    if env_iq_imag:
        ax.add_collection(LineCollection(
            env_iq_imag, colors=INK_PRIMARY, linewidths=ENVELOPE_LINEWIDTH * 0.75,
            alpha=0.45, zorder=3))

    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels([lane_label(trace, ch) for ch in reversed(channels)],
                       fontsize=9, color=INK_PRIMARY)
    ax.set_ylim(-0.7, len(channels) - 0.3)
    ax.set_xlim(t0, t1)
    ax.set_xlabel(f"time [{unit}]", fontsize=9, color=INK_SECONDARY)
    ax.grid(axis="x", color=INK_MUTED, alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.set_facecolor(th["axes_bg"])
    ax.figure.set_facecolor(th["figure_bg"])

    _hide_overflowing_labels(label_marks, ax)

    if legend:
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
        if len(handles) >= 2:
            ax.legend(handles=handles, loc="upper left",
                      bbox_to_anchor=(1.005, 1.0), frameon=False, fontsize=8,
                      labelcolor=INK_PRIMARY)

    if title is not False:
        ax.set_title(title or f"{trace.runtime_class} - sweep point {trace.point}",
                     fontsize=10, color=INK_PRIMARY, loc="left")
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
