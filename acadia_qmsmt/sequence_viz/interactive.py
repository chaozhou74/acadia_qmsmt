"""
Interactive viewport for a traced sequence: drag a box to zoom.

The interaction lives here rather than in a GUI because it is pure matplotlib
event handling -- it works identically on an ipympl canvas in a notebook and on a
``FigureCanvasQTAgg`` inside acadia_gui. The GUI supplies the window, the file
picker and the toolbar; this class supplies the behaviour, so there is one
implementation to keep correct.

Controls
    drag             box-zoom to that time window and lane range
    double-click     reset to the full sequence
    r                reset
    scroll           zoom the time axis about the cursor
    shift+drag       pan

Re-rendering on every viewport change is what makes zooming useful: label
density, the ns/µs unit and the envelope overlay are all chosen for the window
currently in view, and marks outside it are culled.
"""
import time

from .plotting import draw, fit_layout


class SequenceView:
    """Own an axes and keep it showing ``trace`` at the current viewport."""

    MIN_SPAN_NS = 2.0

    # Half a lane height (LANE_HEIGHT / 2), for deciding whether the cursor is on
    # a pulse bar when hovering.
    HOVER_HALF_HEIGHT = 0.31

    # Some backends (ipympl among them) never set MouseEvent.dblclick, so the
    # double click is also detected by timing. Compared in pixels, not data
    # coordinates -- a reset changes the data scale underfoot.
    DOUBLE_CLICK_SECONDS = 0.4
    DOUBLE_CLICK_PIXELS = 8

    def __init__(self, trace, ax, on_viewport=None, **draw_kwargs):
        self.trace = trace
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.draw_kwargs = draw_kwargs
        self.on_viewport = on_viewport

        self.full_xlim = (0.0, max(trace.length_ns, 1.0))
        self.xlim_ns = self.full_xlim
        self.ylim = None

        self._press = None
        self._last_xy = None
        self._last_click = None
        self._rubber = None
        self._cids = []
        self._hover = None          # tooltip annotation, rebuilt on every render
        self._hover_index = []      # (lane y, x0, x1, name, length_ns) per pulse
        self._hover_key = None      # the bar the tooltip is currently on
        self.render()
        self._connect()

    # ---------------- rendering ----------------

    @property
    def divisor(self):
        """Plotted-unit divisor for the current window (1 for ns, 1000 for µs)."""
        from .plotting import _time_scale
        return _time_scale(self.xlim_ns[1] - self.xlim_ns[0])[0]

    def render(self):
        draw(self.ax, self.trace, xlim_ns=self.xlim_ns, **self.draw_kwargs)
        self._install_hover()       # draw() cleared the axes, so rebuild the tooltip
        if self.ylim is not None:
            self.ax.set_ylim(*self.ylim)
        # an interactive canvas gets no bbox_inches="tight" expansion, so the
        # room for the outside legend has to be reserved on every render
        fit_layout(self.ax.figure, self.ax)
        self.canvas.draw_idle()
        if self.on_viewport:
            self.on_viewport(self.xlim_ns)

    def set_window(self, t0_ns, t1_ns, ylim=None):
        """Show ``[t0_ns, t1_ns]``, clamped to the sequence and a sane minimum."""
        lo, hi = sorted((float(t0_ns), float(t1_ns)))
        if hi - lo < self.MIN_SPAN_NS:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - self.MIN_SPAN_NS / 2, mid + self.MIN_SPAN_NS / 2
        self.xlim_ns = (lo, hi)
        if ylim is not None:
            self.ylim = ylim
        self.render()

    def reset(self):
        self.xlim_ns = self.full_xlim
        self.ylim = None
        self.render()

    def set_point(self, index):
        """Show another captured sweep point. No re-tracing -- the schedule is
        shared and only the pulse data and register lengths change."""
        self.trace.select_point(index)
        self.full_xlim = (0.0, max(self.trace.length_ns, 1.0))
        self.render()

    def relayout(self):
        """Re-run the layout after changing ``loop_counts`` or ``path_choices``.

        Both change how many placements there are and therefore the total length, so
        the full-view limits are recomputed. Still no re-trace: the compiled schedule
        is unaffected by which path you choose to draw.
        """
        self.trace.relayout()
        self.full_xlim = (0.0, max(self.trace.length_ns, 1.0))
        self.reset()

    # ---------------- events ----------------

    def _connect(self):
        for event, handler in [
            ("button_press_event", self._on_press),
            ("motion_notify_event", self._on_motion),
            ("button_release_event", self._on_release),
            ("scroll_event", self._on_scroll),
            ("key_press_event", self._on_key),
        ]:
            self._cids.append(self.canvas.mpl_connect(event, handler))

    def disconnect(self):
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        self._cids = []

    def _toolbar_busy(self):
        """True while matplotlib's own zoom/pan tool is armed -- yield to it."""
        toolbar = getattr(self.canvas, "toolbar", None)
        return bool(getattr(toolbar, "mode", "") or "")

    def _on_press(self, event):
        if event.inaxes is not self.ax or self._toolbar_busy():
            return
        if event.button != 1 and not event.dblclick:
            return

        if self._is_double_click(event):
            self._last_click = None      # a third click starts over
            self._press = None
            self.reset()
            return
        if event.button != 1:
            return

        self._last_xy = None        # never fall back to a previous drag's end
        self._press = (event.xdata, event.ydata,
                       bool(event.key and "shift" in event.key))
        self._hide_hover()          # get the tooltip out of the way while dragging

    def _is_double_click(self, event):
        now = time.monotonic()
        previous, self._last_click = self._last_click, (now, event.x, event.y)
        if event.dblclick:
            return True
        if previous is None:
            return False
        was, x, y = previous
        return (now - was < self.DOUBLE_CLICK_SECONDS
                and abs(event.x - x) <= self.DOUBLE_CLICK_PIXELS
                and abs(event.y - y) <= self.DOUBLE_CLICK_PIXELS)

    def _on_motion(self, event):
        if event.xdata is not None:
            # remembered so a drag released outside the axes still zooms --
            # matplotlib reports xdata=None there, and dragging off the edge
            # of the plot is a normal gesture
            self._last_xy = (event.xdata, event.ydata)
        if self._press is None:
            self._update_hover(event)       # not dragging: show the pulse tooltip
            return
        if event.xdata is None:
            return
        x0, y0, panning = self._press
        if panning:
            shift = (x0 - event.xdata) * self.divisor
            lo, hi = self.xlim_ns
            self.xlim_ns = (lo + shift, hi + shift)
            self.render()
            self._press = (event.xdata, event.ydata, True)
            return
        self._show_rubber(x0, y0, event.xdata, event.ydata)

    def _on_release(self, event):
        if self._press is None:
            return
        x0, y0, panning = self._press
        self._press = None
        self._clear_rubber()
        if panning:
            return

        x1, y1 = ((event.xdata, event.ydata) if event.xdata is not None
                  else (self._last_xy or (None, None)))
        if x1 is None:
            self.canvas.draw_idle()
            return

        # a click rather than a drag: ignore, so stray clicks don't zoom
        lo_x, hi_x = self.ax.get_xlim()
        if abs(x1 - x0) < 0.005 * (hi_x - lo_x):
            self.canvas.draw_idle()
            return

        div = self.divisor
        ylim = None
        if y1 is not None and abs(y1 - y0) > 0.75:
            # snap to whole lanes so channels are never half-clipped
            lo, hi = sorted((y0, y1))
            ylim = (round(lo) - 0.5, round(hi) + 0.5)
        self.set_window(x0 * div, x1 * div, ylim=ylim)

    def _on_scroll(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        factor = 0.8 if event.button == "up" else 1.25
        centre = event.xdata * self.divisor
        lo, hi = self.xlim_ns
        self.set_window(centre - (centre - lo) * factor,
                        centre + (hi - centre) * factor)

    def _on_key(self, event):
        if event.key in ("r", "R", "escape"):
            self.reset()

    # ---------------- rubber band ----------------

    def _show_rubber(self, x0, y0, x1, y1):
        from matplotlib.patches import Rectangle

        if self._rubber is None:
            self._rubber = Rectangle((0, 0), 0, 0, facecolor="#2a78d6",
                                     alpha=0.15, edgecolor="#2a78d6",
                                     linewidth=1.0, zorder=10)
            self.ax.add_patch(self._rubber)
        top, bottom = self.ax.get_ylim()[1], self.ax.get_ylim()[0]
        if y0 is None or y1 is None or abs(y1 - y0) <= 0.75:
            y0, y1 = bottom, top
        self._rubber.set_bounds(min(x0, x1), min(y0, y1),
                                abs(x1 - x0), abs(y1 - y0))
        self._rubber.set_visible(True)
        self.canvas.draw_idle()

    def _clear_rubber(self):
        if self._rubber is not None:
            self._rubber.set_visible(False)

    # ---------------- hover tooltip ----------------

    def _install_hover(self):
        """Rebuild the pulse index and the tooltip artist after a render.

        ``draw()`` clears the axes, so both are recreated every time. The index
        holds each pulse's lane and extent in the *plotted* unit (ns or µs, per
        the current zoom) plus its real length in ns.
        """
        channels = list(self.trace.channels)
        lane = {ch: len(channels) - 1 - i for i, ch in enumerate(channels)}
        ns = self.trace.ns_per_cycle / self.divisor
        # entry: (lane_y | None for full-height, x0, x1, name, length_ns, always_time)
        from .plotting import _stretch_groups
        cyc_ns = self.trace.ns_per_cycle
        by_channel = {}
        for c in self.trace.commands:
            if c.channel in lane:
                by_channel.setdefault(c.channel, []).append(c)
        index = []
        for ch, cmds in by_channel.items():
            cmds = sorted(cmds, key=lambda c: c.start)
            # a use_stretch pulse is three commands; hover the whole span, not just
            # the ARB head (else only the rising edge is hoverable) -- mirror draw()
            groups = _stretch_groups(cmds)
            skip = set()
            for i, c in enumerate(cmds):
                if i in skip:
                    continue
                group = groups.get(i)
                if group:
                    skip.update({i + 1, i + 2})
                    head, x0, x1 = group[0], group[0].start, group[2].stop
                else:
                    head, x0, x1 = c, c.start, c.stop
                length_ns = (x1 - x0) * cyc_ns
                if head.pulse:                    # pulse: hovers to length/name (toggle)
                    index.append((lane[ch], x0 * ns, x1 * ns, head.pulse,
                                  length_ns, False))
                elif head.kind == "CONST_CONT" and ch.startswith("ADC"):
                    index.append((lane[ch], x0 * ns, x1 * ns, "capture",
                                  length_ns, False))
                elif head.kind == "DWELL" and not head.is_padding:   # dwell: its time
                    index.append((lane[ch], x0 * ns, x1 * ns, "dwell",
                                  length_ns, True))
        # inter-block dead time spans the full height -- always its time
        for p in (self.trace.placements or self.trace.blocks):
            if getattr(p, "gap_after", 0):
                index.append((None, p.stop * ns, (p.stop + p.gap_after) * ns,
                              "dead time", p.gap_after * cyc_ns, True))
        self._hover_index = index
        self._hover_key = None
        self._hover = self.ax.annotate(
            "", xy=(0, 0), xytext=(10, 12), textcoords="offset points",
            ha="left", va="bottom", fontsize=8, color="#1a1a19", zorder=20,
            bbox=dict(boxstyle="round,pad=0.35", fc="#fbfbe6", ec="#8c8b83",
                      lw=0.6))
        self._hover.set_visible(False)

    def _hit_pulse(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return None
        x, y = event.xdata, event.ydata
        gap_hit = None
        for entry in self._hover_index:
            lane_y, x0, x1 = entry[0], entry[1], entry[2]
            if not (x0 <= x <= x1):
                continue
            if lane_y is None:                 # full-height dead-time band
                gap_hit = gap_hit or entry
            elif abs(y - lane_y) <= self.HOVER_HALF_HEIGHT:
                return entry                   # a lane bar wins over the band
        return gap_hit

    def _update_hover(self, event):
        """Tooltip for whatever is under the cursor. A pulse/capture shows the
        *other* attribute (length when labels are names, name when lengths); a
        dwell or dead-time band always shows its duration."""
        if self._hover is None:
            return
        hit = self._hit_pulse(event)
        if hit is None:
            self._hide_hover()
            return
        lane_y, x0, x1, name, length_ns, always_time = hit
        key = (name, x0, x1, lane_y)
        if key == self._hover_key:      # same bar -- tooltip already up, no redraw
            return
        if always_time:
            text = f"{length_ns:.0f} ns"
        else:
            mode = self.draw_kwargs.get("pulse_label", "name")
            text = f"{length_ns:.0f} ns" if mode == "name" else name
        lo, hi = self.ax.get_ylim()
        self._hover.xy = ((x0 + x1) / 2,
                          lane_y if lane_y is not None else (lo + hi) / 2)
        self._hover.set_text(text)
        self._hover.set_visible(True)
        self._hover_key = key
        self.canvas.draw_idle()

    def _hide_hover(self):
        if self._hover is not None and self._hover.get_visible():
            self._hover.set_visible(False)
            self.canvas.draw_idle()
        self._hover_key = None


def interactive_view(trace, figsize=None, **draw_kwargs):
    """Open an interactive figure for ``trace``. Returns the :class:`SequenceView`.

    In a notebook select a widget backend first, otherwise the canvas receives no
    events::

        %matplotlib widget

    Keep a reference to the returned object -- dropping it drops the callbacks.
    """
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (13, 1.2 + 0.62 * len(trace.channels))
    fig, ax = plt.subplots(figsize=figsize)
    view = SequenceView(trace, ax, **draw_kwargs)   # renders, and fits the layout
    fig._sequence_view = view          # keep alive for the figure's lifetime
    return view
