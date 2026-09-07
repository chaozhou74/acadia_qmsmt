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

That render is not cheap, though -- matplotlib rasters the whole figure, ~50 ms on
a nine-lane sequence -- so a gesture is not one render per event:

* a **pan** shows immediately, by blitting the last drawn frame back shifted by the
  pan (``_preview_pan``). A few ms, so the plot stays under the cursor instead of
  trailing it; the strip scrolling into view is blank until a real frame refills it.
* real frames are paced at the measured cost of a frame, and the ones drawn during
  a gesture drop the per-window labels (``fast``).
* the full frame is drawn once the gesture stops.
* hovering repaints only the tooltip, over a snapshot of the plot.
"""
import math
import time

from .plotting import draw, fit_layout


class SequenceView:
    """Own an axes and keep it showing ``trace`` at the current viewport."""

    MIN_SPAN_NS = 2.0
    #: A lane viewport narrower than one lane shows no channel whole and makes the lane scrollbar
    #: degenerate, so the vertical axis has a floor for the same reason the time axis does.
    MIN_LANE_SPAN = 1.0

    # Half a lane height (LANE_HEIGHT / 2), for deciding whether the cursor is on
    # a pulse bar when hovering.
    HOVER_HALF_HEIGHT = 0.31

    # Some backends (ipympl among them) never set MouseEvent.dblclick, so the
    # double click is also detected by timing. Compared in pixels, not data
    # coordinates -- a reset changes the data scale underfoot.
    DOUBLE_CLICK_SECONDS = 0.4
    DOUBLE_CLICK_PIXELS = 8

    # Floor and ceiling for the gap between re-renders during a zoom/pan gesture.
    # The gap itself tracks the *measured* cost of a frame (see _on_draw): a frame
    # costs 50-150 ms on a real sequence, so a fixed 30 ms budget throttled nothing
    # -- every event still queued a full re-render and the view fell further behind
    # the mouse the longer you dragged. Rendering no faster than frames can be
    # drawn keeps that queue empty; what the user actually watches in between is
    # the pan preview.
    RENDER_INTERVAL_S = 0.03
    # ceiling, so a one-off hiccup in the measurement cannot stall the gesture --
    # and so the full-detail frame is never more than this late
    MAX_INTERVAL_S = 0.6
    # how far a frame may be shifted before it is better to hold it (_preview_pan)
    MAX_PREVIEW_SHIFT = 0.4

    def __init__(self, trace, ax, on_viewport=None, on_lanes=None, **draw_kwargs):
        self.trace = trace
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.draw_kwargs = draw_kwargs
        self.on_viewport = on_viewport   # called with the time window, every render
        self.on_lanes = on_lanes         # called with the lane range, every render

        self.full_xlim = (0.0, max(trace.length_ns, 1.0))
        self.xlim_ns = self.full_xlim
        self._drawn_xlim = self.xlim_ns    # the window the axes currently show
        self._drawn_ylim = None            # and the lane range it was drawn with
        self.ylim = None

        self._press = None
        self._press_px = None
        self._last_xy = None
        self._last_click = None
        self._rubber = None
        self._bg = None             # cached plot background for blitting the rubber band
        self._cids = []
        self._hover = None          # tooltip annotation, rebuilt on every render
        self._readout = None        # bottom-left time/IQ readout, rebuilt on every render
        self._readout_text = None   # its current text, to skip redundant redraws
        self._pulse_iq = []         # (lane y, x0, x1, io_name, pulse) per pulse, for IQ
        self._hover_index = []      # (lane y, x0, x1, name, length_ns) per pulse
        self._branch_regions = []   # (x0, x1, y_bottom, y_top, caption), hovered near edge
        self._hover_key = None      # the bar the tooltip is currently on
        self._layout_key = None     # canvas size the cached margins were fit for
        self._layout_pars = None    # cached subplot margins (tight_layout is slow)
        self._timer = None          # coalesces throttled renders
        self._timer_live = False    # False for a backend whose timers never fire
        self._timer_running = False
        self._pending = False
        self._last_render = 0.0
        self._coarse = False        # the frame on screen is a reduced-detail one
        self._frame_s = self.RENDER_INTERVAL_S   # measured cost of one frame
        self._frame_start = None    # when the frame being measured was requested
        self._clean_bg = None       # last drawn frame, for the tooltip and pan blits
        self._shifted = False       # the canvas is showing a pan preview of it
        self.render()
        self._connect()

    # ---------------- rendering ----------------

    @property
    def divisor(self):
        """Plotted-unit divisor of the frame **on screen** (1 for ns, 1000 for µs).

        The drawn window, deliberately, not ``xlim_ns``: while a gesture is being
        coalesced those two differ, and everything matplotlib hands back -- an
        event's ``xdata``, the hover index, the rubber band -- is in the units of
        the frame the user is actually looking at. Reading the *pending* window
        here is what threw the view out to -2000 µs on a fast scroll: an ``xdata``
        still in ns, scaled by the µs divisor the next frame was going to use, is
        out by 1000x.
        """
        from .plotting import _time_scale
        return _time_scale(self._drawn_xlim[1] - self._drawn_xlim[0])[0]

    def _x_to_ns(self, x_px):
        """Pixel column -> time in ns within the window the view is converging to.

        For gestures, which compose onto ``xlim_ns`` (the pending window) rather
        than onto what is currently drawn. Going through pixels rather than
        ``event.xdata`` keeps that independent of which unit the drawn frame is in.
        """
        box = self.ax.bbox
        lo, hi = self.xlim_ns
        return lo + (x_px - box.x0) / max(box.width, 1.0) * (hi - lo)

    def render(self, coarse=False):
        """Redraw the current window.

        ``coarse`` draws a gesture frame: no per-window labels (see ``draw``'s
        ``fast``), 53 ms against 70 for the full frame on a nine-lane RB sequence.
        The full frame follows as soon as the gesture stops (:meth:`_flush_render`).
        """
        self._coarse = coarse
        self._shifted = False        # this frame supersedes any shifted pixels
        # the previous frame's pixels are deliberately kept: until this one has
        # actually been rastered (see _on_draw) they are the only thing a pan
        # preview can shift, and they are still a correct picture of their own
        # window. What must not be reused stale is the tooltip background --
        # _blit_overlay checks the snapshot against the drawn window for that.
        self._frame_start = time.monotonic()
        kwargs = dict(self.draw_kwargs, fast=True) if coarse else self.draw_kwargs
        draw(self.ax, self.trace, xlim_ns=self.xlim_ns, **kwargs)
        # The lane extent is a property of the FRAME, not of the trace: the tab strip's height
        # comes from the tabs this window actually has, so zooming into a shallower stretch makes
        # the stack shorter. A lane range chosen against a taller stack then sits partly outside
        # the axes, and the lane scrollbar -- which treats the range as a slice of the stack --
        # goes degenerate. Re-clamped here, the first moment the frame's real extent is known, and
        # before anything records what was drawn.
        if self.ylim is not None:
            self.ylim = self.slice_of(self.ylim, self.full_ylim, self.MIN_LANE_SPAN)[0]
        # what is on the axes from here on -- the frame every incoming event's data
        # coordinates will be expressed in, until the next render (see divisor)
        self._drawn_xlim = self.xlim_ns
        self._drawn_ylim = self.ylim or self.full_ylim
        if coarse:
            # a gesture hides the tooltip anyway, so it is not rebuilt until the
            # full frame -- and hover handlers no-op while it is None
            self._drop_hover()
        else:
            self._install_hover()   # the index is rebuilt for the new window
        if self.ylim is not None:
            self.ax.set_ylim(*self.ylim)
        self._fit_layout(coarse)
        self.canvas.draw_idle()
        if self.on_viewport:
            self.on_viewport(self.xlim_ns)
        if self.on_lanes:
            self.on_lanes(self.ylim or self.full_ylim)

    def _fit_layout(self, coarse=False):
        """Reserve room for the outside legend and lane labels.

        ``fit_layout`` runs ``tight_layout``, which is far too slow to call on
        every zoom frame -- but the margins only change when the canvas is
        resized. So compute them once per size and just re-apply the cached
        margins on subsequent renders.

        A coarse frame has no legend, so it must never *measure* the layout: the
        margins it would cache leave no room for the legend that comes back.
        """
        fig = self.ax.figure
        key = (self.canvas.get_width_height()
               if hasattr(self.canvas, "get_width_height") else None)
        if coarse:
            if self._layout_pars is not None:
                fig.subplots_adjust(**self._layout_pars)
            return
        if self._layout_pars is None or key != self._layout_key:
            fit_layout(fig, self.ax)
            sp = fig.subplotpars
            self._layout_pars = dict(left=sp.left, right=sp.right,
                                     top=sp.top, bottom=sp.bottom)
            self._layout_key = key
        else:
            fig.subplots_adjust(**self._layout_pars)

    @property
    def full_ylim(self):
        """Lane limits that show every channel -- what ``draw`` uses when there is
        no restriction. Channel *i* of ``trace.channels`` sits at lane
        ``len(channels) - 1 - i`` (the first channel on top), each lane spanning
        half a unit either side of its centre; the extra 0.2 is draw()'s padding.

        The top includes the CONTROL-FLOW TAB STRIP, which is drawn above the lanes. Without it
        the vertical scrollbar's range stopped at the topmost channel, so the tabs -- the handles
        you click to change a loop -- could sit above the reachable range and be unscrollable to.
        The strip height comes from the same helper ``draw`` uses, so the bar and the picture
        cannot disagree about where the top is."""
        lanes = len(self.trace.channels)
        top = lanes - 0.3
        # Prefer the top the RENDERER actually used, which accounts for tabs bumped to a free row
        # on collision. Falling back to a depth-based estimate would stop short of them.
        drawn = getattr(self.ax, "_seqviz_strip_top", None)
        if drawn is not None:
            return (-0.7, max(top, drawn))
        try:
            from .plotting import branch_regions, tab_strip_top, _lanes_per_pixel

            depth = max((info.get("depth", 1)
                         for *_r, info in branch_regions(self.trace)), default=0)
            if depth:
                top = max(top, tab_strip_top(depth, lanes,
                                             _lanes_per_pixel(self.ax, lanes)))
        except Exception:
            pass                    # the strip is an enhancement; never break panning
        return (-0.7, top)

    def set_lanes(self, y0, y1, throttle=False):
        """Show the lanes in ``[y0, y1]`` -- a vertical pan, leaving the time
        window alone. Asking for the whole stack restores the padded full view.

        ``throttle=True`` for a range being dragged somewhere; see :meth:`set_window`.
        """
        (lo, hi), (full_lo, full_hi) = self.slice_of((y0, y1), self.full_ylim,
                                                     self.MIN_LANE_SPAN)
        self.ylim = None if (lo <= full_lo and hi >= full_hi) else (lo, hi)
        if throttle:
            self._request_render()
        else:
            self.render()

    def set_window(self, t0_ns, t1_ns, ylim=None, throttle=False):
        """Show ``[t0_ns, t1_ns]``, kept on the sequence and above a sane minimum.
        A window that runs off an end slides back rather than being clipped, so the
        span you ask for is the span you get.

        ``throttle=True`` for a window that is being *dragged* somewhere (a
        scrollbar, a slider): the limits take effect at once, but the redraw is
        coalesced with the rest of the gesture instead of one full render per event.
        """
        self._apply_window(t0_ns, t1_ns, ylim, throttle=throttle)

    @staticmethod
    def slice_of(requested, full, minimum):
        """``requested`` forced into a finite, ordered window of at least ``minimum``, inside
        ``full``. Returns ``((lo, hi), (full_lo, full_hi))`` -- the extent too, because an extent
        that was itself unusable has been replaced.

        ONE rule for both axes, because both make the same promise to everything downstream: a
        viewport is a finite, ordered slice of what exists. The scrollbars divide by its span, the
        renderer compares it against what it drew last, and matplotlib refuses NaN or infinite
        limits outright -- which in Qt means an exception escaping a slot and taking the whole
        application down. Nothing sensible can be derived from a NaN, so the answer is to show
        everything rather than to fail.

        Windows that merely run off an end SLIDE back rather than being clipped, so the span you
        asked for is the span you get; a window wider than the extent becomes the extent.
        """
        full_lo, full_hi = float(full[0]), float(full[1])
        if not (math.isfinite(full_lo) and math.isfinite(full_hi) and full_hi > full_lo):
            full_lo, full_hi = 0.0, minimum     # the thing being viewed has no usable size
        lo, hi = float(requested[0]), float(requested[1])
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return (full_lo, full_hi), (full_lo, full_hi)
        lo, hi = sorted((lo, hi))
        if hi - lo < minimum:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - minimum / 2, mid + minimum / 2
        span = hi - lo
        if span >= full_hi - full_lo:
            lo, hi = full_lo, full_hi           # zoomed out past the whole extent
        elif lo < full_lo:
            lo, hi = full_lo, full_lo + span
        elif hi > full_hi:
            lo, hi = full_hi - span, full_hi
        return (lo, hi), (full_lo, full_hi)

    def _apply_window(self, t0_ns, t1_ns, ylim=None, throttle=False):
        (lo, hi), full = self.slice_of((t0_ns, t1_ns), self.full_xlim, self.MIN_SPAN_NS)
        self.full_xlim = full
        self.xlim_ns = (lo, hi)
        if ylim is not None:
            self.ylim = self.slice_of(ylim, self.full_ylim, self.MIN_LANE_SPAN)[0]
        if throttle:
            self._request_render()
        else:
            self.render()

    def _interval(self):
        """How long to leave between gesture frames: one frame's measured cost,
        held between the floor and the ceiling.

        (Leaving extra room for previews between renders was tried and measured:
        it changed neither the frame rate nor the blank strip, so there is no
        slack factor here.)
        """
        return min(max(self._frame_s, self.RENDER_INTERVAL_S), self.MAX_INTERVAL_S)

    def _request_render(self):
        """Show the new window at once if it is a pan, and schedule the real frame.

        Scroll/pan events arrive far faster than a frame can be drawn (a scrollbar
        drag emits one per pixel of travel), so every event is *not* a render:
        intermediate windows are dropped and only the latest one is drawn.
        """
        if not self._ensure_timer():
            self.render()       # no timer to end the gesture, so draw in full now
            return
        self._preview_pan()     # show the pan now; the render below refills it
        interval = self._interval()
        if not self._timer_running and time.monotonic() - self._last_render >= interval:
            self._draw_frame(coarse=True)
        else:
            self._pending = True
        # a timer is always armed after a gesture frame: it is what notices the
        # gesture has stopped and puts the labels and legend back
        self._arm_timer(interval)

    def _ensure_timer(self):
        """Build the coalescing timer, and say whether it actually fires.

        A headless canvas hands out an inert ``TimerBase``: nothing would ever
        flush, so the coarse frame would be the last one drawn. Callers fall back
        to rendering in full when that happens.
        """
        if self._timer is None:
            from matplotlib.backend_bases import TimerBase
            timer = self.canvas.new_timer()
            self._timer_live = (type(timer)._timer_start
                                is not TimerBase._timer_start)
            timer.single_shot = True
            timer.add_callback(self._flush_render)
            self._timer = timer
        return self._timer_live

    def _arm_timer(self, interval):
        if self._timer_running or not self._ensure_timer():
            return
        self._timer.interval = max(int(interval * 1000), 1)
        self._timer_running = True
        self._timer.start()

    def _flush_render(self):
        """Timer tick: another gesture frame if the window moved since the last
        one, otherwise the full-detail frame that ends the gesture."""
        self._timer_running = False
        if self._pending:
            self._pending = False
            self._draw_frame(coarse=True)
            self._arm_timer(self._interval())
        elif self._coarse:
            self._draw_frame(coarse=False)

    def _draw_frame(self, coarse):
        self._last_render = time.monotonic()
        self.render(coarse=coarse)

    def reset(self):
        self.xlim_ns = self.full_xlim
        self.ylim = None
        self.render()

    def set_point(self, index):
        """Show another captured sweep point. No re-tracing -- the schedule is
        shared and only the pulse data and register lengths change."""
        previous = self.full_xlim
        self.trace.select_point(index)
        self.full_xlim = (0.0, max(self.trace.length_ns, 1.0))
        self._refit(previous)

    def _refit(self, previous_full):
        """Place the window after the sequence has changed LENGTH.

        A view that covered the whole sequence keeps covering it; a window you zoomed in yourself
        is kept where it was, clamped to what now exists. Anything else lies about one of the two:
        keeping a stale window on a longer sequence shows part of it as though it were all of it
        (a swept delay takes DualRail_RB from 78 us to 319 us -- the view stayed at 78 and said
        nothing), and resetting a deliberate zoom throws away where the reader was looking.
        """
        lo, hi = self.xlim_ns
        was_lo, was_hi = previous_full
        if lo <= was_lo + 1e-6 and hi >= was_hi - 1e-6:
            self.reset()
        else:
            self.set_window(lo, hi)

    def relayout(self):
        """Re-run the layout after changing ``loop_counts`` or ``path_choices``.

        Both change how many placements there are and therefore the total length, so
        the full-view limits are recomputed. Still no re-trace: the compiled schedule
        is unaffected by which path you choose to draw.
        """
        previous = self.full_xlim
        self.trace.relayout()
        self.full_xlim = (0.0, max(self.trace.length_ns, 1.0))
        self._refit(previous)

    # ---------------- events ----------------

    def _connect(self):
        for event, handler in [
            ("button_press_event", self._on_press),
            ("motion_notify_event", self._on_motion),
            ("button_release_event", self._on_release),
            ("scroll_event", self._on_scroll),
            ("key_press_event", self._on_key),
            ("draw_event", self._on_draw),
        ]:
            self._cids.append(self.canvas.mpl_connect(event, handler))

    def disconnect(self):
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        self._cids = []
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:
                pass
        self._timer_running = False

    def _on_draw(self, _event):
        """Time what a frame really costs, and keep a copy of it.

        The raster happens on the event loop well after ``render`` returns, so the
        only honest measure of a frame is request-to-drawn -- which is also exactly
        the quantity the gesture throttle needs. The canvas holds the finished
        frame at this point, so this is also the cheap moment to snapshot it: a
        memcpy of a buffer that is already there.
        """
        self._capture_frame()
        if self._frame_start is None:
            return
        elapsed = time.monotonic() - self._frame_start
        self._frame_start = None
        # smoothed, so one slow frame does not pin the interval at its cost
        self._frame_s = min(max(0.5 * self._frame_s + 0.5 * elapsed, 0.001),
                            self.MAX_INTERVAL_S)

    def _capture_frame(self):
        """Keep the frame now on the canvas, for the tooltip and pan blits.

        Stored with the window it shows and with the two tick-label strips (whose
        extents need a renderer, so they are measured here rather than per event).
        """
        self._clean_bg = None
        if any(a is not None and a.get_visible()
               for a in (self._hover, self._readout)):
            return          # never snapshot with the tooltip baked into the plot
        try:
            region = self.canvas.copy_from_bbox(self._buffer_bbox())
        except Exception:
            return
        self._clean_bg = (region, self._drawn_xlim,
                          self._drawn_ylim or self.full_ylim, self._tick_strip())

    def _snapshot_is_current(self):
        """True when the snapshot's pixels match the window now on the axes.

        False in the gap between a render and its raster, where the artists have
        already moved on -- fine to *shift* those pixels (a pan preview knows
        which window they show), not fine to paint a tooltip over them.
        """
        if self._clean_bg is None:
            return False
        _region, window, ylim = self._clean_bg[:3]
        return (window == self._drawn_xlim
                and ylim == (self._drawn_ylim or self.full_ylim))

    def _tick_strip(self):
        """The canvas rectangle holding the time labels, or None.

        It sits below the axes but belongs to it: a tick label has to travel with
        the gridline it names, or the two come apart mid-pan.
        """
        box = self.ax.bbox
        try:
            renderer = self.canvas.get_renderer()
            below = [t.label1.get_window_extent(renderer)
                     for t in self.ax.xaxis.get_major_ticks() if t.label1.get_visible()]
        except Exception:
            return None
        return ((box.x0, min(e.y0 for e in below), box.x1, box.y0)
                if below else None)

    def _toolbar_busy(self):
        """True while matplotlib's own zoom/pan tool is armed -- yield to it."""
        toolbar = getattr(self.canvas, "toolbar", None)
        return bool(getattr(toolbar, "mode", "") or "")

    def _claimed(self, event):
        """True when something else owns this press, so the viewport must not act on it.

        The whole plot area is the box-zoom gesture's target, which makes it a poor place to put
        anything else clickable: a click meant for a widget drawn ON the plot also drags out a
        zoom rectangle. `claim_press` lets an owner (the control-flow tabs) take a press before
        the viewport sees it. Set it to a callable taking the mpl event and returning truthy.
        """
        claim = getattr(self, "claim_press", None)
        try:
            return bool(claim and claim(event))
        except Exception:
            return False            # a broken claim must never wedge panning and zooming

    def _on_press(self, event):
        if event.inaxes is not self.ax or self._toolbar_busy() or self._claimed(event):
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
        self._press_px = (event.x, event.y)     # for panning, which works in pixels
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
            # in pixels, against the pending window, for the same reason as the
            # wheel: a drag is composed onto where the view is going, not onto the
            # frame that happens to be drawn
            lo, hi = self.xlim_ns
            shift = -(event.x - self._press_px[0]) / max(self.ax.bbox.width, 1.0) * (hi - lo)
            self._press = (event.xdata, event.ydata, True)
            self._press_px = (event.x, event.y)
            self._apply_window(lo + shift, hi + shift, throttle=True)  # pan fires per pixel
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
        """Zoom the time axis about the cursor.

        The centre comes from the cursor's *pixel* column mapped into the pending
        window, so a burst of wheel events compounds correctly however far behind
        the drawing is -- each notch zooms about the same point on screen.
        """
        if event.inaxes is not self.ax or event.xdata is None:
            return
        factor = 0.8 if event.button == "up" else 1.25
        centre = self._x_to_ns(event.x)
        lo, hi = self.xlim_ns
        self._apply_window(centre - (centre - lo) * factor,
                           centre + (hi - centre) * factor, throttle=True)

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
        # snapshot the clean plot once at the start of a drag, then just repaint the
        # box on top of it -- a full draw_idle per mouse-move can't keep up (the box
        # flickers / rarely shows) and redraws the whole timeline each move
        if self._bg is None:
            self._rubber.set_visible(False)
            try:
                self.canvas.draw()
                self._bg = self.canvas.copy_from_bbox(self.ax.bbox)
            except Exception:
                self._bg = None
        # the box follows the actual drag height (no jump to full-height); a
        # short/flat drag is interpreted as time-only on release (see _on_release)
        if y0 is None or y1 is None:        # drag left the axes -- span full height
            y0, y1 = self.ax.get_ylim()[0], self.ax.get_ylim()[1]
        self._rubber.set_bounds(min(x0, x1), min(y0, y1),
                                abs(x1 - x0), abs(y1 - y0))
        self._rubber.set_visible(True)
        if self._bg is not None:
            try:
                self.canvas.restore_region(self._bg)
                self.ax.draw_artist(self._rubber)
                self.canvas.blit(self.ax.bbox)
                return
            except Exception:
                self._bg = None
        self.canvas.draw_idle()

    def _clear_rubber(self):
        if self._rubber is not None:
            self._rubber.set_visible(False)
        self._bg = None

    # ---------------- pan preview ----------------

    def _preview_pan(self):
        """Show the pending window now, by sliding the pixels already drawn.

        A pan is the same picture at a different offset, so there is no reason to
        wait ~80 ms for a redraw to see it: the last drawn frame is blitted back
        shifted by the pan, and the strip scrolling into view is left blank until
        the real frame refills it. That costs a few milliseconds, so the plot
        tracks the handle instead of trailing it by a fifth of a second.

        Returns False when the pending window is not a pure translation of the
        drawn one -- a zoom changes the picture, not just where it sits -- and the
        caller then has nothing to show until the real frame lands.
        """
        snapshot = self._clean_bg
        if snapshot is None:
            return False
        region, (drawn_lo, drawn_hi), drawn_ylim, x_strip = snapshot
        lo, hi = self.xlim_ns
        span, drawn_span = hi - lo, drawn_hi - drawn_lo
        if abs(span - drawn_span) > 1e-9 * max(abs(span), 1.0):
            return False                        # zoomed, not panned
        # Only the time axis. A lane pan is *not* a translation of the picture:
        # the inter-block gap bands and the barrier and block-start lines span the
        # full height in axes coordinates, so they stay put while the lanes move
        # past them -- shifting the image would drag them along and leave them
        # hanging short of the top. Measured: a one-lane preview came out 2 px off
        # and a fifth of the plot wrong, against 0 px and 2% for a time pan.
        if (self.ylim or self.full_ylim) != drawn_ylim:
            return False
        box = self.ax.bbox
        # whole pixels, rounded: the blit is done in integers, and rounding here
        # rather than letting the backend truncate keeps the shift accurate to half
        # a pixel instead of biasing every pan a pixel short
        dx = round((drawn_lo - lo) / span * box.width)
        dy = 0
        # Past this much of the plot the shift is mostly the blank strip that
        # scrolls in behind it, and holding the last complete frame for a moment
        # reads better than watching an empty axes slide by. The next rendered
        # frame resets the shift, so a fast drag alternates between tracking and
        # a brief hold rather than going blank.
        if abs(dx) > self.MAX_PREVIEW_SHIFT * box.width:
            return False
        try:
            self.canvas.restore_region(region)  # the frame as drawn, to shift from
            self._shift(region, box.extents, dx, self.ax.get_facecolor())
            self._shift(region, x_strip, dx, None)   # the time labels move with it
            self.canvas.blit(self._buffer_bbox())
        except Exception:
            self._clean_bg = None               # backend cannot blit; fall back
            return False
        self._shifted = True                    # the buffer is ahead of the artists
        return True

    def _shift(self, region, rect, dx, fill):
        """Redraw ``rect`` of ``region`` moved ``dx`` pixels sideways, blanking the
        strip that moves into it. ``fill`` is that strip's colour (the figure's own
        when None)."""
        if rect is None or not dx:
            return
        x0, y0, x1, y1 = rect
        # the part of the rectangle still inside it after the move
        src = (x0 + max(-dx, 0), y0, x1 - max(dx, 0), y1)
        if src[2] - src[0] >= 1:
            # ``xy`` translates the copied rectangle -- it is *not* the position to
            # put it at, whatever the wording of the matplotlib docstring suggests
            # (checked against the Agg backend: restoring the whole canvas with
            # xy=(50, 0) moves everything 50 px right).
            self.canvas.restore_region(region, bbox=src, xy=(dx, 0))
        exposed = ((x0, y0, x0 + dx, y1) if dx > 0 else (x1 + dx, y0, x1, y1))
        self._fill(exposed, fill if fill is not None
                   else self.ax.figure.get_facecolor())

    def _fill(self, rect, color):
        """Paint a canvas rectangle, in pixels, straight onto the buffer."""
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import IdentityTransform

        patch = Rectangle((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1],
                          facecolor=color, edgecolor="none", linewidth=0,
                          transform=IdentityTransform())
        patch.set_figure(self.ax.figure)
        patch.draw(self.canvas.get_renderer())

    # ---------------- overlay blitting ----------------

    def _blit_overlay(self):
        """Repaint just the tooltip and the readout, over a cached plot image.

        Moving the mouse changes only these two artists, and a full ``draw_idle``
        for each of them costs a whole frame -- which made simply *hovering* along
        the timeline as expensive as zooming. Painting them over the snapshot taken
        after the last render (see :meth:`_capture_frame`) costs a few ms instead.
        Blitting the figure box rather than the axes box is deliberate: the readout
        sits in the bottom margin, outside the axes.
        """
        shown = [a for a in (self._hover, self._readout)
                 if a is not None and a.get_visible()]
        if not self._snapshot_is_current():
            self._clean_bg = None       # older than the axes: would paint a stale plot
            if not shown:
                self.canvas.draw_idle()   # nothing to paint on top: no snapshot needed
                return
            for artist in shown:          # never bake the tooltip into the snapshot
                artist.set_visible(False)
            try:
                self.canvas.draw()        # _on_draw snapshots the frame it produces
            except Exception:
                pass
            for artist in shown:
                artist.set_visible(True)
        if self._clean_bg is None:              # backend cannot blit -- full redraw
            self.canvas.draw_idle()
            return
        try:
            self.canvas.restore_region(self._clean_bg[0])
            for artist in (self._hover, self._readout):
                if artist is not None and artist.get_visible():
                    self.ax.draw_artist(artist)
            self.canvas.blit(self._buffer_bbox())
        except Exception:
            self._clean_bg = None
            self.canvas.draw_idle()

    def _buffer_bbox(self):
        """The region the canvas buffer actually covers.

        ``figure.bbox`` can be a fraction of a pixel wider than the buffer (the
        widget is sized in whole pixels, the figure in inches), and blitting that
        extra column paints it from nothing -- a white seam down the right edge,
        over the legend. Truncate the way the raster itself does.
        """
        fig = self.ax.figure
        try:
            from matplotlib.transforms import Bbox
            renderer = self.canvas.get_renderer()
            return Bbox.from_bounds(0, 0, int(renderer.width), int(renderer.height))
        except Exception:
            return fig.bbox

    # ---------------- hover tooltip ----------------

    def _drop_hover(self):
        """Take the tooltip and readout off the axes.

        ``draw`` reuses the axes rather than clearing it, so these two have to be
        removed by whoever put them there -- otherwise every render would leave
        another pair behind.
        """
        for artist in (self._hover, self._readout):
            if artist is not None:
                try:
                    artist.remove()
                except (ValueError, NotImplementedError):
                    pass
        self._hover = self._readout = None
        self._hover_key = self._readout_text = None

    def _install_hover(self):
        """Rebuild the pulse index and the tooltip artist after a render.

        The index holds each pulse's lane and extent in the *plotted* unit (ns or
        µs, per the current zoom) plus its real length in ns, so it is rebuilt for
        every window -- and the two artists with it.
        """
        self._drop_hover()
        channels = list(self.trace.channels)
        lane = {ch: len(channels) - 1 - i for i, ch in enumerate(channels)}
        ns = self.trace.ns_per_cycle / self.divisor
        # entry: (lane_y | None for full-height, x0, x1, name, length_ns, kind)
        # kind: "toggle" (pulse/capture: length<->name), "time" (dwell/gap: its
        # duration), "text" (control-flow: the caption)
        from .plotting import _stretch_groups, branch_regions, branch_caption
        cyc_ns = self.trace.ns_per_cycle
        by_channel = {}
        for c in self.trace.commands:
            if c.channel in lane:
                by_channel.setdefault(c.channel, []).append(c)
        index = []
        # pulses only, for the bottom-left time/IQ readout: (lane y, x0, x1 in the plotted
        # unit, io name, pulse name) -- enough to sample the envelope at the hovered time
        self._pulse_iq = []
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
                                  length_ns, "toggle"))
                    if head.io_name:
                        self._pulse_iq.append(
                            (lane[ch], x0 * ns, x1 * ns, head.io_name, head.pulse))
                elif head.kind == "CONST_CONT" and ch.startswith("ADC"):
                    index.append((lane[ch], x0 * ns, x1 * ns, "capture",
                                  length_ns, "toggle"))
                elif head.kind == "DWELL" and not head.is_padding:   # dwell: its time
                    index.append((lane[ch], x0 * ns, x1 * ns, "dwell",
                                  length_ns, "time"))
        # inter-block dead time spans the full height -- always its time
        for p in (self.trace.placements or self.trace.blocks):
            if getattr(p, "gap_after", 0):
                index.append((None, p.stop * ns, (p.stop + p.gap_after) * ns,
                              "dead time", p.gap_after * cyc_ns, "time"))
        self._hover_index = index
        # control-flow regions are hovered near their dashed stroke (not the blank
        # interior) -- kept separate so the box's box geometry is available. The box
        # spans y in [-0.62, len(channels) - 0.67] (see plotting.draw).
        self._branch_regions = []
        if self.draw_kwargs.get("show_branches", True):
            yb, yt = -0.62, len(channels) - 0.67
            for start, stop, context, info in branch_regions(self.trace):
                self._branch_regions.append(
                    (start * ns, stop * ns, yb, yt,
                     branch_caption(self.trace, context, info)))
        self._hover_key = None
        self._hover = self.ax.annotate(
            "", xy=(0, 0), xytext=(10, 12), textcoords="offset points",
            ha="left", va="bottom", fontsize=8, color="#1a1a19", zorder=20,
            bbox=dict(boxstyle="round,pad=0.35", fc="#fbfbe6", ec="#8c8b83",
                      lw=0.6))
        self._hover.set_visible(False)

        # Readout below the time axis: the cursor time, plus the pulse amplitude / IQ when
        # the cursor is over a pulse. A blended transform pins x to the plot's left edge
        # (axes fraction) and y to the figure's bottom margin (figure fraction), so it sits
        # under the x-axis rather than on a lane, and left-aligns with the plot. Colours come
        # from the active theme so it reads in dark mode too.
        from matplotlib.transforms import blended_transform_factory
        from .plotting import LIGHT_THEME
        th = self.draw_kwargs.get("theme") or LIGHT_THEME
        below_axis = blended_transform_factory(
            self.ax.transAxes, self.ax.figure.transFigure)
        self._readout_text = None
        self._readout = self.ax.text(
            0.0, 0.018, "", transform=below_axis, ha="left", va="bottom", clip_on=False,
            fontsize=8, family="monospace", zorder=21,
            color=th.get("ink_primary", "#1a1a19"),
            bbox=dict(boxstyle="round,pad=0.3", fc=th.get("surface", "#ffffff"),
                      ec=th.get("ink_muted", "#8c8b83"), lw=0.6, alpha=0.95))
        self._readout.set_visible(False)

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

    def _hit_branch(self, event):
        """A control-flow region, but only when the cursor is near its dashed
        stroke (any of the four edges) -- not the blank interior."""
        if not self._branch_regions or event.xdata is None:
            return None
        x, y = event.xdata, event.ydata
        ext = self.ax.get_window_extent()
        xlo, xhi = self.ax.get_xlim()
        ylo, yhi = self.ax.get_ylim()
        tol_x = 6.0 * (xhi - xlo) / max(ext.width, 1.0)     # ~6 px, in data units
        tol_y = 6.0 * (yhi - ylo) / max(ext.height, 1.0)
        for region in self._branch_regions:
            x0, x1, yb, yt, _caption = region
            if not (x0 - tol_x <= x <= x1 + tol_x and yb - tol_y <= y <= yt + tol_y):
                continue
            if (abs(x - x0) <= tol_x or abs(x - x1) <= tol_x       # left/right stroke
                    or abs(y - yb) <= tol_y or abs(y - yt) <= tol_y):  # bottom/top
                return region
        return None

    def _update_hover(self, event):
        """The near-cursor tooltip and the bottom-left time/IQ readout, in one redraw."""
        if self._hover is None:
            return
        if self._shifted and self.xlim_ns != self._drawn_xlim:
            # the canvas is showing a pan preview, which the tooltip blit would
            # paint over with the un-shifted snapshot -- jumping the plot back a
            # pan. The render already on its way puts both in step again. (Tested
            # against the view rather than the flag alone, so a preview that is
            # somehow never followed by a render cannot leave hover switched off.)
            return
        tip_changed = self._compute_tooltip(event)
        readout_changed = self._compute_readout(event)
        if tip_changed or readout_changed:
            self._blit_overlay()

    def _compute_tooltip(self, event):
        """Point the near-cursor tooltip at whatever is under the cursor (no redraw).

        A pulse/capture shows the *other* attribute (length when labels are names, name
        when lengths); a dwell or dead-time band shows its duration; a control-flow box
        shows its caption when hovered near the stroke. Returns True if anything changed.
        """
        hit = self._hit_pulse(event)
        if hit is not None:
            lane_y, x0, x1, name, length_ns, kind = hit
            key = (name, x0, x1, lane_y)
            lo, hi = self.ax.get_ylim()
            anchor = ((x0 + x1) / 2, lane_y if lane_y is not None else (lo + hi) / 2)
            if kind == "time":
                text = f"{length_ns:.0f} ns"
            else:                       # toggle: the attribute not shown as a label
                mode = self.draw_kwargs.get("pulse_label", "name")
                text = f"{length_ns:.0f} ns" if mode == "name" else name
        else:
            region = self._hit_branch(event)
            if region is None:
                if self._hover_key is None:
                    return False
                self._hover.set_visible(False)
                self._hover_key = None
                return True
            x0, x1, _yb, yt, text = region
            key = ("branch", x0)
            xlo, xhi = self.ax.get_xlim()
            anchor = (max(x0, xlo) + 0.01 * (xhi - xlo), yt)   # visible top-left
        if key == self._hover_key:      # already showing this
            return False
        self._hover.xy = anchor
        self._hover.set_text(text)
        self._hover.set_visible(True)
        self._hover_key = key
        return True

    def _compute_readout(self, event):
        """Update the bottom-left readout: the cursor time, plus the pulse amplitude and
        IQ when the cursor is on a pulse (just the time otherwise). Returns True if the
        text changed (so a redraw is worthwhile)."""
        if self._readout is None:
            return False
        if event.inaxes is not self.ax or event.xdata is None:
            if self._readout_text is None:
                return False
            self._readout.set_visible(False)
            self._readout_text = None
            return True
        t_ns = event.xdata * self.divisor
        time_str = (f"t = {t_ns / 1000:.4f} µs" if self.divisor >= 1000
                    else f"t = {t_ns:.1f} ns")
        iq = self._iq_at(event)
        text = (time_str if iq is None else
                f"{time_str}   |A| = {abs(iq):.3f}   "
                f"IQ = {iq.real:+.3f}{iq.imag:+.3f}j")
        if text == self._readout_text:
            return False
        self._readout.set_text(text)
        self._readout.set_visible(True)
        self._readout_text = text
        return True

    def _iq_at(self, event):
        """Complex envelope value (DAC full scale) at the hovered pulse and time, or None
        when the cursor is not on a pulse. The envelope is sampled evenly across the bar,
        matching how draw() lays it down."""
        x, y = event.xdata, event.ydata
        if y is None:
            return None
        for lane_y, x0, x1, io_name, pulse in self._pulse_iq:
            if x0 <= x <= x1 and abs(y - lane_y) <= self.HOVER_HALF_HEIGHT:
                samples = self.trace.envelope(io_name, pulse)
                if samples is None or not len(samples):
                    return None
                frac = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
                idx = int(round(min(max(frac, 0.0), 1.0) * (len(samples) - 1)))
                return complex(samples[idx])
        return None

    def _hide_hover(self):
        dirty = False
        if self._hover is not None and self._hover.get_visible():
            self._hover.set_visible(False)
            dirty = True
        self._hover_key = None
        if self._readout is not None and self._readout.get_visible():
            self._readout.set_visible(False)
            self._readout_text = None
            dirty = True
        if dirty:
            self._blit_overlay()


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
