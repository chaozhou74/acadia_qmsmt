# sequence_viz

Visualize the pulse sequence a `QMsmtRuntime` compiles — from a live runtime
object or from an archived data folder. No hardware required.

```python
import sequence_viz as sv

view = sv.explore_folder("/path/to/data_folder")   # interactive, drag to zoom
fig, ax, trace = sv.plot_folder("/path/to/data_folder")   # static figure
```

## Layout

| file | role |
|---|---|
| `tracing.py` | the decompiler: runs the runtime off-hardware, builds a `SequenceTrace` |
| `dryrun.py` | hardware stubs + host-memory `fake_attach` |
| `folder.py` | reconstruct a runtime from a data folder |
| `compiled_log.py` | decode `compiled.log`; cross-check a trace against it |
| `plotting.py` | `draw()` — the zoom-aware renderer; `plot_trace()` for a static figure |
| `interactive.py` | `SequenceView` — drag-box zoom, scroll, pan, reset |
| `notebooks/` | `explore_sequence.ipynb` (interactive), `visualize_sequence.ipynb` (static) |
| `validation/selftest.py` | hardware-free regression: traces archived folders, checks each against its own `compiled.log`, exercises every render option |
| `docs/MAINTENANCE.md` | the workflow to update the visualizer when acadia/acadia_qmsmt change (symptom → coupling point → fix → re-validate) |
| `docs/EMPIRICAL_CONSTANTS.md` | the only two numbers that are measured rather than derived, and the brief for tracking them down in acadia |
| `validation/` | the regression net for acadia/acadia_qmsmt drift: loopback test runtime, deploy/measure/compare scripts, measured results — run these when those packages change (`validation/README.md`) |
| `docs/DEVELOPER_NOTES.md` | maintainer notes: known issues (KI_001–004), gotchas, open items |

`tracing`/`folder`/`compiled_log` are pure data — no matplotlib. Nothing in this package
imports PyQt5; the Qt wrapper lives in acadia_gui (see *In acadia_gui* below).

## How it works

`Acadia.__init__` touches no hardware — every `/dev/mem` mapping happens in
`attach()`. So the module stubs out `attach`, the RFDC/NCO calls and `run()`,
backs each managed memory pool with a host `bytearray`, and lets `main()` run
normally up to the first `acadia.run()`.

Every timing-relevant operation — `schedule_pulse`, `dwell`, the ADC capture
inside `measure` — funnels through `Acadia.channel_synchronizer`. On block exit,
`DMASynchronizer.merge_schedules` does the barrier reconciliation and returns the
per-channel command list with lengths in cycles. **We hook that function rather
than reimplementing its timing model**, so the picture is what the FPGA plays.
It also extends the subschedule lists in place, so reading them back afterwards
exposes the alignment dwells acadia inserted — that is why barrier padding is
visible and distinguishable from your own dwells.

Pulse envelopes come from `InputOutput.compute_pulse`, which is pure numpy.

## Interactive view

```python
%matplotlib widget          # in a notebook; without this the canvas gets no events
view = sv.explore_folder(folder)
view.ax.figure
```

| gesture | action |
|---|---|
| drag a box | zoom to that time window; a tall box also restricts the lane range |
| double-click | reset |
| `r` | reset |
| scroll | zoom the time axis about the cursor |
| shift + drag | pan |

Zooming **re-renders** rather than rescaling: `draw()` receives the visible window
and picks the label density, the ns/µs unit and whether to draw envelopes from it,
culling marks outside. That is what makes deep zoom useful — at a 200 ns window you
can see that two readout pulses have different ramp lengths.

Scriptable: `view.set_window(t0_ns, t1_ns)`, `view.set_lanes(y0, y1)`,
`view.reset()`, `view.xlim_ns`, `view.ylim`. Lane coordinates count from the
bottom — channel *i* of `trace.channels` sits at lane `len(channels) - 1 - i`, so
the first channel is the top one — and `view.full_ylim` is the whole stack.
matplotlib's own toolbar still works; this yields to it while its zoom/pan tool is
armed.

`on_viewport(xlim_ns)` and `on_lanes(ylim)` are called after every render, which is
how acadia_gui keeps its two scrollbars in step with the view.

### Why a gesture is not one render per event

A frame is expensive — ~53 ms for a gesture frame and ~70 ms for a full one on a
9-lane RB sequence, nearly all of it matplotlib rastering the figure. Gestures emit
events far faster than that (a scrollbar drag, one per pixel of travel), so
`SequenceView` never tries to render one per event:

* **a pan is shown immediately, without rendering.** `_preview_pan` blits the last
  drawn frame back shifted by the pan — a few ms — so the plot stays under the
  handle instead of trailing it. The strip scrolling into view is painted with the
  axes background until a real frame refills it. Measured on a 2-screens-per-second
  drag: 54 fps and 4 px of lag, against 8 fps and 130 px before. Past
  `MAX_PREVIEW_SHIFT` of the plot width the shift is mostly blank strip, so the
  last complete frame is held instead.
* **the time axis only.** A lane pan is not a translation of the picture: the gap
  bands and the barrier and block-start lines span the full height in *axes*
  coordinates, so they stay put while the lanes move past them. Shifting the image
  would drag them along — measured 2 px out and a fifth of the plot wrong, against
  0 px and 2% (half-pixel rounding) for a time pan.
* frames are drawn no more often than one *measured* frame time apart
  (`_on_draw` times request-to-drawn, which is the only honest measure since the
  raster is deferred to the event loop); intermediate windows are dropped, never
  queued;
* frames drawn *during* a gesture pass `fast=True` to `draw()`, which skips the
  per-window labels (53 ms against 70). The legend stays up: it costs 4.6 ms and is
  the most conspicuous thing on the plot to have blinking on and off.
* the full-detail frame follows one interval after the last event, so labels and
  the hover index come back as soon as you let go;
* hovering repaints only the tooltip and the readout, blitted over a snapshot of
  the plot, instead of a full redraw per mouse-move.

`draw()` reuses the axes rather than calling `ax.clear()`, which is what made those
numbers reachable: clearing rebuilds the ticks, their labels and the spines from
rcParams every frame — 75 artists and thousands of deepcopies for a window that
owns nine bars, 44 ms of a 70 ms frame. It now replaces only the marks it drew
(tracked on `ax._sequence_viz`) and rebuilds the chrome when the channels or the
theme change. Anything else you add to those axes — the tooltip, the rubber band —
is now yours to remove.

Coalescing means there are **two windows at once**, and mixing them up is the one
real trap here: `xlim_ns` is where the view is going, `_drawn_xlim` is what the axes
currently show. Everything matplotlib hands back — an event's `xdata`, the hover
index, the rubber band — is in the *drawn* frame's units, so it converts with
`view.divisor` (defined on the drawn window for exactly this reason). Gestures
compose onto the *pending* window instead and therefore work in pixels
(`_x_to_ns`), which is unit-free. Scaling a drawn `xdata` by the pending window's
divisor is out by 1000× whenever a scroll crosses the 5000 ns ns/µs boundary
between frames — that bug sent the view to −2000 µs on a fast wheel.

Zooming out and panning are clamped to the sequence: the window slides rather than
clipping, so it keeps the zoom you asked for, and the scrollbars — which model the
window as a slice of the sequence — stay well defined.

`view.set_window(..., throttle=True)` and `view.set_lanes(..., throttle=True)` opt a
caller into that pacing — what acadia_gui's two scrollbars use. Without it, dragging
a scrollbar queued a full render per emitted value and the plot crawled along behind
the handle.

## In acadia_gui

`SequenceView` (in `interactive.py`) is pure matplotlib event handling, so the same class
drives an ipympl canvas in a notebook and a `FigureCanvasQTAgg` in Qt — one implementation of
the zoom/pan behaviour to keep correct. The Qt wrapper that embeds it in acadia_gui lives
**in acadia_gui** (`gui/sequence_view.py`), not in this package, which is why nothing here
imports PyQt5.

The wrapper adds one scrollbar per axis — time along the bottom, lanes down the right — each
enabled only while that axis is zoomed in. Both show where you are (the handle is the
visible fraction of the sequence, or of the channel stack) and let you walk along at a fixed
zoom, which the mouse gestures alone cannot do. Both pan continuously: `QScrollBar` is
int-only, so the time bar counts in whole nanoseconds and the lane bar in thousandths of a
lane, each far finer than a pixel of canvas. The lane bar's value is measured downwards, as
Qt counts a vertical bar: 0 is the top of the channel stack.

That wrapper's `load_folder(path)` never raises — a non-data folder or a trace failure is
reported in the widget. Folder loading goes through `acadia_qmsmt.utils.saved_runtime_loader`
(the same path the GUI's plot view uses); `use_saved_qmsmt` defaults to the folder's own
`acadia_qmsmt.py` and falls back to the installed package, ending up on
`trace.used_saved_qmsmt` and in `trace.summary()`.

## Sweep points

The sequence is compiled **once**, so every sweep point shares one schedule —
only the pulse data and the cache differ. So one dry run captures them all: at
each `acadia.run()` the DAC memories and the sequencer cache are snapshotted, and
any point can then be selected with no re-tracing.

```python
trace = sv.trace_folder(folder)      # 96 points in 0.06 s
trace.n_points                       # 96
trace.select_point(42)               # ~0.05 ms
view.set_point(42)                   # same, and redraws
```

Two things make this practical:

- **`iterations` is forced to 1** for the dry run (restored afterwards). Iterations
  just repeat the same points — the DR-tomography runtime is 96 distinct points but
  1.92M runs otherwise. `trace.iterations_forced` records it.
- **Snapshots are deduplicated** by content, so unchanged memories are shared: the
  beam-splitting run's 3362 points hold 40344 memory references but only 212
  distinct arrays, 0.29 MB total.

`max_points` (default 4096) caps the capture; `trace.truncated_points` and
`summary()` say when it was hit. `capture_points=False` reverts to stopping at a
single point, which is quicker for one look at a huge sweep.

### Register-driven lengths resolve themselves

A dwell driven by a `Register` has no length at compile time (`REG0` in T1). But
the register is loaded *from the cache*, and the cache is captured per point — so
the real value is available:

```
T2 echo:  point  0: REG0 = 50.0 ns    -> sequence   3225 ns
          point 25: REG0 = 50050.0 ns -> sequence 103225 ns
          point 50: REG0 = 100050.0 ns-> sequence 203225 ns
```

The mapping is read out of the compiled program, which spells a register load as
an address write followed by the data landing in the register:

```
001D0000 -> BUS_ADDR  |  REG0 -> NONE      <- cache base + word index
BUS_DATA -> REG0      |  REG0 -> NONE
```

So each register is tied to its *own* cache word — no single-register restriction.
`trace.registers` reports what was found, and a register fed by something other
than the cache (a CMACC accumulator, i.e. a measurement result) is named but has
no static value:

```python
trace.registers        # {"REG0": {"source": "cache[0]", "cache_word": 0}}
trace.register_label("REG0")           # "REG0 = cache[0]"
```

Three ways a length gets its value, in precedence order — `Command.resolution`
records which was used, and `summary()` reports it per register:

| `resolve_registers={"REG0": 400}` | explicit override, in cycles — for a register with no static value |
| the cache | automatic, per sweep point |
| `resolve_indeterminate` | blanket fallback for anything left, e.g. a DSP-driven length |

`register_names={"REG0": "t_echo"}` sets a display alias.

### The streamed-gate loop's bound

A streamed gate loop states its end as `Register.load(cache_base + region + cache[count_word])` —
an address *relative* to the cache, not a value in it — and acadia emits that as arithmetic in a
DSP rather than a bus read, so the pattern above does not match it:

```
001D0168 -> BUS_ADDR                                          <- the count word
...bus latency...
001D0000 -> DSP_AB5                                           <- the immediate addend
DSPConfiguration(mode='AB+C') -> DSP_CFG5  |  BUS_DATA -> DSP_C5
DSP_P5 -> REG3                                                <- the sum lands here
```

`describe_cache_sums` reads those, so `REG3` resolves per point as `addend + cache[word]` and
reports as `cache[360] + 0x1D0000`. That makes `repeat_until(pointer == REG3)` arithmetic: the
count is `target - where the pointer is`, and *where the pointer is* accumulates over the guarded
loops before it — which is what lets the two XEB families share one pointer and one shot enter
exactly one of them (`test(pointer != final)` is decided the same way, so the family this run did
not play is drawn as skipped rather than assumed taken). Before this, both families drew one
assumed pass and every XEB run looked like a 2-cycle circuit whatever its depth.

A cache-pointer **stream** (randomized benchmarking) is deliberately excluded: `_expand_stream`
already unrolls that loop from the same count word, one command per gate inside a single pass, so
resolving its pass count as well would draw the whole train once per pass — N² gates.

## Inter-block dead time

A **blocking** `channel_synchronizer` does not hand off seamlessly. At block exit
`DMASynchronizer.__exit__` emits the DMA trigger, a `bus_read` of `dma_running`,
and a `repeat_until` that holds the PC until the mask clears. Only once that poll
releases does the sequencer push the next block's commands, pad with the
FIFO-latency NOPs from `calculate_trigger_delay`, and trigger again. All of it is
dead air on every channel, and the visualizer accounts for it.

**Validated against hardware.** The 4-channel DAC→ADC loopback
(`validation/timing_validation.py`) measures pulse intervals directly. Coverage is in three
layers, all deployed and measured on the board:

* **62 hand-written cases** — straight-line, barrier-padded, stretched, register-driven, looped,
  branched, batched behind either FIFO drain, and read out through `measure()` /
  `measure_trace()` / two resonators at once;
* **every ordered PAIR of the 10 scheduling primitives** (100 deploys) and **125 ordered
  TRIPLES** over the FIFO/branch-stateful subset — exhaustive adjacency coverage, because
  scheduling errors are properties of a JOIN rather than of a construct in isolation;
* **randomly generated sequences** (`random_seq`), for interactions no hand-written case reaches.

The triple sweep runs **0 failures at 0.23 ns worst**, and re-scoring every archived run against
the model gives the same **0.23 ns**, with no residual above 1 ns that is not a classified
systematic. Documented *measurement* systematics (the
mixed-ramp stretch edge, slow-ramp edge jitter) and the two KI_004 variants are excluded and
listed in `validation/README.md`, which also records the model bugs this net has caught. The
`REGn -> BUS_DATA` streaming idiom (the multi-rail XEB runtimes) resolves as of 2026-08-24: the
loop bound is a register loaded as `immediate + cache[word]` through an `AB+C` DSP, which
`describe_cache_sums` now reads, so those trains draw their real per-point length instead of one
assumed pass — see *The streamed-gate loop's bound* below.

A second harness, `validation/render_validation.py`, closes the other half of the chain: it
renders a trace to a real figure, reads the patches back off the axes and checks every rectangle
against the command it stands for. `board <-> trace` and `trace <-> drawing` are both measured.

The older 29-case table below is kept because its per-case gap breakdown is still the clearest
illustration of how the boundary gap is built up:

```
case                   blocks         gaps (ns)   worst err
single                      3            [80.0]     0.00 ns
two_same_block              3            [80.0]     0.00 ns     back-to-back, one block
two_blocks                  4      [75.0, 80.0]     0.09 ns
two_blocks_1ch              4      [60.0, 80.0]     0.04 ns     1 DMA push in block 2
two_blocks_2ch              4      [65.0, 80.0]     0.04 ns     2 pushes
three_blocks                5 [75.0,75.0,80.0]      0.04 ns     gap recurs per boundary
four_blocks                 6 [75,75,75,80.0]       0.14 ns     compounds correctly
batch_nonblocking           3                []     0.00 ns     block=False, seamless
dwell_between               3            [80.0]     0.12 ns
register_dwell              3            [80.0]     0.04 ns     length from a register
barrier_single_channel      3            [80.0]     0.00 ns     padding aligns all 4
loop_2                      3     [100.0, 90.0]     0.03 ns     unrolled, back-edge costed
loop_3                      3 [100.0,100.0,90.0]    0.09 ns     linear in pass number
test_true                   5 [80.0, 75.0, 80.0]    0.04 ns     body taken
test_false                  4      [95.0, 80.0]     0.10 ns     body skipped, gap predicted
```

Run `timing_validation.py --revalidate` for the full table.

The boundary-gap model (`edge_gap()`, applied by the execution-model layout) has four
contributions per blocking boundary:

| term | where it comes from |
|---|---|
| `detect` | `Acadia._bus_latency("dma_running")` — the poll reads a value that stale, so the deassertion is seen that many cycles late. **3 cycles** on CONFIG_200 |
| `issue` | counted **exactly out of the compiled program**: instructions from the poll through the next `Trigger DMAs`. Varies per boundary — 6 for one pulse, 15 when datamover configuration sits in between |
| `propagate` | `dma_trigger_dataport` pipelining + the cycle the DMA takes to latch the FIFO output (see the `calculate_trigger_delay` docstring). **2 cycles**; counted once, as it applies to both blocks |
| `branch_penalty` | **3 cycles, measured not derived**, and only on an edge crossed by a *taken* branch — a loop back-edge or a skipped `test` body. Fitted on `loop_2`/`loop_3`/`loop_2_double` (backward edges), then it predicted the forward `test_false` skip gap at 95 ns with no retuning. See `docs/EMPIRICAL_CONSTANTS.md` |
| `measured_offset` | **1 cycle, measured not derived.** The three terms above come out exactly one cycle short at every boundary, independent of push count, while intra-block layout is exact. Which term owns it is not separable by timing alone — `detect` one low, `propagate` one low, or an unmodelled DAC start latency — so it is kept as its own named term rather than silently folded in |

At 200 MHz (5 ns/cycle) that lands at **55–100 ns per blocking boundary** on the
runtimes checked. Real cost varies with the sequence:

```
SimulBus  25 blocks   280 ns dead  ( 0.6%)
DR tomo    5 blocks   340 ns dead  ( 1.0%)
Chevron    5 blocks   340 ns dead  (12.1%)
T2E        5 blocks   285 ns dead  ( 9.1%)
```

**Non-blocking blocks get no gap** — their commands queue in the DMA FIFO and play
back-to-back. That is precisely why batching with `block=False` avoids this cost,
and the plot shows the difference directly.

`detect` and `propagate` are read from firmware constants and the documented
hardware behaviour; `issue` is counted, not modelled; the last two are measured. With
all four terms the boundary is good to **a small fraction of a cycle** on every case
tested — but two of the terms are empirical, so a firmware or gateware change can
invalidate them silently. `--revalidate` is the check.
Per-block numbers are on `block.gap_after` / `block.gap_breakdown`, the
total on `trace.dead_time_ns`, and the band is drawn in the plot (`show_gaps=False`
to hide).

## Checking a trace against what actually ran

`compiled.log` in the folder is the compiled sequencer program. It can be decoded
with no execution at all, and used to verify the re-trace:

```python
sv.compare_with_compiled_log(trace, folder)
# {'match': True, 'blocks': 25, 'triggers': 25,
#  'commands_retrace': 217, 'commands_archive': 217, ...}
```

This doubles as a regression test: edit a runtime, re-trace, and diff against a
known-good archived run.

## Reading the plot

### Color

`color_by="memory"` (default) gives **one hue per waveform memory**, keyed on
`(channel, address)`. Two `schedule_pulse` calls that share a memory necessarily
play the same samples, so they share a hue; duplicates made to hold different
phases or amplitudes get their own; and two pulses that merely share a *name* on
different channels no longer collide — `swap` on `qubit1_bs_stimulus` (80 ns,
scale 0.74) and on `qubit2_bs_stimulus` (170 ns, scale 0.21) are different pulses
and now look it.

A run can hold more memories than the palette has slots (12 in one beam-splitting
run, against 8 hues). Rather than invent a 9th hue the palette wraps and the
second generation is drawn with a dark outline, so identity stays unambiguous.
Legend entries are named by pulse, prefixed with the channel only when that name
is ambiguous (`DAC9/swap`).

`color_by="name"` restores name-keyed color (one hue per name everywhere, merged
across channels); `color_by="channel"` gives one hue per lane.

### Envelope

Shapes are real, not decorative — but read the options, because the default trades
amplitude fidelity for shape legibility.

| option | values | meaning |
|---|---|---|
| `envelope_source` | **`memory`**, `config` | `memory` reads the DAC waveform memory, i.e. the samples actually loaded at this sweep point — swept scale, detune and phase included. `config` recomputes from the yaml, which ignores anything `load_pulse` overrode (the nominal pulse) |
| `envelope_scale` | **`per-pulse`**, `channel`, `shared`, `absolute` | `per-pulse` normalises each pulse to its own peak, so every bar is full height and **amplitude is not readable**. The others divide by the loudest pulse on that DAC / anywhere / DAC full scale, making heights comparable |
| `envelope_mode` | **`magnitude`**, `iq` | `magnitude` draws `\|s\|`, which hides detune, phase and the DRAG quadrature entirely — a −131 MHz detuned pulse looks flat-topped. `iq` draws I and Q about the lane centre and shows all three |

`trace.envelope(io_name, pulse, source)` gives the samples directly; the memory
source falls back to the config when a pulse was never loaded.

### Marks

| mark | meaning |
|---|---|
| filled bar | a pulse; the line on top is its envelope |
| dark outline on a bar | palette wrapped — a second-generation hue |
| bar labels | pulse name |
| hatched `///` grey | dwell acadia inserted to align channels at a barrier |
| plain grey | a dwell you scheduled |
| white outlined bar | ADC capture window |
| cross-hatched | register/DSP-driven length, symbolic at compile time (`resolve_indeterminate=` to give it a value in cycles) |
| dotted vertical | barrier inside a synchronizer block |
| dashed violet box | conditional region — one pass shown, real count is data-dependent |
| pink band | inter-block dead time — see above |

Above ~400 bars in view, per-bar labels and envelopes are dropped — they are
unreadable at that density and drawing them makes dragging feel sticky. Zoom in
and they come back.

## Limits

- **Amplitude is hidden by the default `envelope_scale="per-pulse"`.** Switch to
  `channel`/`shared`/`absolute` to read it.
- **A memory reloaded per run shows one point's contents.** A tomography
  placeholder rewritten every run (`swap_copy1` in the DR-tomography runtime) is
  drawn with whatever the selected point loaded; nothing on the plot flags that
  it is 1 of N. Step `point` to see the others.
- **Register resolution needs an unambiguous cache.** Multiple registers or
  multiple cache words in play fall back to `resolve_indeterminate`.
- **Tracing the same runtime object twice fails** (`Processor data is non-empty`) —
  `main()` recompiles onto a sequencer that already holds a program. `trace_folder`
  builds a fresh runtime each call, so this only bites when reusing a live object.
- **Register-driven lengths are symbolic** at compile time. Pass
  `resolve_indeterminate` (in cycles) to render a concrete instance.
- **Two terms in the inter-block gap are empirical** — the +1 cycle boundary offset
  and the 3-cycle taken-branch penalty. They hold across 29 measured cases, but they
  are not derived from anything, so a gateware change could invalidate them without
  any error being raised. `docs/EMPIRICAL_CONSTANTS.md`; re-check with
  `timing_validation.py --revalidate`.
- **`repeat_until` shows one pass.** Its trip count depends on a live measurement,
  so there is nothing to unroll. The region is drawn once, outlined, and captioned
  `repeat_until(REG0 == 0) — 1 pass shown; real count is data-dependent`. With
  active reset the real duration is longer by (extra passes) × the body.
  `loop(N)` does *not* have this limit — it is unrolled exactly (validated to
  ≤0.09 ns) and each pass is captioned `pass k of N`.
- **`test(speculation=False)` is not modelled — KI_004.** That layout puts the body
  out of line, so address order stops being execution order and the instruction
  count does not apply; measured 25 ns out on the taken arm, and the skipped arm
  *hangs the sequencer*. Such a region lands in `trace.unsupported_paths` and is
  captioned `TIMING NOT MODELLED (see KI_004)` rather than drawn as if trusted.
  The default `speculation=True` is validated on both arms (0.04 / 0.10 ns).
- **An undecidable `test` is assumed taken.** If the condition cannot be resolved
  from the cache, the body is drawn and the block is listed in
  `trace.assumed_paths`, with the caption saying `assumed taken`. Force the other
  arm without re-tracing:
  `trace.path_choices[block_index] = False; trace.relayout()` (same for
  `trace.loop_counts[block_index] = n`).
- **Readout kernel `.npy` files are not archived** with the folder. If one has
  moved, window loading fails; this does not affect the timeline.
