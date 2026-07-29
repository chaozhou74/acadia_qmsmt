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
| `qt_widget.py` | `SequenceWidget` — PyQt5 wrapper for acadia_gui |
| `notebooks/` | `explore_sequence.ipynb` (interactive), `visualize_sequence.ipynb` (static) |
| `selftest.py` | hardware-free regression: traces archived folders, checks each against its own `compiled.log`, exercises every render option |
| `docs/VALIDATION.md` | staged record of checking the model against the loopback station — what was tried, what it found |
| `docs/EMPIRICAL_CONSTANTS.md` | the only two numbers that are measured rather than derived, and the brief for tracking them down in acadia |
| `docs/EXAMPLE_FOLDERS.md` | one archived folder per runtime class (39 of them) with its structure — the quickest varied test set |
| `validation/` | the hardware-validation harness: the loopback test runtime, the deploy/measure/compare scripts, and the measured results |
| `docs/DEVELOPER_NOTES.md` | maintainer notes: known issues (KI_001–004), gotchas, open items |

`tracing`/`folder`/`compiled_log` are pure data — no matplotlib. `qt_widget` is the
only file that imports PyQt5, and it is not imported by `__init__`.

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

Scriptable: `view.set_window(t0_ns, t1_ns)`, `view.reset()`, `view.xlim_ns`.
matplotlib's own toolbar still works; this yields to it while its zoom/pan tool is
armed.

## In acadia_gui

The interaction lives here, not in the GUI, because it is pure matplotlib event
handling — the same `SequenceView` drives an ipympl canvas in a notebook and a
`FigureCanvasQTAgg` in Qt. The GUI supplies the window, the folder selection and
the toolbar; there is one implementation of the behaviour to keep correct.

acadia_gui is already matplotlib-in-Qt (`LivePlotWidget` uses `FigureCanvasQTAgg` +
`NavigationToolbar2QT`), so it drops in. In `acadia_gui/gui/main_data_browser.py`,
`RightPanelTabs`:

```python
from sequence_viz.qt_widget import SequenceWidget      # or acadia_qmsmt.sequence_viz

self.sequence_tab = SequenceWidget()
self.addTab(self.sequence_tab, "Pulse Sequence")
```

and wherever the browser learns the selected folder:

```python
self.sequence_tab.load_folder(path)
```

`load_folder` never raises — a non-data folder or a trace failure is reported in
the widget. The widget adds sweep-point / register-cycles / saved-qmsmt controls
and a jump-to-block dropdown.

Folder loading goes through `acadia_qmsmt.utils.saved_runtime_loader`, the same path
`LivePlotWidget` uses, so a folder that opens in the GUI traces here and vice versa.
`use_saved_qmsmt` defaults to `True` — the folder's own `acadia_qmsmt.py`, which is
the faithful choice for an archived run — and falls back to the installed package
with a warning if that import fails, exactly as `LivePlotWidget` does. Whichever
was used ends up on `trace.used_saved_qmsmt` and in `trace.summary()`.

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

## Inter-block dead time

A **blocking** `channel_synchronizer` does not hand off seamlessly. At block exit
`DMASynchronizer.__exit__` emits the DMA trigger, a `bus_read` of `dma_running`,
and a `repeat_until` that holds the PC until the mask clears. Only once that poll
releases does the sequencer push the next block's commands, pad with the
FIFO-latency NOPs from `calculate_trigger_delay`, and trigger again. All of it is
dead air on every channel, and the visualizer accounts for it.

**Validated against hardware.** The 4-channel DAC→ADC loopback
(`validation/timing_validation.py`) measures pulse intervals directly. Across
**29 cases** — straight-line, barrier-padded, stretched, register-driven, looped and
branched — every one agrees to **≤0.14 ns** (0.028 cycles), excluding three documented
*measurement* systematics and the two KI_004 variants:

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

`sequencer_block_gaps()` reports four contributions per boundary:

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

### Colour

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

`color_by="name"` restores name-keyed colour (one hue per name everywhere, merged
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

## Moving this into acadia_qmsmt

Depends only on `acadia`, `acadia_qmsmt`, `numpy`, `matplotlib` and (in
`qt_widget` alone) PyQt5. Move the directory as-is. The only change needed is the
`sys.path` bootstrap in the notebooks (and the `import sequence_viz as sv` in
`selftest.py` / `validation/*`), which becomes `from acadia_qmsmt.sequence_viz import ...`.
