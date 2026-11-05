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
| `docs/TIMING_MODEL.md` | how the tracer times DMA commands, and where each number comes from in the gateware |
| `validation/` | the regression net for acadia/acadia_qmsmt drift: loopback test runtime, deploy/measure/compare scripts, measured results — run these when those packages change (`validation/README.md`) |
| `docs/DEVELOPER_NOTES.md` | maintainer notes: known issues, gotchas, open items |

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

Scriptable: `view.set_window(t0_ns, t1_ns)`, `view.reset()`, `view.xlim_ns`.
matplotlib's own toolbar still works; this yields to it while its zoom/pan tool is
armed.

## In acadia_gui

`SequenceView` (in `interactive.py`) is pure matplotlib event handling, so the same class
drives an ipympl canvas in a notebook and a `FigureCanvasQTAgg` in Qt — one implementation of
the zoom/pan behaviour to keep correct. The Qt wrapper that embeds it in acadia_gui lives
**in acadia_gui** (`gui/sequence_view.py`), not in this package, which is why nothing here
imports PyQt5.

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

### Arithmetic on a register resolves too

A register is often not loaded from the cache directly but computed from one:

```python
flag.load(rounds_reg & (1 << (k - 1)))     # rounds_reg = cache[0]
with a.sequencer().test(flag != 0):        # ... run round k
```

`describe_registers` cannot see this — it reads the compiled program, where the only
thing attributable to a register is a direct `BUS_DATA -> REGn` bus read, and `flag` is
reloaded with a *different* mask before each round's test. So the dry run records what
each register was loaded **with** (the un-evaluated expression, via the register class's
own `load`), and each `test`/`loop` context snapshots that map as it is entered — which
is what ties a test to the value its register held *there*, straight out of Python
execution order. `evaluate_condition` then resolves a condition recursively down to
cache-fed or pinned leaves.

The practical effect: those round-enable tests follow the sweep point. In a runtime that
sweeps the round count this way (`rounds_reg` = 0, 1, 3, 7, 15, 31), stepping `point` grows
the drawn timeline round by round instead of drawing one fixed picture.

### Pinning a measurement-fed register

A register fed by live hardware — `qubit_e = measurement & QUADRANT_2` — has no value
off-hardware, so the branch it gates can only be guessed. Pin it and the branch is decided
outright:

```python
trace.register_overrides["REG0"] = 0        # or the bit the runtime masks for
trace.relayout()
```

`trace.register_summary()` lists what is worth pinning, under the runtime's own variable
name; in the GUI these are the Registers panel's spin boxes, where the minimum reads
`auto` (no override) so a deliberate pin to 0 is distinguishable from an unset box. With
two such registers gating a feedforward, pinning both selects each of the four paths
exactly, with nothing left assumed. Note the units differ by kind: a
length override is in cycles, a condition register's is the raw register word.

`register_names={"REG0": "t_echo"}` sets a display alias — though usually there is
nothing to set: every `Register()`/`DSP()` is named after the runtime's own variable
(`t_echo = a.sequencer().Register()` is enough on its own), read straight off that
call site's source line. `register_names` only overrides one, or fills in a name for
a call site that isn't a plain assignment (a tuple-unpack, an attribute target).

A symbolic length that turns out to be pure arithmetic over already-known numbers —
acadia's own barrier-padding `max(a, b) - c`, most commonly — is not a register at
all, and does not appear in `register_summary()` or the GUI's register panel; only
genuine register/DSP-driven quantities do. It stays symbolic on the timeline itself
(cross-hatched, at a placeholder width) rather than being laid out at that resolved
length: the schedule this is read from can carry such a padding entry without acadia
ever emitting a matching DMA instruction for it, so trusting the folded number for
*layout* invents commands `compare_with_compiled_log` then can't find in the archive.

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

A blocking boundary is a poll on `dma_running`, which the sequencer runs as a PC hold.
The dead time after a block follows from the gateware (`docs/TIMING_MODEL.md`):

| term | where it comes from |
|---|---|
| `issue` | counted **exactly out of the compiled program**: instructions from the poll through the next `Trigger DMAs`, along the path that executes (`edge_gap()`). Varies per boundary — 6 for a block with one pulse, 15 when datamover configuration sits in between |
| `branch_penalty` | 3 cycles per *taken* branch on that path — a loop back-edge or a skipped `test` body; consecutive skipped `test` bodies (each guard jumps only past its own body) are followed as a chain |
| fixed part | 6 cycles: the poll reads `dma_running` low 1 cycle after the playout ends, the instruction after a hold that waited runs 3 cycles later, and a trigger reaches the output 4 cycles after it executes, less the 2 instructions already in `issue`. The pipeline settings come from the firmware configuration (`tracing.gateware_timing`) |

A poll that is already satisfied when it first executes falls through in one cycle, so a
block that ends within a cycle of starting (a one-cycle `dwell`) costs 2 cycles less.

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

Nothing in this is fitted. The boundary is good to **a small fraction of a cycle** on
every loopback case; a gateware change that moves any of these numbers shows up as
whole-cycle errors in `--revalidate`.
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
| dashed violet box | conditional region; captioned with its condition, plus `— guessed` when the branch could not be decided |
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
- **The inter-block gap is read off the gateware**, not measured at run time, so a
  gateware change would not raise any error. `docs/TIMING_MODEL.md` lists what it
  relies on; re-check with `timing_validation.py --revalidate`.
- **`repeat_until` shows one pass.** Its trip count depends on a live measurement,
  so there is nothing to unroll. The region is drawn once, outlined, and captioned
  `repeat_until(REG0 == 0) — 1 of ? passes`. With
  active reset the real duration is longer by (extra passes) × the body.
  `loop(N)` does *not* have this limit — it is unrolled exactly (validated to
  ≤0.09 ns) and each pass is captioned `pass k of N`.
- **`test(speculation=False)` is not modelled — KI_004.** That layout puts the body
  out of line, so address order stops being execution order and the instruction
  count does not apply; measured 25 ns out on the taken arm, and the skipped arm
  *hangs the sequencer*. Such a region lands in `trace.unsupported_paths` and is
  captioned `TIMING NOT MODELLED (see KI_004)` rather than drawn as if trusted.
  The default `speculation=True` is validated on both arms (0.04 / 0.10 ns).
- **An undecidable `test` is assumed taken — except a mutually-exclusive pair only
  draws one side.** If the condition cannot be resolved from the cache, the body is
  drawn and the block is listed in `trace.assumed_paths`, with the caption saying
  `assumed taken`. A runtime often keeps a data-dependent branch's timing symmetric by
  compiling it as TWO
  `test()` blocks whose conditions are exact negations of one another on the same
  live-measurement-fed register (`test(REGn != 0)` / `test(REGn == 0)`) — real
  hardware takes exactly one, but each is independently unresolvable off-hardware.
  Defaulting each independently used to draw BOTH as if they ran;
  now every `test` whose condition has a counterpart that is its exact textual
  negation anywhere in the trace (`trace.mutually_exclusive_paths`) picks one side
  once and every later occurrence of that same physical condition — or its negation,
  including a register reused across separate `test()` pairs, as in a feedforward
  repeated over rounds — mirrors it, so exactly one is
  ever drawn. Nested `test`s are gated correctly too: a block only executes if EVERY
  enclosing `test`/`loop`/`repeat_until` is taken, not just its innermost one. Force
  either arm without re-tracing:
  `trace.path_choices[block_index] = False; trace.relayout()` (same for
  `trace.loop_counts[block_index] = n`).
- **Readout kernel `.npy` files are not archived** with the folder. If one has
  moved, window loading fails; this does not affect the timeline.
