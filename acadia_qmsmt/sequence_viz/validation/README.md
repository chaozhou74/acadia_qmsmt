# validation — the regression net for acadia / acadia_qmsmt drift

`sequence_viz` works by decompiling acadia's *own* compiled program, so it leans on acadia
internals that carry no stability promise: the text of `instruction.pprint()`, the shape of
`DMASynchronizer.merge_schedules`, `acadia._firmware` keys, private memory pools, and the
`compiled.log` line format. When acadia or acadia_qmsmt change, those assumptions can break —
sometimes silently. This module exists to catch that. For the full update recipe (symptom →
coupling point → fix → re-validate) see `../docs/MAINTENANCE.md`.

## Run these when acadia or acadia_qmsmt changes

Paths below are relative to the `sequence_viz/` package root.

| command | needs | catches |
|---|---|---|
| `python validation/timing_validation.py --dryrun all` | packages only | compile-time / trace breakage on every case |
| `python validation/selftest.py` | archived data (see below) | a trace that fails or no longer matches its own `compiled.log` — i.e. any decode/hook/format change |
| `python validation/timing_validation.py --revalidate` | archived measured runs | changes to the *timing model* (gap counting, empirical constants) that a trace alone wouldn't reveal, re-derived against measured intervals |
| `python validation/timing_validation.py --all` | the loopback board | full re-measure — only needed if the gateware itself changed |
| `python validation/render_validation.py` | packages only | a DRAWING that disagrees with the trace — the second half of the chain (see below) |
| `python validation/timing_validation.py --pairs` | the loopback board | every ordered PAIR of scheduling primitives, exhaustively |
| `python validation/timing_validation.py --fuzz-steps N --scan random_seq:fuzz_seed=0,1,...` | the loopback board | randomly composed sequences, for interactions no hand-written case covers |
| `python validation/nesting_boxes.py` | archived data | one dashed box per ENTRY, and a clickable handle on every construct and execution — including ones pinned to zero passes, which used to delete their own control |
| `QT_QPA_PLATFORM=offscreen python validation/gui_validation.py --broad` | archived data | the panel against ONE RUN PER EXPERIMENT TYPE (24). Breadth by sequence shape, which is where the default three deep-nesting folders are blind: straight-line runs with no control flow, streamed RB trains, register sweeps |
| `QT_QPA_PLATFORM=offscreen python validation/gui_robustness.py` | archived data | can the panel be broken? every callback guarded (checked from the source), hostile inputs, and random event storms per seed in their own process |
| `python validation/streaming.py` | archived data | RB and XEB gate trains across sweep points -- the two idioms whose gates are decoded from the CACHE, which `compiled_log.compare` is structurally blind to. Asserts XEB does not decode an identical train at every point |
| `python validation/notebook_path.py` | archived data | the viewer WITHOUT Qt -- `SequenceView` on a bare axes and `plot_trace`, as a notebook uses it. No panel in between, and `interactive.py` is where the stale-tab-text and viewport-crop bugs lived |
| `python validation/path_independence.py [N]` | archived data | the panel must be a pure function of (folder, sweep point, pins) -- three routes to the same state, plus a fresh trace of it, must fingerprint identically. The two knobs interact (a point change decides registers, which decide counts and arms), so click order is a real axis |
| `python validation/zoom_extremes.py [N]` | archived data | the viewport at every width from the whole sequence down past the floor, plus the malformed windows a drag, a wheel or a typed number can produce (inverted, zero, negative, NaN, infinite). Finite, ordered, on-sequence, idempotent, and `reset()` recovers from any of them |
| `python validation/sweep_points.py [N] [--points K]` | archived data | the drawing invariants at a DENSE sample of sweep points rather than the six the GUI gate uses. A point's cache decides register lengths, test arms and loop counts, so the sequence can change length 4x and gain or lose constructs across a sweep |
| `python validation/render_geometry.py [N]` | archived data | the same run at 8 canvas sizes (a cramped dock to a 40-inch strip) x both themes. Layout decisions made in PIXELS behave differently per canvas, and a theme must change colours, never content -- the dark render must hold the same bars, tabs and text as the light one |
| `python validation/hover_truth.py [N]` | archived data | what the cursor SAYS, not just that it survives: the tooltip must report the same length zoomed in and out (the hover index is rebuilt per window and carries the plotted unit), every index entry must match something the trace contains, an empty stretch must not claim a bar, and the bottom-left readout -- the only NUMBER the viewer gives about the data -- must anchor to the envelope's ends, follow it across the bar, and report the same instant identically at either zoom |
| `python validation/navigation.py [N]` | archived data | the controls that MOVE you must land where they say: every jump-list entry leaves its own span inside the window, a window written to the scrollbar and read back is the same window, dragging one way moves the view that way, and the tail is reachable at every zoom |
| `python validation/cache_write_alignment.py` | **board** | which host-side numpy SLICE writes into a `CacheArray` survive. One deploy per (length, offset), nothing else varying; a write that faults takes the board process down with no traceback and returns no data, so the outcome is binary. MEASURED 2026-08-24 over 15 points: survives iff the byte length and byte offset are both multiples of 8, or the write is a single word — an even word count at an even word index. This is the rule the XEB runtimes' `_pad_cache_block` implements and `failure_052`/`failure_054` broke; re-run it after a board image update, because it is a measurement of this libc and this gateware, not a theorem |
| `python validation/prep_selector_timing.py --phase count\|body\|join` | **board** | what a SKIPPED `test` arm costs, in the register-selected prep shape. Marker pulses either side of the chain make the marker interval the chain's cost, so sweeping the arm count gives the per-skip slope and sweeping the arm body answers whether a skip is O(1) or O(body). `--phase join` measures whether the 1-cycle join dwell after `channel_trigger` is load-bearing |
| `python validation/degenerate_sweep.py` | packages only | every case parameter at its edge (0, 1, boundary). This is where `loop_count=0` was found — a board hang AND a wrong model in one combination |

Green `--dryrun` + `--revalidate` + `render_validation.py` means the model AND the drawing still
hold against the installed acadia/acadia_qmsmt. Current state, 2026-08-14:

    --revalidate            0.23 ns worst across every archived run, 0 unexplained; every residual
                            above 1 ns is a classified systematic (see systematic_note), and the
                            run is diffed case-by-case against the last known-good, not just by
                            its headline
    render_validation.py    69 / 69 cases; no drawing disagrees with its trace
    gui_validation.py       3 / 3 folders, 0 problems (23 flow-rows, 39 hovers)
    sweep-point axis        124 (folder, point) combinations over 24 experiment types, 16 of them
                            with a length that genuinely varies (DualRail_RB: 78 us -> 319 us).
                            Found both viewport crop and orphan tabs; now part of gui_validation,
                            reports its own count, and REFUSES to pass if the point range would
                            clamp its inputs (that vacuum wasted a cycle: the harness had bypassed
                            reload(), the range stayed 0..0, and 92 "checks" all read point 0)
    many-runs axis          2616 archived runs across 974 experiments, 221 of them where runs of
                            the SAME runtime resolve to DIFFERENT control-flow shapes (the cache
                            differs per run, and the viewer must read THIS run rather than what
                            that runtime usually does); 0 problems
    streaming.py            20 (streamed experiment, sweep point) combinations over 4 experiments;
                            5 distinct trains from 5 samples on every one. RB sweeps its depth
                            (3 -> 1791 gates), XEB keeps the count and randomises the identities
                            per point -- so an identical train at every point would mean the
                            decode is not following this run's cache
    notebook_path.py        109 operations over 12 folders (set_window, scroll-zoom, reset,
                            plot_trace); 0 problems. The non-Qt path, which two of today's bugs
                            would have shown first
    zoom x pin axis         210 (zoom window x pin state) combinations; 0 problems. Zoom had only
                            ever been swept with nothing pinned, and both tab bugs found earlier
                            lived in that interaction
    gui_validation --broad  24 / 24 experiment types, 0 problems (up to 39 flow-rows and 68
                            hovers per folder; found the missing-execution-rows bug that the
                            three default folders could not reach)
    nesting_boxes.py        644 archived runs; 0 wrong box counts, no handle ever lost
    override_fidelity.py    234 captures; 0 where pinning changed the timeline
    gui_robustness.py       every callback guarded; 0 problems from hostile inputs
    degenerate_sweep.py     759 combinations, 750 ok, 0 viewer defects; 9 findings, all of them
                            the same class -- loop_count=0 is non-terminating in EVERY counter-loop
                            case (nested_cool_n, loop_with_measure, batch_in_loop, stretch_in_loop,
                            repeat_until_op, repeat_until_count_n, test_in_counter_loop,
                            counter_loop_in_test, batch_in_loop_almost), and all nine are refused
                            by unsafe_reason() before they can reach the board
    --triples               125 ordered triples, 0 failures, 0.23 ns worst
    --pairs                 100 ordered pairs; only the stretch measurement systematic left
    random_seq              257 measured intervals over stretch-free sequences, 0.24 ns worst

## Setup

`--all`/`--revalidate`/`selftest.py` need station-specific paths (board IP, data roots), read
from a gitignored `paths.local.yaml`:

    cp validation/paths.local.example.yaml validation/paths.local.yaml   # then edit

`--dryrun` needs none of it. `loopback_timing_cases.py` holds the 64 test sequences, of which
two (`random_seq`, `pair_seq`) are GENERATED rather than hand-written — see "Generated
coverage" below.

## The chain that has to hold

    board  <->  trace  <->  drawing

`timing_validation.py` closes the first link: deploy, capture the DACs in loopback, compare
measured pulse times against the trace. `render_validation.py` closes the second: render a trace
to a real figure, read the patches back off the axes, and check every rectangle against the
command it stands for. A break in either link is equally misleading while debugging — arguably
the second more so, because a rendering bug looks exactly like a physics bug.

One deliberate caveat the render check encodes rather than hides: when a command ends exactly
where the next begins, the earlier bar is drawn short by `SEPARATOR_PIXELS` so the two do not
fuse. Bar STARTS are always exact; the inset is a fixed number of screen pixels, so it shrinks
to nothing as you zoom in, and a bar is never shortened by more than half.

## Generated coverage

Hand-written cases only cover what someone thought to write down. Two generated modes close that
gap over the primitive alphabet in `PRIMITIVES` (block, batch, batch_almost, dwell, reg_dwell,
loop, counter_loop, test_taken, test_skipped, stretch):

- `--pairs` deploys **every ordered pair** — exhaustive adjacency coverage. Scheduling errors are
  properties of a JOIN, not of a construct in isolation: both model bugs found in the 2026-08-13
  sweep were adjacency bugs. A random walk only *probably* produces a given join; the enumeration
  guarantees it.
- `random_seq` composes the alphabet at random from a seed, for longer-range interactions.

## What it's guarding (the coupling surface)

- **Text formats** (most fragile): `pprint()` markers/regexes in `tracing.py`; the `compiled.log`
  format and DMA word encoding in `compiled_log.py`.
- **Internal hooks**: the `DMASynchronizer.{create,merge}_schedules`/`__exit__` monkeypatch, the
  `Sequencer.{test,loop,repeat_until,bus_read}` wraps, `_compiled_program`, `_firmware`,
  `_bus_latency`, the `fake_attach` memory pools, and runtime privates (`_ios`, `_config`,
  `_pulse_cache`, `io.channel`).
- **Gateware constants**: `DMA_STATUS_REGISTER`, `MEASURED_BRANCH_PENALTY` in `tracing.py` — both
  now DERIVED from the VHDL and only *checked* by measurement (see `../docs/EMPIRICAL_CONSTANTS.md`).

Only a thin public-API layer (`Acadia.compile`, `io.compute_pulse`, `saved_runtime_loader`, …)
is stable. Everything else above is what these checks defend.

---

# How the timing model is validated

The results the checks above defend, and the method that makes them trustworthy — against the
4-channel loopback station (each DAC cabled to its own ADC, so a capture is a direct recording
of that DAC), an oracle for the *derived* parts of the model, above all the inter-block dead
time.

## Method

**The rule that makes it work: compare intervals between edges on one channel, never absolute
arrival times.** The DAC→cable→ADC latency (~283 ns) is unknown and unmeasurable but cancels in
a within-channel interval, as does capture-trigger skew. Edges are taken at 50% of the pulse
plateau with linear interpolation; averaging 5000 iterations resolves an edge to ~0.05 ns
despite 5 ns samples.

Two corollaries used throughout:
- **Only compare edges between pulses of identical ramp shape** — a 50%-of-power crossing sits
  at a different point on a 50 ns ramp than a 10 ns one, which shows up as tens of ns of
  *apparent* error that is a measurement systematic, not a model error.
- **Discriminate a constant offset from a miscount by varying the thing a miscount depends on**
  (DMA push count, loop-body size). If the residual stays put while the count-driven term moves
  as predicted, the counting is right and the residual is a real constant.

**Edge detection vs. amplitude-varying back-to-back trains (measurement systematic).** The rising
edge is taken at 50% of the *merged region's* peak — deliberate, because it is robust to ramp shape
and pulse length (a low fixed threshold re-introduces the ramp bias above). But when back-to-back
pulses of *different amplitude* merge into one region (e.g. `rb_stream`'s "8 basic gates" =
lo/mid/hi), the low/mid gates never cross the high gate's half-max, so the detected edge latches
onto the first *high* gate — the region reads late by (leading low gates) × (gate period). This is
a **measurement** artifact, not a model error: the region *start* (first above-threshold sample)
still matches the tracer to a few ns. Amplitude variation exists for gate-**identity** readability,
which is orthogonal to timing — so for **timing** of a back-to-back train, use one amplitude
(`rb_final_gate` makes the 8 final gates uniform; `rb_final_gate="rb_gate_hi"` validates to ~0–5 ns
where the varying pattern reads ~70 ns late). Fixing the detector itself would need per-sub-pulse
segmentation and would trade a well-understood artifact for a subtler one; not worth it.

## Timing (straight-line + barriers)

The model was initially one cycle short at every blocking boundary. Varying the DMA push count
in the second block was the discriminator — `issue` scales with pushes (+5 ns each, exactly as
counted), the residual stayed at +1.00 cycle regardless — proving the instruction counting is
correct and the shortfall is a fixed per-boundary constant. It compounds linearly
(`four_blocks` across 3 boundaries ≤0.14 ns) and is absent intra-block (back-to-back pulses,
`dwell`, `block=False` batching all exact). Carried as `MEASURED_BOUNDARY_OFFSET` (KI_003)
rather than folded into a derived term, since timing alone cannot say which term owns it; fitted
on one case, it then predicted `two_blocks_3ch`, `four_blocks`, `barrier_uneven` with no
retuning.

**Since resolved — it is now derived, and it owns the `already_true` term too.** The cycle is the
state register inside `acadia_dma.vhd`. `running_int_proc` sets `running_int` one clock *after*
`trigger` (and clears it one clock after `descriptor_done and fifo_empty`), and `bus_miso_proc`
registers the FIFO flags the same way. `Acadia._bus_latency(port)` models only the **read** path (1
sequencer register + 1 decoder MISO + 1 decoder device pipeline = 3 here); it adds exactly this extra
cycle for `datamover_controller` ports, with the comment *"its MISO is driven in a synchronous
process"*, and `acadia_dma` drives its MISO the same way but gets no such term. The dataport's own
`"pipeline": 1` is not a second stage — `BusDataport.generate_hdl` emits registers for
`range(1, pipeline)`, so 1 means none — which is why exactly one stage is uncounted rather than two.

That single fact predicts both halves of the old model: a wait pays the extra cycle because it must
observe a *transition*, and a poll whose condition is **already true** pays nothing extra because
there is no transition to observe. The two empirical constants (`MEASURED_BOUNDARY_OFFSET` and
`MEASURED_POLL_ALREADY_TRUE`, which cancelled each other in that branch) are now one
`DMA_STATUS_REGISTER = 1`, charged only on a transition. Arithmetically identical — every
measurement above still holds — but it names the flip-flop instead of a fitted number.

It matters only for *predicting* timing. Acadia's wait loops poll until the condition holds, so the
under-modelled cycle costs the board nothing at runtime.

Register-driven dwell (the T1/T2 shape) checked against a measured interval: predicted 420.0,
measured ≤0.04 ns. A `use_stretch` pulse is three DMA commands (first half / park / second half)
summed by the visualizer; validated with `stretch_two_blocks_same` (identical pulses, so the
ramp systematic cancels): ≤0.15 ns. `--dryrun` calls the real `acadia.compile()`, so it
reproduces board compile errors off-hardware — that is how KI_002 (the barrier `max()` bug,
since fixed in acadia) was found with zero hardware runs.

```
case                   blocks              gaps (ns)  worst err
single                      3                 [80.0]     0.00 ns
two_same_block              3                 [80.0]     0.00 ns
two_blocks                  4           [75.0, 80.0]     0.09 ns
two_blocks_1ch/2ch/3ch      4        [60/65/70, 80.0]  ≤0.13 ns
three_blocks                5     [75.0, 75.0, 80.0]     0.04 ns
four_blocks                 6 [75.0,75.0,75.0,80.0]      0.14 ns
batch_nonblocking           3                     []     0.00 ns
dwell_between               3                 [80.0]     0.12 ns
register_dwell              3                 [80.0]     0.04 ns
stretch / stretch_then_pulse 3                [80.0]     0.00 ns
stretch_two_blocks_same     4          [100.0, 80.0]     0.15 ns
barrier_single/uneven/2ch   3                 [80.0]   ≤0.02 ns
```

`stretch_two_blocks` is kept as a demonstration of the mixed-ramp systematic and listed in
`KNOWN_SYSTEMATIC` so it is not scored as a failure.

## Frequency, phase, shape

From `InputOutputWaveforms.scale_detune_pulse`: `phase` is radians (`exp(1j·phase)`), `detune`
is Hz (`exp(2πi·detune·t)`, `t` from each pulse's own start). Because the DAC and ADC NCOs share
a frequency, the residual carrier in the capture *is* the SSB detune.

| quantity | method | result |
|---|---|---|
| frequency | slope of unwrapped IQ phase, samples >50% peak | 10/25 MHz to ≤6/≤12 kHz |
| phase | *difference* between two pulses in one trace (absolutes carry unknown NCO/propagation phase) | π/2 to within 1.2° on all 4 ch |
| shape | RMS of peak-normalised `\|IQ\|` vs `trace.envelope(...,"memory")`, xcorr-aligned | 0.02–0.11 RMS |

Shape RMS rises with detune (0.02→0.05→0.10 at 0/10/25 MHz): an ideal SSB envelope is flat, so
this is analog SSB/mixer imbalance, a property of the chain, not the visualizer. These slow-ramp
cases show 1–2.5 ns *timing* error in `--revalidate` — **mixed sign and sub-cycle**, i.e.
edge-estimation noise (50 ns ramps buy ~5× the jitter of the 10 ns timing cases), not a model
error (a real one is one sign and one magnitude on all four channels, as the +1 offset was).
Hence the split: **timing conclusions rest on fast-ramp cases, shape/phase/frequency on
slow-ramp ones.** All are in `KNOWN_SYSTEMATIC`.

## Branch-aware layout

A compiled `Block` exists once but a loop body *runs* N times, so the trace carries two lists:
`trace.blocks` (compiled structure) and `trace.placements` (what executes — loops unrolled,
skipped `test` bodies dropped). `execution_plan()` → `relayout()` builds placements with their
own command copies, so changing a loop count or branch choice is a re-layout, not a re-trace.

Gaps on executed edges are counted from the compiled program along the path actually taken (for
a loop, backwards through the branch). A **taken branch costs 3 cycles beyond the count**
(`MEASURED_BRANCH_PENALTY`), established by the same discrimination trick — a 4-push body
measured 20 cycles vs 11 counted, an 8-push body 21 vs 12, and `loop_3`'s intervals are exactly
linear (220/440 ns), so the cost is per edge. The strongest single result is `test_false`: the
skipped-body gap `detect 3 + issue 10 + propagate 2 + boundary 1 + branch 3 = 19 cyc = 95 ns`
was **predicted**, not fitted.

| claim | result |
|---|---|
| loop unrolled to executed timeline | ✅ ≤0.09 ns (`loop_2/3/2_double`) |
| loop back-edge gap | ✅ derived from the compiled program |
| `test` taken / skipped (`speculation=True`) | ✅ 0.04 / 0.10 ns (skip **predicted**) |
| taken-branch cost | ⚠️ 3 cycles, empirical — see `../docs/EMPIRICAL_CONSTANTS.md` |
| `test(speculation=False)` | ❌ unsupported (KI_004); one arm hangs the board |
| `repeat_until` (counter-driven) | ✅ count resolved from the cache and unrolled (`nested_cool_n`, ≤0.11 ns over the 2..6 x 2..6 grid) |
| `repeat_until` (fifo drain) | ✅ both senses, see below |
| data-dependent `test` (active reset) | ✅ arm resolved from the CAPTURE, then scored: `feedback_reset` 0.05 ns |

`speculation=False` places the body out of line, so address order stops being execution order:
`test_true_nospec` mistimes 25 ns and `test_false_nospec` hangs the sequencer (do not redeploy).
Both are marked UNSUPPORTED; the renderer captions them "TIMING NOT MODELLED (see KI_004)".
`speculation=True` (the default) is correct on both arms. See `../docs/DEVELOPER_NOTES.md` for
the KI_004 root cause and open items, and `../docs/EMPIRICAL_CONSTANTS.md` for the two constants.


## Model bugs this harness found (2026-08-13)

Both were **adjacency** bugs — wrong because of what a construct sat next to, not wrong in
isolation — which is why the `--pairs` enumeration exists.

### 1. `fifo_almost_empty` was modelled as `fifo_empty`

The two drain primitives compile to **byte-identical poll instructions**; only the polled status
MASK differs (`0x2` vs `0x8`). `acadia_dma.vhd` publishes `miso(1) <= fifo_empty` and
`miso(3) <= fifo_almost_empty`, and the XPM FIFO asserts `almost_empty` while ONE word is still
queued — one descriptor earlier. The model released both at last-pulled, so everything after an
`almost_empty` drain was drawn one descriptor LATE: measured 119.92 ns per drain (24 cycles =
exactly one 120 ns descriptor), 239.97 ns over two.

This is not a corner case. Of the 121 qudit-branch runtimes, **zero** use `channel_is_fifo_empty`
and all seven that stream use `channel_is_fifo_almost_empty` (`dualrail_rb`, `xeb_1DR/2DR/3DR`,
`beamsplitter_amp_detune_calibration`) — refilling the FIFO while it plays is the point of the
primitive. Fixed in `machine.drain_block_issue` + the layout; `batch_drain_almost` now 0.08 ns.

### 2. A drain inside a loop was costed in address order

`drain_block_issue` counts the issue span from the drain poll to the next trigger **in address
order**. Inside a loop the next block executed is the loop head, reached backwards over the
branch, so the address-order span charges the wrong instructions — exactly one cycle too many per
pass, which the loop then multiplies: `batch_in_loop` measured 5.05 / 10.12 / 15.12 / 19.97 /
25.01 ns at loop_count 2..6. `edge_gap` already walks the executed path, but drain polls were
never recorded in `control_flow["polls"]` (only blocking ones were), so it returned None and the
layout silently fell back. Fixed by recording both poll senses; `batch_in_loop` now 0.05–0.12 ns
with **no growth in loop count**, which is the signature of a fixed per-iteration miscount.

### 3. A cache-pointer stream was detected only when it was the sole `P+1` counter

`describe_cache_stream` took the FIRST `P+1` DSP config in the program as the walking pointer.
The real `DualRailRBRuntime` configures four DSPs `P+1` — cooling and gate-sequence counters come
first — and its cache pointer is the fourth, so detection failed and the whole gate train stayed
one symbolic `BUS_DATA` command, drawn as a single grey block instead of the individual gates.
The harness's own `rb_stream` case has only one `P+1` counter, so it passed throughout and hid
this completely. Fixed by selecting the pointer whose `DSP_AB` immediate lands **inside the cache
address range** — pointing into the cache is what makes a counter a cache pointer.

### 4. A pointer-driven bus read was attributed to the wrong device

`describe_registers` walks back from a `BUS_DATA -> REGn` load to find the bus address it read.
It only recognised a LITERAL address, so `bus_read(pointer)` -- which compiles to
`BUS_ADDR <- DSP_P` -- was walked straight past, and the read was attributed to whatever literal
address came earlier. In `DualRail2XEBRuntime` that was a neighbouring FIFO poll, so the viewer
labelled the streamed gate commands **`REG0 = dac3_dma`**: a channel they have nothing to do
with. A confident wrong label is worse than no label. The walk now stops at a bus-address write
of either kind and reports `cache[pointer]`.

Deliberately NOT reported: which DSP drives the pointer. The decoded record carries a minor only
for destinations, and on that instruction it is the bus port, not the counter -- naming a
specific `DSP0` from it would repeat the mistake being fixed.

### 5. Playback did not continue seamlessly after an `almost_empty` drain

A channel whose FIFO still holds a descriptor keeps playing, and commands pushed before it
finishes continue with no bubble. The layout always started the next block at the TRIGGER, so it
inserted a gap the hardware does not have: `batch_almost -> batch` predicted 725 ns and measured
720.0.

The state that decides this is per-CHANNEL and outlives the drain block (an unrelated marker
block usually runs in between, while the batched channel is still playing). It is also specific
to `almost_empty`: a channel drained to `fifo_empty` has nothing queued by definition and does
restart at the trigger. Two intermediate versions of this fix were wrong and were caught by the
regression set -- one applied the rule to every channel (which broke `batch_uneven`, 0.08 ->
24.99 ns), the other gated it on the immediately-preceding block (which missed the case entirely).
`machine.machine_layout` now tracks `queued[channel]`.

### 7. The seamless rule drifted 6 cycles per LOOP PASS

Bug 5's window was `push_start <= cursor < t_seq`. On a loop back-edge the cursor lands exactly
ON `push_start` -- the channel finished as the pushes began, so nothing was queued behind it --
and treating that as seamless shortened every pass after the first. Measured on
`batch_in_loop_almost` (dualrail_rb's exact shape): hardware is a uniform 390 ns/pass at
loop_count 3,4,5 while the model gave 390 then 360, 360, 360, i.e. +30/+60/+90 ns of drift.
The window is now strict (`push_start < cursor < t_seq`).

Worth noting how it hid: `--revalidate` showed those cases as `DROPPED PULSES`, a structural
flag that takes display precedence over `MISMATCH`, and the flag had an innocent explanation at
the time (the case's own descriptor count had just been made configurable). Two plausible stories
for one symptom is exactly when to go back to the raw measured edge times.

### 8. A replayed single-word stream was drawn as zero-length commands

`describe_cache_stream` only recognises the WALKING-pointer idiom (a `P+1` DSP stepping through a
cache region, one gate per pass). `BeamsplitterAmpDetuneCalibrationRuntime` instead loads a plain
`Register` with ONE cache address and replays that word inside a deterministic `loop(4N)`, so
detection bailed out and all 64 plays were left symbolic at `resolve_indeterminate` (0 cycles).

Zero collapses the train: the cached word `0x19` decodes as a 26-cycle (130 ns) ARB, so the real
train is 8320 ns, and drawing it as nothing put the readout block after it **~1.2 us early**.
`direct_command_words()` now resolves a direct command read from a FIXED cache address -- through
either `BUS_ADDR <- IMM` or `BUS_ADDR <- REG` where that register was loaded once with a constant
-- and the layout takes the length from the captured cache. The folder's sequence length goes
7310 ns -> 11580 ns and still matches its own `compiled.log`.

Which register drives `BUS_ADDR` is not in the decoded record (the minor is the destination bus
port), so this resolves only when exactly ONE register in the program holds a constant cache
address. Otherwise it is left unresolved rather than guessed.

### 9. The register-sourced gate idiom drew grey boxes instead of gates (XEB)

`DualRail2XEBRuntime` / `DualRail3XEBRuntime` latch each gate word into a register before issuing
it -- `regs[n].load(bus_read(pointers[n]))` then `schedule_direct(channel, regs[n])` -- so the
compiled command is `REGn -> BUS_DATA`, not the `BUS_DATA -> BUS_DATA` form
`describe_cache_stream` recognises. Detection bailed out and every gate stayed symbolic: an
indeterminate grey box captioned with the register name, no identity and no duration.

`machine._register_gate` now decodes those words from the captured cache. The 2DR reference run
draws `XEBrand_DR1_p0/p45/p90` at 185 ns on DAC10 and `XEBrand_DR2_*` at 265 ns on DAC3; 3DR
gains 110 named pulses across all three rails.

WHICH cache region belongs to WHICH rail is stated nowhere in the compiled program -- the pointer
DSP's index is not in the decoded record, the same trap that produced the false `REG0 =
dac3_dma` caption in item 4. So the pairing is established from the DATA and checked: a region
belongs to a channel only if its word decodes to an address that names a pulse ON THAT CHANNEL.
A wrong pairing fails to resolve and is rejected. It independently recovers offsets 0/32 for 2DR
and 0/120/240 for 3DR -- the stream stride falling out for three rails is the evidence the
association is right.

**What this does NOT fix: how MANY gates are drawn.** In the 2DR reference run the gate blocks
(19-22) execute once, and two of them are in `assumed_paths`: the gate loop's count and its
`test` branches depend on runtime data no static trace can resolve. Gate IDENTITY and DURATION
are now right; the train LENGTH still reflects one pass, which the trace flags rather than hides.
Making that faithful needs the count supplied (the GUI's register panel, `resolve_indeterminate`
or `path_choices`), not more decoding.

A latent issue remains in `describe_cache_stream` for the walking-pointer path it does own:
`count_word` is taken from the FIRST `DSP_C` bus load in the program, and the XEB runtimes have
two (`num_cycles_cache[0]`/`[1]`) -- the same "first match wins" shape as item 3. It does not
fire today because that path is not reached for these runtimes.

`compiled_log.compare` cannot catch any of this either way: the archive records these as
`; Command DMA for DAC10, type 1: REG0`, which is not 8 hex digits, so the comparison skips them
on both sides. The oracle is structurally blind to exactly the commands that were wrong -- which
is why the loopback measurements, not the archive comparison, are the authority.

## Results of the exhaustive sweeps

- **100 ordered pairs** of the 10 scheduling primitives, deployed and measured. After the fixes
  above: the only remaining failures are the 16 pairs containing `stretch`, which are the
  documented mixed-ramp MEASUREMENT systematic (~25 ns), classified by `systematic_note()`.
  Substituting a same-ramp stretchable pulse was tried on the board and was worse (370 ns),
  because ramp/flat set the half/hold/half stretch geometry itself.
- **125 ordered triples** over the FIFO/branch-stateful subset, for interactions that need a
  particular predecessor AND successor -- which is how bug 5 hid from the pair sweep.
- **Random sequences** (`random_seq`) at several lengths. Score them with
  `validation/fuzz_split.py`, which replays each seed's generator draw order to recover the
  primitive list and splits the results by whether the sequence contains a `stretch`. That
  matters: a stretch-containing sequence inherits the ~25 ns mixed-ramp measurement systematic,
  so reporting the two groups together lets a known artifact masquerade as a model error, while
  reporting only the clean group hides how much of the sweep is affected. Sequences WITHOUT a
  stretch are a clean model result.

### 6. A drain poll whose condition is already true costs one cycle less (partial)

`detect` is the latency between the FIFO status changing and the sequencer seeing it. When the
drain condition is already satisfied as the poll executes there is no transition to wait for, and
the poll costs a cycle less (`MEASURED_POLL_ALREADY_TRUE`).

This is modelled, but it does not close the whole gap. Sweeping an `almost_empty` drain's
descriptor count on two independent cases puts every remaining discrepancy at **n=2 only** --
`batch_drain_almost` and `batch_in_loop_almost` both fail there and both agree to <=0.19 ns at
n=3,4,5,6,8,10. With two descriptors the "one word left" level is reached the instant the batch
starts playing, so the poll never blocks at all and the gap terms -- calibrated on drains that DO
block -- over-count by 1-2 cycles.

Deliberately not fitted further: no rule that also explains the third measured instance fits the
rest, so inventing one would be curve-fitting a degenerate case. It is classified in
`systematic_note()` as a stated limitation. No qudit runtime batches two descriptors behind an
`almost_empty` drain -- the streaming runtimes push a whole RB/XEB cycle precisely to keep the
FIFO fed.

## UI and drawing bugs this round found (2026-08-14)

Every one of these was reported or reproduced against a real run, and each fix is a RULE rather than
a special case. `validation/nesting_boxes.py` (644 archived runs) and `validation/gui_robustness.py`
now hold them in place.

| # | Bug | Cause |
|---|---|---|
| 10 | Tab labels smeared after zoom in → zoom out | The tab TEXT was the one artist not registered for removal between frames, so old labels stayed and the next frame drew over them |
| 11 | An outer loop that is entered once drew several boxes, gaining a vertical dashed edge per pass | Spans were grouped by consecutive `block.index`; a loop replays its body (11,12,11,12…) so the grouping broke at every pass boundary. Grouping is now by position in the execution PLAN plus enclosing-pass identity, so one box == one entry |
| 12 | Editing the inner cooling loop moved the outer one too | The panel wrote `loop_counts[block]`, the pre-depth fallback key, which matches every construct starting at that block. The tabs already used `(block, depth)`; both do now |
| 13 | A zoomed-out view silently stopped showing the whole sequence | Pinning a count lengthens the timeline and the previous window was restored afterwards, cropping a third of the sequence and 10 of 21 constructs off a view that looked complete |
| 14 | One tab sat alone on a fourth row of a three-deep sequence | Colliding tabs were bumped across depth bands, so a depth-2 tab pushed a depth-3 tab above every level. Rows are now banded per depth |
| 15 | A construct set to 0 passes lost its own tab | The handle came from the drawn span, and there is no span. Elided constructs now keep a hollow, still-clickable tab labelled `x0`/`skip` |
| 16 | `?` disappeared as soon as you pinned a value | Resolvability was only computed when no override existed, so a hypothesis became indistinguishable from a measurement. It is now resolved from the data on every layout and reported independently of any pin |
| 17 | `x1?` at 6 pt reads as `x17` | Flags now sit in their own token (`x1 ?`) -- a marker that can be misread as a digit is worse than no marker |
| 18 | A box covering four passes was captioned "1 pass shown", and an outer 3-pass loop was captioned "4 passes" | The caption ignored the drawn count for pinned constructs, and `passes` counted the placement's `iteration` scalar -- the INNERMOST loop's index -- at every level. It now counts this construct's own slot in the execution path |
| 19 | The panel's own spin box could hang the GUI | Its range was `0..100000`; typing that expanded the plan to 100 000 copies of the body. The range is now derived from what that body costs to draw |
| 20 | Setting ONE execution of a construct to 0 passes removed its tab, its panel row, and the tabs of everything nested inside it | Executions were enumerated from the drawn placements, and an execution drawn zero times produces none -- so it vanished from the list and the setting deleted its own control. They are now recorded where the layout REACHES them, before the count can suppress the body |
| 21 | Pinning an outer loop gave the constructs inside it more EXECUTIONS, and those appeared as tabs on the diagram with no row in the panel (7 of 24 experiment types) | Rows were built once per trace and only updated in place afterwards. How many executions a construct has is not fixed -- it follows the enclosing pass count. The panel now adds and removes rows to match, defers only while a spin box has focus (a half-typed number is real state; a combo selection is not), and remembers which execution lists were expanded so a rebuild does not collapse them |
| 22 | Stepping the SWEEP POINT left the viewport at the old sequence length -- 78 us of a 319 us DualRail_RB shown with nothing to say the rest existed (10 of 124 folder/point combinations) | Two paths for one event: `relayout()` updated the extent and called `reset()`, `set_point()` updated it and called `render()`, which keeps the existing window. Both now go through one `_refit()` -- a view that covered everything keeps covering it, a window you zoomed yourself is kept and clamped |
| 23 | Stepping the sweep point could leave a tab on the diagram with no row in the panel | A register read from the cache decides a test's arm and a repeat_until's count, so a different point can add or remove CONSTRUCTS entirely (Readout_Fidelity grows one at point 1). `_select_point` refreshed the registers and the block list but not the control-flow panel |
| 24 | A construct that is NOT drawn had its handle placed in the wrong spot: skipping a test put its tab to the RIGHT of a construct that comes after it, and zeroing an execution put #1 after #2 | The marker was positioned by sorting placements on `(path, index)` as a stand-in for execution order. A nested placement sorts after every top-level one no matter when it runs, so block 21 (nested, 345 ns earlier) lost to block 22. It is now read off the PLAN, which is already in execution order, and anchored inside the construct's own enclosing pass -- falling back past the pass boundary is what put execution #1 after #2 |
| 25 | Tab labels became unreadable with nesting depth, and hollow tabs vanished in the dark theme | The tab palette lightens deliberately with depth, but the label colour was fixed at white: depth 3 measured 2.48:1 and depth 4 1.65:1 against their own fill. A hollow tab (a construct drawn zero times) is inked straight onto the page, and depth-1 ink on the dark theme's background is 1.71:1 -- the handle for something not drawn was invisible exactly where you would look for it. Both now choose by CONTRAST: the label takes whichever of white/dark ink reads better on its fill, and a hollow tab's ink is blended toward the page's opposite until it clears 4.5:1, which keeps the hue that identifies the nesting level |
| 26 | Guarding the callbacks broke every checkbox | The guard's `*args` wrapper hid each slot's arity, so Qt stopped truncating surplus signal arguments and `stateChanged(int)` hit a no-argument `_redraw`. Caught by the guard's own fault list, which is why a fired guard fails the suite |
| 27 | Setting a control-flow pin and THEN stepping the sweep point left the sequence at the old point's length -- DualRail_RB drawn at 79.7 us with the header reading point 279, whose real length is 321 us | Not the point step itself: `adopt_trace` deliberately keeps the point you were reading across a reload, but a freshly traced folder holds point 0, so it SHOWED the old index without moving the trace to it. Every later step then compared against a stale current value and could no-op. It now applies the point it displays, at the one place every load goes through. Found by asking whether the panel is a pure function of (folder, point, pins) rather than of the order you turned the knobs |
| 28 | A non-finite window took the whole application down | `set_window` passed its arguments through to matplotlib, which refuses NaN/Inf axis limits with a `ValueError`. In Qt an exception escaping a slot calls `qFatal()`, so a viewport that could not be computed killed the panel rather than showing something. `_apply_window` now sanitises at its one entry point: a window that cannot be made finite and ordered becomes the full extent, and a sequence whose own length is not a number gets a floor-width extent |
| 29 | A depth-1 tab's label shipped below the small-text contrast threshold (4.19:1 light, 3.72:1 dark) | The earlier fix chose the label as the better of two inks, which is not the same as choosing a readable one: a mid-tone fill is far from both ends of the ink range at once, so neither candidate could reach 4.5:1 and 'least bad' shipped. Contrast is a property of the fill-and-ink PAIR, so `legible_tab` now nudges the FILL -- 4% toward black for depth 1, which keeps the ladder that makes nesting depth readable -- once the text is already at the extreme. Found by sweeping depth 0..12 instead of the four the palette names; the gate's own bar had also been set at 4.0, below WCAG AA for 6 pt, and is now 4.5 |
| 30 | On a small canvas the legend crowded out the sequence: a 5x3 inch figure gave the plot a quarter of its width and still clipped the key | The legend was a fixed 8 pt however big the canvas, and `fit_layout` would rather not reserve the space than shrink the plot below 30% -- so the key overflowed instead. It now scales to the canvas (`legend_fontsize`, estimated from the figure and the labels), and `fit_layout` measures the result and drops a key that still would not FIT INSIDE the figure: the colours it names are also on the bars |
| 31 | Lane labels overlapped each other -- the axis read `readout1_stimulusADC0`, naming a lane something no lane is called | Three causes in one place. The labels were sized against the LANE COUNT, but the tab strip's headroom means nine lanes can span seventeen units, so each lane had half the room assumed; the budget was computed in points and compared against pixels (a 3-line 9 pt label is 34 pt but 47 px at 100 dpi); and a blanket `ax.tick_params(labelsize=8)` two lines later silently overrode whatever size was chosen, so the code read one way and drew another. Now sized from the real y span in pixels, after the headroom is known: shrink, then drop io lines (marked with an ellipsis), then label every Nth lane -- an unlabelled lane is honest, two names on top of each other is not |
| 32 | The LANE viewport had none of the time axis's protection: `set_lanes` sorted its arguments and stored them, so a NaN reached matplotlib and, in Qt, took the application down | The rule existed but on one axis. Both now go through one `slice_of` -- finite, ordered, at least a floor wide, inside the extent, sliding rather than clipping -- and the sweep exercises both, because a rule written once but tested on one axis is a rule that holds on one axis |
| 33 | A lane range could sit outside the lane stack after a re-render | The lane extent is a property of the FRAME, not the trace: the tab strip's height comes from the tabs the current window has, so zooming into a shallower stretch shortens the stack under a range that was chosen against a taller one. The lane scrollbar treats the range as a slice of the stack and goes degenerate outside it. Now re-clamped inside `render`, at the first moment the frame's real extent is known and before anything records what was drawn -- the same rule `_refit` applies on the time axis when the sequence changes length |
| 34 | A CHAIN of skipped `test` arms was over-counted: every skip after the first was costed as though its body had run | The prep-selector shape -- one `test(sel == i)` per prep state, at most one taken (`resonator_number_measurement`). `edge_gap` counted an address range and then subtracted the ONE skipped body it could find between the poll and the first branch, so skips 2..N were counted in full. Measured on the loopback with the new `test_chain` case (arm 0 taken, 1 pulse per arm): the board pays a flat **25 ns per skipped arm** and the old model charged +40 ns extra for each skip beyond the first -- 365/390/**520**/**780** ns predicted against 365/390/440/540 measured at 1/2/4/8 arms. `edge_gap` now WALKS the executed path (`_executed_path`), so every skip costs its own condition plus one taken branch and nothing for the body it jumps over, which is what the hardware does and why the measured slope does not care how big the arm is. Worst error 240 ns -> 0.2 ns; no other case moved |
| 35 | A `channel_synchronizer(trigger=False)` block was costed as taking NO sequencer time at all | It emits no DMA trigger, so it has no trigger address -- and the gap accounting looked up the next executed block, found nothing, and gave up. Everything on that stretch went uncharged: the queued block's own command pushes, the `channel_trigger` write that fires them, and the condition-plus-branch of every arm skipped along the way. The model predicted a flat 305 ns for 1, 2, 4 and 8 arms alike, against 380/380/430/530 measured. Two parts to the fix: the gap now looks ahead to the next block that IS anchored, so those instructions land inside one edge; and the path walk is told which trigger-less arms ran (they cannot be inferred from the program, so the execution plan names them, consumed in program order because branches and blocks share it). Slope now exact in both modes -- 25 ns per skipped arm -- with a residual CONSTANT of 10 ns (all arms skipped) to 25 ns (one taken) that does not grow with the chain: 240 ns worst error -> 25 ns |
| 36 | A POINTER-counted `repeat_until` was drawn as one assumed pass, hiding a whole 13-round counting train | `repeat_until_count` took the loop's exit value AS the pass count, which is right only because the usual idiom loads the counter with 0. A pointer loop starts at a cache ADDRESS (`length_pointer.load(base + index)`) and exits at that address plus the round count, so the exit value alone is a ~1.9-million-pass loop -- the resolver rejected it and fell back to one data-dependent pass. Passes are now `exit - start`, and both endpoints are compile-time immediates the program carries (`describe_immediates`). On resonator_number_measurement the drawn sequence went from 6.79 us / 1 pass to **39.28 us / 13 passes**, which is what the board runs |
| 37 | A register holding a LENGTH was decoded as if it were a packed DMA command word | `_register_gate` exists for the XEB idiom where a register carries a whole `(address << 16) | (length - 1)` command. It keyed only on "symbolic name starts with REG" -- so it also caught the CONST_CONT hold of a `use_stretch` pulse, whose symbolic value is a hold LENGTH. A ladder length of 39 cycles decoded as address 0, length 40: every counting round drew one cycle too long, read one cache word too early (word 0 was the prep selector, not a length), and was attributed to a pulse the word never named. Plausible enough to be believed, which is why it needed a guard rather than a reader. A continuation command's symbolic value is a length, never a command word |
| 38 | ...and once it stopped being mis-decoded, the per-round length was not resolved at all | A `cache[pointer]` register has no single value -- that is the point of it -- so `register_cycles` has no entry and the length fell back to 0 (drawn indeterminate). Now derived per pass: the pointer advances once per `pulse_cep()`, so pass r reads cache word `index + r`. WHICH pointer feeds the register is not in the compiled record (`bus_read(pointer)` emits `BUS_ADDR <- DSP_P`, whose minor is the bus port), so it is taken from the enclosing loop's exit condition, which names the counter -- and in this idiom the counter IS the pointer. When it is not, the length stays honestly indeterminate. All 13 ladder lengths now match the captured cache exactly |
| 39 | A BARRIER costs 20 ns more than the equivalent hand-built dwells when the block also contains a `measure()`, and the model shows the two as identical | Measured on the loopback with a register-driven stretch and a real readout in one block (`stretch_measure_dwell` vs `stretch_measure_barrier`, reg = 1/5/20/60/115): the readout arrives **+20.05, +19.88, +19.83, +20.06, +20.09 ns** later with the barrier -- a constant 4 cycles, independent of the register, and both variants track the register at exactly +5.000 ns per cycle. The DAC-only contrast (`stretch_barrier_align` vs `stretch_dwell_align`) shows **0.03 ns**, so the cost appears only when the barrier must also reconcile the capture chain. The traced picture puts the readout at the same 200.0 ns in both, so SeeQuence currently under-states the barrier variant by those 4 cycles. Two-point contrast, so the capture chain is the indicated cause rather than a proven one |

Two performance bugs came out of the same pass: `machine_layout` rebuilt a whole-plan set once per
placement (9.1 M dict lookups at 3000 placements; a 1000-pass pin took **11.9 s**, now **0.48 s**),
and patches were attached with `add_patch`, whose data-limit bookkeeping this drawing never uses.

## Hazards found outside the visualizer

- **`Acadia._bus_latency` under-counts a DMA status read by one cycle.** The polled bits are
  produced by clocked processes inside `acadia_dma.vhd` (`running_int_proc` for `running`,
  `bus_miso_proc` for the FIFO flags), and `_bus_latency` models only the read path -- it adds
  exactly this cycle for `datamover_controller` ports, with the comment *"its MISO is driven in a
  synchronous process"*, and the DMA drives its MISO the same way. This is what the old
  `MEASURED_BOUNDARY_OFFSET` was compensating for; it is now `DMA_STATUS_REGISTER`, derived. It costs
  the board nothing at runtime (the wait loops poll until the condition holds) and matters only when
  PREDICTING timing.

- **A register-driven length of 0 is not zero.** `Acadia.command_dma` emits `length - 1`, so a
  register holding 0 becomes an all-ones length field: 2**16-1 cycles for an ARB command,
  **2**32-1 (~21 s)** for a 32-bit dwell. `dual_rail_ramsey._delay_cycles` already floors its
  register dwell at one cycle and says why; `dynamical_chi_ramsey`, `dynamical_chi_parity`,
  `crosstalk_to_qubit` and `collapse_and_revival` call `seconds_to_cycles` with no such floor.
  The tracer now MODELS the underflow and records it in `trace.length_underflows`, so a sweep
  point that does this is visible instead of silently drawn as an empty command, and
  `underflow_reason()` refuses to deploy one.
- **`repeat_until` runs its body at least once, so a target of 0 never returns.** Measured
  2026-08-14: `repeat_until(counter == 0)` on a counter loaded 0 and incremented +1 per pass hangs
  the board ("Timeout occurred waiting for line", repeating, until killed), while the same loop at
  1,2,3,4,5,6,8 measures to ~0.1 ns. This is also the **only** measurement that distinguishes
  test-before-body from test-after-body: both predict N passes for every N ≥ 1 and differ only at 0.
  The hang settles it — the body always runs, and the counter can never return to 0.
  The tracer now reports such a loop as `nonterminating` (drawn `x∞`, with a caption saying so)
  instead of drawing a tidy empty body, and `unsafe_reason()` refuses to deploy one. A runtime
  simply cannot express "zero passes" with a counter loop; guard the parameter instead.
- **`three_deep_nest`** (three nested counter loops, each `configure`d once up front and only
  reloaded) never returns from the board. `nested_cool_n` re-issues `configure()` as well as
  `load()` on re-entry and works. No qudit runtime reaches three counter levels — `cool_modes`
  uses two counters plus a `test` — so nothing shipped is affected. See `UNSAFE_TO_DEPLOY`.
