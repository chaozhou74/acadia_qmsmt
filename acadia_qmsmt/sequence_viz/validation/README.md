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

Green `--dryrun` + `--revalidate` + `selftest.py` (currently: worst 0.14 ns, 6/6 folders) means
the model still holds against the installed acadia/acadia_qmsmt.

## Setup

`--all`/`--revalidate`/`selftest.py` need station-specific paths (board IP, data roots), read
from a gitignored `paths.local.yaml`:

    cp validation/paths.local.example.yaml validation/paths.local.yaml   # then edit

`--dryrun` needs none of it. `loopback_timing_cases.py` holds the 29 test sequences.

## What it's guarding (the coupling surface)

- **Text formats** (most fragile): `pprint()` markers/regexes in `tracing.py`; the `compiled.log`
  format and DMA word encoding in `compiled_log.py`.
- **Internal hooks**: the `DMASynchronizer.{create,merge}_schedules`/`__exit__` monkeypatch, the
  `Sequencer.{test,loop,repeat_until,bus_read}` wraps, `_compiled_program`, `_firmware`,
  `_bus_latency`, the `fake_attach` memory pools, and runtime privates (`_ios`, `_config`,
  `_pulse_cache`, `io.channel`).
- **Gateware constants**: `MEASURED_BOUNDARY_OFFSET`, `MEASURED_BRANCH_PENALTY` in `tracing.py`
  (see `../docs/EMPIRICAL_CONSTANTS.md`).

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
| `repeat_until` | ➖ data-dependent count; one pass drawn and labelled as such |

`speculation=False` places the body out of line, so address order stops being execution order:
`test_true_nospec` mistimes 25 ns and `test_false_nospec` hangs the sequencer (do not redeploy).
Both are marked UNSUPPORTED; the renderer captions them "TIMING NOT MODELLED (see KI_004)".
`speculation=True` (the default) is correct on both arms. See `../docs/DEVELOPER_NOTES.md` for
the KI_004 root cause and open items, and `../docs/EMPIRICAL_CONSTANTS.md` for the two constants.
