# Validating sequence_viz against hardware

A running record of how the visualizer's claims are checked against the physical output of
the DACs, stage by stage: what was asked for, what was tried, and what it turned up.

**Method.** The 4-channel loopback station cables each DAC to its own ADC, so an ADC capture
is a direct recording of that DAC's output. That makes it an oracle for the parts of the
visualizer that are *derived* rather than measured — above all the inter-block dead time.

**The one rule that makes it work:** compare *intervals between edges on one channel*, never
absolute arrival times. The DAC→cable→ADC latency (~283 ns) is unknown and unmeasurable, but
it cancels in a within-channel interval, as does any residual capture-trigger skew. Edges are
taken at 50% of the pulse plateau with linear interpolation; averaging 5000 iterations
resolves an edge to ~0.05 ns despite 5 ns samples.

**Tools.** `validation/timing_validation.py` (`--dryrun`, `--case`, `--analyse`,
`--regions`, `--logs`, `--revalidate`) driving
`validation/loopback_timing_cases.py` (29 selectable DAC sequences).
`sequence_viz/validation/selftest.py` is the hardware-free regression.

---

## Stage 0 — station setup

**Asked for:** confirm the runner works and that the loopback data is usable.

**Found.** The first run reproduced **KI_001** exactly: with the default `capture_order`, ch3
(last in the ADC-trigger loop) arrived 36.4 ns early on its own axis. Enabling the sacrificial
dummy capture fixed it — ch3 went to −1.1 ns, in line with ch1/ch2:

```
                first-edge skew vs ch0                  interval (all channels)
dummy OFF   ch1 −1.38  ch2 −1.60  ch3 −36.42 ns         195.0 ns
dummy ON    ch1 −1.53  ch2 −1.40  ch3  −1.14 ns         195.0 ns
```

Cross-channel alignment is therefore trustworthy to ~1.5 ns with the dummy on. Note the
*interval* was unaffected either way — the reason within-channel intervals are the primary
metric.

---

## Stage 1 — timing (complete)

**Asked for:** simple sequences, no real-time branching, power only; check relative timing in
as many scenarios as existing runtimes suggest; calibrate the global offset.

### What was found, in order

**1. The model was wrong by exactly one cycle per blocking boundary.** The very first
comparison, on an existing run: predicted 190.0 ns pulse-to-pulse, measured 195.00 ns on all
four channels (agreeing to ±0.05 ns).

**2. The error is a constant, not a counting mistake.** Varying how many DMA pushes the second
block issues was the discriminator — `issue` scales with pushes, the other terms don't:

| pushes in block 2 | predicted gap | error |
|---|---|---|
| 1 | 55 ns | +4.96 ns |
| 2 | 60 ns | +4.96 / +4.99 ns |
| 4 | 70 ns | +5.00 ns (all 4 ch) |

Each extra push added exactly 5 ns as predicted, so **the instruction counting from the
compiled program is correct** — which matters because that is the machinery stage 3 needs.

**3. It is per boundary.** `three_blocks` gave +1.000 cycle on the first interval and +2.003
on the second.

**4. Intra-block layout was already exact.** `two_same_block` merged into one region of
245 ns vs 125 ns for a single pulse — the second pulse added exactly one pulse length, zero
gap. `batch_nonblocking` merged to 365 ns = 3 × 120 + threshold. `dwell_between` predicted
320.0 and measured 320.00. So the shortfall is localised entirely to blocking boundaries.

**5. Correction, kept honest.** Which term owns the cycle is not separable by timing alone, so
it is carried as its own named term `MEASURED_BOUNDARY_OFFSET` in the gap breakdown rather
than silently folded into `detect` or `propagate`. Recorded as **KI_003** with a TODO to find
the source in the acadia gateware.

**6. Out-of-sample check** (the correction was fitted on one-boundary/4-push data only):

| case, never previously measured | predicted | measured | error |
|---|---|---|---|
| `two_blocks_3ch` (3 pushes) | 190.0 ns | 190.13 / 190.03 / 189.99 | ≤0.13 ns |
| `four_blocks` (3 boundaries) | 195 / 390 / 585 ns | 195.11 / 390.14 / 584.99 | ≤0.14 ns |
| `barrier_uneven` (4-ch barrier) | 320.0 ns | 320.01 on all four | 0.01 ns |

A wrong per-boundary constant would have made the `four_blocks` error grow 5 ns per boundary.
It didn't, so the correction generalises.

**7. A new acadia limitation, since fixed.** `barrier_uneven` originally would not *compile*:
alignment builds `max()` over per-channel lengths and the sequencer rejected it (2 channels →
`max(48, 24)`; 4 channels → `max(64, 24, 24, 24)`). Only the single-active-channel barrier
shape worked — which is the shape every existing runtime happens to use. Filed as **KI_002**;
the 2026-07-27 acadia pull fixed it, and the 4-channel barrier now matches to 0.01 ns.

Characterising it took four iterations and **zero hardware runs**, because `--dryrun` calls the
real `acadia.compile()` and so reproduces board compile errors off-hardware. That turned out
to be the most useful by-product of this stage.

**8. A latent bug in my own test runtime.** Cases that schedule only some channels had
`load_pulse` allocating waveform memory *after* `attach()`, which never maps it. Fixed by
allocating all four before compile. Would have failed on hardware too.

**9. Register-driven dwell, against a measured interval.** The auto-resolve was previously
only checked against the cache off-hardware, which is not the same claim. `register_dwell`
loads a `Register` from `cache[0]` and dwells it between two pulses — the shape of a T1/T2
sweep. Predicted 420.0 ns, measured 420.00 / 420.01 / 420.04 → **≤0.04 ns**.

**10. Stretch geometry.** A `use_stretch` pulse is three DMA commands (first half, park
mid-waveform, second half) and the visualizer sums them. Validating it needed care:

- Comparing *widths* does not work. A width taken at a low threshold shrinks for a slow ramp —
  nominal 500 ns measured 490, while a 20 ns-ramp pulse measures 125 for a nominal 120. The
  threshold is crossed well into a 50 ns ramp.
- Comparing a stretch pulse's rising edge against a *test* pulse's (`stretch_two_blocks`) gave
  −25.5 ns, because a 50%-of-power crossing sits at a different point on a 50 ns ramp than on
  a 10 ns one. **That was a measurement systematic, not a model error.**
- `stretch_two_blocks_same` uses the same stretchable pulse for both edges, so the systematic
  cancels exactly: predicted 595.0 ns, measured 594.98 / 594.99 / 595.15 → **≤0.15 ns**.

The lesson generalises to stage 2: only compare edges between pulses with identical ramp
shape, or model the crossing point explicitly.

### Stage 1 status

All 19 cases compile; every case with measurable intervals agrees to **≤0.15 ns**
(0.03 cycles), which is measurement noise. Stage 1 is complete — nothing outstanding.

```
case                   blocks              gaps (ns)  worst err
single                      3                 [80.0]     0.00 ns
two_same_block              3                 [80.0]     0.00 ns
two_blocks                  4           [75.0, 80.0]     0.09 ns
two_blocks_1ch              4           [60.0, 80.0]     0.04 ns
two_blocks_2ch              4           [65.0, 80.0]     0.04 ns
two_blocks_3ch              4           [70.0, 80.0]     0.13 ns
three_blocks                5     [75.0, 75.0, 80.0]     0.04 ns
four_blocks                 6 [75.0,75.0,75.0,80.0]      0.14 ns
batch_nonblocking           3                     []     0.00 ns
dwell_between               3                 [80.0]     0.12 ns
register_dwell              3                 [80.0]     0.04 ns
stretch                     3                 [80.0]     0.00 ns
stretch_then_pulse          3                 [80.0]     0.00 ns
stretch_two_blocks          4           [75.0, 80.0]    25.84 ns  measurement systematic
stretch_two_blocks_same     4          [100.0, 80.0]     0.15 ns
barrier_single_channel      3                 [80.0]     0.00 ns
barrier_uneven              3                 [80.0]     0.01 ns
barrier_uneven_2ch          3                 [80.0]     0.02 ns
barrier_uneven_pulses       3                 [80.0]     0.01 ns
```

`stretch_two_blocks` is retained deliberately as a demonstration of the mixed-ramp
systematic; `timing_validation.py` lists it in `KNOWN_SYSTEMATIC` so it is not counted as a
model failure.

---

## Stage 2 — phase, frequency, pulse shape (complete)

**Asked for:** check phase, frequency and shape agreement; per-channel relative comparison is
enough for amplitude.

Tooling: `validation/shape_validation.py`, cases `shape`, `detune_pair`,
`phase_pair`, using `long_ramp_pulse` (100 ns ramps — a 20 ns ramp is only 4 samples at the
5 ns capture spacing) and detunes well under the 100 MHz Nyquist.

Conventions, from `InputOutputWaveforms.scale_detune_pulse`: **`phase` is in radians**
(`exp(1j·phase)`), `detune` in Hz (`exp(2πi·detune·t)`, with `t` measured from the pulse's own
start — so each pulse's detune phase restarts at zero).

### Frequency

Fitted as the slope of the unwrapped captured IQ phase, restricted to samples above 50% of
peak magnitude (on the ramps the amplitude is small and the phase is noise-dominated):

| configured | measured across 4 channels | error |
|---|---|---|
| 10.000 MHz | 9.994 – 10.000 MHz | ≤6 kHz (0.06%) |
| 25.000 MHz | 24.988 – 24.995 MHz | ≤12 kHz (0.05%) |

This works because the DAC and ADC NCOs are set to the same frequency, so the residual
carrier in the capture *is* the SSB detune.

### Phase

Only a *difference* between two pulses in one trace is meaningful — the absolute captured
phase also contains the propagation delay and the DAC/ADC NCO phase offset, neither of which
is known. `phase_pair` plays `long_ramp_pulse` (no phase) then `phase_half_pi` (π/2):

```
ch0 +1.5678 rad (89.83 deg)      ch2 +1.5923 rad (91.23 deg)
ch1 +1.5793 rad (90.49 deg)      ch3 +1.5880 rad (90.99 deg)
```

Within 1.2° of the configured π/2 on all four channels.

### Shape

RMS difference between peak-normalised measured `|IQ|` and the normalised envelope from
`trace.envelope(io, pulse, "memory")`, time-aligned by cross-correlation:

```
long_ramp_pulse   ch0 0.074   ch1 0.024   ch2 0.026   ch3 0.019
detune_10MHz      ch0 0.080   ch1 0.083   ch2 0.049   ch3 0.048
detune_25MHz      ch0 0.107   ch1 0.102   ch2 0.098   ch3 0.087
```

ch0 is consistently worst on the undetuned pulse, which is expected rather than surprising:
it is the 7 GHz `mix_reconstruction` channel and its peak is 3.8e3 against 5e4 on the others,
so its SNR is ~13x lower.

The **detune-dependent** rise (0.02 → 0.05 → 0.10 as detune goes 0 → 10 → 25 MHz) is more
interesting. The envelope of an ideal SSB pulse is flat regardless of detune, so a ripple that
grows with detune points at SSB/mixer imbalance: a residual image at −detune beating against
the wanted tone at +detune produces amplitude ripple at 2·detune. The monotonic ordering
supports that, but it has not been demonstrated directly (fitting the ripple frequency would
settle it). Either way it is a property of the analog chain, not of the visualizer — the
envelope it reports is the ideal one computed from the samples in DAC memory.

**Not a test:** the pulse[1] − pulse[0] phase difference printed for `detune_pair` is
meaningless, because the two pulses have different detunes and each one's phase is referenced
to its own start. Only `phase_pair` compares like with like — `shape_validation.py` now says so
in its output rather than printing a number that looks like a result.

### Measurement resolution on slow ramps

The stage-2 cases show 1–2.5 ns of *timing* error in `timing_validation --revalidate`, against
≤0.15 ns for stage 1. That is the measurement, not the model:

- the errors are **mixed sign and sub-cycle** (−1.14 / +0.70 / +0.82 / +0.26 ns on `phase_pair`).
  A real model error appears with one sign and one magnitude on all four channels — which is
  precisely how the +1 cycle boundary offset presented (+4.96…+5.00 everywhere);
- these pulses have 50 ns ramps against 10 ns for the stage-1 pulses, so the same amplitude
  noise buys ~5× the timing jitter at the 50%-of-power crossing;
- `detune_pair` adds the SSB ripple, which perturbs the plateau peak that the half level is
  derived from, and differs between the 10 and 25 MHz pulses.

So: **timing conclusions rest on the fast-ramp cases; frequency, phase and shape conclusions
rest on the slow-ramp ones**, which is the right split — slow ramps are what make the shape and
phase fits well-conditioned, and fast edges are what make the timing fits well-conditioned. All
three are listed in `KNOWN_SYSTEMATIC` so they are not scored as model failures.

While chasing this, one real harness bug did surface and is fixed: the half level was taken from
a fixed 40-sample window, which never reaches the plateau of a 300 ns pulse and so biased the
crossing by pulse length. It now uses each region's own peak (`stretch_two_blocks_same`
improved 0.15 → 0.05 ns).

---

## Stage 3 — branch-aware layout, "B" (complete, with one unsupported variant)

**Asked for:** finish B, then check the branching cases. Later: *"yeah do test"*; for the loop,
*"1 pass shown is fine, but label it properly"*; and *"take down notes for the empirical
numbers/delays, so we can look into acadia later to find the actual source."*

Stage 1 cleared the dependency: B has to walk the compiled program to work out which
instructions actually execute on a chosen path, and stage 1 proved that instruction counting
against hardware. Order of attack was `loop(N)` first (deterministic count, so the expected
unrolled timeline is exact), then `test()` forced true and forced false, then `repeat_until`,
where the count is genuinely data-dependent.

### 1. What B actually is: `Placement`

The blocker was structural, not numerical. A compiled `Block` exists once; a loop body *runs*
N times. Trying to express that by stretching one block's start/stop cannot work, because each
pass needs its own pulses at its own times. So the trace now carries two lists:

| | what it is |
|---|---|
| `trace.blocks` | the **compiled** structure, one entry per `channel_synchronizer` block |
| `trace.placements` | what **executes** — loops unrolled, skipped `test` bodies dropped |

`execution_plan()` returns `[(block index, iteration), ...]` in run order and `relayout()` turns
it into placements, each with its own copies of the commands. Everything downstream (renderer,
zoom, `length_cycles`) reads `placements` when present. Because layout is split out from
tracing, changing a loop count or a branch choice is a re-layout, not a re-trace — which also
matters for register-driven lengths, only knowable once a sweep point is picked.

### 2. Loops: exact, once the back-edge is costed

`loop_2` / `loop_3` / `loop_2_double` all land at **0.03 / 0.09 / 0.04 ns**. Two findings got
them there.

**The loop-back gap is derived, not fitted.** It is read out of the compiled program the same
way stage 1's forward gap was: count the instructions from the poll to the next `Trigger DMAs`
along the path actually taken. For a loop that path runs backwards through the branch, so the
count includes the loop's own decrement/compare instructions.

**A taken branch costs 3 cycles beyond the count.** Same discrimination trick as stage 1 —
vary the thing that would change a miscount and see whether the residual moves:

| case | body pushes | counted instructions | measured cycles | implied extra |
|---|---|---|---|---|
| `loop_2`, `loop_3` | 4 | 11 | 20 | 3 |
| `loop_2_double` | 8 | 12 | 21 | 3 |

Doubling the body changed the count by exactly +1 (8 pushes also drop the FIFO-latency NOPs
from 3 to 0) and the measurement agreed, so the count is right and the residual is a constant.
`loop_3`'s two intervals are **220.0 and 440.0 ns** — exactly linear, so the cost is per *edge*,
not per loop. Recorded as `MEASURED_BRANCH_PENALTY = 3`; see `EMPIRICAL_CONSTANTS.md`.

### 3. `test(speculation=True)`: both arms right, and the skip gap was a prediction

`test_true` **0.04 ns**, `test_false` **0.10 ns**.

`test_false` is the strongest single result of the stage, because nothing was fitted to it. The
skipped-body gap comes straight out of the compiled program: poll at 107 → branch at 108,
target 123 → trigger at 130 gives `issue = 10`, so
`detect 3 + issue 10 + propagate 2 + boundary 1 + branch 3 = 19 cycles = 95 ns`. Measured: 95 ns.
The same 3-cycle penalty derived from a *backward* loop edge predicted a *forward* skip edge
with no retuning — which is what makes it credible as a branch cost rather than a loop cost.

Getting there needed one parser fix worth noting: the branch target is printed in two formats,
and the regex only knew one. `value=0x7B` was being missed, so the skip target was unknown and
the whole 95 ns gap collapsed. Fixed with a two-format regex matched against the instruction
tail only.

### 4. `test(speculation=False)`: unsupported, and it hangs the board — KI_004

The prediction from reading `Sequencer.test` held up: with `speculation=False` the body is
placed **out of line**, so address order stops being execution order and the instruction-count
model does not apply. Measured:

- `test_true_nospec` — mistimes by **25.02 ns**
- `test_false_nospec` — **hangs the sequencer.** `DataManager.sync` raises
  `TimeoutError: Timeout occurred waiting for line`; no `t_data` group is written.

The board recovers on the next deploy — confirmed by killing the hung job and running `single`
cleanly rather than retrying the hanging case. Filed as **KI_004**. Both variants are marked
UNSUPPORTED in `timing_validation.py` so they read as a known acadia limitation, not a model
failure, and `test_false_nospec` is flagged do-not-redeploy. The tracer now adds such a block to
`trace.unsupported_paths` and the renderer captions the region **"TIMING NOT MODELLED (see
KI_004)"** instead of quietly drawing a timeline it cannot justify. `speculation=True` is the
default and is correct in both arms — that is the recommendation.

### 5. `repeat_until`: one pass, honestly labelled

Its trip count depends on a live measurement, so there is no timeline to validate against.
Per *"1 pass shown is fine, but label it properly"*, `branch_caption()` draws one pass and says
`repeat_until(cond) — 1 pass shown; real count is data-dependent`. A `loop(N)` keeps its exact
unrolling and is labelled `pass k of N`; an unbounded `loop()` says `of unbounded`. Nothing is
inferred silently.

### Stage 3 status

| claim | result |
|---|---|
| loop unrolled to real executed timeline | ✅ ≤ 0.09 ns (`loop_2`, `loop_3`, `loop_2_double`) |
| loop back-edge gap | ✅ derived from the compiled program |
| `test` body taken | ✅ 0.04 ns |
| `test` body skipped | ✅ 0.10 ns, gap **predicted** not fitted |
| taken-branch cost | ⚠️ 3 cycles, empirical — `EMPIRICAL_CONSTANTS.md` |
| `test(speculation=False)` | ❌ unsupported (KI_004); one arm hangs the board |
| `repeat_until` | ➖ not validatable; 1 pass drawn and labelled as such |

Full re-derivation of every recorded run: **29 cases, worst 0.14 ns** excluding the three
documented measurement systematics and the two KI_004 variants
(`timing_validation.py --revalidate`). `selftest.py` passes all 6 folders × 48 render combos.

---

## Open items

- Find the source of `MEASURED_BOUNDARY_OFFSET` (KI_003) and `MEASURED_BRANCH_PENALTY` in the
  acadia gateware/sequencer. `EMPIRICAL_CONSTANTS.md` is the brief: what each number is, what
  evidence a candidate must fit, and how to test one (`--revalidate` must hold all 29 cases).
- KI_004 wants a fix in acadia, not in the visualizer.
