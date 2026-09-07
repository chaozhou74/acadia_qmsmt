# Acadia / sequence_viz stress campaign

Findings appended as they are found. Each entry says what was run, what was
observed, and whether the fault is the viewer's or the run's. Nothing here is
inferred without evidence; where something could not be determined, it says so.

**How to read this file.** Each `###` section is one sweep. A sweep reports what it deployed or
traced, then either "no unexplained failures" or a table of the ones that need attention. Two
categories are separated out deliberately, because mixing them makes the file useless:

- *the run's own failure* -- the folder is the wreckage of a run that never compiled on the
  board, and the viewer is faithfully reproducing it. Not a viewer bug.
- *the ~25 ns stretch systematic* -- a generated sequence containing a stretchable pulse measures
  a 100 ns ramp against 20 ns markers, which moves the 50%-of-power crossing. A property of the
  measurement, not of the model; the same-pulse cases agree to 0.05 ns.

Anything left over is a real disagreement between the picture and the hardware.

### Archive sweep  
*2026-08-13 23:18*

- Traced **400** archived runs across **15** runtime classes.
- `compiled.log` cross-check: **366 match, 0 mismatch**, 34 raised.
- Of the 34 that raised, **22** were the original run failing the same way on the board (not viewer faults).

Problems needing attention:

| folder | kind | detail |
|---|---|---|
| `160604` | raised | ValueError: Empty synchronizer |
| `101535` | raised | ValueError: Empty synchronizer |
| `101722` | raised | ValueError: Empty synchronizer |
| `155510` | raised | ValueError: Empty synchronizer |
| `155439` | raised | ValueError: Empty synchronizer |
| `134851` | raised | ValueError: Empty synchronizer |
| `121922` | raised | ValueError: Empty synchronizer |
| `214527` | raised | KeyError: 'qb2_stimulus' |
| `222157` | raised | ValueError: Empty synchronizer |
| `202746` | raised | ValueError: Empty synchronizer |
| `134746` | raised | ValueError: Empty synchronizer |
| `212804` | raised | ValueError: Empty synchronizer |


### Archive sweep  
*2026-08-13 23:23*

- Traced **600** archived runs across **30** runtime classes.
- `compiled.log` cross-check: **586 match, 4 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

Problems needing attention:

| folder | kind | detail |
|---|---|---|
| `141634` | mismatch | {('DAC6', 'DWELL', 30): 1, ('ADC2', 'DWELL', 30): 1, ('DAC5', 'DWELL', 30): 1, ('ADC1', 'DWELL', 30): 1, ('DAC |
| `163440` | mismatch | {('DAC6', 'DWELL', 30): 1, ('ADC2', 'DWELL', 30): 1, ('DAC5', 'DWELL', 30): 1, ('ADC1', 'DWELL', 30): 1, ('DAC |
| `182904` | mismatch | {('DAC10', 'DWELL', 30): 4, ('DAC12', 'DWELL', 30): 4, ('DAC14', 'DWELL', 30): 4} |
| `163544` | mismatch | {('DAC6', 'DWELL', 30): 1, ('ADC2', 'DWELL', 30): 1, ('DAC5', 'DWELL', 30): 1, ('ADC1', 'DWELL', 30): 1, ('DAC |
| `105608` | compare-raised | FileNotFoundError: [Errno 2] No such file or directory: '/home/boson/data/sophia/sophia\\sophia_0706/Qubit_ef_ |


### Barrier padding: a fix that had to be half-reverted
*2026-08-14*

**What was wrong.** The GUI's register panel offered rows captioned
`Operation(<built-in function max>, (30, 30), {})` as settable values. They are barrier padding:
acadia builds the padding length as a compile-time expression, which arrives at the tracer as a
symbolic length, and the panel then invited the user to pin a number the compiler had already
fixed.

**The obvious fix was wrong, and the sweep caught it.** Resolving those Operations and laying the
commands out at their computed length looked right -- symbolic commands went 9 to 0 on a
tomography folder and the padding gained real lengths. But the breadth-first archive sweep then
found **4 runs whose re-trace no longer matched their own `compiled.log`**
(DualRailCCZGateTomographyBasisDebug x3, DRCCZPhaseCalibration), all with the same signature:

    only_in_archive: {}
    only_in_retrace: {('DAC6','DWELL',30): 1, ('ADC2','DWELL',30): 1, ...}

Nothing missing, six commands INVENTED. The archive does contain 30-cycle dwells -- twenty of
them -- so the length itself is plausible; the re-trace simply had six more than the board
played. **acadia keeps these padding entries in its in-memory schedule and does not emit a DMA
command for all of them**, the same way a zero-length command is unrepresentable and never
emitted. The tracer reads the in-memory schedule, so it sees entries the hardware never runs.

**What was kept.** The compile-time value is still computed, but only to record it on the command
(`Command.static_length`) so `register_summary` can decline to offer it as settable. The layout is
unchanged, all four runs match again, and the panel is empty for runtimes with no real registers.

**The lesson.** "This value is knowable, so draw it" does not follow. The question is whether the
BOARD acts on it. The archive is the authority, and a change that improves one folder's picture
while contradicting another's compiled program is a regression however sensible it looks -- this
one survived a full timing revalidation and a 64-case render check before the archive sweep
caught it.

**Why it was found at all.** Only because the sweep ordering was changed from group-by-group to
breadth-first. The first 400-run sweep covered 15 runtime classes and found nothing; the same
budget spread across experiments covered 30 classes and found this immediately. A sweep that
looks broad because the run count is large can be narrow where it matters.

### acadia limit: `repeat_until` accepts only `==` and `!=`
*2026-08-14*

Swept the comparison operator of `repeat_until(counter <op> 3)` with everything else held fixed:

| comparison | acadia | the viewer |
|---|---|---|
| `==` | compiles | count **resolves** (3 passes drawn) |
| `!=` | compiles | count not resolvable -> one pass drawn, flagged `assumed` |
| `<`  | **rejected**: "Less-than comparisons can only check x < 0 or 0 <= x." | n/a |
| `>`  | **rejected**: "Greater-than comparisons can only check 0 > x or x >= 0." | n/a |
| `>=` | **rejected**: same message as `>` | n/a |
| `<=` | **rejected**: `__bool__ should return bool, returned Operation` | n/a |

Two things worth knowing before writing a loop:

1. **Only `==` gives a picture with a real count.** `!=` compiles and runs, but nothing static can
   say how many passes it makes, so the viewer draws one and says so. If you want the sequence
   drawn at its true length, count with `==`.
2. **`<=` fails with a message that does not mention comparisons at all.** The other three
   ordered comparisons name the restriction; `<=` surfaces as
   `__bool__ should return bool, returned Operation`, which reads like a Python bug in your own
   code. It is the same restriction.

The viewer's behaviour is correct on both supported forms: it resolves the `==` count from the
condition and refuses to invent one for `!=`.

### `repeat_until` and `test`, exhaustively on hardware
*2026-08-14*

Every form acadia accepts, measured on the loopback. **No unexplained failure; worst 0.16 ns.**

| sweep | points | worst |
|---|---|---|
| operator sweep, nested test, test-in-loop, loop-in-test | 4 cases | 0.03-0.13 ns |
| `repeat_until` resolved count, 1 -> 8 | 7 | 0.16 ns |
| `test` inside a counter loop, counts 2 -> 5 | 4 | 0.09 ns |
| counter loop inside a `test`, counts 2 -> 4 | 3 | 0.06 ns |
| both arms x nested test | 2 | 0.04 ns |
| both arms x loop-inside-test | 2 | 0.06 ns |
| both arms x test-inside-loop | 2 | 0.12 ns |

Three things this establishes that a single-count test could not:

* the resolved count is right at **every** value including the degenerate 1, and the error does
  not grow with the count -- an error that scales with what it counts is a per-pass miscount,
  which is how the drain-in-loop bug was identified;
* **both arms** of every conditional are correct, in both nesting directions (a loop inside a
  conditional must vanish entirely when the arm is skipped, not run once);
* nesting a conditional inside a loop pays the branch cost on every pass, and that compounds
  correctly.

### Test economics: most of a sweep measures nothing new
*2026-08-14*

The 1000-triple enumeration was spending its board time badly. Roughly 271 of its cases contain a
`stretch`, and every one reports the same classified ~25 ns measurement systematic. Many of the
rest differ only in a label: `block__block__block` and `block__block__loop` both reduce to
blocking edges paying a fall-through gap, so measuring the second after the first adds nothing.

`validation/novel_cases.py` inverts the economics. A deploy costs ~20 s; a trace costs almost
nothing. Every candidate is traced first and reduced to a SIGNATURE of the model features it
exercises -- gap kinds, drain sense, seamless continuation, branch penalty, nesting depth, stream
unrolling -- and only an unseen signature is deployed. Redundant candidates are counted, not
silently dropped, so the coverage claim stays honest.

Measured on the first 60 candidates: **20 distinct signatures, 40 redundant -- 67% of the board
time saved**, before widening to structures the enumeration never reached.

### repeat_until and test, exhaustively on hardware -- all clean
*2026-08-14*

Every legal form of both constructs, measured on the loopback. **0 failures**, worst 0.16 ns:

| sweep | points | worst |
|---|---|---|
| the four new cases as built | 4 | 0.13 ns |
| `repeat_until` count 1,2,3,4,5,6,8 | 7 | 0.16 ns |
| `test` inside a counter loop, count 2..5 | 4 | 0.09 ns |
| a counter loop inside a `test`, count 2..4 | 3 | 0.06 ns |
| nested `test`, both arms | 2 | 0.04 ns |
| loop-in-test, both arms | 2 | 0.06 ns |
| test-in-loop, both arms | 2 | 0.12 ns |

What this establishes that the earlier sweeps did not:

- The resolved count is right at **every** value including the degenerate 1, and the error does
  not grow with the count -- so there is no per-pass miscount left in either nesting direction.
- **Both arms** of every conditional are correct, in all three nesting arrangements (test in loop,
  loop in test, test in test). A skipped arm drops its whole body, including a loop nested inside
  it, and the branch cost is charged once per pass rather than once per sequence.
- The forms acadia REJECTS were established offline and cost no board time, which is the point of
  probing compile-time behaviour separately from timing.

### Archive sweep  
*2026-08-14 09:31*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 10:00*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 10:33*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 11:03*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 11:33*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 12:02*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 12:30*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 12:59*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 13:27*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 13:56*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 14:36*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 15:12*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 15:48*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 16:17*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 16:46*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 17:14*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 17:43*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 18:12*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 18:40*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 19:18*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 19:48*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 20:17*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 20:46*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 21:16*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 21:45*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 22:16*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 22:56*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-14 23:32*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 00:02*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 00:34*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 01:07*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 01:36*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 02:05*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 02:44*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 03:12*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 03:47*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 04:22*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 04:52*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 05:27*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 05:57*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 06:49*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 07:24*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 07:57*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 08:31*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 09:05*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 09:38*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 10:12*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 10:50*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 11:24*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 11:58*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 12:30*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 13:04*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 13:37*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 14:10*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 14:43*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 15:17*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 15:50*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 16:23*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 16:59*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 17:32*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 18:05*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 18:38*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 19:11*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 19:44*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 20:16*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 20:50*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 21:23*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 21:57*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 22:31*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 23:07*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-15 23:41*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 00:13*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 00:46*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 01:20*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 01:53*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 02:26*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 03:01*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 03:34*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 04:11*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 04:45*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 05:22*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 05:55*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 06:28*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 07:01*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 07:34*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 08:07*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 08:40*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 09:14*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 09:47*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 10:21*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 10:54*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 11:31*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 12:06*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 12:41*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 13:17*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 13:52*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 14:25*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 15:00*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 15:40*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 16:20*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 16:56*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 17:30*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 18:07*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 18:41*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 19:14*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 19:48*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 20:21*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 20:56*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 21:30*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 22:04*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 22:38*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 23:12*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-16 23:50*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 00:23*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 00:57*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 01:33*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 02:06*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 02:40*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 03:15*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 03:51*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 04:26*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 05:02*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 05:37*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 06:14*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 06:48*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 07:22*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 07:56*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 08:31*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 09:05*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 09:39*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 10:13*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 10:46*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**


### Archive sweep  
*2026-08-17 11:20*

- Traced **300** archived runs across **29** runtime classes.
- `compiled.log` cross-check: **291 match, 0 mismatch**, 9 raised.
- Of the 9 that raised, **9** were the original run failing the same way on the board (not viewer faults).

**No viewer-side problem found.**

