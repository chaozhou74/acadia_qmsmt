# SeeQuence — what the picture means, and what you can change

This is the practical guide to the sequence viewer: how to read it, which parts are measured fact
and which are the tracer's best guess, and what the controls do. If you are debugging odd data and
want to know whether the board really played what you think it played, start here.

---

## 1. The one thing to understand first

The viewer does **not** simulate your sequence. It decompiles **acadia's own compiled program** —
the same instruction stream the board runs — and lays it out on a two-clock model of the
sequencer and each channel's DMA FIFO. That is why it can be checked against hardware, and it is.

Every number it draws falls into one of three categories, and the difference matters:

| Category | Meaning | How much to trust it |
|---|---|---|
| **Measured** | Timing verified against a 4-channel DAC→ADC loopback | Agreement is **0.23 ns** — about 5% of one 5 ns cycle |
| **Resolved** | Read out of the run's own captured cache (a register value, a loop count) | As good as the run's own data |
| **Assumed** | The sequencer decides it at runtime from live data, so a static trace cannot know | **One possibility, not fact** — the viewer says so |

The third category is the one that bites. A feedback-reset arm depends on what the qubit did; a
loop count can come from a value the trace cannot see. The viewer never silently picks one and
presents it as truth — it marks it `assumed`, and now lets you change it.

---

## 2. The Control flow panel (new)

One row per `loop`, `repeat_until` and `test` in the sequence, nested and indented the way the
sequence nests.

**For a loop or `repeat_until`** you get a spin box: *how many passes of this body to draw*. Type
a number and the timeline re-lays out immediately — every block after it moves, and all the gaps
are recounted.

**For a `test`** you get `auto / taken / skipped`. Pick an arm and the body is drawn or dropped,
with the timing that follows from that choice, including the branch cost.

Each row is prefixed with where its value came from:

- **`resolved`** — read from the run's cache. Leave it alone unless you are exploring.
- **`(assumed)`** — the tracer had to guess. **This is the flag to look for.** If a sequence looks
  wrong and the count is assumed, set it to what you expect and see if the picture becomes the one
  you had in mind.
- **`(pinned)`** — you set it.

Nothing is re-traced when you change these. The compiled program, the pulses and the cache are
untouched; only the *layout* is recomputed. So you can explore freely — you cannot corrupt the
trace, and switching back to `auto` restores exactly what the run implies.

> **How a construct is named — and why it now has a depth.** A row is `repeat_until @11.2 x3`:
> `@11` is the first block of the body, `.2` is its nesting depth, `x3` is the count. Nested
> constructs regularly begin at the *same* block — a cooling round and the active reset inside it
> both start at `@11` — so the block alone named both of them. The depth is what tells them apart,
> and it is shown only where a block really is shared, to keep the common case quiet.
>
> This was a **real bug**, not cosmetics. Editing a row wrote `loop_counts[block]`, which matches
> *every* construct starting at that block, so raising the inner cooling loop raised the outer one
> with it — the reported "I changed one and the others changed too". The tabs on the diagram already
> used the full key `(block, depth)`; the panel did not, so the two disagreed as well. Both write
> the same key now, and `validation/nesting_boxes.py` checks the resulting independence over every
> archived run: pinning a construct changes that construct and the things nested inside it, and
> nothing else.

### How to read a label: `repeat_until @11.2#2.1 x4 *?`

Every construct is named the same way in the panel and on its tab, so you can match them by eye.
Each piece is one fact, and the optional pieces appear only when they are needed:

| piece | meaning |
|---|---|
| `@11` | the first block of the body — the construct's name |
| `.2` | its **nesting depth**. Shown when that block keys more than one construct (a cooling round and the reset inside it both begin at `@11`) — and always past depth 4, because the box inset and the tab ink stop changing there, so the drawing alone can no longer tell you which level you are looking at |
| `#2` | which **execution** — a construct inside a loop runs once per enclosing pass, and each run is separately settable. Shown only when there is more than one |
| `x4` | how many passes are **drawn**. `x0` means drawn zero times; for a `test`, `skip` means the arm is not drawn |
| `*` | **you pinned this.** The drawing is your hypothesis, not what the run did |
| `?` | **the board decides this at runtime** — the count or arm cannot be read from this run's data |

The last two are the ones that matter when you are debugging real data. `?` is a property of the
*construct*, so it stays there when you pin a value: choosing a number does not make the hardware
deterministic, it just picks which possibility is on screen. And `*` exists so a pinned `x5` can
never be mistaken for a measured `x5`.

The flags sit in their own token, after a space. Run together with the count they were genuinely
misread — at the tab's 6 pt, `x1?` looks like `x17`, and a marker that reads as a digit is worse
than no marker at all.

### Per-execution rows, and folding

A construct that runs several times gets one row for the construct — which sets **all** of its
executions — and one row per execution beneath it, which set **one each**. An active-reset loop
genuinely takes a different number of rounds every time it runs, so a single number for all of them
would be a fiction. The execution rows start folded, behind the small arrow on the construct's row;
the whole panel folds away from the checkbox in its "Control flow" title.

### Editing from the diagram itself — the tabs

Every control-flow block draws a small **tab at its top-left**, labelled with its count (`x3`) or
`test`. That tab is the handle:

* **hover it** — the block it governs is shaded, and a tooltip names the construct and where its
  value came from (`repeat_until @8 — 2 passes (resolved)`);
* **click it** — a box opens to change the iteration count, or to pick the arm of a `test`.

The panel and the diagram stay in sync whichever you use.

**Why a tab rather than the block itself.** The plot area belongs to the box-zoom gesture, so
clicking a span to edit it also dragged out a zoom rectangle — two things fighting over the same
press. A tab is a small, unambiguous target, and the viewport explicitly yields to it, so a click
on a tab edits and does not zoom while a click anywhere else still zooms normally.

**Nested constructs each get their own tab.** They are drawn at their own inset, so an outer loop
and the loop inside it have tabs at different heights and you can point at either one
independently. Deeper tabs sit on top where they overlap, so the one you see is the one you get.

**One band of rows per nesting depth.** A row of tabs is a nesting level, and constructs that start
close together add a row *inside their own level's band* rather than climbing into the level above.
Bumping across levels was tried first and read wrong: one tab ended up alone on a fourth row of a
three-deep sequence, which said "four levels of nesting" when there were three.

**Hiding the tabs.** The *control-flow tabs* checkbox in the Marks group turns the strip of handles
off while keeping the dashed boxes. The boxes say what the structure is; the tabs are for changing
it, and on a sequence with twenty constructs the handles are clutter when you are reading pulses.
Hiding them also gives their vertical strip back to the lanes, and leaves no invisible click
targets — a press up there drags a zoom box like anywhere else. The Control flow panel still edits
everything.

**Tab colours are chosen for contrast, not fixed.** The palette lightens with nesting depth on
purpose, so a label fixed at white became unreadable on the paler deep tabs (2.5:1 at depth 3,
1.7:1 at depth 4), and a hollow tab inked straight onto a dark page disappeared (1.7:1). The label
now takes whichever of white/dark ink reads better on its own tab, and a hollow tab's ink is moved
toward the page's opposite until it is readable — keeping the hue that tells you which nesting
level it belongs to.

**A construct that is drawn zero times still has a tab** — hollow instead of solid, and labelled
`x0` or `skip`. Setting a loop to 0 passes used to remove the very handle you would use to put it
back, leaving the panel as the only way out. A setting that hides its own control is a trap. This
also cascades correctly: zero the outer loop and the constructs nested inside it go hollow too,
because they genuinely are not reached.

**Executions are tagged by their pass path, not by position** — `@11#2.1` is "the 1st pass of this
construct during the 2nd pass of the loop around it". Numbering them 1, 2, 3 in drawn order was
tried and is unstable: skip the first of two and the second becomes "#1", so the row you just
changed appears to be a different construct.

**And so does each individual execution.** Setting *one* execution to 0 keeps that execution's own
hollow tab, its panel row, and its siblings' tabs unchanged. This needed the execution list to come
from what the layout **reaches** rather than from what it draws: an execution drawn zero times
produces no placements, so reading the placements dropped it from the list entirely — its row and its
tab both disappeared, and the setting deleted its own control. `nesting_boxes.py` now pins every
construct *and* every execution to zero in turn, across the whole archive, and fails if any handle
goes missing.

### Why this was worth adding

The backend has always accepted `loop_counts[block] = N` and `path_choices[block] = True/False`,
and `relayout()` has always re-timed in place. There was simply no way to reach them without
editing block indices by hand. That meant the single most important question about a drawn
sequence — *"is this the run, or one possibility the tracer had to pick?"* — had no answer in the
UI, and an `assumed` branch looked exactly like a certain one.

---

## 3. The Registers panel

Register- and DSP-driven **lengths** (a swept T1 delay, a stretched pulse). Values read from the
run's cache are shown read-only; ones the trace cannot recover are editable in cycles.

**A trap this panel will now show you.** A register-driven length of **zero** is not "nothing".
`Acadia.command_dma` emits `length - 1`, so 0 wraps to an all-ones length field: ~328 µs for an
ARB command, and **~21 seconds** for a 32-bit dwell. On the board that reads as a hung run
(`Timeout occurred waiting for line`), not as a bad sweep point. The viewer now draws the wrap at
its true length instead of drawing nothing, so a sweep that starts at zero is visible *before* you
deploy. `np.linspace(0, stop, num)` is the usual way in — floor the value at one cycle, as
`dual_rail_ramsey._delay_cycles` does.

---

## 4. Reading the timeline

- **Lanes** are channels (`DAC7 / q1_stimulus`), stimulus and capture.
- **Coloured bars** are pulses, named if the tracer can name them.
- **Hatched grey** is barrier padding — dead time inserted so channels stay in lock step.
- **Plain grey** is a dwell you scheduled.
- **Cross-hatched** is an *indeterminate* length: a register value the trace could not recover.
- **Pale yellow** is the inter-block gap — the sequencer's own overhead between blocks.
- **Dashed blue** spans are control flow; the label says which construct.

**The sweep point is kept across a reload, and applied.** Reloading a folder, or loading another
one, keeps the point you were reading (clamped to what the new run captured) and moves the trace to
it — so the number beside *Point* is always the point on screen. It used to be kept as a *label*
only, which drew point 0's data under someone else's index; see bug 27 in `validation/README.md`.
A point change re-reads that point's cache, and the cache decides register-driven lengths, test
arms and `repeat_until` counts, so the sequence can genuinely change length and even gain or lose
constructs between points.

**Bar widths have one deliberate white lie.** When a command ends exactly where the next begins,
the earlier bar is drawn short by 1.5 screen pixels so the two do not fuse into one block. Three
things bound it: a bar's **start is never moved**, the inset is a fixed number of *pixels* so it
shrinks to nothing as you zoom in, and a bar is never shortened by more than half. Measure
durations zoomed in, or from the trace.

---

## 5. When a folder will not open

The viewer re-runs `runtime.main()` to rebuild the sequence. So a sequence that could not compile
**on the board** cannot compile in the viewer either. When that happens it now tells you
explicitly:

> `[sequence_viz] This is NOT a viewer error: the original run failed the same way on the board`

That means the folder is the wreckage of a run that never produced data — fix the sequence or the
config and re-run. A real viewer bug looks different: the folder has data and the trace still
fails.

---

## 6. What is verified, and what is not

**Verified against hardware** (see `README.md` in this directory for the full record):

- 62 hand-written cases, every ordered **pair** of the 10 scheduling primitives (100 deploys),
  **125 ordered triples**, and randomly generated sequences — **0.23 ns worst**.
- The **drawing** itself: every rectangle checked against the command it represents, on all cases
  and on real archived runs.
- 161 archived runs reproduce their own `compiled.log` exactly.

**`repeat_until` runs its body at least once — so a target of 0 never exits.**
Measured 2026-08-14: `repeat_until(counter == 0)`, on a counter loaded 0 and incremented once per
pass, never returns from the board — repeated "Timeout occurred waiting for line" until the run is
killed. The same loop at 1,2,3,4,5,6,8 measures to ~0.1 ns.

That one measurement also settles what `repeat_until` *means*, which nothing else could: testing the
condition before the body and testing it after predict the same pass count for every target ≥ 1, and
differ only at 0 — test-first would exit immediately, test-after must go round again. The hang says
test-after.

The viewer used to draw such a loop as a tidy empty body, which is a picture the hardware cannot
produce. It now labels it **`x∞`**, draws one pass, and captions it *"NEVER EXITS: the counter starts
at 0 and is incremented before this test is next evaluated, so it cannot reach 0"*. The panel row
reads `(never exits)`, and `validation/timing_validation.py` refuses to deploy any sequence
containing one — asked of the model, so it covers cases nobody has written yet.

*A runtime cannot express "run this loop zero times" with a counter loop.* Guard the parameter.
Pinning 0 in the panel is a different thing and stays allowed: that is a drawing hypothesis, marked
`*`, not a compiled program.

**Known limits, stated rather than hidden:**

- A **2-descriptor** batch behind a `fifo_almost_empty` drain is 1–2 cycles optimistic. Confined
  to that one case by two independent scans; no runtime does it.
- Sequences containing a stretchable pulse carry a **~25 ns measurement** systematic (the 100 ns
  stretch ramp against 20 ns markers moves the 50%-power crossing). This is the measurement, not
  the model — the same-pulse cases agree to 0.05 ns.
- For the multi-rail XEB runtimes the gate **identities and durations** are decoded, but **how many
  gates play** depends on a loop count and `test` arms that are data-dependent, so the train drawn
  is one pass.

  Pinning that loop in the Control flow panel draws the extra passes, but it **cannot invent their
  gates**: a run's cache holds exactly the words its own circuit played, so asking for more passes
  than it ran leaves nothing to decode. Measured on `DualRail_XEB_2DR` — pin `(19,2)` to 2 and you
  get 4 decoded gates plus 2 unidentified; pin it to 5 and you get 4 decoded plus 8 unidentified.
  The unidentified ones are drawn as hatched **`indeterminate (register)`** markers, so they are
  visible and honestly not-a-gate rather than invented pulses. Pinning tells you where those passes
  SIT, not what they play.

**One oracle you should not lean on.** `compiled_log.compare` reports a clean match on the XEB
runs even when the gate commands are wrong, because the archive records them as `REG0` rather
than as hex and both sides skip them. It is structurally blind to exactly those commands. The
loopback measurements are the authority.

---

## 7. How the GUI itself is tested

`validation/gui_validation.py` drives the panel headlessly on real archived folders: it builds
every control-flow row, changes each one **through the widget** (not the backing dict), lets the
event loop run, and checks the timeline actually moved and that clearing the overrides restores
it exactly. It then synthesises a hover over the middle of every control-flow span and asserts
the innermost construct is the one picked.

This exists because the first version of the panel **crashed on its first edit** -- it rebuilt
itself from inside a spin box's own `valueChanged` handler, which deletes the widget currently
emitting the signal. Nothing that inspects the trace, the layout or the figure can see that; only
pushing the widget can. The same harness then caught a second problem: rebuilding on every edit
destroyed and recreated the widgets, losing focus and the caret mid-typing. Rows are now only
built when the trace changes and are updated in place otherwise.

It also reports its own coverage (`11 controls + 70 hovers driven`) and **fails if it exercised
nothing** -- an earlier version passed while refusing every synthesised hover, because the panel
had no axes until a redraw. A test that silently tests nothing is worse than no test.

### Can it be broken? `validation/gui_robustness.py`

Correct on real data and unbreakable are different properties, so they are checked differently.

* **Every callback is guarded, checked from the source.** PyQt5 turns an exception that escapes a
  slot into `qFatal()` — the whole data browser aborts, not just this panel. So each bound method
  the panel hands to Qt or matplotlib is wrapped, and the harness re-reads the source with `ast` to
  prove none was missed. Adding a callback next month is covered without anyone remembering to
  list it here.
* **A guard that fires is a failure, not a shrug.** Every catch is recorded on the widget and shown
  in the status line, and both `gui_validation.py` and `gui_robustness.py` fail on a non-empty
  list. The guard converts a crash into a test failure; it is not permission to ship a broken path.
* **Hostile and degenerate inputs**: counts of `0` and `100000`, a folder that is not data, a
  zero-width and an inverted viewport, a widget resized to 1×1, the trace swapped underneath a
  built panel and then removed, and override keys that name constructs which do not exist.
* **Random event storms**: hundreds of clicks, drags, wheel events, key presses, resizes and
  control changes per seed, in random interleavings, each seed in its **own process** so that a
  hard abort shows up as a signal instead of taking the harness with it. The seed is printed, so
  anything found is reproducible with `--storm <seed>`.

### The four things looking at the picture caught that no assertion had

Both were found by rendering the panel to PNG and reading it, which is why that is now part of the
routine rather than an afterthought.

1. **Tab labels piled up on zoom.** Every artist in the drawing is registered for removal between
   frames — every one except the tab *text*. Zooming in and back out left the old labels behind for
   the next frame to draw over, so `@8 x3` became an unreadable smear. The rectangles were always
   cleaned up, which is why only the text misbehaved.
2. **A zoomed-out view silently stopped being zoomed out.** Re-timing changes the length of the
   sequence — drawing four passes of a loop instead of one makes it longer — and the panel restored
   the previous window afterwards to protect your zoom. Raising one `repeat_until` therefore left a
   third of the sequence, and 10 of its 21 constructs, outside a view that looked complete. The rule
   now: a window that covered the whole sequence keeps covering the whole sequence; a window you
   zoomed in yourself is kept exactly as it was.
3. **A box covering four passes was captioned "1 pass shown".** The caption for a pinned
   `repeat_until` never consulted the count that had been pinned, so it contradicted the four boxes
   drawn beside it. It now reads *"4 passes drawn (pinned); this run's cache says 3"* — both numbers,
   and whose each one is.
4. **An outer 3-pass loop was captioned "4 passes".** The pass count came from each placement's
   `iteration` field, which is a single scalar — the *innermost* loop's index — so at any outer level
   it counted the wrong loop. It now counts that construct's own slot in the execution path.

### What a dashed box and its vertical edges mean

A dashed rectangle is **one execution** of a construct, and a vertical dashed edge is where that
construct was **entered or left**. That is worth stating because it was wrong: spans were grouped by
consecutive *block index*, and a loop replays its body, so the indices run `11,12,11,12…` and the
grouping broke at every pass boundary. One execution came out as one rectangle per pass — so raising
an inner count grew extra vertical edges on the **outer** construct, which is entered exactly once.

The rule is now derived from the execution plan alone:

    rectangles(construct) == number of distinct enclosing-pass paths

An outermost construct has exactly one, so it draws one box however many passes it runs. A construct
one level in runs once per enclosing pass and draws one box per pass — those edges are real
re-entries. `validation/nesting_boxes.py` checks this over the whole archive, against entry counts
computed straight from the placements rather than from the drawing code.

## 8. Speed, and the limit it puts on the panel

Asking the fuzzer to type the largest number each spin box allowed found **three** compounding
problems. All were quadratic, none was the drawing:

* **`machine_layout` rebuilt a whole-plan set once per placement** -- 9.1 M dictionary lookups at
  3000 placements. It does not depend on the placement, so it is computed once: a 1000-pass pin went
  from **11.9 s to 0.48 s**.
* **The execution tag was scanned, not looked up.** Pinning an outer loop to N passes gives every
  inner construct N executions *and* N spans, and the tag for each span was found by scanning that
  construct's execution list -- O(spans x executions). One edit took **475 s**; with a dict built
  once per draw, **1.35 s**.
* **The pass limit was derived from the wrong number.** A construct's per-pass cost was taken from
  the count of *blocks* in its body, but a body containing a loop expands to far more: 83 passes of a
  "3-block" body produced 583 placements. The cost is now measured from the plan itself (placements
  inside the construct / its drawn passes), and -- importantly -- grouped by the construct's logical
  `id`, because **every block carries its own copy of the context dicts**, so grouping by object
  identity counted a single block and called a three-block body "one".

There was a fourth, subtler one. The limit lived only in the spin box's `maximum`, and that was
re-derived in the panel refresh — which Qt runs on a **deferred** timer. Anything firing edits faster
than the event loop turns (a script, a held arrow key, the fuzzer) landed several values before any
limit updated, and the plan ballooned in the gap. The budget is now enforced **where the count is
applied**, which nothing can outrun; if a value is capped the status line says so rather than
silently changing it. Checked with the adversarial case — nine edits to nine constructs' maxima with
no event-loop turn between them — which now settles at 317 placements in 9.5 s, the plain cost of
nine redraws.

After all three, the **worst edit the panel can be asked to make is 1.29 s** (it was 475 s), and the
spin box maximum is a derived number with a tooltip that states it. Drawing was never the bottleneck:
`add_patch` was replaced with `add_artist` (identical pixels, skips a data-limit update this drawing
never uses) for a ~10% saving, and that is all there was to get.

**Batching the artists was measured and rejected.** The obvious next step is folding the rectangles
into `PatchCollection`s the way the envelopes already are. Benchmarked directly: 3500 rectangles take
**2.74 s** individually and **0.92 s** as one collection -- a 3x saving on a part that is only about
**17%** of the draw. So a full batching refactor buys 10-15%, paid for by rewriting the inner loops
of the one file whose correctness everything else here exists to defend. If drawing hundreds of
passes ever becomes a real requirement, the thing to change is the *number* of artists -- summarising
repeated passes instead of drawing N copies -- not how they are attached.

---

## 9. Suggested further work

Not done, listed so it is not rediscovered from scratch:
- **A "what changed" diff** between two sweep points, to see where a sequence stopped matching.
- **Resolving the XEB gate count** from the cache so the train draws at full length automatically.
  A latent bug sits in the way: `describe_cache_stream` takes `count_word` from the *first*
  `DSP_C` load in the program, and the XEB runtimes have two.
