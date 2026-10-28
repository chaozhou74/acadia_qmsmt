# Developer notes

Maintenance notes for `sequence_viz`: known issues, gotchas, and open items. For
usage see `README.md`; for the timing model and its hardware validation see
`../validation/README.md` and `EMPIRICAL_CONSTANTS.md`.

## Known issues

- **KI_001** — the last channel in the ADC trigger loop arrives ~36 ns early. Workaround is
  a sacrificial dummy capture channel, which the loopback test config enables.
- **KI_002** — barrier `max()` bug; **fixed** in acadia.
- **KI_003** — every blocking boundary costs +1 cycle beyond `detect + issue + propagate`,
  carried as `MEASURED_BOUNDARY_OFFSET = 1` in `tracing.py`. Best-guess source (not yet
  confirmed in gateware): the DMA-poll/hold takes one registered cycle to *act* on the
  de-asserted `dma_running` it read, beyond the cycles `_bus_latency` counts to get the value
  into `BUS_DATA`. See `EMPIRICAL_CONSTANTS.md`.
- **KI_004** — `test(..., speculation=False)`: the taken arm mistimes by ~25 ns and the
  skipped arm **hangs the sequencer** (`TimeoutError` in `DataManager.sync`, no `t_data`;
  the board recovers on the next deploy). Root cause found by reading the compiled program:
  with `speculation=False` the body is relocated to the program tail, but the `STACK → PC`
  return lands inline on the guard's fall-through path — so the skip path pops an empty
  return stack and jumps to garbage. Wants a fix in acadia (`compiler.block_end` return
  placement), not a workaround here. The tracer flags such blocks in `unsupported_paths`;
  `speculation=True` (the default) is validated on both arms.

Plus `MEASURED_BRANCH_PENALTY = 3` cycles on any taken branch (loop back-edge or skipped
`test`) — fitted on loop back-edges, then it predicted the forward `test` skip gap (95 ns)
with no retuning. Best guess: a fixed pipeline-flush depth in the sequencer core, in gateware
(the instruction-memory pipeline knobs are 0, so it is not memory latency).

## Instrument safety — do not weaken this

Some runtimes set **external** instruments in `main()` (e.g. a flux-sweep spectroscopy ramps
a bias source through `instrumentserver`'s proxy client). Those calls are real and move lab
hardware. `dryrun.hardware_stubbed` blocks `instrumentserver.client.proxy.Client.__init__`
and raises `InstrumentAccessBlocked`, so such runtimes fail to trace by design.
`allow_instruments=True` is an escape hatch for when the instrument server is known offline;
it must not become a default.

That block also removed a side effect: a leaked client left a zmq socket with `LINGER=-1`,
and garbage-collecting it blocked `zmq.Context.term()` forever — a "hang" that was really a GC
waiting on a dead socket. Do not fix that by destroying zmq contexts from inside the library:
in a Jupyter kernel those contexts belong to ipykernel, and destroying them kills the kernel.

## Breadth test

Every distinct runtime class in one station's full archive (39 classes) was traced as a
breadth test — **37 succeed**, and `compare_with_compiled_log` matches on all 37; the other 2
are the instrument-touching ones above, blocked by design. A good regression set spans the
runtime shapes: single-block, register-driven (T1/T2), loop, `test`, `repeat_until`, stretch,
and multi-rail — the `selftest_folders` in `validation/paths.local.yaml` are picked that way.

Two bugs the sweep found, now fixed:

- `set_frequency` is defined on `MeasurableResonator` and `Qubit`, not just `InputOutput`, and
  an archived `acadia_qmsmt.py` has its own copies — so stubs are discovered by scanning the
  modules in the runtime's MRO rather than a hard-coded class list.
- A DMA command whose whole word comes from the bus carries `{"command": BUS_DATA}` and *no*
  `length` key (randomized benchmarking picks its Clifford that way); it is treated as a
  symbolic length rather than raising `KeyError`.

## Gotchas

- **A trace point is one `acadia.run()` call.** Repeated shots remain separate points. Do not
  infer distinct experimental settings from waveform equality or silently collapse the points.
- **Check the point before diagnosing a missing envelope.** In the q3 AllXY archive
  `261023/113114`, point 41 is still I·I; point 50 begins X180·X180 because each pair fires
  50 shots. Memory capture and the compiled schedule were correct. The proposed hollow
  zero-waveform style was reverted at the user's request; pulse backgrounds remain solid.
- **A runtime object can only be traced once** (`Processor data is non-empty`). `trace_folder`
  builds a fresh one each call; `already_traced()` gives the explanatory error.
- **Tracing must not mutate the runtime.** `duplicate_pulse` appends to `io._config["pulses"]`,
  which `_dump_fields` would archive; `preserved_runtime_state` undoes that.
- **`envelope_mode` is `"magnitude"` or `"iq"`** — not `"power"`.
- **Trace one runtime at a time in a process.** Tracing many and keeping them all alive was
  what surfaced the zmq hang.
- **Screenshots: never `savefig(bbox_inches="tight")` when checking layout.** It silently
  hides a clipped legend. `fit_layout()` reserves the margin, and the interactive canvas gets
  no tight-bbox expansion.

## Open items

1. Trace `MEASURED_BOUNDARY_OFFSET` (KI_003) and `MEASURED_BRANCH_PENALTY` to their source in
   the acadia gateware. `EMPIRICAL_CONSTANTS.md` is the brief; `--revalidate` holding all
   cases at ≤0.15 ns is the acceptance test for any candidate explanation.
2. KI_004 wants a fix in acadia (`compiler.block_end` return placement), not a workaround here.
3. `repeat_until` has no hardware validation and cannot have one; if a way to bound the trip
   count from archived data turns up, that changes.
4. The Qt wrapper now lives in acadia_gui (`gui/sequence_view.py`); finish wiring it into the
   data browser (a toggle between the live plot view and the sequence view).

## Immediate register-store selections

`tracing.describe_constant_register_writes` records the source instruction address,
immediate value (or unknown) and supported plain-register predicate of stores.
`machine._constant_register_gate` starts from the last unconditional assignment
before a DMA block and applies following conditional assignments. Unknown source
writes invalidate the result. Pulse identities come only from the packed DMA
address and the run's memory map; lengths come from the word's low bits.

Do not advance a streamed cache pointer for such a selection. Forward PC-branch
assignments are deliberately excluded, because their latest executed write must
be resolved by the path walker. Unsupported arithmetic/masked predicates are
not simplified into a guessed truth value. Keep `validation/conditional_register_dma.py`
and the streamed-register checks when changing this priority.

## Loop-header words end at the next register write (2026-10-27)

`tracing.linear_header_words` derives the words a counted pointer loop's header
latches. They are valid only up to the next write of that register inside the
body. Z2HiggsSimulation (`261016/231217`) loads REG0-2 in the header for the
loc pulses, then reloads them from another pointer for the A layer; applying the
header words to the A block decoded raw cache words as lengths (61 M cycles) and
stretched a 21 us sequence to ~1 s, overflowing the GUI's int32 time scrollbar.
Registers written between the loop's first block and the current block are now
dropped from the header result, so the executed-path walk keeps its word.
