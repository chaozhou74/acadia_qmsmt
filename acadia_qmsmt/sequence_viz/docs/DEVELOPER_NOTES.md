# Developer notes

Maintenance notes for `sequence_viz`: known issues, gotchas, and open items. For
usage see `README.md`; for the timing model and its hardware validation see
`../validation/README.md` and `TIMING_MODEL.md`.

## Known issues

- **KI_001** — the last channel in the ADC trigger loop arrives ~36 ns early. Workaround is
  a sacrificial dummy capture channel, which the loopback test config enables.
- **KI_002** — barrier `max()` bug; **fixed** in acadia.
- **KI_004** — `test(..., speculation=False)`: the taken arm mistimes by ~25 ns and the
  skipped arm **hangs the sequencer** (`TimeoutError` in `DataManager.sync`, no `t_data`;
  the board recovers on the next deploy). Root cause found by reading the compiled program:
  with `speculation=False` the body is relocated to the program tail, but the `STACK → PC`
  return lands inline on the guard's fall-through path — so the skip path pops an empty
  return stack and jumps to garbage. Wants a fix in acadia (`compiler.block_end` return
  placement), not a workaround here. The tracer flags such blocks in `unsupported_paths`;
  `speculation=True` (the default) is validated on both arms.

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

- **Never compare two `conditional` context dicts with plain `==`.** Since 2026-09-25
  each carries the raw condition `Operation`/`Register` tree (`condition_obj`, next to
  the already-rendered `condition` text) so `evaluate_condition` can substitute this
  point's register values into it. `Operation`/`Register` overload comparison
  operators to build ANOTHER symbolic `Operation` rather than a bool, so `dict.__eq__`
  -- which compares values with `==` -- raises `TypeError: __bool__ should return
  bool, returned Operation` the moment it reaches that key, which it will as soon as
  two *different* `test()`/`repeat_until()` calls happen to render identical text
  (the same register reloaded and re-tested every pass of a Python loop, e.g. the
  `flag != 0` idiom of a round loop). Use `tracing._same_context`
  (a single context) or `tracing._same_conditional` (a whole `conditional` tuple) --
  both compare `condition_obj` by identity instead, which is also the semantically
  right test (every block compiled inside one `with test(...):` shares the exact same
  `condition_obj` reference; a different call gets a fresh one even with identical
  text). `execution_plan`/`_plan_level` and `plotting.branch_regions` both need this.
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

1. KI_004 wants a fix in acadia (`compiler.block_end` return placement), not a workaround here.
2. `repeat_until` has no hardware validation and cannot have one; if a way to bound the trip
   count from archived data turns up, that changes.
3. The Qt wrapper now lives in acadia_gui (`gui/sequence_view.py`); finish wiring it into the
   data browser (a toggle between the live plot view and the sequence view).
