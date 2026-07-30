# Updating the visualizer when acadia / qmsmt change

`sequence_viz` decompiles acadia's own compiled program, so it is coupled to acadia/acadia_qmsmt
internals with no stability guarantee (full surface in `validation/README.md`). When those
packages change, follow this workflow — it is the intended recipe for a maintainer, human or
Claude, and assumes only the checks already in `validation/`.

## Workflow

1. **Localize the breakage — run the checks in order** (from the `sequence_viz/` package root;
   details and what each catches in `validation/README.md`):
   - `python validation/timing_validation.py --dryrun all` — compile/trace breakage, no data needed.
   - `python validation/selftest.py` — trace failure or a trace that no longer matches its own
     `compiled.log` (any decode/hook/format change). Needs archived data.
   - `python validation/timing_validation.py --revalidate` — timing-model drift vs measured
     intervals, even when a trace still succeeds. Needs archived measured runs.

2. **Map the symptom to the coupling point** (see the table below), then read the owning module.

3. **Fix in that module**, matching the existing style. The hook mechanism is `dryrun.py`
   (`hardware_stubbed`, `fake_attach`) + `tracing.py` (`spy_*` around `DMASynchronizer`); the
   string parsing is the marker/regex constants at the top of `tracing.py` and `compiled_log.py`.

4. **Re-validate.** Re-run the three checks; `--revalidate` must hold all cases at ≤0.15 ns and
   `selftest.py` must be green. If the change is in the *gateware* (not the Python API) — the
   timing shifts by whole cycles on every boundary — re-measure on the loopback board
   (`--all`, see `validation/README.md`) and update the constants in `tracing.py` and
   `EMPIRICAL_CONSTANTS.md`.

5. **Update the docs** the change touched: `DEVELOPER_NOTES.md` (known issues / gotchas),
   `validation/README.md` (method + results), `EMPIRICAL_CONSTANTS.md` (the two constants).

## Symptom → likely cause → where to look

| symptom | likely cause | where |
|---|---|---|
| trace raises, or `selftest` reports a `compiled.log` mismatch | acadia changed the compiled-program or `compiled.log` string format | marker/regex constants at top of `tracing.py`; `CMD_RE`/`TRIGGER_RE` + word encoding in `compiled_log.py` |
| `AttributeError` / `KeyError` during trace | a renamed private attr or `_firmware` key | `dryrun.py` (patched entry points, memory pools), `tracing.py` (`_sequencer_type`, `_firmware`, `_bus_latency`, runtime `_ios`/`_config`/`_pulse_cache`) |
| dry run touches hardware, or a new hardware call slips through | acadia added/renamed a hardware entry point not in the no-op list | `dryrun.hardware_stubbed` patch list |
| trace succeeds but `--revalidate` drifts by whole cycles on every boundary | gateware timing change | re-measure (`--all`); update `MEASURED_BOUNDARY_OFFSET` / `MEASURED_BRANCH_PENALTY` in `tracing.py` per `EMPIRICAL_CONSTANTS.md` |
| condition/loop/test mis-read | `Sequencer.{test,loop,repeat_until}` or `Operation` repr changed | `dryrun.branch_recorder`, `tracing.evaluate_condition`/`execution_plan` |
| envelopes wrong or missing | `io.compute_pulse` / pulse-cache shape changed | `tracing._compute_config_envelopes`, `_snapshot`, `_pulse_address_map` |

## What each check can and cannot see

`--dryrun` and `selftest` prove the *decode* still works (structure matches `compiled.log`).
`--revalidate` proves the *timing model* still matches previously-measured hardware. Neither can
catch a gateware change that alters real timing — only a fresh board measurement (`--all`) can,
which is why the loopback harness is kept rather than replaced by the offline checks alone.
