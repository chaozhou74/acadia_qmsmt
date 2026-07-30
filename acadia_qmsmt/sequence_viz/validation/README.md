# validation — the regression net for acadia / acadia_qmsmt drift

`sequence_viz` works by decompiling acadia's *own* compiled program, so it leans on acadia
internals that carry no stability promise: the text of `instruction.pprint()`, the shape of
`DMASynchronizer.merge_schedules`, `acadia._firmware` keys, private memory pools, and the
`compiled.log` line format. When acadia or acadia_qmsmt change, those assumptions can break —
sometimes silently. This module exists to catch that.

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

## Setup

`--all`/`--revalidate`/`selftest.py` need station-specific paths (board IP, data roots), read
from a gitignored `paths.local.yaml`:

    cp validation/paths.local.example.yaml validation/paths.local.yaml   # then edit

`--dryrun` needs none of it. See `../docs/VALIDATION.md` for the staged hardware-validation
record and `loopback_timing_cases.py` for the test sequences.
