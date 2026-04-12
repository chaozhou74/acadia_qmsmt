# acadia_qmsmt

`acadia_qmsmt` is a Python package for quantum measurement experiments based on the [Acadia](https://github.com/BillyKalfus/acadia.git) RFSoC control framework.

## Features

### YAML-based parameter management

The workflow is designed around human-editable YAML files that describe:

- physical DAC/ADC channels
- channel configurations including NCO (numerically controlled oscillator) frequencies, reconstruction modes, etc.
- pulse configurations including shape, length, amplitude, detuning, etc.
- readout memories, windows, and state-classifier configurations

The example config in [`acadia_qmsmt/examples/config.yaml`](acadia_qmsmt/examples/config.yaml) shows this style in practice.

The package also includes helpers for loading and updating experiment configuration files:

- [`acadia_qmsmt/utils/yaml_editor.py`](acadia_qmsmt/utils/yaml_editor.py)

### Abstract classes

The core classes in `qmsmt.py` provide higher-level abstractions that make it easier to build measurement code on top of `acadia`.

- `InputOutput`: a wrapper around an Acadia `Channel` together with its configuration. It provides convenience methods for working with pulses, waveform memory, capture windows, and other channel-specific resources.
- `QMsmtRuntime`: a base runtime class that extends Acadia's `Runtime` with YAML-aware channel setup. It can automatically build and configure `InputOutput` objects from config dictionaries or YAML entries, which makes it easier to write experiment runtimes with less repeated setup code.
- Quantum objects including `Qubit`, `MeasurableResonator`, `TwoQubit`, etc., and related classes: these bundle together the relevant control and readout channels so experiment code can work with physical objects rather than only raw DAC/ADC channels.

These abstractions are meant to reduce boilerplate in experiment scripts and make common measurement patterns easier to reuse.

### Prebuilt experiment runtimes

The [`acadia_qmsmt/runtimes`](acadia_qmsmt/runtimes) package contains ready-made runtime modules for common experiments, including:

- resonator and qubit spectroscopy
- amplitude and length Rabi measurements
- readout window (aka kernel/weight function) calibration
- readout fidelity metrics
- qubit `T1` and `T2`
- single-qubit gate randomized benchmarking
- beam-splitter and SWAP-related experiments
- other experiment templates that can be adapted for new measurements

These runtimes can be used directly for simple measurements, but they are also meant to serve as examples of how to write experiments with `acadia` using the abstractions provided in `qmsmt`.

### Analysis and plotting utils

The package also includes:

- [`acadia_qmsmt/analysis`](acadia_qmsmt/analysis) for processing, classification, tomography, sweep prediction, and fitting routines
- [`acadia_qmsmt/plotting`](acadia_qmsmt/plotting) for commonly used plotting helpers
- [`acadia_qmsmt/utils`](acadia_qmsmt/utils) for logging, path handling, saved-runtime loading, annotations (for GUI usage), and config manipulation

## Getting Started

### Install from source

```bash
cd path/to/acadia_qmsmt
pip install -e .
```

### Explore notebooks and examples

The `examples` folder includes notebooks and example yaml files for getting started with experiments.


## Important Notes on Board Deployment

`acadia` follows a "code on host, run on board" pattern: the runtime script is sent to the board, executed there, and data is sent back through `DataManager`.

`acadia_qmsmt` follows the same pattern, but we do not want to require a separate installation of `acadia_qmsmt` on the board. If we did that, local edits to `acadia_qmsmt` would not automatically be reflected on the board, which would quickly become a version-control headache.

To avoid this, `QMsmtRuntime` does something slightly sneaky when preparing files for deployment:

- it copies the user runtime script to the board, as `acadia` normally does
- it also copies [`acadia_qmsmt/qmsmt.py`](acadia_qmsmt/qmsmt.py) to the board
- on the board, that file is renamed to `acadia_qmsmt.py`

Locally, many of the public classes are exposed through [`acadia_qmsmt/__init__.py`](acadia_qmsmt/__init__.py). In practice, this means that the important imports on the host side and the board side both end up resolving to the same implementation file.

TL;DR: when you run a `QMsmtRuntime`, you are not relying on a separately installed copy of `acadia_qmsmt` on the board. The relevant `qmsmt` implementation is automatically shipped together with the runtime, which keeps the board behavior in sync with your local working copy.

This also means that, for any custom function you want to use in the `main` function of a runtime, it must be written in `qmsmt.py` for the board to have access to it.
