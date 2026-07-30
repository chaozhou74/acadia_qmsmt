"""
Regression self-test for sequence_viz. No hardware, no deploy.

Traces a set of archived data folders and checks each one against its own
``compiled.log`` -- the independent artifact of what the FPGA actually ran -- then
exercises every render option, point selection, and the interactive viewport.

    $ACADIA_ENV/bin/python validation/selftest.py

The archived runs it traces are read from the gitignored
``validation/paths.local.yaml`` (``selftest_folders``), so no data path is
committed; copy ``validation/paths.local.example.yaml`` to configure them.
"""
import itertools
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _name in ("acadia", "qmsmt_runtime_loader", "sequence_viz"):
    logging.getLogger(_name).setLevel(logging.ERROR)

from acadia_qmsmt import sequence_viz as sv

sys.path.insert(0, str(Path(__file__).resolve().parent))  # paths_local (sibling)
import paths_local

# Archived runs to trace, from the gitignored validation/paths.local.yaml
# (label -> data folder). One folder per runtime class is a good varied set;
# see validation/paths.local.example.yaml.
FOLDERS = list(paths_local.load().get("selftest_folders", {}).items())

RENDER_OPTIONS = list(itertools.product(
    ("memory", "name", "channel"), ("memory", "config"),
    ("per-pulse", "channel", "shared", "absolute"), ("magnitude", "iq")))


def main():
    failures = []
    if not FOLDERS:
        print("no selftest_folders configured -- copy "
              "validation/paths.local.example.yaml to validation/paths.local.yaml "
              "and add archived runs under 'selftest_folders'.")
        return 1
    print(f"{'folder':10s} {'trace':>7s} {'pts':>6s} {'blocks':>7s} {'match':>6s} "
          f"{'dead ns':>8s}  registers")
    for label, folder in FOLDERS:
        if not Path(folder).is_dir():
            print(f"{label:10s}  MISSING {folder}")
            failures.append(f"{label}: folder missing")
            continue
        started = time.perf_counter()
        try:
            trace = sv.trace_folder(folder)
            elapsed = time.perf_counter() - started
            check = sv.compare_with_compiled_log(trace, folder)

            for color_by, source, scale, mode in RENDER_OPTIONS:
                figure, _ = sv.plot_trace(
                    trace, xlim_ns=(0, min(3000, trace.length_ns)),
                    color_by=color_by, envelope_source=source,
                    envelope_scale=scale, envelope_mode=mode)
                plt.close(figure)

            for point in (0, trace.n_points // 2, trace.n_points - 1):
                trace.select_point(point)

            view = sv.interactive_view(trace)
            view.set_point(0)
            view.set_window(0, min(1000, trace.length_ns))
            view.reset()
            plt.close(view.ax.figure)

            registers = ", ".join(f"{n}={i['source']}"
                                  for n, i in trace.registers.items()) or "none"
            print(f"{label:10s} {elapsed:6.2f}s {trace.n_points:6d} "
                  f"{len(trace.blocks):7d} {str(check['match']):>6s} "
                  f"{trace.dead_time_ns:8.0f}  {registers}")
            if not check["match"]:
                failures.append(f"{label}: compiled.log mismatch {check}")
        except Exception as exc:
            print(f"{label:10s}  FAILED {type(exc).__name__}: {exc}")
            failures.append(f"{label}: {type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for failure in failures:
            print("  " + failure)
        return 1
    print(f"all {len(FOLDERS)} folders OK "
          f"({len(RENDER_OPTIONS)} render combos + point switching each)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
