#!/usr/bin/env python3
"""
Sweep the tracer over ARCHIVED runs of every runtime class in the data root.

The loopback cases (loopback_timing_cases.py) verify TIMING against a physical recording, but
they only cover the primitives someone wrote a case for. This covers the other axis: for every
real runtime class that has ever been deployed, re-trace an archived run and check the result
against that run's own ``compiled.log`` -- the record of what the board's compiler actually
emitted. No hardware, no board time.

It answers "does the visualizer reproduce what really ran, for every runtime we have", which is
the only check that scales to ~90 runtime classes.

    $ACADIA_ENV/bin/python validation/archive_sweep.py --data-root ~/data --max-per-class 2

Discovery is deliberately cheap on an NFS archive: find every ``compiled.log``, collapse to one
run per experiment path, then read each run's 767-byte ``run.py`` for its
``from runtime import <Class>`` line rather than the 110 kB inlined ``runtime.py``. The
class -> folders map is cached (``--cache``) so re-runs skip the walk.
"""
import argparse
import json
import logging
import re
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _name in ("acadia", "qmsmt_runtime_loader", "sequence_viz", "acadia_qmsmt"):
    logging.getLogger(_name).setLevel(logging.ERROR)

CLASS_RE = re.compile(r"from runtime import (\w+)")


def discover(data_root, cache=None, workers=48):
    """{class name: [folder, ...]} for every traceable archived run under ``data_root``."""
    if cache and Path(cache).exists():
        raw = json.loads(Path(cache).read_text())
        return {k: [Path(x) for x in v] for k, v in raw.items()}

    root = Path(data_root).expanduser()
    folders = [p.parent for p in root.rglob("compiled.log")]
    # all runs under one experiment path share a class; one representative each is enough
    per_experiment = defaultdict(list)
    for f in folders:
        per_experiment[str(f.parent.parent)].append(f)
    reps = [sorted(v)[-1] for v in per_experiment.values()]        # newest per experiment

    def classify(folder):
        try:
            if not (folder / "kwargs.json").exists() or not (folder / "runtime.py").exists():
                return None
            match = CLASS_RE.search((folder / "run.py").read_text(errors="replace"))
            return match.group(1) if match else None
        except OSError:
            return None

    by_class = defaultdict(list)
    with ThreadPoolExecutor(workers) as pool:
        for folder, name in zip(reps, pool.map(classify, reps)):
            if name:
                by_class[name].append(folder)
    if cache:
        Path(cache).write_text(json.dumps(
            {k: [str(x) for x in v] for k, v in by_class.items()}, indent=1))
    return by_class


def check(folder, row, render=True, points=(0, 1)):
    """Trace one folder into ``row`` (mutated, so ``row["stage"]`` survives a raise)."""
    from acadia_qmsmt.sequence_viz import compare_with_compiled_log, draw, trace_folder

    row["stage"] = "trace"
    started = time.time()
    trace = trace_folder(folder, point=0)
    row.update(trace_s=round(time.time() - started, 2), blocks=len(trace.blocks),
               placements=len(trace.placements), n_points=trace.n_points,
               assumed_paths=sorted(trace.assumed_paths),
               unsupported_paths=sorted(getattr(trace, "unsupported_paths", ())))

    row["stage"] = "compare"
    result = compare_with_compiled_log(trace, folder)
    row["match"] = result["match"]
    row["cmp"] = {k: result[k] for k in
                  ("blocks", "triggers", "commands_retrace", "commands_archive",
                   "symbolic_retrace", "zero_length_retrace")}
    if not result["match"]:
        row["only_in_archive"] = {str(k): v for k, v in result["only_in_archive"].items()}
        row["only_in_retrace"] = {str(k): v for k, v in result["only_in_retrace"].items()}

    if render:
        # every render option and a deep zoom, because the GUI exposes all of them
        row["stage"] = "render"
        for kwargs in ({}, {"color_by": "name"}, {"envelope_mode": "iq"},
                       {"envelope_scale": "absolute", "envelope_source": "config"},
                       {"show_gaps": False, "show_branches": False, "legend": False}):
            figure, axes = plt.subplots(figsize=(14, 6))
            try:
                draw(axes, trace, **kwargs)
            finally:
                plt.close(figure)

        row["stage"] = "zoom"
        span = max(trace.length_ns, 1.0)
        for window in ((0.0, span * 0.02), (span * 0.5, span * 0.52), (0.0, span)):
            figure, axes = plt.subplots(figsize=(14, 6))
            try:
                draw(axes, trace, xlim_ns=window)
            finally:
                plt.close(figure)

        row["stage"] = "points"
        for point in points:
            if point < trace.n_points:
                trace.select_point(point)

    row["stage"] = "ok"
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="~/data")
    parser.add_argument("--cache", default=str(Path(__file__).with_name("archive_classes.json")))
    parser.add_argument("--max-per-class", type=int, default=2)
    parser.add_argument("--only", default=None, help="substring filter on the class name")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    by_class = discover(args.data_root, cache=args.cache)
    print(f"{sum(len(v) for v in by_class.values())} archived runs across "
          f"{len(by_class)} runtime classes; up to {args.max_per_class} per class\n")
    print(f"{'runtime class':<44} {'result':<9} {'blk':>4} {'plc':>4} {'pts':>5}  detail")
    print("-" * 118)

    results, failures = [], []
    for klass in sorted(by_class):
        if args.only and args.only.lower() not in klass.lower():
            continue
        for folder in sorted(by_class[klass], reverse=True)[:args.max_per_class]:
            row = {"class": klass, "folder": str(folder)}
            try:
                check(folder, row, render=not args.no_render)
                verdict = "OK" if row["match"] else "MISMATCH"
                detail = "" if row["match"] else str(row.get("cmp"))
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc()[-1500:]
                verdict, detail = "FAIL", f"[{row['stage']}] {row['error']}"[:190]
                failures.append(row)
            results.append(row)
            print(f"{klass[:43]:<44} {verdict:<9} {row.get('blocks', '-'):>4} "
                  f"{row.get('placements', '-'):>4} {row.get('n_points', '-'):>5}  {detail}")
            sys.stdout.flush()

    matched = sum(1 for r in results if r.get("match"))
    mismatched = sum(1 for r in results if r.get("stage") == "ok" and not r.get("match"))
    print("-" * 118)
    print(f"{len(results)} runs: {matched} match compiled.log, {mismatched} mismatch, "
          f"{len(failures)} failed")
    if failures:
        grouped = defaultdict(list)
        for f in failures:
            grouped[(f["stage"], f["error"][:110])].append(f["class"])
        print("\nfailure classes:")
        for (stage, message), classes in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
            print(f"  [{stage}] {message}\n      x{len(classes)}: {sorted(set(classes))}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
        print(f"\nfull results -> {args.out}")
    return 1 if (failures or mismatched) else 0


if __name__ == "__main__":
    sys.exit(main())
