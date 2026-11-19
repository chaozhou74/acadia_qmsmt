"""Does the instruction-level emulator's CLOCK (sequence_viz/emulator.py) place every DMA command
where the older execution-model layout (machine.py, loopback-validated) places it?

    python validation/emulator_timing.py --selftest-folders [--points N]
    python validation/emulator_timing.py --dryrun all            # timing_validation's 83 cases
    python validation/emulator_timing.py FOLDER [FOLDER ...]

Per trace (and sampled point): the emulator runs the program on its clock; the layout lays the
same point out. Both are reduced to {channel: [(start, length, kind)]} with zero-length entries
dropped (a zero-cycle command is not emitted by acadia), and the emulator's times are shifted by
one constant so the first command on any channel coincides (the program start is not drawn by
the layout). A channel agrees when every command has the same start, length and kind.

Disagreements name the channel, the first command that differs, and both values. Which side is
right is then a board question (timing_validation --revalidate / model_vs_board).
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent.parent))

KIND = {0: "ARB_CONT", 1: "ARB", 2: "CONST_CONT", 3: "DWELL"}


def layout_commands(trace):
    out = {}
    for c in trace.commands:
        if not c.length:
            continue
        kind = "ARB" if c.kind in ("ARB", "DIRECT") else c.kind
        out.setdefault(str(c.channel), []).append((int(c.start), int(c.length), kind))
    return {ch: sorted(v) for ch, v in out.items()}


def emulator_commands(run):
    out, unplayed = {}, 0
    for ch, descriptors in run.played().items():
        for d in descriptors:
            if d.start is None:
                unplayed += 1
                continue
            out.setdefault(ch, []).append((d.start, d.length, KIND[d.ctype]))
    return {ch: sorted(v) for ch, v in out.items()}, unplayed


def compare(trace, run):
    """(n commands compared, [(channel, message)])."""
    mine, unplayed = emulator_commands(run)
    theirs = layout_commands(trace)
    if not mine or not theirs:
        return 0, [("-", f"nothing to compare (emulator {len(mine)} channels, layout {len(theirs)})")]
    shift = min(v[0][0] for v in mine.values()) - min(v[0][0] for v in theirs.values())
    problems, n = [], 0
    if unplayed:
        problems.append(("-", f"{unplayed} descriptors pushed but never triggered"))
    for ch in sorted(set(mine) | set(theirs)):
        a = [(s - shift, l, k) for s, l, k in mine.get(ch, [])]
        b = theirs.get(ch, [])
        n += min(len(a), len(b))
        if a == b:
            continue
        k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        problems.append((ch, f"emulator {len(a)} commands, layout {len(b)}; first difference at "
                             f"{k}: emulator {a[k] if k < len(a) else '-'} layout "
                             f"{b[k] if k < len(b) else '-'}"))
    return n, problems


def check(trace, label, points=None):
    picks = [None]
    if points and trace.n_points > 1:
        n = trace.n_points
        picks = sorted({0, n - 1, *range(0, n, max(1, n // points))})[:points + 1]
    agreed = failed = commands = 0
    for p in picks:
        if p is not None:
            trace.select_point(p)
        run = trace.emulate()
        n, problems = compare(trace, run)
        commands += n
        tag = f"{label} point {trace.point}"
        head = (f"  {tag}: {n} commands, {run.steps} steps, {run.cycles} cycles, "
                f"assumed {len(run.assumed)}, layout assumed {len(trace.assumed_paths)}")
        for problem in run.problems:
            problems.append(("EMULATOR", problem))
        if problems:
            failed += 1
            print(head)
            for ch, message in problems:
                print(f"    {ch}: {message}")
        else:
            agreed += 1
            print(f"{head}  -- agrees")
    return agreed, failed, commands


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--points", type=int, default=4)
    ap.add_argument("--selftest-folders", action="store_true")
    ap.add_argument("--dryrun", default=None, help="'all' or comma-separated timing_validation cases")
    args = ap.parse_args(argv)
    from acadia_qmsmt import sequence_viz as sv

    jobs = [(f, None) for f in args.folders]
    if args.selftest_folders:
        import yaml
        cfg = yaml.safe_load(open(HERE / "paths.local.yaml"))
        jobs += [(f, None) for f in cfg.get("selftest_folders", {}).values()]
    if args.dryrun:
        from loopback_timing_cases import CASES
        cases = list(CASES) if args.dryrun == "all" else args.dryrun.split(",")
        if args.dryrun == "all":
            # cases the board cannot run (timing_validation.UNSAFE_TO_DEPLOY); three_deep_nest's
            # dry-run TRACE itself never finishes, emulator or not
            from timing_validation import UNSAFE_TO_DEPLOY
            print(f"skipping unsafe cases: {sorted(set(cases) & set(UNSAFE_TO_DEPLOY))}")
            cases = [c for c in cases if c not in UNSAFE_TO_DEPLOY]
        jobs += [(None, c) for c in cases]

    totals = [0, 0, 0]
    errors = []
    for folder, case in jobs:
        started = time.time()
        try:
            if case is not None:
                import timing_validation as tv
                trace = sv.trace_runtime(tv.build_runtime(case), envelopes=False)
                result = check(trace, case)
            else:
                print(f"\n=== {folder}")
                trace = sv.trace_folder(folder, envelopes=False)
                result = check(trace, Path(folder).name, points=args.points)
        except Exception as exc:                                       # noqa: BLE001
            print(f"  {case or folder}: ERROR {type(exc).__name__}: {exc}")
            errors.append(case or folder)
            continue
        for i in range(3):
            totals[i] += result[i]
        if case is None:
            print(f"  {time.time() - started:.1f} s")
    print(f"\n{len(jobs)} traces; comparisons agreed {totals[0]}, disagreed {totals[1]}, "
          f"commands compared {totals[2]}, errors {len(errors)} {errors}")
    return 1 if totals[1] or errors else 0


if __name__ == "__main__":
    sys.exit(main())
