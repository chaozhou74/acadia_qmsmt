"""Does the instruction-level emulator (sequence_viz/emulator.py) play the same DMA words as the
tracer's layout, point by point and channel by channel?

    python validation/emulator_vs_trace.py FOLDER [FOLDER ...] [--points N]
    python validation/emulator_vs_trace.py --selftest-folders      # paths.local.yaml's list

Per folder: the re-traced program against the archived assembled.log, instruction by instruction
(the emulator decodes the assembled bytes, so this says it runs what the board ran). Then per
sampled point:
  * emulate the compiled program on that point's cache;
  * take the tracer's laid-out commands (trace.commands, loops unrolled) per channel;
  * compare the two command lists per channel, each command as (kind, address, length).

The emulator executes the program; the tracer infers it from idioms. Where the tracer is right
the lists agree. A disagreement names the channel, the first index that differs and both values,
so it can be settled from the program itself (prog dump: --dump).

`--no-layout` skips the tracer's layout (for a program it cannot lay out, e.g. the first
MultiDualRailRB form) and prints the emulator's per-channel summary only.
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))

from acadia_qmsmt import sequence_viz as sv                            # noqa: E402
from acadia_qmsmt.sequence_viz import emulator as E                    # noqa: E402
from acadia_qmsmt.sequence_viz.tracing import SequenceTrace           # noqa: E402

EMU_KIND = {0: "ARB_CONT", 1: "ARB", 2: "CONST_CONT", 3: "DWELL"}


def trace_commands(trace):
    """{channel: [(kind, address, length)]} of the tracer's laid-out commands, in time order."""
    out = {}
    for c in sorted(trace.commands, key=lambda c: (c.channel, c.start)):
        kind = "ARB" if c.kind in ("ARB", "DIRECT") else c.kind
        address = c.address if kind == "ARB" else None
        out.setdefault(str(c.channel), []).append((kind, address, int(c.length)))
    return out


def emulator_commands(run):
    """{channel: [(kind, address, length)]} from the emulator's executed DMA words."""
    out = {}
    for channel, words in run.dma_words().items():
        out[channel] = [E.decode_word(ctype, word) for ctype, word in words]
    return out


def compare(trace, run):
    """[(channel, message)] for every channel whose lists differ; [] when all agree."""
    mine, theirs = emulator_commands(run), trace_commands(trace)
    problems = []
    for channel in sorted(set(mine) | set(theirs)):
        a, b = mine.get(channel, []), theirs.get(channel, [])
        if a == b:
            continue
        k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        problems.append((channel, f"emulator {len(a)} commands, tracer {len(b)}; first difference "
                                  f"at {k}: emulator {a[k] if k < len(a) else '-'} tracer "
                                  f"{b[k] if k < len(b) else '-'}"))
    return problems


def archive_check(trace, folder):
    """The re-traced program against the folder's archived assembled.log (what the board loaded),
    instruction by instruction, every field but the comment. Returns (compared, differing)."""
    path = Path(folder) / "assembled.log"
    if not path.exists() or trace.program is None:
        print(f"  archive: no assembled.log or no program ({trace.program_error}) -- not compared")
        return 0, 0
    archived = E.load_assembled_log(path)
    strip = lambda op: {k: v for k, v in vars(op).items() if k != "comment"}   # noqa: E731
    differing = sum(1 for a, b in zip(archived, trace.program) if strip(a) != strip(b))
    differing += abs(len(archived) - len(trace.program))
    print(f"  archive: {len(archived)} archived instructions, {len(trace.program)} re-traced, "
          f"{differing} differ")
    return len(archived), differing


def check_folder(folder, points, layout=True):
    started = time.time()
    if not layout:
        original = SequenceTrace.relayout
        SequenceTrace.relayout = lambda self: None
    try:
        trace = sv.trace_folder(folder, envelopes=False)
    finally:
        if not layout:
            SequenceTrace.relayout = original
    _n_archive, archive_differ = archive_check(trace, folder)
    n = trace.n_points
    picks = sorted({0, n - 1, *range(0, n, max(1, n // max(points, 1)))})[:points + 1]
    compared, failed = 0, 0
    for p in picks:
        if layout:
            trace.select_point(p)
        else:
            trace.point_cache = trace.snapshots[p]["cache"]
        run = trace.emulate()
        words = sum(len(v) for v in run.dma_words().values())
        header = (f"  point {p:4d}: {run.steps} steps, {words} DMA words, halted {run.halted}, "
                  f"assumed {len(run.assumed)} branches, waits "
                  f"{sum(1 for e in run.events if e.kind == 'wait')}")
        for problem in run.problems:
            print(f"{header}\n    EMULATOR PROBLEM {problem}")
        if not layout:
            print(header)
            for channel, cmds in sorted(emulator_commands(run).items()):
                print(f"    {channel}: {len(cmds)} commands")
            continue
        diffs = compare(trace, run)
        compared += 1
        if diffs:
            failed += 1
            print(header)
            for channel, message in diffs:
                print(f"    {channel}: {message}")
        else:
            print(f"{header}  -- agrees on {len(emulator_commands(run))} channels")
    failed += 1 if archive_differ else 0
    print(f"  {folder}: {n} points, compared {compared}, disagree {failed}, "
          f"{time.time() - started:.1f} s")
    return compared, failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--points", type=int, default=4)
    ap.add_argument("--selftest-folders", action="store_true")
    ap.add_argument("--no-layout", action="store_true")
    args = ap.parse_args(argv)
    folders = list(args.folders)
    if args.selftest_folders:
        import yaml
        cfg = yaml.safe_load(open(HERE / "paths.local.yaml"))
        folders += list(cfg.get("selftest_folders", {}).values())
    total_compared = total_failed = 0
    for folder in folders:
        print(f"\n=== {folder}")
        try:
            c, f = check_folder(folder, args.points, layout=not args.no_layout)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  ERROR {type(exc).__name__}: {exc}")
            total_failed += 1
            continue
        total_compared += c
        total_failed += f
    print(f"\n{len(folders)} folders; point comparisons {total_compared}; disagreements {total_failed}")
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
