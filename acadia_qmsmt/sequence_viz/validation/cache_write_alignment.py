"""Which host-side numpy SLICE writes into a CacheArray survive on the board? MEASURED.

A CacheArray's backing store is ``np.frombuffer`` over an mmap of ``/dev/mem`` (system.py, in
``Acadia._attach_resource``), so ``cache[a:b] = words`` is a memcpy straight into DEVICE memory,
where an access that is not naturally aligned faults. SIGBUS is not a Python exception: the board
process dies mid-statement, remote_stderr.log is empty, the acadia screen disappears and the
client reports "Socket peer closed connection" and then a missing data group out of finalize.
That is the whole signature of failure_052 (``length_cache[0:8]`` at byte offset 4) and of
failure_054 (XEB's ``seq_lengths`` acquiring a 45, i.e. a 180-byte write).

Rather than infer the rule from those two crashes, MEASURE it. Each point below is one deploy of
the loopback runtime with ``slice_probe_words``/``slice_probe_align`` set: the runtime performs
exactly one slice write of that length at that alignment BEFORE its first ``acadia.run()``, and
logs either side of it. The outcome is binary and unambiguous:

    survived -> the log carries "SURVIVED", and the run returns its usual traces
    faulted  -> no "SURVIVED" line, and NO DATA comes back at all

Nothing about the sequence changes between points, so nothing else can explain a difference.

MEASURED, 2026-08-24, 15 deploys::

    bytes    4    8    8   12   16   16   20   24   32   40   44  120  180  180  192
    off%16   0    0    4    0    0    8    0    0    4    0    0    0    0    8    0
             ok   ok  FAIL FAIL  ok   ok  FAIL  ok  FAIL  ok  FAIL  ok  FAIL FAIL  ok

THE RULE, fitting 15/15 points::

    a slice write survives  <=>  (bytes % 8 == 0 and byte_offset % 8 == 0)  or  bytes == 4

i.e. an EVEN number of int32 words at an EVEN word index; a single word is one 4-byte access and
is always fine, which is why element-wise writes never hit this at all. A 16-byte rule misfits 6
of these points (8@4, 12@0, 20@0, 32@4, 40@0, 120@0) and a plain 4-byte rule misfits 7, so 8 is
the granularity -- not a guess about which registers glibc's memcpy happens to use.

    python validation/cache_write_alignment.py              # the whole grid
    python validation/cache_write_alignment.py 45           # just the 45-word write
    python validation/cache_write_alignment.py 8 --align 4  # 8 words starting at byte 4 (mod 16)

Predictions are recorded per point so the report shows agreement rather than just an outcome; a
point whose prediction is None is one the rule does not settle in advance.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import paths_local
from timing_validation import build_runtime, capture_window_for

#: (words, align, prediction) -- prediction True = survives, False = faults, None = undecided.
#: Predictions are the RULE ABOVE, so a re-run re-tests the rule rather than restating it. Every
#: point here has been measured once; a disagreement means the rule has stopped being true (a
#: libc, kernel or gateware change), which is exactly what this file exists to catch.
GRID = (
    (1, 0, True),      # 4 B -- one word, one 4-byte access: the element-wise case
    (2, 0, True),      # 8 B, even offset
    (2, 4, False),     # 8 B at byte 4 -- the OFFSET alone is enough to fault
    (3, 0, False),     # 12 B -- odd word count, smallest faulting length
    (4, 0, True),      # 16 B
    (4, 8, True),      # 16 B at byte 8 -- 8-byte-aligned start is enough
    (5, 0, False),     # 20 B -- odd
    (6, 0, True),      # 24 B -- even but not a 16 B multiple: survives, so 16 is not the rule
    (8, 4, False),     # 32 B at byte 4 -- failure_052's shape
    (10, 0, True),     # 40 B -- from the killer seq_lengths, and SAFE
    (11, 0, False),    # 44 B -- odd
    (30, 0, True),     # 120 B -- from the killer seq_lengths, and SAFE
    (45, 0, False),    # 180 B -- THE write that killed XEB three deploys in a row
    (45, 8, False),    # 180 B at byte 8 -- the length faults whatever the offset
    (48, 0, True),     # 192 B
)

ITERATIONS = 20        # the probe runs before the first run(); the traces only prove it survived


def survived(folder):
    """(survived, evidence) for one deployed run folder."""
    folder = Path(folder)
    log = folder / "remote_main.log"
    text = log.read_text(errors="replace") if log.is_file() else ""
    if "SURVIVED" in text:
        return True, "logged SURVIVED"
    meta = folder / "metadata.txt"
    if meta.is_file():
        rows = [line for line in meta.read_text().splitlines()
                if line.startswith("trace_") and "num_records=0" not in line]
        if rows:
            return True, f"{len(rows)} trace group(s) with data"
    traces = [p for p in folder.glob("trace_*.bin") if p.stat().st_size]
    if traces:
        return True, f"{len(traces)} trace file(s)"
    return False, "no SURVIVED line and no data"


def probe(words, align, prediction=None):
    board_ip = paths_local.require("board_ip")
    save_root = paths_local.require("loopback_data_root")
    case = "single"
    window = capture_window_for(build_runtime(case, iterations=10))
    runtime = build_runtime(case, iterations=ITERATIONS)
    if not runtime.capture_length_override:
        runtime.capture_length_override = window
    runtime.slice_probe_words = int(words)
    runtime.slice_probe_align = int(align)
    tag = f"slice_{words}w_a{align}"
    print(f"\n>>> {words} words = {words * 4} bytes, start byte {align} (mod 16)"
          f"{'' if prediction is None else '   predicted ' + ('SURVIVES' if prediction else 'FAULTS')}",
          flush=True)
    runtime.deploy(board_ip, local_directory=f"{save_root}/{tag}/%y%m%d_%H%M%S")
    runtime.wait_for_deploy_completion()
    ok, evidence = survived(runtime.local_directory)
    verdict = "SURVIVED" if ok else "FAULTED "
    agree = "" if prediction is None else ("  (as predicted)" if ok == prediction
                                          else "  *** DISAGREES WITH THE PREDICTION ***")
    print(f"    {verdict}  {evidence}{agree}\n    {runtime.local_directory}", flush=True)
    return {"words": words, "bytes": words * 4, "align": align, "survived": ok,
            "predicted": prediction, "evidence": evidence,
            "folder": str(runtime.local_directory)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("words", nargs="?", type=int, default=None,
                        help="int32 words to write in one slice (default: the whole grid)")
    parser.add_argument("--align", type=int, default=0,
                        help="byte alignment (mod 16) of the write's first byte")
    args = parser.parse_args()

    grid = GRID if args.words is None else ((args.words, args.align, None),)
    results = []
    for words, align, prediction in grid:
        try:
            results.append(probe(words, align, prediction))
        except Exception as exc:                     # a deploy that never ran is not a result
            print(f"    DEPLOY ERROR {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            results.append({"words": words, "bytes": words * 4, "align": align,
                            "survived": None, "predicted": prediction,
                            "evidence": f"deploy error: {type(exc).__name__}"})

    print(f"\n{'=' * 72}\n  bytes  start%16   outcome    predicted")
    for r in results:
        outcome = {True: "survived", False: "FAULTED", None: "error"}[r["survived"]]
        predicted = {True: "survived", False: "FAULTED", None: "-"}[r["predicted"]]
        print(f"  {r['bytes']:>5}  {r['align']:>7}   {outcome:<9}  {predicted}")
    disagree = [r for r in results if r["predicted"] is not None
                and r["survived"] is not None and r["survived"] != r["predicted"]]
    # Only the full grid is a record. A single-point run must not overwrite the table with one row
    # -- that reads as "this is what was measured" while hiding the other fourteen points.
    if args.words is None:
        report = Path(__file__).with_name("cache_write_alignment_results.json")
        report.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {report}")
    else:
        print("\n(single point -- the results table on disk was left alone)")
    if disagree:
        print(f"{len(disagree)} point(s) disagree with the rule as stated -- the rule is wrong, "
              f"not the board")
        return 1
    survivors = [r for r in results if r["survived"]]
    print(f"{len(survivors)}/{len(results)} survived; every prediction held")
    return 0


if __name__ == "__main__":
    sys.exit(main())
