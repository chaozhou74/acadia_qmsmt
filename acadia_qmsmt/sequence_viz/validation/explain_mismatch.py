"""Explain a `compiled.log` mismatch: what the re-trace has that the archive does not, and why.

`compiled_log.compare` reports mismatching multisets of ``(channel, kind, length)``. That says a
disagreement exists but not what kind of disagreement, and the distinction decides who is at
fault:

* commands in the ARCHIVE but not the re-trace -- the tracer is MISSING something the board ran.
  Usually the serious direction.
* commands in the RE-TRACE but not the archive -- the tracer invented or mis-sized something.
* the same command at a slightly different LENGTH -- a length model problem, and the pairing of
  near-misses shows by how much.

The third is easy to mistake for the first two, because a length that is off by one appears as
one "missing" and one "extra" entry. Pairing them up is most of the work, so it is done here.

Usage: ``python validation/explain_mismatch.py [stress_archive.json]``
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_JSON = Path(__file__).parent / "stress_archive.json"


def explain(folder):
    from acadia_qmsmt import sequence_viz as sv
    from acadia_qmsmt.sequence_viz import compiled_log

    trace = sv.trace_folder(folder)
    result = compiled_log.compare(trace, folder)
    if result["match"]:
        return [f"{folder}: now MATCHES (fixed since the sweep)"]

    only_archive = Counter(
        {eval(k) if isinstance(k, str) else k: v
         for k, v in result["only_in_archive"].items()}) \
        if isinstance(result["only_in_archive"], dict) else Counter()
    only_retrace = Counter(
        {eval(k) if isinstance(k, str) else k: v
         for k, v in result["only_in_retrace"].items()}) \
        if isinstance(result["only_in_retrace"], dict) else Counter()

    lines = [f"{Path(folder).parent.parent.name} / {Path(folder).name}  "
             f"[{trace.runtime_class}]",
             f"   blocks {result['blocks']} vs triggers {result['triggers']}, "
             f"symbolic {result['symbolic_retrace']}, zero-length {result['zero_length_retrace']}"]

    # pair up entries that differ only in length -- an off-by-N shows as one missing + one extra
    paired = []
    for key, count in list(only_archive.items()):
        channel, kind, length = key
        for other in list(only_retrace):
            if other[0] == channel and other[1] == kind and other[2] != length:
                paired.append((channel, kind, length, other[2],
                               min(count, only_retrace[other])))
                break
    if paired:
        lines.append("   LENGTH differences (archive -> retrace):")
        for channel, kind, archive_len, retrace_len, count in paired[:10]:
            lines.append(f"      {channel} {kind}: {archive_len} -> {retrace_len} cycles "
                         f"({retrace_len - archive_len:+d}) x{count}")
    missing = [k for k in only_archive if not any(p[0] == k[0] and p[1] == k[1]
                                                  for p in paired)]
    extra = [k for k in only_retrace if not any(p[0] == k[0] and p[1] == k[1] for p in paired)]
    if missing:
        lines.append(f"   MISSING from the re-trace (the board ran these): {missing[:6]}")
    if extra:
        lines.append(f"   ONLY in the re-trace (invented or mis-sized): {extra[:6]}")
    return lines


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    data = json.loads(path.read_text())
    folders = [p["folder"] for p in data["problems"] if p["kind"] == "mismatch"]
    if not folders:
        print("no mismatches recorded in", path)
        return 0
    print(f"{len(folders)} mismatching folder(s)\n")
    for folder in folders:
        try:
            for line in explain(folder):
                print(line)
        except Exception as exc:
            print(f"{folder}: could not re-explain ({type(exc).__name__}: {exc})")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
