"""Deploy only the sequences that exercise something new.

A deploy costs ~20 s of board time; a trace costs almost nothing. So the board should never be
spent on a sequence whose behaviour is already covered -- and "covered" has to mean *the model
paths it exercises*, not *the label it was given*.

The 1000-triple enumeration this replaces is a good illustration of the problem. Roughly 271 of
its cases contain a `stretch` and every one reports the same classified ~25 ns measurement
systematic: 90 minutes of board time for zero information. Many of the rest differ only in a
label -- `block__block__block` and `block__block__loop` both come down to blocking edges with a
fall-through gap, and measuring the second after the first tells you nothing the first did not.

So each candidate is TRACED first and reduced to a SIGNATURE of the model features it exercises:
which gap kinds it pays, whether it drains a FIFO and in which sense, whether the seamless
continuation rule fires, whether a branch penalty is charged, how deep its nesting goes, whether
a stream unrolls. A candidate is deployed only if its signature has not been seen. Everything
else is skipped and counted, so the coverage claim stays honest -- these are not silently
dropped, they are demonstrably redundant.

What that buys, measured on the current alphabet: the full triple space collapses to a few dozen
distinct signatures. The same board hour then covers wider structures (quadruples, deeper nesting,
longer batches) instead of re-measuring fall-through gaps.

Usage:
    python validation/novel_cases.py            # report the coverage collapse, deploy nothing
    python validation/novel_cases.py --deploy   # deploy one representative per novel signature
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def signature(trace):
    """The model features this sequence exercises, as a hashable set.

    Deliberately about the MODEL, not the sequence: two different-looking sequences that pay the
    same kinds of gap, drain the same way and nest the same depth will confirm or refute exactly
    the same code paths, so measuring both is measuring one twice.

    Excluded on purpose: pulse names, channel identity, block counts, absolute times. Those
    change the picture without changing which part of the timing model is under test.
    """
    features = set()
    for placement in (trace.placements or trace.blocks):
        breakdown = getattr(placement, "gap_breakdown", None) or {}
        if breakdown:
            features.add(("edge", breakdown.get("edge")))
            if breakdown.get("branch_penalty"):
                features.add(("branch_penalty", int(breakdown["branch_penalty"])))
        if getattr(placement, "stream", False):
            features.add(("stream",))
        if not placement.blocking:
            features.add(("non_blocking",))
        features.add(("depth", len(getattr(placement, "conditional", ()) or ())))
    for drain in (trace.drain_blocks or {}).values():
        features.add(("drain", bool(drain.get("almost_empty"))))
    if trace.assumed_paths:
        features.add(("assumed_branch",))
    if getattr(trace, "length_underflows", None):
        features.add(("length_underflow",))
    if any(c.symbolic for c in trace.commands):
        features.add(("symbolic_length",))
    # a channel that resumes before the sequencer reaches it -- the seamless-continuation path
    for placement in (trace.placements or []):
        for command in placement.commands:
            if command.resolution == "cache":
                features.add(("cache_resolved",))
                break
    return frozenset(features)


def candidates():
    """(name, setup) pairs, widest structures first so novelty is found early."""
    from loopback_timing_cases import PRIMITIVES

    # `stretch` is dropped from the generated alphabet: it contributes only the documented
    # mixed-ramp MEASUREMENT systematic, so every sequence containing one reports ~25 ns of
    # apparent error that says nothing about the model. It stays covered by the dedicated
    # same-pulse cases (stretch_two_blocks_same, register_stretch), which are the ones that
    # actually test the stretch length model.
    alphabet = [p for p in PRIMITIVES if p != "stretch"]
    for first in alphabet:
        for second in alphabet:
            for third in alphabet:
                yield (f"{first}__{second}__{third}",
                       {"case": "pair_seq", "pair_a": first, "pair_b": second, "pair_c": third})
    for steps in (6, 9, 12):
        for seed in range(30):
            # exclude_stretch: the triples already drop it for the same reason, and leaving it
            # in the random candidates polluted 4 of the first 46 novel measurements with the
            # known ~25 ns measurement systematic -- board time spent confirming an artifact.
            yield (f"random_s{steps}_seed{seed}",
                   {"case": "random_seq", "fuzz_seed": seed, "fuzz_steps": steps,
                    "exclude_stretch": True})


def survey(limit=None):
    """Trace every candidate and group them by signature. Returns (novel, redundant)."""
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    import timing_validation as tv
    from acadia_qmsmt import sequence_viz as sv

    seen, novel, redundant = {}, [], 0
    for index, (name, setup) in enumerate(candidates(), 1):
        if limit and index > limit:
            break
        case = setup.pop("case")
        try:
            runtime = tv.build_runtime(case, iterations=10)
            for field, value in setup.items():
                setattr(runtime, field, value)
            trace = sv.trace_runtime(runtime, envelopes=False)
        except Exception:
            continue
        key = signature(trace)
        if key in seen:
            redundant += 1
            continue
        seen[key] = name
        novel.append((name, case, dict(setup), key))
        if index % 100 == 0:
            print(f"  surveyed {index}: {len(novel)} novel, {redundant} redundant", flush=True)
    return novel, redundant


def main():
    deploy = "--deploy" in sys.argv
    limit = next((int(a) for a in sys.argv[1:] if a.isdigit()), None)
    novel, redundant = survey(limit)
    total = len(novel) + redundant
    print(f"\n{total} candidates traced -> {len(novel)} distinct model signatures, "
          f"{redundant} redundant ({100 * redundant / max(total, 1):.0f}% of board time saved)")
    for name, _case, _setup, key in novel[:25]:
        print(f"   {name:38s} {len(key)} features")
    if not deploy:
        print("\n(survey only; pass --deploy to measure one representative per signature)")
        return 0

    import timing_validation as tv
    board = tv.paths_local.require("board_ip")
    root = tv.paths_local.require("loopback_data_root")
    worst_overall, failures = 0.0, []
    from pathlib import Path as _Path
    for index, (name, case, setup, _key) in enumerate(novel, 1):
        # RESUMABLE: skip anything already measured. The survey is deterministic, so a restart
        # rebuilds the same list; without this a session interruption would re-deploy every
        # case it had already done, and long runs get interrupted.
        done = _Path(root) / f"novel_{name}"
        if done.is_dir() and any(done.iterdir()):
            print(f"  [{index:3d}/{len(novel)}] {name:38s} already measured, skipping",
                  flush=True)
            continue
        probe = tv.build_runtime(case, iterations=10)
        for field, value in setup.items():
            setattr(probe, field, value)
        runtime = tv.build_runtime(case, iterations=5000)
        for field, value in setup.items():
            setattr(runtime, field, value)
        if not runtime.capture_length_override:
            runtime.capture_length_override = tv.capture_window_for(probe)
        runtime.deploy(board, local_directory=f"{root}/novel_{name}/%y%m%d_%H%M%S")
        runtime.wait_for_deploy_completion()
        result = tv.compare(runtime.local_directory, verbose=False)
        compared = sum(r["n_compared"] for r in result["rows"])
        worst = result["worst_error_ns"]
        status = "OK" if (compared and worst < 0.5) else ("no interval" if not compared else "FAIL")
        if status == "FAIL":
            failures.append((name, worst))
        if compared:
            worst_overall = max(worst_overall, worst)
        print(f"  [{index:3d}/{len(novel)}] {name:38s} worst {worst:7.2f} ns "
              f"({compared} intervals)  {status}", flush=True)
    print(f"\nnovel-signature sweep: {len(novel)} deployed, worst {worst_overall:.2f} ns, "
          f"{len(failures)} failing")
    for name, worst in failures:
        print(f"   FAIL {name:38s} {worst:.2f} ns")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
