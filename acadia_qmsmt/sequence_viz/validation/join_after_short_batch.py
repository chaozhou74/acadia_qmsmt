"""A one-cycle join on a channel whose block=False pulse ends right where the join starts.

    python validation/join_after_short_batch.py            # deploy on the loopback, then score
    python validation/join_after_short_batch.py --dry-run  # both engines, no board

join_dwell_n with join_after_batch=True: ch1 plays a block=False pulse, then the one-cycle dwell join
on ch1, between two markers on ch0. With a 30 ns (6-cycle) batch pulse the pulse ends on the very
cycle the join would start (the GKP frame runtime's shape, 2026-11-18), and the two layout engines
disagree: the older layout treats the channel as idle and releases the join ONE_CYCLE_JOIN_EARLY (2)
cycles early; the emulator has the join descriptor follow the still-playing pulse seamlessly, which
the original measurement says is NOT early. The ch0 marker interval decides (285 vs 295 ns).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths_local                                              # noqa: E402
import timing_validation as tv                                  # noqa: E402

PULSE = "rb_len_030"


def build(iterations):
    runtime = tv.build_runtime("join_dwell_n", iterations=iterations)
    runtime.join_after_batch = True
    runtime.join_batch_pulse = PULSE
    return runtime


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from acadia_qmsmt import sequence_viz as sv
    for engine in ("machine", "emulator"):
        os.environ["SEQVIZ_LAYOUT"] = engine
        trace = sv.trace_runtime(build(10), envelopes=False)
        starts, _ = tv.regions_of(trace)
        print(f"model {engine}: ch0 marker starts {[round(x, 1) for x in starts['ch0']]} ns")
    if args.dry_run:
        return 0
    board_ip = paths_local.require("board_ip")
    save_root = paths_local.require("loopback_data_root")
    probe = build(10)
    runtime = build(5000)
    if not runtime.capture_length_override:
        runtime.capture_length_override = tv.capture_window_for(probe)
    runtime.deploy(board_ip, local_directory=f"{save_root}/join_dwell_n__after_{PULSE}/%y%m%d_%H%M%S")
    runtime.wait_for_deploy_completion()
    print("SAVED", runtime.local_directory)
    for engine in ("machine", "emulator"):
        os.environ["SEQVIZ_LAYOUT"] = engine
        result = tv.compare(runtime.local_directory, verbose=False)
        compared = sum(r["n_compared"] for r in result["rows"])
        print(f"board vs {engine}: worst {result['worst_error_ns']:.2f} ns over {compared} intervals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
