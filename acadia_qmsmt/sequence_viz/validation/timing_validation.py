"""
Validate sequence_viz's timing model against the 4-channel loopback.

Each DAC is cabled to its own ADC, so a capture is a direct recording of that DAC's
output. That makes the loopback a physical oracle for the parts of the visualizer that
are *derived* rather than measured -- above all the inter-block dead time.

Method
------
Compare INTERVALS BETWEEN EDGES ON ONE CHANNEL, never absolute arrival times. The
constant DAC->cable->ADC latency (~283 ns) is unknown and unmeasurable, but it cancels
in a within-channel interval, as does any residual capture-trigger skew. Edges are taken
at 50% of the pulse plateau with linear interpolation; averaging 5000 iterations gets the
edge to ~0.05 ns even though the samples are 5 ns apart.

Usage
-----
    $ACADIA_ENV/bin/python timing_validation.py --case two_blocks
    $ACADIA_ENV/bin/python timing_validation.py --all
    $ACADIA_ENV/bin/python timing_validation.py --analyse /path/to/test_loopback/two_blocks/<run>

`--analyse` re-reads an existing folder without deploying.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # loopback_timing_cases, paths_local

import paths_local

# Board IP and data root are machine/board-specific -- read from the gitignored
# paths.local.yaml (or SEQVIZ_* env vars), never committed. May be None when only
# an off-hardware --dryrun is being run; the deploy/analyse paths require() them.
_LOCAL = paths_local.load()
BOARD_IP = _LOCAL.get("board_ip")
SAVE_ROOT = _LOCAL.get("loopback_data_root")
YAML_PATH = _LOCAL.get("yaml_path") or str(Path(__file__).parent / "validation_board_config.yaml")
REPORT = Path(__file__).parent / "timing_validation_results.json"

CHANNEL_OF = {"ch0": "DAC0", "ch1": "DAC2", "ch2": "DAC4", "ch3": "DAC6"}

#: A channel is treated as IDLE (no pulses at all) unless its peak stands this many baseline sigmas
#: above the baseline. Measured on this station: a driven loopback channel peaks at 4.1e3-5.5e4
#: counts, an undriven one at 0.4-0.7, so the true separation is ~5 orders of magnitude and any
#: value in the tens is safe. Guards pulse_regions_ns, whose other thresholds are all relative to
#: the channel's own maximum and therefore meaningless on a channel carrying only noise.
SILENT_CHANNEL_SIGMAS = 30.0

#: A region must peak at least this fraction of its channel's maximum to count as a pulse.
#: Was 0.05, which admitted CROSSTALK: a pulse on one DAC shows up on a neighbouring cabled
#: channel at ~7% of that channel's own maximum, and in feedback_reset such a blip stood in for a
#: conditional reset pulse that never fired -- turning a missing pulse into a spurious 0.17 ns
#: "OK". Chosen from the data, not guessed: over every archived loopback run the region peaks fall
#: into two disjoint groups, 10 regions below 15% (all crosstalk) and 592 at 25% or above (all real
#: pulses), with NOTHING between 15% and 25%. 0.20 sits in that empty band.
REGION_FLOOR_FRACTION = 0.20

#: A channel is IDLE if its peak is below this fraction of the LOUDEST channel's peak in the same
#: run. The within-channel sigma test above is not enough on its own: an idle channel's noise
#: distribution can be narrow in its lower band while still throwing outliers 30 sigma up, which
#: left 6 phantom "pulses" on ch3 (peak 0.705) next to a driven ch2 at 5.5e4. Across channels the
#: separation is unambiguous -- driven peaks measured 4.1e3-5.5e4, idle ones 0.37-0.71, i.e. 4-5
#: orders of magnitude -- while the spread BETWEEN driven channels is only ~13x (different
#: attenuation and NCO per channel), so 1e-3 sits far from both.
IDLE_CHANNEL_FRACTION = 1e-3

#: Seconds of ADC window to request, overriding the yaml's 1.2e-6. The yaml window records only
#: 1070 ns, but these sequences run 2400-2610 ns, so more than half of every case -- including the
#: post-batch discriminator pulses the batch cases exist to measure -- was never captured. Must
#: cover the longest sequence plus the ~283 ns DAC->cable->ADC latency plus one pulse length.
CAPTURE_WINDOW_S = 4.0e-6

# Cases whose measured error is a known MEASUREMENT systematic, not a model error.
#: Cases that must NOT be deployed by default. ``test_false_nospec`` HANGS the sequencer
#: (KI_004: with speculation=False the body is relocated to the program tail, but the STACK->PC
#: return lands on the guard's fall-through path, so the skip path pops an empty return stack and
#: jumps to garbage). The board recovers on the next deploy, but the run produces no data and
#: costs a power-cycle of confidence. --all used to include it, which contradicted the
#: KNOWN_SYSTEMATIC note telling you not to re-deploy it. Opt in with --include-unsafe.
UNSAFE_TO_DEPLOY = {
    "test_false_nospec": "KI_004: hangs the sequencer (skipped arm returns to garbage)",
    "three_deep_nest":
        "never returns from the board -- repeated 'Timeout occurred waiting for line', i.e. a "
        "counter loop that does not terminate. It configures all three DSP counters ONCE up "
        "front and then only reloads them on re-entry; nested_cool_n (which works) re-issues "
        "configure() as well as load() each time. three_deep_nest_reconfig is the same sequence "
        "with that one difference and is the version to deploy. No qudit runtime reaches three "
        "counter levels -- cool_modes uses two counters plus a test -- so nothing shipped is "
        "affected.",
}

def systematic_note(name, trace=None):
    """The note explaining a residual error, or None if it needs explaining.

    Decided from what the sequence CONTAINS, not from what the folder is called. Name matching was
    tried first and is the wrong shape for this: it only ever covers the cases someone already
    thought to name, so a new case with the same physics reads as an unexplained failure while an
    unrelated case that happens to match the pattern gets excused. Both of those happened -- a
    generated sequence needed a second rule the moment the naming changed from `random_seq...` to
    `novel_random...`, which is a rename, not a physical difference.

    The two systematics have crisp physical signatures, so they are detected as such:

    * a MIXED-RAMP pair -- the sequence measures a stretchable pulse against a marker with a
      different ramp, so the 50%-of-power crossings sit at different points on the two edges;
    * a TWO-DESCRIPTOR almost_empty drain -- the "one word left" level is reached as the batch
      starts playing, so the poll never blocks and the gap terms (calibrated on drains that do
      block) over-count.

    Hand-written cases keep their explicit notes in KNOWN_SYSTEMATIC, which are about a fixture's
    construction rather than about physics.
    """
    note = KNOWN_SYSTEMATIC.get(name)
    if note:
        return note
    if trace is None:
        return None

    stretched = {c.pulse for c in trace.commands if c.pulse and "stretch" in str(c.pulse)}
    plain = {c.pulse for c in trace.commands if c.pulse and "stretch" not in str(c.pulse)}
    if stretched and plain:
        return (f"mixed ramp shapes: this sequence measures {sorted(stretched)[0]} against "
                f"{len(plain)} pulse(s) of a different shape, so the 50%-of-power crossing sits "
                f"at a different point on each edge -- ~25 ns of apparent error from the "
                f"MEASUREMENT, not the model. The same-pulse cases agree to 0.05 ns.")

    for nth, drain in (trace.drain_blocks or {}).items():
        if not drain.get("almost_empty"):
            continue
        # how many descriptors the drained block pushed: with two, `almost empty` is true as soon
        # as the first is pulled, so the poll never blocks
        block = next((b for i, b in enumerate(trace.blocks) if i == nth), None)
        pushed = len([c for c in (block.commands if block else []) if c.pulse])
        if pushed and pushed <= 2:
            return (f"two-descriptor almost_empty drain (block {nth} pushed {pushed}): the "
                    f"'one word left' level is reached as the batch starts playing, so the poll "
                    f"never blocks and the gap terms over-count by 1-2 cycles. Confined to this "
                    f"case by descriptor-count scans; no runtime batches two behind almost_empty.")
    return None


KNOWN_SYSTEMATIC = {
    "stretch_two_blocks":
        "mixed ramp shapes (50 ns stretch vs 10 ns test pulse): a 50%-of-power crossing "
        "sits at a different point on each ramp, costing ~20-25 ns of apparent error. "
        "stretch_two_blocks_same uses identical pulses and agrees to 0.05 ns.",
    "phase_pair":
        "edge-estimation limited, not a model error: the stage-2 pulses have 50 ns ramps, "
        "5x slower than the 10 ns-ramp cases, so the same amplitude noise buys 5x the "
        "timing jitter. Errors are MIXED SIGN and sub-cycle (-1.14/+0.70/+0.82/+0.26 ns); "
        "a real model error shows one sign and one magnitude on all four channels, as the "
        "+1 cycle boundary offset did (+4.96..+5.00 everywhere).",
    "test_true_nospec":
        "UNSUPPORTED, see KI_004: test(speculation=False) places the body out of line so "
        "address order stops being execution order. The taken arm mistimes by ~25 ns and "
        "the skipped arm (test_false_nospec) hangs the sequencer. The default "
        "speculation=True is correct in both arms (0.04 / 0.10 ns) -- use that.",
    "test_false_nospec":
        "UNSUPPORTED, see KI_004: this variant HANGS the board. Do not re-deploy it.",
    "detune_pair":
        "same slow-ramp edge limit as phase_pair, plus SSB imbalance ripple that differs "
        "between the 10 MHz and 25 MHz pulses and perturbs the plateau peak the half level "
        "is taken from. Timing conclusions rest on the fast-ramp cases.",
    "feedback_reset":
        "the conditional-reset arm is DATA-DEPENDENT: whether the reset pulse fires depends on the "
        "live CMACC quadrant of the loopback signal, which no static trace can know (the tracer "
        "correctly reports the block in assumed_paths). Only ch0's unconditional readout pulses are "
        "a timing result; a count mismatch on the conditional channel is expected, not an error.",
    "rb_stream":
        "the final '8 basic gates' block plays lo/mid/hi amplitudes back-to-back, merging into "
        "one region; the 50%-of-region-peak rising edge then latches onto the first HIGH gate, "
        "reading the block ~70 ns (2 gates) late. Measurement systematic, not a model error -- "
        "the region START matches the tracer to ~5 ns. The amplitude variation is for gate "
        "IDENTITY; for timing of that block, set rb_final_gate to one shape (rb_stream_uniform / "
        "rb_final_gate='rb_gate_hi' validates to ~0-5 ns). The stream unroll itself is exact.",
}


# ---------------------------------------------------------------- measurement

def pulse_regions_ns(runtime, label, threshold_sigmas=10.0):
    """(start, stop, peak) of every above-threshold region, in ns.

    Width matters for the seamless-join test: two pulses butted together with no gap
    merge into ONE region of twice the width, which is the signature of a correct
    zero-gap prediction rather than a missing pulse.
    """
    t = np.asarray(runtime.t_data) * 1e9
    y = np.asarray(runtime.avg_trace_pwr[label])
    # Baseline from a LOW percentile, not the first 30 samples: a capture_start_delay can
    # slide a pulse into those early samples (a first-samples baseline would then inflate the
    # threshold and hide every region), while a median baseline sits ON the pulse when a
    # pulse fills most of the trace (the stretch cases). The 20th percentile tracks the quiet
    # level as long as some of the trace is quiet; the noise comes from the lower band so a
    # plateau cannot inflate it.
    level = float(np.percentile(y, 20))
    band = y[y <= np.percentile(y, 40)]
    noise = float(band.std()) if band.size > 1 else float(y.std())
    if noise == 0.0:                                        # degenerate flat trace
        noise = 1.0
    # SILENT-CHANNEL GUARD. Every threshold below is RELATIVE to this channel, so on a channel
    # the case never drives, "5% of the maximum" is 5% of the noise floor and noise sails through:
    # batch_uneven reported 7 "pulses" on ch1 whose peak was 0.37 counts against 4120 on the driven
    # ch0 -- five orders of magnitude down -- which then looked like the tracer had put pulses on
    # the wrong channels. A real loopback pulse sits thousands of sigma above baseline, so requiring
    # a large dynamic range separates "driven" from "idle" without any absolute count threshold
    # (which would depend on attenuation and ADC gain).
    if y.max() < level + SILENT_CHANNEL_SIGMAS * noise:
        return []
    # only the channels that were actually captured (trace_channels); a case may trace a subset
    loudest = max((float(np.asarray(trace).max())
                   for trace in runtime.avg_trace_pwr.values()), default=0.0)
    if loudest > 0 and y.max() < IDLE_CHANNEL_FRACTION * loudest:
        return []
    above = np.where(y > level + threshold_sigmas * noise)[0]
    if not len(above):
        return []
    regions, start = [], above[0]
    for i in range(1, len(above)):
        if above[i] - above[i - 1] > 1:
            regions.append((start, above[i - 1]))
            start = above[i]
    regions.append((start, above[-1]))

    # A baseline-sigma threshold alone lets single-sample noise through: on a quiet
    # channel it flagged a 1-sample "pulse" at peak 0.77 against a real peak of 4.8e4.
    # Require a real pulse to be at least 2 samples wide and 5% of the channel maximum.
    floor = REGION_FLOOR_FRACTION * y.max()
    return [(float(t[a]), float(t[b]), float(y[a:b + 1].max()))
            for a, b in regions if b > a and y[a:b + 1].max() >= floor]


def pulse_edges_ns(runtime, label, threshold_sigmas=10.0):
    """Rising-edge times of every pulse on one channel, in ns, sub-sample resolved."""
    t = np.asarray(runtime.t_data) * 1e9
    y = np.asarray(runtime.avg_trace_pwr[label])
    edges = []
    for start_ns, stop_ns, peak in pulse_regions_ns(runtime, label, threshold_sigmas):
        # Take the half level from THIS region's own peak. A fixed-width window does not
        # reach the plateau of a long pulse, which biases the half level -- and therefore
        # the crossing -- by pulse length, costing ~1-2 ns on 300 ns pulses.
        a = int(np.searchsorted(t, start_ns))
        b = int(np.searchsorted(t, stop_ns))
        if b <= a:
            continue
        half = peak / 2.0
        rising = np.where(y[max(a - 4, 1):b + 1] > half)[0]
        if not len(rising):
            continue
        j = max(a - 4, 1) + int(rising[0])
        if j < 1:
            continue
        edges.append(float(np.interp(half, [y[j - 1], y[j]], [t[j - 1], t[j]])))
    return edges


def measure(folder):
    """{label: [edge_ns, ...]} for a finished loopback folder."""
    from acadia_qmsmt.utils.saved_runtime_loader import load_runtime_from_data_dir

    # str(): --revalidate/--all pass Path objects, and some DataManager.load
    # builds reject a Path outright ("Load location must be a string"). The loader
    # coerces too, but don't depend on that -- folder.py does the same for its own
    # load_runtime_from_data_dir call.
    runtime = load_runtime_from_data_dir(str(folder))
    runtime.process_current_data()
    # A case that traced only some channels (trace_channels, e.g. the measure_* cases needing a
    # cmacc for the resonator) has no data for the others -- report them empty rather than raising.
    return {label: (pulse_edges_ns(runtime, label)
                    if label in getattr(runtime, "avg_trace_pwr", {}) else [])
            for label in ("ch0", "ch1", "ch2", "ch3")}


# ---------------------------------------------------------------- prediction

def measure_spans(folder):
    """{label: [region width ns, ...]} -- the other measurable quantity.

    A case with ONE region per channel has no interval, so the interval metric says nothing
    (7 of the 45 cases were reporting a vacuous "0.0 ns"). Its claim is still testable: two
    pulses butted together in one block MERGE into a single region of twice the width, so
    predicting "no gap" predicts that width. Width is a within-channel measurement, so the
    cable latency cancels exactly as it does for an interval.
    """
    from acadia_qmsmt.utils.saved_runtime_loader import load_runtime_from_data_dir

    runtime = load_runtime_from_data_dir(str(folder))
    runtime.process_current_data()
    out = {}
    for label in ("ch0", "ch1", "ch2", "ch3"):
        if label not in getattr(runtime, "avg_trace_pwr", {}):
            out[label] = []
            continue
        out[label] = [stop - start
                      for start, stop, _peak in pulse_regions_ns(runtime, label)]
    return out


def regions_of(trace):
    """{label: ([region start ns], [region width ns])} for the trace's CURRENT layout."""
    starts, spans = {}, {}
    for label, channel in CHANNEL_OF.items():
        pulses = sorted((c for c in trace.commands
                         if c.pulse and c.channel == channel), key=lambda c: c.start)
        merged = []
        for command in pulses:
            if merged and command.start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], command.stop)
            else:
                merged.append([command.start, command.stop])
        starts[label] = [r[0] * trace.ns_per_cycle for r in merged]
        spans[label] = [(r[1] - r[0]) * trace.ns_per_cycle for r in merged]
    return starts, spans


def resolve_assumed_arm(trace, measured):
    """Decide an undecidable ``test`` arm from the MEASURED region counts, and re-lay out.

    A branch on a live measurement result (``test(reg == CMACC_QUADRANT_1)`` -- active reset, and
    the same shape ``_cool_single_qubit`` uses) cannot be resolved off-hardware: it depends on what
    the qubit did. The trace therefore guesses "taken" and records the block in ``assumed_paths``.

    But the capture says which arm actually ran: the conditional pulse is either in the trace or it
    is not, so the number of REGIONS on each channel distinguishes the arms. Selecting the arm from
    region COUNTS and then scoring INTERVALS is not circular -- they are different observables. The
    count fixes *which* timeline to check; the model still has to predict *when* every edge lands,
    which is what gets scored.

    Returns the chosen mapping (block -> taken) or None when neither arm reproduces the counts.
    """
    counts = {label: len(v) for label, v in measured.items()}
    for taken in (False, True):
        trace.path_choices = {block: taken for block in trace.assumed_paths}
        trace.relayout()
        starts, _ = regions_of(trace)
        if all(len(starts[label]) == counts[label] for label in counts):
            return {block: taken for block in trace.assumed_paths}
    trace.path_choices = {}
    trace.relayout()
    return None


def predict(folder):
    """{label: [region start ns, ...]} from sequence_viz's trace of the same folder.

    Pulses that are contiguous in the prediction are MERGED into one region, because that
    is what the measurement sees: two pulses butted together with no gap cross the threshold
    once and come back as a single wide region. Without merging, a case like
    ``two_same_block`` compares nothing (fewer measured regions than predicted pulses) and
    ``loop_2_double`` compares the wrong pairs.
    """
    from acadia_qmsmt import sequence_viz as sv

    trace = sv.trace_folder(folder)
    starts, spans = {}, {}
    for label, channel in CHANNEL_OF.items():
        pulses = sorted((c for c in trace.commands
                         if c.pulse and c.channel == channel),
                        key=lambda c: c.start)
        merged = []
        for command in pulses:
            if merged and command.start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], command.stop)
            else:
                merged.append([command.start, command.stop])
        starts[label] = [region[0] * trace.ns_per_cycle for region in merged]
        spans[label] = [((region[1] - region[0]) * trace.ns_per_cycle) for region in merged]

    # Channels whose predicted pulses come from a block the tracer could NOT decide.
    # `test(reg == CMACC_QUADRANT_1)` on a live measurement result is genuinely undecidable
    # off-hardware -- the branch depends on what the qubit did -- so the tracer assumes the body
    # runs and records the block in `assumed_paths`. Comparing such a channel scores the ASSUMPTION,
    # not the model: feedback_reset's conditional-reset channel reported 165.05 ns purely because
    # the shot did not take the branch the trace had to guess. An assumed path must never pass as a
    # verified one (the same principle as capture_points in dry_run), and equally must never fail as
    # one, so these channels are reported and excluded from the score rather than silently compared.
    # The rule is TEMPORAL, not per-channel. A block the trace guessed wrong about does not
    # merely misplace its own channel: if the branch is not taken, every later block slides
    # earlier by the skipped duration, on EVERY channel. feedback_reset shows this exactly --
    # the undecidable conditional sits on DAC2, yet the 165.05 ns error appeared on DAC0, whose
    # second readout pulse simply started earlier than the guess implied. So everything at or
    # after the earliest assumed block is unscorable.
    assumed_start = min((pl.start for pl in (trace.placements or [])
                         if pl.index in trace.assumed_paths), default=None)
    assumed_labels = set()
    if assumed_start is not None:
        cutoff = assumed_start * trace.ns_per_cycle
        assumed_labels = {label for label in starts
                          if any(start >= cutoff for start in starts[label])}
    return starts, trace, spans, assumed_labels


# ---------------------------------------------------------------- comparison

def compare(folder, verbose=True):
    """Predicted vs measured intervals for one run. Returns a result dict."""
    measured = measure(folder)
    measured_spans = measure_spans(folder)
    predicted, trace, predicted_spans, assumed_labels = predict(folder)
    # An undecidable branch: ask the CAPTURE which arm ran, then re-lay out and score that
    # timeline. Only if the data cannot distinguish the arms does the channel stay unscored.
    resolved_arm = None
    if assumed_labels and trace.assumed_paths:
        resolved_arm = resolve_assumed_arm(trace, measured)
        if resolved_arm is not None:
            predicted, predicted_spans = regions_of(trace)
            assumed_labels = set()

    rows, worst = [], 0.0
    for label in ("ch0", "ch1", "ch2", "ch3"):
        m, p = measured[label], predicted[label]
        merged_note = False
        # intervals relative to the channel's own first pulse -- the latency cancels
        m_int = [x - m[0] for x in m[1:]] if len(m) > 1 else []
        p_int = [x - p[0] for x in p[1:]] if len(p) > 1 else []
        # SEGMENTATION MUST AGREE BEFORE AN INTERVAL MEANS ANYTHING.
        # These are two lists of REGION starts, and interval k is only the same physical gap on
        # both sides if both sides split the signal into the same regions. When they disagree,
        # pairing by position compares unrelated pulses and manufactures a huge "error" out of
        # nothing: batch_uneven reported 549.83 ns purely because predict() merges the whole
        # contiguous batch into ONE region (10 pulses -> 3 regions) while the detector resolved 9,
        # so predicted region 2 (the post-batch discriminator, +835 ns) was compared against
        # measured region 2 (the second batch pulse, +285 ns). The same shape flattered other
        # cases: batch_resync/simulbus_transition/loop_batch each compared ZERO pairs and printed
        # a reassuring "worst error 0.0 ns". So a count mismatch is now a reported verdict, never
        # an error number, and it never feeds `worst`.
        # MERGED REGIONS. predict() merges only pulses that exactly touch, but the detector
        # merges any pair whose envelope never dips below REGION_FLOOR_FRACTION between them --
        # so a small real gap (stretch_then_pulse's join, back-to-back batch pulses) is 2
        # predicted regions against 1 measured. That is the measurement resolving less, not the
        # model being wrong, and the case docstrings say as much ("the two merge into one
        # above-threshold region, so this only confirms the join is seamless").
        # Collapse consecutive PREDICTED regions that fall inside one measured region's span, so
        # the comparison is between the same physical groups. Uses the measured spans, so there
        # is no invented gap threshold; if it cannot align, the mismatch still stands.
        if len(m) < len(p) and m and measured_spans[label]:
            grouped, gi = [], 0
            for start, width in zip(m, measured_spans[label]):
                stop = start + width
                members = [x for x in p[gi:] if x <= stop + 1.0]
                if not members:
                    break
                grouped.append(members[0])          # the group's FIRST predicted region
                gi += len(members)
            if len(grouped) == len(m) and gi == len(p):
                p = grouped
                p_int = [x - p[0] for x in p[1:]] if len(p) > 1 else []
                merged_note = True

        segmentation_ok = len(m) == len(p)
        if segmentation_ok:
            n = min(len(m_int), len(p_int))
            errors = [m_int[i] - p_int[i] for i in range(n)]
        else:
            errors = []
        if errors and label not in assumed_labels:
            worst = max(worst, max(abs(e) for e in errors))
        # Count and span, which the interval metric alone cannot see: a dropped or collapsed
        # train (e.g. an un-unrolled cache-pointer stream) merges to one region with no
        # intervals, so the interval error is a misleading 0. The tell-tale is the hardware
        # showing MORE pulses than the trace predicted -- `dropped` -- which means the tracer
        # is missing pulses. (Measured FEWER than predicted is the opposite, benign case of
        # two pulses merging into one region, e.g. stretch_then_pulse.)
        # region WIDTHS, comparable whenever both sides found the same regions -- this is what
        # makes a single-region case (single, two_same_block, stretch, shape, barrier_single_channel,
        # batch_nonblocking, stretch_then_pulse) an actual test instead of a vacuous 0.0 ns.
        mw, pw = measured_spans[label], predicted_spans[label]
        width_errors = ([mw[i] - pw[i] for i in range(len(mw))]
                        if segmentation_ok and len(mw) == len(pw) else [])
        if width_errors:
            worst_width = max(abs(e) for e in width_errors)
        else:
            worst_width = 0.0
        span_m = (m[-1] - m[0]) if len(m) > 1 else 0.0
        span_p = (p[-1] - p[0]) if len(p) > 1 else 0.0
        rows.append({"channel": label, "n_measured": len(m), "n_predicted": len(p),
                     "count_ok": segmentation_ok,
                     "predicted_merged": merged_note,
                     # what was actually checked: 0 means this channel proves nothing
                     "n_compared": len(errors),
                     "segmentation_ok": segmentation_ok,
                     # only where the tracer drew SOME pulses: a channel it (correctly) left
                     # empty can still pick up a stray noise region, which is not a drop
                     "dropped": (max(0, len(m) - len(p)) if len(p) > 0 else 0),
                     "span_measured_ns": round(span_m, 1),
                     "span_predicted_ns": round(span_p, 1),
                     "span_error_ns": round(abs(span_m - span_p), 1),
                     # absolute first edge = the constant DAC->cable->ADC latency;
                     # only meaningful as a per-channel calibration constant
                     "first_edge_ns": round(m[0], 2) if m else None,
                     "measured_widths_ns": [round(x, 1) for x in mw],
                     "predicted_widths_ns": [round(x, 1) for x in pw],
                     "width_error_ns": [round(e, 2) for e in width_errors],
                     "worst_width_error_ns": round(worst_width, 2),
                     # this channel's pulses come from an undecidable branch: reported, not scored
                     "assumed_path": label in assumed_labels,
                     "measured_intervals_ns": [round(x, 2) for x in m_int],
                     "predicted_intervals_ns": [round(x, 2) for x in p_int],
                     "error_ns": [round(e, 2) for e in errors],
                     "error_cycles": [round(e / trace.ns_per_cycle, 3) for e in errors]})

    result = {
        "folder": str(folder),
        "runtime": trace.runtime_class,
        "case": getattr(trace, "case", None),
        "blocks": len(trace.blocks),
        # gaps live on placements (what executes, loops unrolled), not on blocks
        "gaps_ns": [round(p.gap_after * trace.ns_per_cycle, 1)
                    for p in (trace.placements or trace.blocks) if p.gap_after],
        "gap_breakdowns": [p.gap_breakdown
                           for p in (trace.placements or trace.blocks) if p.gap_after],
        "executed_blocks": len(trace.placements or trace.blocks),
        # which arm of an undecidable branch the capture showed, when it could be decided
        "resolved_arm": resolved_arm,
        # kept so a GENERATED sequence can be classified by what it scheduled, not by its name
        "trace": trace,
        "worst_error_ns": round(worst, 2),
        "worst_error_cycles": round(worst / trace.ns_per_cycle, 3),
        # channels where hardware shows more pulses than the trace drew: a structural miss
        # (the tracer dropped pulses), separate from and invisible to the interval error
        "pulses_dropped": [r["channel"] for r in rows if r["dropped"] > 0],
        "rows": rows,
    }

    # Cache-pointer stream (rb_stream / randomized benchmarking) guard: the loop MUST be
    # unrolled from the cache to the count the cache holds, and the post-loop blocks MUST sit
    # after the train. This is exactly the regression the tracer fix prevents (a collapsed
    # stream would leave one symbolic placeholder and drag the readout/tail up to the front).
    if getattr(trace, "stream", None):
        streamed = [p for p in trace.placements if getattr(p, "stream", False)]
        gates = sum(len(p.commands) for p in streamed)
        expected = int(trace.point_cache.get(trace.stream["count_word"], 0))
        train_stop = max((p.stop for p in streamed), default=0)
        after = any(not getattr(p, "stream", False) and p.start >= train_stop
                    and any(c.pulse for c in p.commands) for p in trace.placements)
        result["stream_gates"] = gates
        result["stream_expected"] = expected
        result["stream_ok"] = bool(gates == expected and expected > 0
                                    and not result["pulses_dropped"] and after)

    if verbose:
        print(f"\n=== {folder}")
        print(f"    {trace.runtime_class}: {len(trace.blocks)} blocks, "
              f"gaps {result['gaps_ns']} ns")
        for row in rows:
            if not row["segmentation_ok"]:
                # Not a timing result: the two sides split the signal differently, so no interval
                # pair is the same physical gap. Say so instead of printing a bogus error.
                flag = (f"  <-- SEGMENTATION MISMATCH: measured {row['n_measured']} regions vs "
                        f"predicted {row['n_predicted']} (span meas/pred "
                        f"{row['span_measured_ns']}/{row['span_predicted_ns']} ns) -- "
                        f"NOT COMPARED")
            elif row.get("assumed_path"):
                # An undecidable branch: the trace had to guess which arm ran, so this channel
                # says nothing about the timing model either way (see predict()).
                flag = ("  <-- ASSUMED PATH (undecidable branch) -- REPORTED, NOT SCORED"
                        f"; would read {max(abs(e) for e in row['error_ns']):.2f} ns"
                        if row["error_ns"] else "  <-- ASSUMED PATH -- NOT SCORED")
            elif row["error_ns"] and max(abs(e) for e in row["error_ns"]) >= 1.0:
                flag = "  <-- MISMATCH"
            elif row["error_ns"]:
                flag = (f"  OK ({row['n_compared']} intervals"
                        + (", predicted regions merged to match the detector"
                           if row["predicted_merged"] else "") + ")")
            elif row["width_error_ns"]:
                # NOT a pass. Region width is threshold-biased and cannot be scored: the measured
                # width is where the envelope crosses REGION_FLOOR_FRACTION of the peak, while the
                # prediction is the DMA command length, so the difference carries the ramp shape
                # and the floor. Measured here at 5-25 ns for fast ramps and 445 ns for `stretch`
                # (whose mid-pulse park dips under the floor) -- all artifact, no model content.
                # Reported as a diagnostic so a single-region case is not silently "0.0 ns OK";
                # such cases are verified STRUCTURALLY against compiled.log instead.
                flag = (f"  NO INTERVAL -- width diag only "
                        f"(meas {row['measured_widths_ns']} vs pred {row['predicted_widths_ns']} ns, "
                        f"threshold-biased, NOT scored)")
            else:
                flag = "  <-- NOTHING COMPARED (no interval and no comparable region width)"
            print(f"    {row['channel']}  pulses meas/pred {row['n_measured']}/{row['n_predicted']}"
                  f"  measured {row['measured_intervals_ns']}"
                  f"  predicted {row['predicted_intervals_ns']}"
                  f"  error {row['error_ns']} ns "
                  f"({row['error_cycles']} cyc){flag}")
        widths = sum(len(r["width_error_ns"]) for r in rows)
        worst_w = max((r["worst_width_error_ns"] for r in rows), default=0.0)
        if widths:
            print(f"    region widths (diagnostic only, threshold-biased): "
                  f"{widths} channels, worst {worst_w} ns")
        compared = sum(r["n_compared"] for r in rows)
        if compared:
            print(f"    worst error: {result['worst_error_ns']} ns "
                  f"= {result['worst_error_cycles']} cycles  ({compared} intervals compared)")
        elif widths:
            print(f"    NO TIMING RESULT: no interval exists on any channel. Region widths are "
                  f"reported above as a diagnostic only -- they are threshold-biased and are not "
                  f"a check. This case's claim is covered structurally (trace vs compiled.log).")
        else:
            print(f"    NO TIMING RESULT: 0 intervals and 0 widths compared "
                  f"(worst error 0.0 would be vacuous)")
        if "stream_ok" in result:
            print(f"    cache-pointer stream: {result['stream_gates']} gates unrolled "
                  f"(cache says {result['stream_expected']}), tail after train: "
                  f"{'OK' if result['stream_ok'] else '<-- FAIL (stream not unrolled)'}")
    return result


# ---------------------------------------------------------------- running

FUZZ_STEPS = None       # set from --fuzz-steps; applies to the random_seq generator only


def build_runtime(case, iterations=5000, dwell_length=200e-9):
    from loopback_timing_cases import (READOUT_CASES,
        LoopbackTimingCaseRuntime)

    runtime = LoopbackTimingCaseRuntime(
        stimulus0="stimulus0", stimulus1="stimulus1",
        stimulus2="stimulus2", stimulus3="stimulus3",
        capture0="capture0", capture1="capture1",
        capture2="capture2", capture3="capture3",
        capture_dummy="capture_dummy",
        case=case,
        dwell_length=dwell_length,
        iterations=iterations,
        # the test_* cases compare a cached register against 0, so 0 forces the
        # condition true and 1 forces it false
        test_register_value=0 if "true" in case else 1,
        # the measure_* cases put the resonator's capture on ADC1, so the sacrificial dummy
        # capture must release it
        use_dummy_channel=case not in READOUT_CASES,
        # measure()/feedback cases need a cmacc for the resonator, so they trace only the two
        # channels they actually read (see trace_channels on the runtime)
        trace_channels=((0, 1) if case in READOUT_CASES else (0, 1, 2, 3)),
        yaml_path=YAML_PATH,
    )
    if FUZZ_STEPS is not None:
        runtime.fuzz_steps = FUZZ_STEPS
    return runtime


def dry_run(case, **kwargs):
    """Trace a case off-hardware. Catches compile errors before any deploy.

    The dry run calls the real ``acadia.compile()``, so anything the board would
    reject at compile time is rejected here too -- barrier_uneven's multi-argument
    max() is the example.

    ``capture_points`` is deliberately LEFT ON. It used to be False here, which looks like a
    harmless speed-up (these cases have exactly one point) but silently disabled branch
    resolution: ``evaluate_condition`` reads the ``test`` register out of the captured cache, so
    with no cache every ``test`` body is *assumed taken*. ``test_false`` then printed
    "5 blocks, gaps [80.0, 75.0, 80.0]" -- byte-identical to ``test_true`` -- instead of the
    skipped-body "4 executed, gaps [95.0, 80.0]" that is the harness's strongest single result.
    ``assumed``/``UNSUPPORTED`` are reported below for the same reason: an assumed path must never
    pass as a verified one.
    """
    from acadia_qmsmt import sequence_viz as sv

    trace = sv.trace_runtime(build_runtime(case, **kwargs), envelopes=False)
    # gaps live on placements (what executes), not on blocks (what was compiled)
    executed = trace.placements or trace.blocks
    gaps = [round(p.gap_after * trace.ns_per_cycle, 1) for p in executed if p.gap_after]
    unrolled = ("" if len(executed) == len(trace.blocks)
                else f" -> {len(executed)} executed")
    flags = ""
    if trace.assumed_paths:
        flags += f"  ASSUMED blocks {sorted(trace.assumed_paths)}"
    if getattr(trace, "unsupported_paths", None):
        flags += f"  UNSUPPORTED blocks {sorted(trace.unsupported_paths)}"
    print(f"  {case:20s} OK   {len(trace.blocks)} blocks{unrolled}, "
          f"{trace.length_ns:.0f} ns, gaps {gaps} ns{flags}")
    return trace


def unsafe_reason(runtime):
    """Why this runtime must not be deployed, or None if it is fine.

    Two pre-flight checks, both asked of the MODEL rather than of a list of case names -- which is
    the point of having a model you trust, and which covers cases nobody has thought of yet.

    Traced ONCE and both checks applied to that trace: a runtime can only be traced once (its Acadia
    holds the compiled program afterwards), so asking two questions that each traced it meant the
    second silently returned "fine" for every runtime in existence.
    """
    from acadia_qmsmt import sequence_viz as sv

    try:
        # capture_points=True is REQUIRED: the register value comes from the per-point cache, and
        # without it every register length falls back to `resolve_indeterminate` (0) and looks
        # like an underflow. That false positive would have skipped every valid stretch point.
        trace = sv.trace_runtime(runtime, capture_points=True, envelopes=False)
    except Exception:
        return None                              # cannot trace it; let the deploy decide
    return underflow_note(trace) or nontermination_note(trace)


def nontermination_note(trace):
    """Refuse a sequence containing a loop that cannot exit.

    MEASURED 2026-08-14: ``repeat_until(counter == 0)`` on a counter loaded 0 and incremented +1
    per pass never returns from the board -- "Timeout occurred waiting for line", repeating, until
    the run is killed. The same scan at 1..8 measures clean, so this is the degenerate target, not
    the case. It also settles the semantics: test-before-body and test-after-body both predict N
    passes for every N >= 1 and differ only at 0, so a hang proves the body always runs at least
    once and the counter can never come back to 0.

    Costing minutes of board time to rediscover that is avoidable, so the model says no first.
    """
    stuck = [e for e in trace.control_flow_summary() if e.get("nonterminating")]
    if not stuck:
        return None
    first = stuck[0]
    return (f"{first['kind']} @{first['block']} exits when its counter reaches 0, but the counter "
            f"starts at 0 and is incremented before the test is next evaluated, so it never can. "
            f"The board hangs (measured 2026-08-14). Use a target of 1 or more.")


def underflow_note(trace):
    """A register-driven length of zero, which acadia emits as an all-ones length field.

    ``Acadia.command_dma`` writes ``length - 1``, so 0 wraps to ~328 us for an ARB command and
    ~21 s for a 32-bit dwell. Deploying it wedges the run (repeated "Timeout occurred waiting for
    line") and costs minutes of board time for no data. Cheaper to ask the model.
    """
    if not trace.length_underflows:
        return None
    first = trace.length_underflows[0]
    return (f"register {first['register']} on {first['channel']} resolves to a ZERO-length "
            f"{first['kind']}, which acadia emits as {first['cycles']} cycles "
            f"({first['cycles'] * 5e-9:.1f} s). Floor it at one cycle "
            f"(see dual_rail_ramsey._delay_cycles) rather than sweeping to 0.")


def underflow_reason(runtime):
    """``unsafe_reason`` restricted to the length-underflow check, for callers that want only it."""
    from acadia_qmsmt import sequence_viz as sv

    try:
        trace = sv.trace_runtime(runtime, capture_points=True, envelopes=False)
    except Exception:
        return None
    return underflow_note(trace)


def capture_window_for(runtime):
    """Seconds of ADC window that will actually cover this runtime's sequence.

    Sized from an off-hardware trace rather than a constant, because the window has to hold the
    whole sequence plus the ~283 ns cable latency plus a final pulse -- and a generated sequence
    (random_seq) or a swept count (blocks_n=10, loop_count=8) has no fixed length. A window
    shorter than the sequence silently truncates the recording: the yaml's 1.2e-6 recorded only
    1070 ns of 2400-2610 ns cases, which is why the post-batch discriminator pulses those cases
    exist to measure were missing entirely.
    """
    from acadia_qmsmt import sequence_viz as sv

    try:
        trace = sv.trace_runtime(runtime, capture_points=False, envelopes=False)
    except Exception:
        return CAPTURE_WINDOW_S                      # fall back to the constant
    return max(CAPTURE_WINDOW_S, (trace.length_ns + 1200.0) * 1e-9)


def run_case(case, iterations=5000, dwell_length=200e-9):
    """Deploy one timing case to the board. Returns the local data directory."""
    board_ip = paths_local.require("board_ip")
    save_root = paths_local.require("loopback_data_root")
    window = capture_window_for(build_runtime(case, iterations=10, dwell_length=dwell_length))
    runtime = build_runtime(case, iterations=iterations, dwell_length=dwell_length)
    # Record the WHOLE sequence, not just its first 1070 ns.
    if not runtime.capture_length_override:
        runtime.capture_length_override = window
    runtime.deploy(board_ip, local_directory=f"{save_root}/{case}/%y%m%d_%H%M%S")
    runtime.wait_for_deploy_completion()
    return runtime.local_directory


def run_scan(spec):
    """Sweep one field of one case over values, deploying and checking each point.

    ``spec`` is ``CASE:FIELD=v1,v2,...``. Reports every point plus the worst error over the whole
    scan, so a constant offset is distinguishable from one that grows with the swept quantity --
    the discrimination the whole timing model rests on (a miscount scales with what it counts; a
    fixed hardware latency does not).
    """
    case, _, rest = spec.partition(":")
    field, _, values = rest.partition("=")
    # Cast to the FIELD's type, not to whatever the literal looks like. Parsing "0" as an int
    # because it has no "." or "e" sent a bare 0 into a seconds-valued field, and
    # seconds_to_cycles rejects a non-float outright ("must be a float or numpy array of
    # floats") -- so the one scan point that mattered, a zero-length register command, died in
    # the runtime instead of being measured.
    probe_type = type(getattr(build_runtime(case, iterations=10), field, 0.0))
    cast = float if probe_type is float else (int if probe_type is int else float)
    points = [cast(v) for v in values.split(",") if v.strip()]
    if not (case and field and points):
        raise SystemExit(f"--scan needs CASE:FIELD=v1,v2,...; got {spec!r}")

    board_ip = paths_local.require("board_ip")
    save_root = paths_local.require("loopback_data_root")
    print(f"scan: {case}.{field} over {points}\n")
    rows, worst_overall = [], 0.0
    for value in points:
        probe = build_runtime(case, iterations=10)
        setattr(probe, field, value)
        unsafe = unsafe_reason(probe)
        if unsafe:
            # skip this POINT, not the scan: the rest of the sweep is still worth measuring
            print(f"  {field}={value:<12g} SKIPPED -- {unsafe}")
            continue
        window = capture_window_for(probe)
        runtime = build_runtime(case, iterations=5000)
        setattr(runtime, field, value)
        if not runtime.capture_length_override:
            runtime.capture_length_override = window
        tag = f"{field}_{value:g}".replace(".", "p").replace("-", "m")
        runtime.deploy(board_ip, local_directory=f"{save_root}/{case}__{tag}/%y%m%d_%H%M%S")
        runtime.wait_for_deploy_completion()
        result = compare(runtime.local_directory, verbose=False)
        worst = result["worst_error_ns"]
        compared = sum(r["n_compared"] for r in result["rows"])
        rows.append((value, worst, compared))
        worst_overall = max(worst_overall, worst if compared else 0.0)
        status = "OK" if (compared and worst < 0.5) else ("no interval" if not compared else "FAIL")
        print(f"  {field}={value:<12g} worst {worst:7.2f} ns  ({compared} intervals)  {status}")
    print(f"\nscan worst over {len(rows)} points: {worst_overall:.2f} ns")
    return 0 if worst_overall < 0.5 else 1


#: Triple enumeration defaults to the primitives whose timing depends on FIFO/branch state --
#: the ones every bug found so far involved. 5**3 = 125 deploys, against 1000 for the full
#: alphabet; the rest (block/dwell/stretch) are covered as neighbours by the pair sweep.
TRIPLE_SUBSET = ("block", "batch", "batch_almost", "loop", "test_taken")


def run_pairs(only=None, triples=None):
    """Deploy every ordered pair of scheduling primitives and check each.

    Exhaustive rather than probabilistic. Scheduling errors are properties of a JOIN -- what
    runs immediately before or after a construct -- not of a construct in isolation: both bugs
    this harness found were adjacency bugs (a drain followed by a loop back-edge, and a drain's
    release level feeding whatever came next). A random walk only probably produces any given
    join; enumerating the pairs guarantees every one of them is measured at least once.
    """
    from loopback_timing_cases import PRIMITIVES

    board_ip = paths_local.require("board_ip")
    save_root = paths_local.require("loopback_data_root")
    if triples is not None:
        alphabet = list(triples) if triples else list(TRIPLE_SUBSET)
        pairs = [(a, b, c) for a in alphabet for b in alphabet for c in alphabet]
    else:
        pairs = [(a, b) for a in PRIMITIVES for b in PRIMITIVES
                 if only is None or only in (a, b)]
    print(f"enumeration: {len(pairs)} ordered "
          f"{'triples' if triples is not None else 'pairs'}\n")

    worst_overall, failures, skipped = 0.0, [], []
    for index, combo in enumerate(pairs, 1):
        first, second, third = (list(combo) + [""])[:3]
        tag = "__".join(k for k in (first, second, third) if k)
        try:
            probe = build_runtime("pair_seq", iterations=10)
            probe.pair_a, probe.pair_b, probe.pair_c = first, second, third
            window = capture_window_for(probe)
            runtime = build_runtime("pair_seq", iterations=5000)
            runtime.pair_a, runtime.pair_b, runtime.pair_c = first, second, third
            if not runtime.capture_length_override:
                runtime.capture_length_override = window
            runtime.deploy(board_ip,
                           local_directory=f"{save_root}/pair_{tag}/%y%m%d_%H%M%S")
            runtime.wait_for_deploy_completion()
            result = compare(runtime.local_directory, verbose=False)
        except Exception as exc:                      # a pair that will not build or deploy
            skipped.append((tag, f"{type(exc).__name__}: {exc}"))
            print(f"  [{index:3d}/{len(pairs)}] {tag:34s} SKIPPED ({type(exc).__name__})")
            continue
        worst = result["worst_error_ns"]
        compared = sum(r["n_compared"] for r in result["rows"])
        status = "OK" if (compared and worst < 0.5) else ("no interval" if not compared else "FAIL")
        if status == "FAIL":
            failures.append((tag, worst))
        if compared:
            worst_overall = max(worst_overall, worst)
        print(f"  [{index:3d}/{len(pairs)}] {tag:34s} worst {worst:7.2f} ns "
              f"({compared} intervals)  {status}")

    print(f"\npairs: {len(pairs)}  worst {worst_overall:.2f} ns  "
          f"failures {len(failures)}  skipped {len(skipped)}")
    for tag, worst in failures:
        print(f"   FAIL    {tag:34s} {worst:.2f} ns")
    for tag, why in skipped:
        print(f"   SKIPPED {tag:34s} {why}")
    return 1 if failures else 0


def reportable(result):
    """A result without the live objects, for the JSON report.

    ``compare()`` hands back the SequenceTrace itself so callers can ask it further questions
    (systematic_note does). It is not serialisable, so every path that writes the report has to
    drop it -- and `--case`/`--cases` did not, which raised TypeError AFTER the deploy and the
    comparison had both succeeded: the measurement was done and thrown away at the last step.
    Stripped here, at the one boundary where serialisation happens, rather than at each caller.
    """
    return {k: v for k, v in result.items() if k != "trace"}


def main():
    global FUZZ_STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--cases", type=str, default=None,
                        help="comma-separated list, run in one invocation")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--scan", default=None, metavar="CASE:FIELD=v1,v2,...",
                        help="sweep one runtime field over values for one case, deploying and "
                             "checking each. e.g. --scan dwell_n:dwell_length=100e-9,200e-9,300e-9 "
                             "or --scan blocks_n:n_blocks=2,3,4,5,6. Turns a spot check into a "
                             "curve: a model error that is a fixed offset and one that scales with "
                             "the swept quantity look identical at a single point.")
    parser.add_argument("--triples", default=None, metavar="P1,P2,...",
                        help="deploy every ordered TRIPLE over the given primitives "
                             "(default: the interaction-prone subset)")
    parser.add_argument("--pairs-only", default=None, metavar="PRIMITIVE",
                        help="restrict --pairs to ordered pairs involving this primitive")
    parser.add_argument("--pairs", action="store_true",
                        help="deploy EVERY ordered pair of scheduling primitives (exhaustive "
                             "adjacency coverage; see PRIMITIVES in loopback_timing_cases)")
    parser.add_argument("--fuzz-steps", type=int, default=None,
                        help="random_seq: how many primitive steps to compose per sequence")
    parser.add_argument("--include-unsafe", action="store_true",
                        help="also deploy cases in UNSAFE_TO_DEPLOY (test_false_nospec hangs "
                             "the sequencer -- see KI_004)")
    parser.add_argument("--analyse", type=str, default=None,
                        help="analyse an existing folder without deploying")
    parser.add_argument("--regions", type=str, default=None,
                        help="dump above-threshold regions per channel for a folder")
    parser.add_argument("--logs", type=str, default=None,
                        help="print the remote logs of the newest run of a case")
    parser.add_argument("--dryrun", type=str, default=None,
                        help="trace cases off-hardware ('all' for every case)")
    parser.add_argument("--revalidate", action="store_true",
                        help="re-compare the newest run of every case, compactly")
    parser.add_argument("--program", type=str, default=None,
                        help="dump the compiled sequencer program for a case (dry run)")
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()
    if args.fuzz_steps is not None:
        FUZZ_STEPS = args.fuzz_steps

    if args.triples is not None:
        subset = [s for s in args.triples.split(",") if s.strip()] or None
        return run_pairs(triples=subset)

    if args.pairs or args.pairs_only:
        return run_pairs(only=args.pairs_only)

    if args.program:
        # Tagged dump of the compiled program, so control-flow edges can be read off.
        dry_run(args.program)   # prints a summary + reproduces any compile error
        from acadia_qmsmt import sequence_viz as sv
        runtime = build_runtime(args.program)
        # dry_run already compiled a runtime, but it is not returned -- recompile a fresh
        # one purely to hold the program (a runtime can only be traced once, by design)
        sv.trace_runtime(runtime, capture_points=False, envelopes=False)
        program = [instruction.pprint() for sequencer
                   in runtime.acadia._sequencer_type.instances
                   for instruction in sequencer._compiled_program]
        for index, line in enumerate(program):
            # Tag on the instruction's OWN destination, which is the tail of the line --
            # a Symbol repr earlier in the line quotes some *other* instruction and must
            # not decide the tag. This is what hid the `test` skip branch before.
            tail = line[-90:]
            tag = ""
            if "; Trigger DMAs" in line:
                tag = "TRIGGER"
            elif "PC (absolute hold) if BUS_DATA AND MASK != 0" in tail:
                tag = "DMA-POLL"
            elif "PC (absolute branch)" in tail:
                tag = "BRANCH"
            elif "PC (absolute hold)" in tail:
                tag = "hold"
            elif "Command DMA" in line:
                tag = "push"
            elif "FIFO latency" in line:
                tag = "nop"
            elif "DSP" in line:
                tag = "dsp"
            if tag:
                print(f"  {index:04d} {tag:9s} ...{tail}")
        print(f"  ({len(program)} instructions total)")
        return

    if args.revalidate:
        paths_local.require("loopback_data_root")   # SAVE_ROOT must be set
        # regression check: newest run of each case, one line each
        print(f"{'case':22s} {'blocks':>6s} {'gaps (ns)':>22s} {'worst err':>10s}")
        worst_overall = 0.0
        for case_dir in sorted(p for p in Path(SAVE_ROOT).glob("*") if p.is_dir()):
            # a case dir holds run dirs; a run dir is one with kwargs.json in it.
            # SAVE_ROOT also holds legacy flat runs from before cases existed.
            runs = sorted((p for p in case_dir.glob("*")
                           if p.is_dir() and (p / "kwargs.json").is_file()),
                          key=lambda p: p.stat().st_mtime)
            if not runs:
                continue
            try:
                result = compare(runs[-1], verbose=False)
            except KeyError:
                # no t_data group: main() raised on the board before writing any.
                # Expected for the KI_002 cases, which cannot compile at all.
                print(f"{case_dir.name:22s} {'-':>6s} {'-':>22s}   no data "
                      f"(run failed; see --logs {case_dir.name})")
                continue
            except Exception as exc:
                print(f"{case_dir.name:22s} {'-':>6s} {'-':>22s}  "
                      f"{type(exc).__name__}")
                continue
            note = systematic_note(case_dir.name, trace=result.get("trace"))
            if not note:
                worst_overall = max(worst_overall, result["worst_error_ns"])
            # dropped pulses / a collapsed cache-pointer stream are structural misses the
            # interval error is blind to -- flag first, even when the interval error is small
            if result.get("stream_ok") is False:
                flag = (f"   <-- STREAM NOT UNROLLED "
                        f"({result['stream_gates']}/{result['stream_expected']} gates)")
            elif result["pulses_dropped"]:
                flag = f"   <-- DROPPED PULSES on {','.join(result['pulses_dropped'])}"
            elif result["worst_error_ns"] < 1.0:
                flag = ""
            elif note:
                flag = "   <-- measurement systematic, see note"
            else:
                flag = "   <-- MISMATCH"
            print(f"{case_dir.name:22s} {result['blocks']:6d} "
                  f"{str(result['gaps_ns']):>22s} "
                  f"{result['worst_error_ns']:8.2f} ns{flag}")
        print(f"\nworst across all cases (excluding known systematics): "
              f"{worst_overall:.2f} ns")
        for case, note in KNOWN_SYSTEMATIC.items():
            print(f"\nnote on {case}: {note}")
        return

    if args.dryrun:
        from loopback_timing_cases import CASES
        cases = list(CASES) if args.dryrun == "all" else [
            c.strip() for c in args.dryrun.split(",") if c.strip()]
        print("dry run (no hardware):")
        for case in cases:
            try:
                dry_run(case)
            except Exception as exc:
                print(f"  {case:20s} FAIL {type(exc).__name__}: "
                      f"{str(exc)[:110]}")
        return

    if args.logs:
        paths_local.require("loopback_data_root")   # SAVE_ROOT must be set
        # newest run folder for a case, plus the tail of its remote logs -- kept here
        # rather than done in the shell so every invocation stays a single command
        root = Path(SAVE_ROOT) / args.logs
        folders = sorted((p for p in root.glob("*") if p.is_dir()),
                         key=lambda p: p.stat().st_mtime)
        if not folders:
            print(f"no runs under {root}")
            return
        folder = folders[-1]
        print(f"--- {folder}")
        for name in ("remote_stderr.log", "remote_main.log", "runtime.log"):
            path = folder / name
            if not path.is_file() or not path.stat().st_size:
                continue
            lines = path.read_text(errors="replace").splitlines()
            print(f"  [{name}] last {min(len(lines), 20)} of {len(lines)} lines")
            for line in lines[-20:]:
                print("    " + line)
        return

    if args.regions:
        from acadia_qmsmt.utils.saved_runtime_loader import load_runtime_from_data_dir
        for folder in args.regions.split(","):
            runtime = load_runtime_from_data_dir(folder.strip())
            runtime.process_current_data()
            print(f"--- {folder.strip()}")
            for label in ("ch0", "ch1", "ch2", "ch3"):
                regions = pulse_regions_ns(runtime, label)
                print("  %-4s %s" % (label, " | ".join(
                    f"{a:.0f}-{b:.0f} (w={b - a:.0f}, pk={p:.3g})" for a, b, p in regions)))
        return

    if args.analyse:
        compare(args.analyse)
        return

    from loopback_timing_cases import CASES
    if args.scan:
        return run_scan(args.scan)

    if args.all:
        cases = [c for c in CASES if c not in UNSAFE_TO_DEPLOY or args.include_unsafe]
        for case, why in UNSAFE_TO_DEPLOY.items():
            if case in CASES and not args.include_unsafe:
                print(f"  SKIPPING {case} -- {why}. Use --include-unsafe to deploy it anyway.")
    elif args.cases:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        cases = [args.case or "two_blocks"]

    if not args.include_unsafe:
        blocked = [c for c in cases if c in UNSAFE_TO_DEPLOY]
        if blocked:
            raise SystemExit(
                "refusing to deploy " + ", ".join(blocked) + ":\n  "
                + "\n  ".join(f"{c}: {UNSAFE_TO_DEPLOY[c]}" for c in blocked)
                + "\nPass --include-unsafe if you really mean to.")

    results = []
    for case in cases:
        print(f"\n>>> deploying case {case!r} ...")
        try:
            folder = run_case(case, iterations=args.iterations)
            results.append(compare(folder))
        except Exception as exc:              # keep the batch going unattended
            print(f"    FAILED {type(exc).__name__}: {exc}")
            results.append({"case": case, "error": f"{type(exc).__name__}: {exc}"})

    previous = json.loads(REPORT.read_text()) if REPORT.exists() else []
    REPORT.write_text(json.dumps(previous + [reportable(r) for r in results], indent=2))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
