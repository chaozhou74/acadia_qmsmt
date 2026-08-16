"""
Loopback timing cases -- a physical oracle for the sequence_viz timing model.

Same 4-channel DAC->ADC loopback scaffolding as LoopbackMultichannelRuntime (each DAC
is cabled to its own ADC, so a capture is a direct recording of that DAC's output), but
the DAC block structure is chosen by the `case` parameter. Each case isolates one claim
the visualizer makes about timing, so the prediction can be checked against hardware.

Measure INTERVALS BETWEEN EDGES ON ONE CHANNEL, never absolute arrival: the constant
DAC->cable->ADC latency (~283 ns here) and any residual capture-trigger skew both cancel
in a within-channel interval. `use_dummy_channel=True` additionally removes the KI_001
last-triggered-channel skew, so cross-channel comparison is good to ~1.5 ns, but
within-channel intervals remain the primary metric.

The capture-trigger block and the closing wait block are identical in every case, so the
only thing that varies between cases is the DAC sequence under test.

Cases (see `dac_sequence`):
  single              one pulse, one block                 -- per-channel offset calibration
  two_same_block      two pulses back-to-back, ONE block   -- null test: predicts zero gap
  two_blocks          two pulses, two blocking blocks      -- the inter-block gap
  two_blocks_1ch      block 2 drives ch0 only              -- fewer DMA pushes  \\ discriminator:
  two_blocks_2ch      block 2 drives ch0+ch1               -- more pushes       // does the gap
  three_blocks        three blocking blocks                -- does the gap accumulate linearly
  batch_nonblocking   three pulses, block=False            -- predicts seamless concatenation
  dwell_between       pulse, dwell, pulse in one block     -- dwell() honours its argument
  barrier_uneven      barrier with unequal channel lengths -- alignment padding
"""

import numpy as np

from acadia import Acadia, DataManager
from acadia.runtime import annotate_method
from acadia_qmsmt import QMsmtRuntime, IOConfig

import logging

logger = logging.getLogger("acadia")

CASES = ("single", "two_same_block", "two_blocks", "two_blocks_1ch", "two_blocks_2ch",
         "two_blocks_3ch", "three_blocks", "four_blocks", "batch_nonblocking",
         "dwell_between", "register_dwell", "stretch", "stretch_then_pulse",
         "stretch_two_blocks", "stretch_two_blocks_same",
         "shape", "detune_pair", "phase_pair", "loop_2", "loop_3", "loop_2_double",
         "test_true", "test_false", "test_true_nospec", "test_false_nospec",
         "barrier_single_channel", "rb_stream", "rb_stream_uniform", "batch_resync",
         "simulbus_transition", "batch_two_channels", "batch_uneven", "batch_interleaved",
         "loop_batch", "stream_then_batch", "batch_concurrent_blocking",
         # cooling shape: nested counter loops, and a test nested inside a loop
         "nested_cool_2x2", "nested_cool_3x2", "test_in_loop_true", "test_in_loop_false",
         # the readout path: measure(), two-round readout, active reset
         "measure_readout", "measure_two_rounds", "feedback_reset", "measure_trace_case",
         "measure_multi",
         # parametric variants, for sweeps (see timing_validation --scan)
         "loop_n", "nested_cool_n", "blocks_n", "dwell_n",
         # COMPOSITIONS: the interactions, which is where the model broke before
         "loop_with_measure", "batch_in_loop", "test_then_batch", "stretch_in_loop",
         "three_deep_nest",
         # isolate the stretch-length and FIFO-drain models from the loop model
         "register_stretch", "batch_drain_twice", "three_deep_nest_reconfig",
         # repeat_until and test, exhaustively
         "repeat_until_op", "repeat_until_count_n", "test_nested",
         "test_in_counter_loop", "counter_loop_in_test",
         # the almost_empty drain -- the ONLY drain variant the real runtimes use
         "batch_drain_almost", "batch_in_loop_almost",
         # generated sequences: random compositions, and the exhaustive pair enumeration
         "random_seq", "pair_seq",
         # KI_002 cases: these could not compile before the 2026-07-27 acadia pull
         "barrier_uneven_pulses", "barrier_uneven_2ch", "barrier_uneven")

# Cases built on the cache-pointer stream idiom: they share the command-cache setup, the
# rb-pulse allocation, and the cache fill. rb_stream_uniform is rb_stream with uniform-amplitude
# final gates (clean timing of the back-to-back block; see README on the edge-detection artifact).
STREAM_CASES = ("rb_stream", "rb_stream_uniform", "stream_then_batch")

# Cases that read out through MeasurableResonator. They need the resonator built BEFORE attach()
# (its window/accumulation memories must exist to be mapped), one cmacc module of their own, and
# so only two raw trace captures. Kept as one list because a case that is in some of those lists
# and not others half-exists: self._resonator stays None, resonator.measure() raises
# AttributeError inside a channel_synchronizer, and the synchronizer's __exit__ masks it with
# "ValueError: Empty synchronizer" -- which is what happened when loop_with_measure was added.
#: The scheduling alphabet: every construct the qudit runtimes build sequences out of, and
#: that the timing model has a distinct term for. `pair_seq` enumerates the ordered PAIRS of
#: these so every adjacency is covered exhaustively; `random_seq` composes them at random for
#: longer-range interactions. Both `test` arms and both FIFO drain senses are listed separately
#: because they are genuinely different scheduling events, not parameters of one event.
PRIMITIVES = ("block", "batch", "batch_almost", "dwell", "reg_dwell",
              "loop", "counter_loop", "test_taken", "test_skipped", "stretch")


READOUT_CASES = ("measure_readout", "measure_two_rounds", "feedback_reset",
                 "measure_trace_case", "loop_with_measure", "measure_multi")


class LoopbackTimingCaseRuntime(QMsmtRuntime):
    """4-channel loopback with a selectable DAC sequence, for timing validation."""

    # ========== IOConfig declarations ==========
    stimulus0: IOConfig
    stimulus1: IOConfig
    stimulus2: IOConfig
    stimulus3: IOConfig
    capture0: IOConfig
    capture1: IOConfig
    capture2: IOConfig
    capture3: IOConfig
    capture_dummy: IOConfig

    # ========== Parameters ==========
    iterations: int
    case: str = "two_blocks"
    stimulus_pulse_name: str = "test_pulse"
    capture_memory_name: str = "loopback_trace"
    dwell_length: float = 200e-9      # used by dwell_between / barrier_uneven
    register_dwell: float = 300e-9    # used by register_dwell; written into the cache
    test_register_value: int = 0      # cache[0] for the test_* cases; 0 makes the
                                      # condition (REG == 0) true, anything else false
    # rb_stream mirrors QubitRBRuntime on ch0: (0) an initial pulse, (1) the cache-pointer
    # gate loop, (2) a readout-marker block, (3) the "8 basic gates" back-to-back. The loop
    # plays one gate per entry of rb_pattern (indices into rb_gates); its LAST entry is the
    # undo/recovery gate (played inside the loop, as in the real RB). Distinct-amplitude
    # gates make the played sequence readable off the power trace. rb_num_pulses, if set,
    # tiles rb_pattern to that length (so a scaling sweep needs only one number).
    rb_gates: tuple = ("rb_gate_lo", "rb_gate_mid", "rb_gate_hi")
    rb_pattern: tuple = (2, 0, 1, 2, 1)   # last entry = the undo/recovery gate
    rb_num_pulses: int = None         # None -> len(rb_pattern); else tile to this length
    rb_initial_pulse: str = "rb_gate_hi"   # block 0, mirrors the initial pi pulse
    rb_readout_pulse: str = "test_pulse"   # readout-marker block after the loop (wide, distinct)
    rb_final_pattern: tuple = (0, 1, 2, 0, 1, 2, 0, 1)  # the 8 basic gates, one block, back-to-back
    rb_final_gate: str = None         # if set, the 8 final gates all use this ONE shape (uniform
                                      # amplitude) instead of rb_final_pattern. The varying pattern
                                      # is for gate-IDENTITY readability, but it defeats the edge
                                      # detector on the merged back-to-back region (see README:
                                      # the 50%-of-region-peak edge latches onto the first HIGH
                                      # gate). Set this for TIMING validation of the final block.
    rb_loop_gate: str = None          # if set, the loop plays only this gate (a single shape),
                                      # for the pulse-length sweep that maps the ~110 ns floor
    loop_count: int = 2               # loop_n / nested_cool_n: outer deterministic loop passes
    inner_loop_count: int = 2         # nested_cool_n: inner passes per outer pass
    n_blocks: int = 3                 # blocks_n: how many blocking blocks in a row
    register_stretch: float = 100e-9   # stretch length driven from a register (see cache[0])
    fuzz_seed: int = 0                # random_seq: seed for the generated sequence
    pair_a: str = "block"             # pair_seq: first primitive of the ordered pair
    pair_b: str = "block"             # pair_seq: second primitive
    pair_c: str = ""                  # pair_seq: optional THIRD primitive (triple enumeration)
    repeat_operator: str = "=="       # repeat_until_op: which comparison the loop comes out on
    exclude_stretch: bool = False     # generated cases: leave `stretch` out of the alphabet
    fuzz_steps: int = 6               # random_seq: how many primitive steps to compose
    batch_resync_pulses: int = 8      # batch_resync: how many pulses the block=False batches play
                                      # before the dwell(pulse_length) + barrier + readout re-sync
    capture_length_override: float = None  # seconds; lengthen the ADC window past the yaml
    capture_start_delay: float = 0.0  # seconds; dwell the ADC before it triggers, so the
                                      # dead lead-in is skipped and more pulses fit the frame.
                                      # Measured 1:1 (100/200/300/400 ns -> 100/200/300/400 ns
                                      # frame shift); pulse_regions_ns uses a robust median/MAD
                                      # baseline so a pulse shifted near t=0 stays detectable.
    run_delay: int = 200_000
    tail_trim_samples: int = 25       # see KI_001 note in loopback_multichannel.py
    trace_channels: tuple = (0, 1, 2, 3)   # which channels get a raw trace capture. Each costs
                                      # one cmaccModule and the firmware has only 4, so a case
                                      # that also runs MeasurableResonator.measure() (its own
                                      # cmacc) must give one up -- otherwise compile fails with
                                      # "instance limit reached for cmaccModule". The measure_*
                                      # cases trace only the channels they read: ch0 (which is
                                      # cabled to the resonator's stimulus, so it records the
                                      # readout pulse) and ch2 (the reference marker).
    use_dummy_channel: bool = True    # removes the KI_001 capture skew; keep True
    dummy_memory_name: str = "dummy_trace"
    figsize: tuple = None
    yaml_path: str = None

    def main(self):
        if self.case not in CASES:
            raise ValueError(f"unknown case {self.case!r}; expected one of {CASES}")

        # rb_stream_uniform is rb_stream with the 8 final gates forced to one amplitude, so the
        # merged back-to-back region's edge is the true first gate (see README). Set it here,
        # before pulse allocation reads rb_final_gate, and treat it as rb_stream below.
        if self.case == "rb_stream_uniform" and not self.rb_final_gate:
            self.rb_final_gate = "rb_gate_hi"

        self._labels = ["ch0", "ch1", "ch2", "ch3"]
        stimuli = [self.io(f"stimulus{i}") for i in range(4)]
        captures = [self.io(f"capture{i}") for i in range(4)]
        dummy_cap = self.io("capture_dummy") if self.use_dummy_channel else None

        # Lengthen the ADC capture window past what the yaml declares, so a long rb_stream
        # train fits. get_config reads self._config, so setting it here (before compile,
        # before any get_waveform_memory) resizes both the capture memory and the stream.
        if self.capture_length_override:
            for cap in captures + ([dummy_cap] if dummy_cap else []):
                mem = self.dummy_memory_name if cap is dummy_cap else self.capture_memory_name
                cap.get_config("memories", mem)["length"] = float(self.capture_length_override)

        capture_length = captures[0].get_config("memories", self.capture_memory_name, "length")

        # batch_resync: the dwell that waits for the last queued pulse, sized flat + ramp,
        # exactly as the SWAP runtime computes bs_pulse_length.
        resync_dwell = (stimuli[0].get_config("pulses", self.stimulus_pulse_name, "flat")
                        + stimuli[0].get_config("pulses", self.stimulus_pulse_name, "ramp"))

        # rb_stream: resolve the gate pattern (indices into rb_gates) and its length.
        rb_pattern = list(self.rb_pattern)
        if self.rb_num_pulses:
            rb_pattern = [rb_pattern[k % len(rb_pattern)]
                          for k in range(int(self.rb_num_pulses))]
        rb_count = len(rb_pattern)

        # Cache-backed dwell length for the register_dwell case. The visualizer resolves
        # REG0 by finding the cache word it is loaded from and reading that word out of the
        # per-point snapshot, so this checks the auto-resolve against a measured interval
        # rather than only against the cache itself.
        cache = self.acadia.CacheArray(shape=(1,), dtype=np.dtype("<i4"))

        # rb_stream: the randomized-benchmarking idiom. A DSP holds a bus POINTER into a
        # command cache and walks it, issuing each cached word straight to the DMA, until
        # it reaches a final pointer held in a register. The pulse COUNT lives in
        # rb_num_cache and each played command in rb_cmd_cache -- both captured per point,
        # so both are recoverable off-hardware even though the tracer today drops them.
        rb_cmd_cache = rb_num_cache = None
        if self.case in STREAM_CASES:
            rb_cmd_cache = self.acadia.CacheArray(shape=int(rb_count),
                                                  dtype=np.dtype("<i4"))
            rb_num_cache = self.acadia.CacheArray(shape=1, dtype=np.dtype("<i4"))

        # Only the channels actually captured. A uniform group that is declared and never
        # written saves a 0-byte file that DataManager cannot parse -- here it aborted the deploy
        # outright with "ValueError: Error loading number of groups" once trace_channels stopped
        # covering all four (a MISSING group loads fine; an EMPTY one does not). Same failure the
        # tomography runtimes hit with their confusion groups.
        for idx, label in enumerate(self._labels):
            if idx in self.trace_channels:
                self.data.add_group(f"trace_{label}", uniform=True)
        self.data.add_group("t_data", uniform=False)

        def dac_sequence(a: Acadia):
            """The part under test. Everything else is identical across cases."""
            pulse = self.stimulus_pulse_name

            if self.case == "single":
                with a.channel_synchronizer():
                    for stim in stimuli:
                        stim.schedule_pulse(pulse)

            elif self.case == "two_same_block":
                # back-to-back inside one block: no trigger/poll in between, so the
                # visualizer predicts the second pulse starts exactly at pulse_length
                with a.channel_synchronizer():
                    for stim in stimuli:
                        stim.schedule_pulse(pulse)
                        stim.schedule_pulse(pulse)

            elif self.case in ("two_blocks", "two_blocks_1ch", "two_blocks_2ch",
                               "two_blocks_3ch"):
                second = {"two_blocks": stimuli,
                          "two_blocks_1ch": stimuli[:1],
                          "two_blocks_2ch": stimuli[:2],
                          "two_blocks_3ch": stimuli[:3]}[self.case]
                with a.channel_synchronizer():
                    for stim in stimuli:
                        stim.schedule_pulse(pulse)
                with a.channel_synchronizer():
                    for stim in second:
                        stim.schedule_pulse(pulse)

            elif self.case in ("three_blocks", "four_blocks"):
                for _ in range(3 if self.case == "three_blocks" else 4):
                    with a.channel_synchronizer():
                        for stim in stimuli:
                            stim.schedule_pulse(pulse)

            elif self.case == "batch_nonblocking":
                # the Berry / beam-splitting batching idiom: commands queue in the DMA
                # FIFO and should play with no gap between them
                with a.channel_synchronizer(block=False):
                    for stim in stimuli:
                        for _ in range(3):
                            stim.schedule_pulse(pulse)

            elif self.case == "dwell_between":
                with a.channel_synchronizer():
                    for stim in stimuli:
                        stim.schedule_pulse(pulse)
                        stim.dwell(self.dwell_length)
                        stim.schedule_pulse(pulse)

            elif self.case in ("shape", "detune_pair", "phase_pair"):
                # Stage 2: each block plays one long-ramp pulse, so every pulse is its own
                # above-threshold region and its envelope / IQ can be fitted separately.
                names = {"shape": ["long_ramp_pulse"],
                         "detune_pair": ["detune_10MHz", "detune_25MHz"],
                         "phase_pair": ["long_ramp_pulse", "phase_half_pi"]}[self.case]
                for name in names:
                    with a.channel_synchronizer():
                        for stim in stimuli:
                            stim.schedule_pulse(name)

            elif self.case in ("test_true", "test_false", "test_true_nospec",
                               "test_false_nospec"):
                # Conditional middle pulse. The condition is a Register loaded from the
                # cache and compared to 0, so it is BOTH deterministic (we set the cache)
                # and evaluable by the visualizer (it captures the cache per sweep point).
                # test_register_value picks the outcome; *_nospec flips `speculation`,
                # which decides whether the body is inline or out-of-line and therefore
                # which arm pays for a taken branch.
                counter = a.sequencer().Register()
                counter.load(cache[0])
                speculation = not self.case.endswith("_nospec")
                with a.channel_synchronizer():
                    for stim in stimuli:
                        stim.schedule_pulse(pulse)
                with a.sequencer().test(counter == 0, speculation=speculation):
                    with a.channel_synchronizer():
                        for stim in stimuli:
                            stim.schedule_pulse(pulse)
                with a.channel_synchronizer():
                    for stim in stimuli:
                        stim.schedule_pulse(pulse)

            elif self.case == "loop_2_double":
                # Same loop-back edge but a body with twice the DMA pushes, so `issue`
                # changes while any taken-branch penalty should not. Discriminates a
                # constant branch cost from a miscount, the same way two_blocks_Nch did
                # for the forward edge. loop(2) not (3): 3 iterations would overrun the
                # 1.2 us capture window.
                with a.sequencer().loop(2):
                    with a.channel_synchronizer():
                        for stim in stimuli:
                            stim.schedule_pulse(pulse)
                            stim.schedule_pulse(pulse)

            elif self.case in ("loop_2", "loop_3"):
                # Stage 3: a deterministic loop count, so the expected unrolled timeline is
                # exact. The body runs once in Python at compile time but N times on the
                # sequencer, which is precisely what the visualizer has to model.
                count = 2 if self.case == "loop_2" else 3
                with a.sequencer().loop(count):
                    with a.channel_synchronizer():
                        for stim in stimuli:
                            stim.schedule_pulse(pulse)

            elif self.case in ("nested_cool_2x2", "nested_cool_3x2"):
                # THE COOLING SHAPE, which is what CustomRuntime.cool_modes compiles and what no
                # previous case covered: an OUTER counter loop whose body is {a pulse, then an
                # INNER counter loop of its own}. 53 of the qudit runtimes build this via
                # cool_modes/cool_qubits, and getting the nesting wrong is not hypothetical --
                # SequenceTrace.execution_plan used to group a loop body by the blocks at the
                # loop's OWN depth, so the outer body collapsed to just the swap and the inner
                # cooling was drawn once, after all the swaps, instead of interleaved. Fixed
                # 2026-08-11; this case is the hardware check of that fix.
                #
                # Structure per outer pass:   ch0 swap-marker, then N_inner x (ch1 cool-marker)
                # so BOTH the outer period (ch0 interval) and the inner period (ch1 interval) are
                # directly measurable within their own channel, and the two are independent.
                # DSP counters + pulse_cep() are the same primitives _cool_single_qubit uses.
                n_outer = 3 if self.case == "nested_cool_3x2" else 2
                n_inner = 2
                swap_ch, cool_ch = stimuli[0], stimuli[1]
                outer = a.sequencer().DSP()
                inner = a.sequencer().DSP()
                outer.load(0)
                outer.configure(mode="P+1", dsp_cep="reset")
                with a.sequencer().repeat_until(outer == n_outer):
                    with a.channel_synchronizer():
                        swap_ch.schedule_pulse(pulse)          # the "mode swap"
                    inner.load(0)
                    inner.configure(mode="P+1", dsp_cep="reset")
                    with a.sequencer().repeat_until(inner == n_inner):
                        with a.channel_synchronizer():
                            cool_ch.schedule_pulse(pulse)      # the "cool round"
                        inner.pulse_cep()
                    outer.pulse_cep()

            elif self.case in ("test_in_loop_true", "test_in_loop_false"):
                # A `test` nested INSIDE a counter loop. BOTH arms are cases, because
                # build_runtime picks the register from the case NAME ("true" in the name -> 0,
                # which makes REG0 == 0 hold): _true runs the conditional body on every pass,
                # _false skips it on every pass, and the loop must still unroll either way -- the other half of the cooling shape
                # (_cool_single_qubit puts repeat_until(feedback == target) inside the round
                # loop, and 46 runtimes use sequencer().test()). Checks that an unrolled loop
                # and a conditional compose: the tracer must apply the loop count to a body that
                # itself contains a branch, and the branch decision must not be re-evaluated
                # per pass in a way that changes the count.
                sel = a.sequencer().Register()
                sel.load(cache[0])                              # test_register_value
                counter = a.sequencer().DSP()
                counter.load(0)
                counter.configure(mode="P+1", dsp_cep="reset")
                with a.sequencer().repeat_until(counter == 2):
                    with a.channel_synchronizer():
                        stimuli[0].schedule_pulse(pulse)        # unconditional, every pass
                    with a.sequencer().test(sel == 0):
                        with a.channel_synchronizer():
                            stimuli[1].schedule_pulse(pulse)    # only when the register says so
                    counter.pulse_cep()

            elif self.case in READOUT_CASES:
                # THE READOUT PATH. MeasurableResonator.measure() is used by 94 of the qudit
                # runtimes and by no other loopback case -- every other case captures a raw trace
                # via stream_cmacc, which is a different command shape (measure() schedules the
                # readout pulse on the stimulus AND a capture_cmacc with a window on the capture).
                #
                # What is physically measured here is the readout PULSE: the resonator's stimulus
                # is ch0's DAC, which is cabled to ch0's ADC, so the trace records it and the
                # marker->readout->marker intervals are measurable within ch0. The resonator's own
                # capture goes to ADC1 (capture_dummy), so it never contends with the four trace
                # captures -- build_runtime sets use_dummy_channel=False for these cases.
                resonator = self._resonator            # built before attach(); see main()
                # Everything must land on a TRACED channel or it cannot be measured: these cases
                # trace ch0 (the resonator's stimulus, so it records the readout pulses) and ch1
                # (markers, the inter-round swap and the conditional reset). ch2/ch3 are not
                # captured here -- their cmacc modules are what the resonator needs.
                marker = stimuli[1]

                if self.case == "measure_multi":
                    # SIMULTANEOUS multi-resonator readout: two measure() calls inside ONE
                    # channel_synchronizer, the shape readout_confusion and the joint dual-rail
                    # readouts use. Distinct from measure_two_rounds, which is two readouts in
                    # SEQUENCE on one line; here both fire in the same barrier, so the barrier
                    # has to pad two independent capture_cmacc command chains against each other.
                    # Both readout pulses land on traced DACs (ch0 and ch1), so both are measured.
                    second = self._resonator2          # on stimuli[1]; built before attach()
                    with a.channel_synchronizer():
                        stimuli[0].schedule_pulse(pulse)
                        marker.schedule_pulse(pulse)
                    with a.channel_synchronizer():
                        resonator.measure(pulse, "readout_accumulated", "boxcar")
                        second.measure(pulse, "readout_accumulated", "boxcar")
                    with a.channel_synchronizer():
                        stimuli[0].schedule_pulse(pulse)
                        marker.schedule_pulse(pulse)

                elif self.case == "loop_with_measure":
                    # A readout INSIDE a counter loop -- repeated single-shot readout
                    # (qubit_repeated_readout, cavity_temperature, the tomography confusion
                    # rounds). Composes the loop unroll with measure()'s command shape, which no
                    # other case did. Must live in this branch: `resonator` only exists here.
                    counter = a.sequencer().DSP()
                    counter.load(0)
                    counter.configure(mode="P+1", dsp_cep="reset")
                    with a.sequencer().repeat_until(counter == int(self.loop_count)):
                        with a.channel_synchronizer():
                            marker.schedule_pulse(pulse)
                        with a.channel_synchronizer():
                            resonator.measure(pulse, "readout_accumulated", "boxcar")
                        counter.pulse_cep()

                elif self.case == "measure_trace_case":
                    # measure_trace() instead of measure(): a raw windowed TRACE capture rather
                    # than a CMACC accumulation. Used by readout_window_calibration (the runtime
                    # that calibrates the kernel every other readout depends on), and a different
                    # capture command shape from both measure() and the stream_cmacc traces.
                    with a.channel_synchronizer():
                        marker.schedule_pulse(pulse)
                    with a.channel_synchronizer():
                        resonator.measure_trace(pulse, "dummy_trace")
                    with a.channel_synchronizer():
                        marker.schedule_pulse(pulse)

                elif self.case == "measure_readout":
                    # marker | readout | marker  -- the plain single-shot readout shape
                    with a.channel_synchronizer():
                        stimuli[0].schedule_pulse(pulse)
                        marker.schedule_pulse(pulse)
                    with a.channel_synchronizer():
                        resonator.measure(pulse, "readout_accumulated", "boxcar")
                    with a.channel_synchronizer():
                        stimuli[0].schedule_pulse(pulse)
                        marker.schedule_pulse(pulse)

                elif self.case == "measure_two_rounds":
                    # TWO measures on ONE line with DISTINCT capture memories -- the two-round
                    # dual-rail readout that 36 runtimes build (rule capture_memory_per_readout:
                    # N readouts need N memories or the second overwrites the first). The second
                    # memory is a duplicate, i.e. an OBJECT not a name, so its window must carry
                    # real kernel data -- hence `matched` (see the config comment and failure_019).
                    mem2 = self._resonator_mem2        # duplicated before attach()
                    with a.channel_synchronizer():
                        marker.schedule_pulse(pulse)               # t0 reference on ch2
                    with a.channel_synchronizer():
                        resonator.measure(pulse, "readout_accumulated", "matched")
                    with a.channel_synchronizer():
                        marker.schedule_pulse(pulse)               # the inter-round "swap"
                    with a.channel_synchronizer():
                        resonator.measure(pulse, mem2, "matched")

                else:   # feedback_reset
                    # measure -> get_measurement() -> test(quadrant) -> conditional pulse: the
                    # active-reset shape 24 runtimes use (and what _cool_single_qubit does inside
                    # its round loop). The branch decision comes from a real CMACC result, so the
                    # tracer cannot resolve it statically -- it must report the block in
                    # assumed_paths rather than silently drawing one arm as certain.
                    feedback = a.sequencer().Register()
                    with a.channel_synchronizer():
                        marker.schedule_pulse(pulse)
                    with a.channel_synchronizer():
                        resonator.measure(pulse, "readout_accumulated", "boxcar")
                    feedback.load(resonator.get_measurement(classifier="quadrant"))
                    resonator.wait_until_measurement_done()
                    with a.sequencer().test(feedback == getattr(a, "CMACC_QUADRANT_1")):
                        with a.channel_synchronizer():
                            marker.schedule_pulse(pulse)          # the conditional reset pi
                    # A SECOND unconditional readout, so ch0 always has a comparable interval
                    # whichever way the branch goes -- the branch itself depends on a live CMACC
                    # result, so its arm is legitimately unpredictable (reported in assumed_paths).
                    with a.channel_synchronizer():
                        resonator.measure(pulse, "readout_accumulated", "boxcar")

            elif self.case == "loop_n":
                # loop_2/loop_3 with the count as a PARAMETER, so the unrolled timeline can be
                # swept instead of spot-checked. The back-edge gap must stay constant per pass and
                # the period must be exactly linear in the count.
                with a.sequencer().loop(int(self.loop_count)):
                    with a.channel_synchronizer():
                        for stim in stimuli:
                            stim.schedule_pulse(pulse)

            elif self.case == "nested_cool_n":
                # nested_cool with BOTH counts parametric -- sweeps the cooling shape over the
                # (outer, inner) grid instead of the two hand-written 2x2 / 3x2 points.
                swap_ch, cool_ch = stimuli[0], stimuli[1]
                outer, inner = a.sequencer().DSP(), a.sequencer().DSP()
                outer.load(0)
                outer.configure(mode="P+1", dsp_cep="reset")
                with a.sequencer().repeat_until(outer == int(self.loop_count)):
                    with a.channel_synchronizer():
                        swap_ch.schedule_pulse(pulse)
                    inner.load(0)
                    inner.configure(mode="P+1", dsp_cep="reset")
                    with a.sequencer().repeat_until(inner == int(self.inner_loop_count)):
                        with a.channel_synchronizer():
                            cool_ch.schedule_pulse(pulse)
                        inner.pulse_cep()
                    outer.pulse_cep()

            elif self.case == "blocks_n":
                # two_blocks/three_blocks/four_blocks with the count as a parameter: the
                # per-boundary gap must compound exactly linearly, which is the strongest test of
                # the boundary model because the error would grow with n if a term were wrong.
                for _ in range(max(int(self.n_blocks), 1)):
                    with a.channel_synchronizer():
                        for stim in stimuli:
                            stim.schedule_pulse(pulse)

            elif self.case == "dwell_n":
                # dwell_between with the dwell as a parameter -- dwell() must honour its argument
                # exactly across the whole range, not just at 200 ns.
                with a.channel_synchronizer():
                    for stim in stimuli:
                        stim.schedule_pulse(pulse)
                        stim.dwell(self.dwell_length)
                        stim.schedule_pulse(pulse)

            elif self.case == "batch_in_loop":
                # A block=False batch inside a counter loop: the FIFO drain and the loop back-edge
                # interact, and the drain releases when the last command is PULLED (not played).
                # dualrail_rb / xeb stream batches inside their circuit loops.
                counter = a.sequencer().DSP()
                counter.load(0)
                counter.configure(mode="P+1", dsp_cep="reset")
                with a.sequencer().repeat_until(counter == int(self.loop_count)):
                    with a.channel_synchronizer():
                        stimuli[0].schedule_pulse(pulse)          # marker for this pass
                    with a.channel_synchronizer(block=False):
                        for _ in range(3):
                            stimuli[1].schedule_pulse(pulse)
                    with a.sequencer().repeat_until(
                            a.channel_is_fifo_empty(stimuli[1].channel)):
                        pass
                    counter.pulse_cep()

            elif self.case == "test_then_batch":
                # A conditional followed by a batch in the same pass: the skip branch and the
                # non-blocking batch share a boundary, so the branch penalty and the "no gap for a
                # non-blocking block" rule must both apply and not double-count.
                sel = a.sequencer().Register()
                sel.load(cache[0])
                with a.channel_synchronizer():
                    stimuli[0].schedule_pulse(pulse)
                with a.sequencer().test(sel == 0):
                    with a.channel_synchronizer():
                        stimuli[2].schedule_pulse(pulse)
                with a.channel_synchronizer(block=False):
                    for _ in range(3):
                        stimuli[1].schedule_pulse(pulse)
                with a.sequencer().repeat_until(a.channel_is_fifo_empty(stimuli[1].channel)):
                    pass
                with a.channel_synchronizer():
                    stimuli[0].schedule_pulse(pulse)

            elif self.case == "register_stretch":
                # A register-driven stretch with NO loop around it. This isolates the stretch
                # length model from the loop model: stretch_in_loop failed by exactly 1 cycle per
                # iteration, and a loop simply multiplies whatever a single pass gets wrong, so a
                # per-iteration error and a per-stretch error are indistinguishable there. Here
                # the interval marker->marker spans exactly ONE stretch, so any error is the
                # stretch's own.
                length_reg = a.sequencer().Register()
                length_reg.load(cache[0])
                with a.channel_synchronizer():
                    stimuli[0].schedule_pulse(pulse)
                with a.channel_synchronizer():
                    stimuli[1].schedule_pulse("stretch_pulse", stretch_length=length_reg)
                with a.channel_synchronizer():
                    stimuli[0].schedule_pulse(pulse)

            elif self.case == "batch_drain_twice":
                # Two non-blocking batches, each drained by repeat_until(fifo_empty), with NO loop.
                # The isolating counterpart to batch_in_loop for the same reason as above: the
                # existing batch_* cases do have drains, but their measured intervals do not SPAN
                # a drain release, so a wrong release point never showed up. Here the ch0 markers
                # bracket each drain, so interval 1 and interval 2 each contain exactly one.
                for _ in range(2):
                    with a.channel_synchronizer():
                        stimuli[0].schedule_pulse(pulse)
                    with a.channel_synchronizer(block=False):
                        for _ in range(3):
                            stimuli[1].schedule_pulse(pulse)
                    with a.sequencer().repeat_until(
                            a.channel_is_fifo_empty(stimuli[1].channel)):
                        pass
                with a.channel_synchronizer():
                    stimuli[0].schedule_pulse(pulse)

            elif self.case in ("batch_drain_almost", "batch_in_loop_almost"):
                # THE DRAIN PRIMITIVE THE RUNTIMES ACTUALLY USE. An audit of all 121 qudit-branch
                # runtimes found `channel_is_fifo_empty` in ZERO of them and
                # `channel_is_fifo_almost_empty` in all 7 that stream (dualrail_rb, xeb_1DR/2DR/3DR,
                # beamsplitter_amp_detune_calibration): they refill the FIFO while it still holds
                # commands, so the release level -- and therefore the release TIME -- is different
                # from a drain-to-empty. Every batch case here used the empty variant, so the
                # variant production depends on was reached only via the stream cases.
                #
                # batch_in_loop_almost is dualrail_rb's exact shape: drain, issue, increment, repeat.
                target = stimuli[1]
                # Descriptor COUNT is scannable, because the release level depends on it: an
                # almost_empty drain asserts with one word left, so a 2-descriptor batch releases
                # at its own start while a 5-descriptor batch releases three descriptors in. The
                # count was hard-coded at 3, which made `--scan batch_drain_almost:...` vacuous --
                # every point deployed the identical sequence and reported a reassuring 0.10 ns.
                n_batch = max(int(self.batch_resync_pulses), 2)
                if self.case == "batch_in_loop_almost":
                    counter = a.sequencer().DSP()
                    counter.load(0)
                    counter.configure(mode="P+1", dsp_cep="reset")
                    with a.sequencer().repeat_until(counter == int(self.loop_count)):
                        with a.channel_synchronizer():
                            stimuli[0].schedule_pulse(pulse)
                        with a.channel_synchronizer(block=False):
                            for _ in range(n_batch):
                                target.schedule_pulse(pulse)
                        with a.sequencer().repeat_until(
                                a.channel_is_fifo_almost_empty(target.channel)):
                            pass
                        counter.pulse_cep()
                else:
                    for _ in range(2):
                        with a.channel_synchronizer():
                            stimuli[0].schedule_pulse(pulse)
                        with a.channel_synchronizer(block=False):
                            for _ in range(n_batch):
                                target.schedule_pulse(pulse)
                        with a.sequencer().repeat_until(
                                a.channel_is_fifo_almost_empty(target.channel)):
                            pass
                    with a.channel_synchronizer():
                        stimuli[0].schedule_pulse(pulse)

            elif self.case == "stretch_in_loop":
                # A register-stretched pulse inside a loop: an indeterminate length and a loop
                # back-edge together (the chevron/rate-match shape run repeatedly).
                length_reg = a.sequencer().Register()
                length_reg.load(cache[0])
                counter = a.sequencer().DSP()
                counter.load(0)
                counter.configure(mode="P+1", dsp_cep="reset")
                with a.sequencer().repeat_until(counter == int(self.loop_count)):
                    with a.channel_synchronizer():
                        stimuli[0].schedule_pulse(pulse)
                    with a.channel_synchronizer():
                        stimuli[1].schedule_pulse("stretch_pulse", stretch_length=length_reg)
                    counter.pulse_cep()

            elif self.case == "three_deep_nest":
                # THREE levels of control flow, which is the deepest any qudit runtime reaches
                # (cool_modes: mode loop -> qubit-round loop -> active-reset loop). Each level is a
                # counter loop so every period is deterministic and measurable on its own channel.
                l1, l2, l3 = (a.sequencer().DSP() for _ in range(3))
                for dsp in (l1, l2, l3):
                    dsp.load(0)
                    dsp.configure(mode="P+1", dsp_cep="reset")
                with a.sequencer().repeat_until(l1 == 2):
                    with a.channel_synchronizer():
                        stimuli[0].schedule_pulse(pulse)
                    l2.load(0)
                    with a.sequencer().repeat_until(l2 == 2):
                        with a.channel_synchronizer():
                            stimuli[1].schedule_pulse(pulse)
                        l3.load(0)
                        with a.sequencer().repeat_until(l3 == 2):
                            with a.channel_synchronizer():
                                stimuli[2].schedule_pulse(pulse)
                            l3.pulse_cep()
                        l2.pulse_cep()
                    l1.pulse_cep()

            elif self.case in ("repeat_until_op", "repeat_until_count_n"):
                # REPEAT_UNTIL, exhaustively. The tracer resolves exactly one form -- a DSP
                # counter loaded 0, incremented +1 per pass, compared `== target` -- and draws a
                # single data-dependent pass for everything else. Both halves of that claim need
                # measuring: the resolved form must be right at every count (including the
                # degenerate 0 and 1), and the UNRESOLVED forms must be honestly unresolved
                # rather than confidently wrong.
                #
                # repeat_until_op sweeps the comparison operator with the same counter and
                # target, so the only thing that changes is whether the tracer can resolve it.
                counter = a.sequencer().DSP()
                counter.load(0)
                counter.configure(mode="P+1", dsp_cep="reset")
                target = int(self.loop_count)
                # Only `==` and `!=` are legal here. acadia rejects every ordered comparison
                # ("Less-than comparisons can only check x < 0 or 0 <= x"), and `<=` fails with
                # a message that does not mention comparisons at all. Measured, not assumed --
                # see ACADIA_FINDINGS.md.
                condition = {"==": counter == target,
                             "!=": counter != target}[self.repeat_operator]
                with a.channel_synchronizer():
                    stimuli[0].schedule_pulse(pulse)
                with a.sequencer().repeat_until(condition):
                    with a.channel_synchronizer():
                        stimuli[1].schedule_pulse(pulse)
                    counter.pulse_cep()
                with a.channel_synchronizer():
                    stimuli[0].schedule_pulse(pulse)

            elif self.case == "test_nested":
                # A test INSIDE a test. Nothing in the runtimes nests conditionals, but the
                # tracer's context walk is depth-based and a skipped OUTER arm must drop the
                # inner one with it -- a body that survives its own parent being skipped would
                # be drawn out of nothing.
                outer = a.sequencer().Register()
                outer.load(cache[0])
                inner = a.sequencer().Register()
                inner.load(cache[0])
                with a.channel_synchronizer():
                    stimuli[0].schedule_pulse(pulse)
                with a.sequencer().test(outer == int(self.test_register_value)):
                    with a.channel_synchronizer():
                        stimuli[1].schedule_pulse(pulse)
                    with a.sequencer().test(inner == int(self.test_register_value)):
                        with a.channel_synchronizer():
                            stimuli[2].schedule_pulse(pulse)
                    with a.channel_synchronizer():
                        stimuli[1].schedule_pulse(pulse)
                with a.channel_synchronizer():
                    stimuli[0].schedule_pulse(pulse)

            elif self.case == "test_in_counter_loop":
                # A conditional inside a counter loop: the branch cost is paid on EVERY pass, so
                # a per-pass miscount compounds -- the shape that exposed the drain-in-loop bug.
                counter = a.sequencer().DSP()
                counter.load(0)
                counter.configure(mode="P+1", dsp_cep="reset")
                sel = a.sequencer().Register()
                sel.load(cache[0])
                with a.sequencer().repeat_until(counter == int(self.loop_count)):
                    with a.channel_synchronizer():
                        stimuli[0].schedule_pulse(pulse)
                    with a.sequencer().test(sel == int(self.test_register_value)):
                        with a.channel_synchronizer():
                            stimuli[1].schedule_pulse(pulse)
                    counter.pulse_cep()
                with a.channel_synchronizer():
                    stimuli[0].schedule_pulse(pulse)

            elif self.case == "counter_loop_in_test":
                # The mirror image: a whole counter loop inside a conditional arm. If the arm is
                # skipped the loop must vanish entirely, not run once.
                sel = a.sequencer().Register()
                sel.load(cache[0])
                counter = a.sequencer().DSP()
                counter.load(0)
                counter.configure(mode="P+1", dsp_cep="reset")
                with a.channel_synchronizer():
                    stimuli[0].schedule_pulse(pulse)
                with a.sequencer().test(sel == int(self.test_register_value)):
                    with a.sequencer().repeat_until(counter == int(self.loop_count)):
                        with a.channel_synchronizer():
                            stimuli[1].schedule_pulse(pulse)
                        counter.pulse_cep()
                with a.channel_synchronizer():
                    stimuli[0].schedule_pulse(pulse)

            elif self.case == "three_deep_nest_reconfig":
                # three_deep_nest, but each counter is RE-CONFIGURED (not merely reloaded) every
                # time its enclosing loop re-enters -- the pattern nested_cool_n uses and that
                # works. three_deep_nest configures all three DSPs once up front and then only
                # reloads them, and that version never returns from the board: repeated
                # "Timeout occurred waiting for line", i.e. a loop that never terminates.
                #
                # The pair isolates ONE difference, so whichever runs tells us whether a counter's
                # `configure` survives re-entry or has to be re-issued. Nothing in the qudit
                # runtimes reaches three counter levels (cool_modes uses two counters plus a
                # `test`), so this is about acadia's limits rather than about a shipped sequence.
                l1, l2, l3 = (a.sequencer().DSP() for _ in range(3))
                l1.load(0)
                l1.configure(mode="P+1", dsp_cep="reset")
                with a.sequencer().repeat_until(l1 == 2):
                    with a.channel_synchronizer():
                        stimuli[0].schedule_pulse(pulse)
                    l2.load(0)
                    l2.configure(mode="P+1", dsp_cep="reset")
                    with a.sequencer().repeat_until(l2 == 2):
                        with a.channel_synchronizer():
                            stimuli[1].schedule_pulse(pulse)
                        l3.load(0)
                        l3.configure(mode="P+1", dsp_cep="reset")
                        with a.sequencer().repeat_until(l3 == 2):
                            with a.channel_synchronizer():
                                stimuli[2].schedule_pulse(pulse)
                            l3.pulse_cep()
                        l2.pulse_cep()
                    l1.pulse_cep()

            elif self.case in ("random_seq", "pair_seq"):
                # GENERATED SEQUENCES. Everything above is a shape someone thought to write down;
                # these compose the primitive alphabet mechanically, so coverage stops depending
                # on imagination. Two modes share one emitter:
                #
                # * ``random_seq`` -- a seeded random composition (--fuzz-steps / fuzz_seed).
                #   Covers long-range interactions no hand-written case would think to try.
                # * ``pair_seq``   -- ONE ordered pair A-then-B (pair_a, pair_b). Enumerating the
                #   pairs covers every primitive ADJACENCY exhaustively rather than probabilistically:
                #   a random walk only *probably* produces "counter_loop immediately after a
                #   drain", while the enumeration guarantees it. Scheduling bugs live exactly at
                #   these joins -- both bugs found here (the almost_empty release level and the
                #   drain/back-edge cost) are properties of what a construct is ADJACENT to.
                #
                # The alphabet is every primitive the qudit runtimes actually use and that the
                # timing model has a term for: a blocking block, a non-blocking batch drained by
                # either FIFO primitive, a literal dwell, a register dwell, a deterministic loop, a
                # counter loop, a conditional (both arms), and a register-stretched pulse.
                #
                # Every step is bracketed by a marker pulse on ch0, so each step's duration is a
                # within-channel interval on one channel -- the metric stays valid whatever the
                # generator produced.
                import random as _random
                rng = _random.Random(int(self.fuzz_seed))
                cycles = self.acadia.seconds_to_cycles(self.register_stretch)
                length_reg = a.sequencer().Register()
                length_reg.load(cache[0])
                sel = a.sequencer().Register()
                sel.load(cache[0])

                def emit(kind, chans):
                    """Lay down one primitive on `chans`. Shared by both generated cases."""
                    if kind == "block":
                        for _ in range(rng.randint(1, 3)):
                            with a.channel_synchronizer():
                                for stim in chans:
                                    stim.schedule_pulse(pulse)
                    elif kind in ("batch", "batch_almost"):
                        # both drain senses -- they release at different FIFO levels
                        target = chans[0]
                        with a.channel_synchronizer(block=False):
                            for _ in range(rng.randint(2, 5)):
                                target.schedule_pulse(pulse)
                        cond = (a.channel_is_fifo_almost_empty(target.channel)
                                if kind == "batch_almost"
                                else a.channel_is_fifo_empty(target.channel))
                        with a.sequencer().repeat_until(cond):
                            pass
                    elif kind == "dwell":
                        with a.channel_synchronizer():
                            for stim in chans:
                                stim.schedule_pulse(pulse)
                                stim.dwell(rng.choice((50e-9, 100e-9, 250e-9, 500e-9)))
                                stim.schedule_pulse(pulse)
                    elif kind == "reg_dwell":
                        with a.channel_synchronizer():
                            for stim in chans:
                                stim.schedule_pulse(pulse)
                                stim.dwell(length_reg)
                                stim.schedule_pulse(pulse)
                    elif kind == "loop":
                        with a.sequencer().loop(rng.randint(2, 4)):
                            with a.channel_synchronizer():
                                for stim in chans:
                                    stim.schedule_pulse(pulse)
                    elif kind == "counter_loop":
                        dsp = a.sequencer().DSP()
                        dsp.load(0)
                        dsp.configure(mode="P+1", dsp_cep="reset")
                        with a.sequencer().repeat_until(dsp == rng.randint(2, 4)):
                            with a.channel_synchronizer():
                                for stim in chans:
                                    stim.schedule_pulse(pulse)
                            dsp.pulse_cep()
                    elif kind in ("test_taken", "test_skipped"):
                        # Compare against the cache value itself (TAKEN) or a value it cannot
                        # hold (SKIPPED). Both arms are separate primitives so the enumeration
                        # covers each next to everything else.
                        want = cycles if kind == "test_taken" else cycles + 7
                        with a.sequencer().test(sel == want):
                            with a.channel_synchronizer():
                                for stim in chans:
                                    stim.schedule_pulse(pulse)
                    elif kind == "stretch":
                        # stretch_pulse, deliberately. A same-ramp stretchable pulse was tried to
                        # remove the ramp-mismatch systematic and made things WORSE (370 ns on the
                        # board): shortening ramp/flat changes the half/hold/half stretch geometry
                        # itself, so the cure was bigger than the disease. The systematic is
                        # bounded and understood instead -- see KNOWN_SYSTEMATIC.
                        with a.channel_synchronizer():
                            chans[0].schedule_pulse("stretch_pulse", stretch_length=length_reg)

                def marker():
                    with a.channel_synchronizer():
                        stimuli[0].schedule_pulse(pulse)

                if self.case == "pair_seq":
                    # marker | A | marker | B | [marker | C] | marker -- each primitive's span is
                    # an interval on ch0, and the joins between them are what is under test.
                    # With pair_c set this is a TRIPLE: a pair only ever puts a construct next to
                    # one neighbour, so it cannot catch anything that needs a particular
                    # predecessor AND successor -- the batch_almost bug, for instance, only
                    # surfaced once a marker block sat between the drain and the next batch.
                    for kind in (k for k in (self.pair_a, self.pair_b, self.pair_c) if k):
                        marker()
                        emit(kind, list(stimuli[1:3]))
                    marker()
                else:
                    # A generated sequence containing a stretchable pulse measures a 100 ns ramp
                    # against 20 ns markers, which moves the 50%-of-power crossing and costs
                    # ~25 ns of apparent error. That is a property of the MEASUREMENT, so a run
                    # meant to test the timing MODEL is better off without it -- the stretch
                    # length model is covered by the same-pulse cases instead.
                    alphabet = ([p for p in PRIMITIVES if p != "stretch"]
                                if self.exclude_stretch else list(PRIMITIVES))
                    for _ in range(max(int(self.fuzz_steps), 1)):
                        marker()
                        emit(rng.choice(alphabet), rng.sample(stimuli[1:], rng.randint(1, 3)))
                    marker()

            elif self.case == "register_dwell":
                # dwell length comes from a Register loaded out of the cache, so it is
                # indeterminate at compile time -- the same shape as a T1/T2 delay sweep
                counter = a.sequencer().Register()
                counter.load(cache[0])
                with a.channel_synchronizer():
                    for stim in stimuli:
                        stim.schedule_pulse(pulse)
                        stim.dwell(counter)
                        stim.schedule_pulse(pulse)

            elif self.case == "stretch":
                # one stretchable pulse: the DMA plays first half, parks mid-waveform for
                # the stretch length, then plays the second half
                with a.channel_synchronizer():
                    for stim in stimuli:
                        stim.schedule_pulse("stretch_pulse")

            elif self.case == "stretch_then_pulse":
                # stretch followed by a normal pulse in the same block. The two merge into
                # one above-threshold region, so this only confirms the join is seamless --
                # use stretch_two_blocks to measure the stretched length itself.
                with a.channel_synchronizer():
                    for stim in stimuli:
                        stim.schedule_pulse("stretch_pulse")
                        stim.schedule_pulse(pulse)

            elif self.case == "stretch_two_blocks_same":
                # Both pulses are the SAME stretchable pulse, so the two rising edges have
                # identical ramp shape and the 50%-of-power crossing sits at the same point
                # on each -- the edge systematic cancels exactly in the interval. Mixing a
                # 50 ns ramp with a 10 ns ramp (stretch_two_blocks) does not cancel and cost
                # ~20 ns of apparent error.
                for _ in range(2):
                    with a.channel_synchronizer():
                        for stim in stimuli:
                            stim.schedule_pulse("stretch_pulse")

            elif self.case == "stretch_two_blocks":
                # Separate blocks so the two pulses are distinct regions and the rising-edge
                # interval is measurable: interval = stretched length + the (already
                # validated) inter-block gap. Rising 50%-of-plateau edges are robust to the
                # slow 50 ns ramp, unlike a width taken at a low threshold.
                with a.channel_synchronizer():
                    for stim in stimuli:
                        stim.schedule_pulse("stretch_pulse")
                with a.channel_synchronizer():
                    for stim in stimuli:
                        stim.schedule_pulse(pulse)

            elif self.case in ("rb_stream", "rb_stream_uniform"):
                # Mirror of QubitRBRuntime on ch0. Four DAC blocks matching the real RB:
                #   (0) an initial pulse                       (the initial pi)
                #   (1) the cache-pointer gate loop, last cache entry = the undo gate
                #   (2) a readout-marker block                 (stands in for the measure)
                #   (3) the 8 basic gates, one block, back-to-back
                # The loop collapses in the tracer, so blocks (2) and (3) are what get
                # dragged early -- this exercises the exact multi-block tail. The loop gates
                # are fifo-almost-empty gated (period floors at ~110 ns), while the 8 final
                # gates are pushed up front in one block (back-to-back at pulse length): the
                # contrast confirms the floor is the loop's alone.
                stim = stimuli[0]
                with a.channel_synchronizer():                       # (0) initial pulse
                    stim.schedule_pulse(self.rb_initial_pulse)

                base = a._firmware.sequencer_bus_decoder["cache"].address().value()
                pointer = a.sequencer().DSP()
                pointer.load(base + rb_cmd_cache.index)
                pointer.configure(mode="P+1", dsp_cep="reset")
                final = a.sequencer().Register()
                final.load(base + rb_cmd_cache.index + rb_num_cache[0])
                with a.sequencer().test(pointer != final):          # (1) gate loop
                    with a.sequencer().repeat_until(pointer == final):
                        with a.sequencer().repeat_until(
                                a.channel_is_fifo_almost_empty(stim.channel)):
                            pass
                        command = a.sequencer().bus_read(
                            pointer, latency=a._bus_latency("cache"))
                        with a.channel_synchronizer(block=False):
                            a.schedule_direct(stim.channel, command)
                        pointer.pulse_cep()
                with a.sequencer().repeat_until(
                        a.channel_is_fifo_almost_empty(stim.channel)):
                    pass

                with a.channel_synchronizer():                       # (2) readout marker
                    stim.schedule_pulse(self.rb_readout_pulse)
                with a.channel_synchronizer():                       # (3) the 8 basic gates
                    for idx in self.rb_final_pattern:
                        # rb_final_gate (if set) makes every final gate one shape -- uniform
                        # amplitude, so the merged region's leading edge is the true first gate
                        stim.schedule_pulse(self.rb_final_gate or self.rb_gates[idx])

            elif self.case == "batch_resync":
                # Mirror the SWAP-calibration idiom (BSAmpNcoSweepPulseNcoBlobsRuntime): N
                # pulses queued back-to-back with block=False in batches of 5, each batch
                # drained by repeat_until(channel_is_fifo_empty); then dwell(pulse_length) to
                # "wait for the last queued pulse to finish", a barrier, and the readout on
                # ANOTHER channel. With block=False the fifo empties when the last command is
                # PULLED (while the last pulse still plays), so that dwell runs concurrently
                # with the last pulse -- the question this case answers is whether the readout
                # lands right after the last pulse (dwell concurrent) or a pulse-length later
                # (the tracer draws the latter). A reference marker on the readout channel,
                # fired with the flip, makes the readout position measurable within-channel.
                bs, ro = stimuli[0], stimuli[1]
                with a.channel_synchronizer():
                    bs.schedule_pulse(pulse)           # the flip
                    ro.schedule_pulse(pulse)           # reference marker (t0 on the ro channel)
                n = int(self.batch_resync_pulses)
                for idx in range(n)[::5]:
                    with a.channel_synchronizer(block=False):
                        for _ in range(idx, min(idx + 5, n)):
                            bs.schedule_pulse(pulse)
                    with a.sequencer().repeat_until(a.channel_is_fifo_empty(bs.channel)):
                        pass
                with a.channel_synchronizer():
                    bs.dwell(resync_dwell)             # wait for the last queued pulse (flat+ramp)
                    a.barrier()
                    ro.schedule_pulse(pulse)           # the "readout"

            elif self.case == "simulbus_transition":
                # Mirror SimulBus's batch-train -> multi-channel transition, the ONE place
                # the execution-model layout (machine.py) diverges from relayout on a real
                # runtime. A block=False batch train plays on ch0 (each batch drained by
                # repeat_until(channel_is_fifo_empty)); ch2 stays IDLE with only a reference
                # marker at t0. After the train, a blocking block plays a pulse on ch0 (which
                # WAS batched) AND on ch2 (which was idle). relayout starts the whole block --
                # including the ch2 pulse -- only after ch0's train fully drains; the machine
                # starts the ch2 pulse at last-pulled + one boundary gap, because an idle
                # channel plays its own FIFO when triggered rather than waiting for ch0's tail.
                # The ch2 reference->post-train interval measures which model is right.
                bs, idle = stimuli[0], stimuli[2]
                with a.channel_synchronizer():
                    bs.schedule_pulse(pulse)           # ch0: start-of-train marker
                    idle.schedule_pulse(pulse)         # ch2: reference marker (t0)
                n = int(self.batch_resync_pulses)
                for idx in range(n)[::5]:
                    with a.channel_synchronizer(block=False):
                        for _ in range(idx, min(idx + 5, n)):
                            bs.schedule_pulse(pulse)
                    with a.sequencer().repeat_until(a.channel_is_fifo_empty(bs.channel)):
                        pass
                with a.channel_synchronizer():
                    bs.schedule_pulse(pulse)           # ch0: post-train pulse (was batched)
                    idle.schedule_pulse(pulse)         # ch2: post-train pulse (the discriminator)

            elif self.case == "batch_two_channels":
                # TWO channels batched together (block=False) and drained together, then a
                # post-train block on both batched channels AND a fresh idle channel. Stresses
                # per-channel FIFO cursors when several channels desync at once (simulbus_
                # transition desynced only one). The idle channel's ref->post-train interval
                # is the discriminator, measurable within-channel.
                a_ch, b_ch, idle = stimuli[0], stimuli[1], stimuli[2]
                with a.channel_synchronizer():
                    a_ch.schedule_pulse(pulse)
                    b_ch.schedule_pulse(pulse)
                    idle.schedule_pulse(pulse)         # ch2 reference marker (t0)
                n = int(self.batch_resync_pulses)
                for idx in range(n)[::5]:
                    with a.channel_synchronizer(block=False):
                        for _ in range(idx, min(idx + 5, n)):
                            a_ch.schedule_pulse(pulse)
                            b_ch.schedule_pulse(pulse)
                    with a.sequencer().repeat_until(a.channel_is_fifo_empty(a_ch.channel)):
                        pass
                with a.channel_synchronizer():
                    a_ch.schedule_pulse(pulse)
                    b_ch.schedule_pulse(pulse)
                    idle.schedule_pulse(pulse)         # ch2 discriminator

            elif self.case == "batch_uneven":
                # A batch whose pulses ALTERNATE length (long test_pulse vs short rb_gate_hi),
                # so the LAST descriptor's length -- which sets last-pulled = playout_end -
                # last_len for the drain -- differs from the others. Checks the machine reads
                # the correct per-drain last_len rather than assuming a uniform pulse length.
                bs, idle = stimuli[0], stimuli[2]
                shapes = [pulse, "rb_gate_hi"]         # long, short
                with a.channel_synchronizer():
                    bs.schedule_pulse(pulse)
                    idle.schedule_pulse(pulse)         # ch2 reference marker (t0)
                n = int(self.batch_resync_pulses)
                for idx in range(n)[::5]:
                    with a.channel_synchronizer(block=False):
                        for k in range(idx, min(idx + 5, n)):
                            bs.schedule_pulse(shapes[k % 2])
                    with a.sequencer().repeat_until(a.channel_is_fifo_empty(bs.channel)):
                        pass
                with a.channel_synchronizer():
                    bs.schedule_pulse(pulse)
                    idle.schedule_pulse(pulse)         # ch2 discriminator

            elif self.case == "batch_interleaved":
                # Batch on ch0 (drain), then a SEPARATE batch on ch1 (drain), then a block
                # touching ch0, ch1 and idle ch2. ch0 and ch1 desync by DIFFERENT amounts --
                # the hardest test of independent per-channel FIFO cursors. Both ch0 and ch1
                # carry a reference marker so each channel's post-train interval is measurable.
                c0, c1, idle = stimuli[0], stimuli[1], stimuli[2]
                with a.channel_synchronizer():
                    c0.schedule_pulse(pulse)
                    c1.schedule_pulse(pulse)
                    idle.schedule_pulse(pulse)         # ch2 reference marker (t0)
                with a.channel_synchronizer(block=False):
                    for _ in range(5):
                        c0.schedule_pulse(pulse)
                with a.sequencer().repeat_until(a.channel_is_fifo_empty(c0.channel)):
                    pass
                with a.channel_synchronizer(block=False):
                    for _ in range(3):
                        c1.schedule_pulse(pulse)
                with a.sequencer().repeat_until(a.channel_is_fifo_empty(c1.channel)):
                    pass
                with a.channel_synchronizer():
                    c0.schedule_pulse(pulse)
                    c1.schedule_pulse(pulse)
                    idle.schedule_pulse(pulse)         # ch2 discriminator

            elif self.case == "loop_batch":
                # A loop whose body is a block=False batch + a fifo_empty drain, so the DRAIN
                # BLOCK REPEATS. drain_blocks is keyed by the compiled trigger index and reused
                # each unrolled iteration, so this checks the machine advances the sequencer
                # clock to last-pulled + gap correctly on every pass, not just the first. Idle
                # ch2 gets a reference marker before the loop and a readout after it.
                bs, idle = stimuli[0], stimuli[2]
                with a.channel_synchronizer():
                    bs.schedule_pulse(pulse)
                    idle.schedule_pulse(pulse)         # ch2 reference marker (t0)
                with a.sequencer().loop(2):
                    with a.channel_synchronizer(block=False):
                        for _ in range(3):
                            bs.schedule_pulse(pulse)
                    with a.sequencer().repeat_until(a.channel_is_fifo_empty(bs.channel)):
                        pass
                with a.channel_synchronizer():
                    bs.schedule_pulse(pulse)
                    idle.schedule_pulse(pulse)         # ch2 discriminator

            elif self.case == "stream_then_batch":
                # The RB cache-pointer stream, but followed IMMEDIATELY by a block=False batch
                # (drained) instead of a blocking readout. Tests the stream's trailing-period
                # cursor feeding into a following batch, plus a real fifo_empty drain right
                # after the stream's own almost_empty drain. Idle ch2 carries a reference marker
                # (t0) and the post-batch discriminator.
                stim, idle = stimuli[0], stimuli[2]
                with a.channel_synchronizer():
                    stim.schedule_pulse(self.rb_initial_pulse)   # ch0 initial pulse
                    idle.schedule_pulse(pulse)                   # ch2 reference marker (t0)
                base = a._firmware.sequencer_bus_decoder["cache"].address().value()
                pointer = a.sequencer().DSP()
                pointer.load(base + rb_cmd_cache.index)
                pointer.configure(mode="P+1", dsp_cep="reset")
                final = a.sequencer().Register()
                final.load(base + rb_cmd_cache.index + rb_num_cache[0])
                with a.sequencer().test(pointer != final):       # the gate loop
                    with a.sequencer().repeat_until(pointer == final):
                        with a.sequencer().repeat_until(
                                a.channel_is_fifo_almost_empty(stim.channel)):
                            pass
                        command = a.sequencer().bus_read(
                            pointer, latency=a._bus_latency("cache"))
                        with a.channel_synchronizer(block=False):
                            a.schedule_direct(stim.channel, command)
                        pointer.pulse_cep()
                with a.sequencer().repeat_until(
                        a.channel_is_fifo_almost_empty(stim.channel)):
                    pass
                with a.channel_synchronizer(block=False):        # NEW: a batch right after
                    for _ in range(5):
                        stim.schedule_pulse(pulse)
                with a.sequencer().repeat_until(a.channel_is_fifo_empty(stim.channel)):
                    pass
                with a.channel_synchronizer():
                    stim.schedule_pulse(pulse)         # ch0 post-batch pulse
                    idle.schedule_pulse(pulse)         # ch2 discriminator

            elif self.case == "batch_concurrent_blocking":
                # A block=False batch on ch0 (drained), then a BLOCKING pulse on ch1 -- which
                # the sequencer cannot issue until ch0's fifo_empty drain releases (last pulse
                # pulled), so ch1's pulse plays CONCURRENTLY with ch0's still-draining last
                # batch pulse. relayout, blind to the drain, fires ch1 right after its own
                # reference instead. ch1's ref->pulse interval discriminates; idle ch2 gets a
                # final readout too.
                c0, c1, idle = stimuli[0], stimuli[1], stimuli[2]
                with a.channel_synchronizer():
                    c0.schedule_pulse(pulse)
                    c1.schedule_pulse(pulse)           # ch1 reference marker (t0)
                    idle.schedule_pulse(pulse)
                with a.channel_synchronizer(block=False):
                    for _ in range(8):
                        c0.schedule_pulse(pulse)       # long batch on ch0
                with a.sequencer().repeat_until(a.channel_is_fifo_empty(c0.channel)):
                    pass
                with a.channel_synchronizer():
                    c1.schedule_pulse(pulse)           # ch1 blocking pulse (concurrent w/ ch0 tail)
                with a.channel_synchronizer():
                    idle.schedule_pulse(pulse)         # ch2 discriminator

            elif self.case == "barrier_single_channel":
                # The barrier pattern real runtimes actually use: exactly ONE channel
                # is active before the barrier, so acadia takes its single-element
                # path and never builds a max(). ch1..3 should each be padded by ch0's
                # pulse length, putting every post-barrier pulse at the same time.
                with a.channel_synchronizer():
                    stimuli[0].schedule_pulse(pulse)
                    a.barrier()
                    for stim in stimuli:
                        stim.schedule_pulse(pulse)

            elif self.case == "barrier_uneven_pulses":
                # Barrier alignment driven purely by differing PULSE counts, with no
                # explicit dwell() in the pre-barrier region. Isolates whether KI_002
                # is caused by the dwell length's type rather than by the max() arity.
                with a.channel_synchronizer():
                    stimuli[0].schedule_pulse(pulse)
                    stimuli[0].schedule_pulse(pulse)
                    stimuli[1].schedule_pulse(pulse)
                    a.barrier()
                    stimuli[0].schedule_pulse(pulse)
                    stimuli[1].schedule_pulse(pulse)

            elif self.case == "barrier_uneven_2ch":
                # Same intent as barrier_uneven but with only TWO channels in the
                # block. With four, barrier alignment builds max(64, 24, 24, 24) and
                # the sequencer rejects it -- see KI_002; two channels stay binary.
                # ch0 runs long before the barrier; ch1 should be padded to match, so
                # both post-barrier pulses start together.
                with a.channel_synchronizer():
                    stimuli[0].schedule_pulse(pulse)
                    stimuli[0].dwell(self.dwell_length)
                    stimuli[1].schedule_pulse(pulse)
                    a.barrier()
                    stimuli[0].schedule_pulse(pulse)
                    stimuli[1].schedule_pulse(pulse)

            elif self.case == "barrier_uneven":
                # ch0 is longer before the barrier; the others should be padded so every
                # channel leaves the barrier together
                with a.channel_synchronizer():
                    stimuli[0].schedule_pulse(pulse)
                    stimuli[0].dwell(self.dwell_length)
                    for stim in stimuli[1:]:
                        stim.schedule_pulse(pulse)
                    a.barrier()
                    for stim in stimuli:
                        stim.schedule_pulse(pulse)

        def sequence(a: Acadia):
            # 1. all four ADC captures triggered together, non-blocking, shared t=0.
            #    capture_start_delay dwells each ADC before it triggers: block 0 is
            #    non-blocking so it does not advance the sequencer for the DAC, hence the
            #    DAC still starts at t=0 while the capture frame begins `delay` later --
            #    which slides the fixed ~cable-latency lead-in out of view so more of the
            #    pulse train lands inside the window.
            with a.channel_synchronizer(block=False):
                for idx, cap in enumerate(captures):
                    if idx not in self.trace_channels:
                        continue
                    if self.capture_start_delay:
                        cap.dwell(self.capture_start_delay)
                    self.acadia.stream_cmacc(
                        src=cap.channel,
                        dst=cap.get_waveform_memory(self.capture_memory_name),
                        length=capture_length,
                        write_mode="input",
                        last_only=False,
                        kernel=None,
                        preload=(0, 0),
                        reset_fifo=True,
                    )
                if dummy_cap is not None:
                    # sacrificial 5th capture, triggered last, discarded (KI_001 workaround)
                    if self.capture_start_delay:
                        dummy_cap.dwell(self.capture_start_delay)
                    dummy_cap.capture_trace(self.dummy_memory_name)

            # 2. the DAC sequence under test
            dac_sequence(a)

            # 3. hold the program open until every capture has flushed its full trace
            with a.channel_synchronizer():
                for idx, cap in enumerate(captures):
                    if idx in self.trace_channels:
                        cap.dwell(capture_length)
                if dummy_cap is not None:
                    dummy_cap.dwell(capture_length)

        # Allocate every stimulus's waveform memory before attach(). Cases that schedule
        # only some channels would otherwise have load_pulse() below allocate a memory
        # *after* attach, and a memory created after attach is never mapped -- attach only
        # iterates the instances that exist when it runs.
        pulses_used = [self.stimulus_pulse_name]
        if self.case.startswith("stretch"):
            pulses_used.append("stretch_pulse")
        pulses_used += {"shape": ["long_ramp_pulse"],
                        "detune_pair": ["detune_10MHz", "detune_25MHz"],
                        "phase_pair": ["long_ramp_pulse", "phase_half_pi"]}.get(self.case, [])

        # rb_stream plays its gate set, initial pulse and readout marker on ch0 only, so
        # those extra pulses are allocated/loaded on stimulus0 alone (allocating them on the
        # idle channels would just waste DAC memory).
        rb_pulses = ([] if self.case not in STREAM_CASES
                     else list(dict.fromkeys(
                         list(self.rb_gates) + [self.rb_initial_pulse, self.rb_readout_pulse]
                         + ([self.rb_loop_gate] if self.rb_loop_gate else [])
                         + ([self.rb_final_gate] if self.rb_final_gate else []))))

        def names_for(index):
            return pulses_used + (rb_pulses if index == 0 else [])

        for i, stim in enumerate(stimuli):
            for name in names_for(i):
                stim.get_waveform_memory(name)

        # The measure_*/feedback_* cases read out through MeasurableResonator. Build it HERE,
        # not inside the sequence function: measure() allocates the CMACC window memory and the
        # accumulating capture memory on first use, and a memory created after attach() is never
        # mapped (attach only walks the instances that exist when it runs) -- which surfaces as
        # "MemoryError: Attempted access of unattached memory" at load_windows(). Same reason the
        # stimulus memories are touched above.
        self._resonator = None
        if self.case in READOUT_CASES:
            from acadia_qmsmt import MeasurableResonator
            self._resonator = MeasurableResonator(stimuli[0], self.io("capture_dummy"))
            self._resonator_mem2 = self.io("capture_dummy").get_waveform_memory(
                "readout_accumulated").duplicate()
            # Second resonator for measure_multi, on the OTHER traced stimulus and its own
            # capture IO. Built here for the same reason as the first: a memory allocated after
            # attach() is never mapped.
            self._resonator2 = (MeasurableResonator(stimuli[1], self.io("capture3"))
                                if self.case == "measure_multi" else None)

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        for i, stim in enumerate(stimuli):
            for name in names_for(i):
                stim.load_pulse(name)

        if self._resonator is not None:
            self._resonator.load_windows()      # fills the CMACC window memory
        if getattr(self, "_resonator2", None) is not None:
            self._resonator2.load_windows()

        if self.case == "register_dwell":
            cache[0] = self.acadia.seconds_to_cycles(self.register_dwell)
        elif self.case in ("register_stretch", "stretch_in_loop", "random_seq", "pair_seq"):
            # The register-stretch cases MUST set this. They load their stretch length from
            # cache[0], and an unset CacheArray is zero -- so the board was being asked to stretch
            # by 0 cycles, the one length acadia cannot encode (command_dma writes `length - 1`,
            # see compiled_log.parse). That degenerate fixture, not the loop, is what made
            # stretch_in_loop miss by exactly 1 cycle per pass.
            cache[0] = self.acadia.seconds_to_cycles(self.register_stretch)
        elif (self.case.startswith("test_") or self.case in
              ("counter_loop_in_test", "repeat_until_op", "repeat_until_count_n")):
            # These compare a cached register against test_register_value, so the cache MUST
            # carry it or only one arm is ever reachable -- counter_loop_in_test silently drew
            # its skipped arm every time because cache[0] defaulted to 0 while the comparison
            # asked for 1, which looks like a passing test of a branch that was never varied.
            cache[0] = int(self.test_register_value)
        elif self.case in STREAM_CASES:
            # Load one DMA command per pattern slot: each is the cached word that plays
            # that gate's waveform. Decoding it back to (address -> pulse, length) is
            # exactly what the tracer fix will do off the captured cache.
            rb_num_cache[0] = int(rb_count)
            for k, gate_idx in enumerate(rb_pattern):
                # rb_loop_gate (if set) forces every loop slot to one shape -- the
                # length-sweep mode; otherwise each slot plays its patterned gate.
                gate = self.rb_loop_gate or self.rb_gates[gate_idx]
                rb_cmd_cache[k] = self.acadia.waveform_dma_command(
                    stimuli[0].get_waveform_memory(gate))

        t_data = None
        configure_streams = True
        for i in range(self.iterations):
            self.acadia.run(minimum_delay=self.run_delay, configure_streams=configure_streams)
            configure_streams = False

            for idx, (label, cap) in enumerate(zip(self._labels, captures)):
                if idx not in self.trace_channels:
                    continue          # never captured -> its memory was never attached
                wf = cap.get_waveform_memory(self.capture_memory_name)
                self.data[f"trace_{label}"].write(wf.array)

            if t_data is None:
                t_data = np.linspace(0, capture_length, len(wf.array), endpoint=False)
                self.data["t_data"].write(t_data)

            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return

        self.final_serve()

    def initialize(self):
        pass

    def update(self):
        self.data.save(self.local_directory)

    def finalize(self):
        super().finalize()
        from acadia_qmsmt.plotting import save_registered_plots
        save_registered_plots(self)

    @annotate_method(is_data_processor=True)
    def process_current_data(self):
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes

        labels = ["ch0", "ch1", "ch2", "ch3"]
        t_data_full = np.array(self.data["t_data"].records()).astype(float).squeeze()
        n_trim = self.tail_trim_samples
        self.t_data = t_data_full[:-n_trim] if n_trim > 0 else t_data_full

        self.traces_iq = {}
        self.avg_trace_iq = {}
        self.avg_trace_pwr = {}

        completed_iterations = None
        # only the captured channels have a group -- see trace_channels and the add_group note
        for idx, label in enumerate(labels):
            if idx not in self.trace_channels:
                continue
            traces_iq = reshape_iq_data_by_axes(self.data[f"trace_{label}"].records(), t_data_full)
            if traces_iq is None:
                return
            if n_trim > 0:
                traces_iq = traces_iq[:, :-n_trim, :]

            completed_iterations = len(traces_iq)
            avg_trace_iq = np.mean(traces_iq, axis=0)

            self.traces_iq[label] = traces_iq
            self.avg_trace_iq[label] = avg_trace_iq
            self.avg_trace_pwr[label] = avg_trace_iq[:, 0] ** 2 + avg_trace_iq[:, 1] ** 2

        return completed_iterations

    @annotate_method(plot_name="loopback traces", axs_shape=(4, 1))
    def plot_all_traces(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(4, 1), figsize=self.figsize)

        for ax, label in zip(axs, self._labels if hasattr(self, "_labels")
                             else ["ch0", "ch1", "ch2", "ch3"]):
            avg_iq = self.avg_trace_iq[label]
            ax.plot(self.t_data, avg_iq[:, 0], label="Re")
            ax.plot(self.t_data, avg_iq[:, 1], label="Im")
            ax.plot(self.t_data, np.sqrt(self.avg_trace_pwr[label]), label="mag")
            ax.set_ylabel(f"{label}\nVoltage [a.u.]")
            ax.grid(True)
        axs[0].legend(fontsize="small")
        axs[0].set_title(f"case = {self.case}")
        axs[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        return fig, axs
