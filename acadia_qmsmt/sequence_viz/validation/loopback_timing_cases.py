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
         # KI_002 cases: these could not compile before the 2026-07-27 acadia pull
         "barrier_uneven_pulses", "barrier_uneven_2ch", "barrier_uneven")

# Cases built on the cache-pointer stream idiom: they share the command-cache setup, the
# rb-pulse allocation, and the cache fill. rb_stream_uniform is rb_stream with uniform-amplitude
# final gates (clean timing of the back-to-back block; see README on the edge-detection artifact).
STREAM_CASES = ("rb_stream", "rb_stream_uniform", "stream_then_batch")


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

        for label in self._labels:
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
                for cap in captures:
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
                for cap in captures:
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

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        for i, stim in enumerate(stimuli):
            for name in names_for(i):
                stim.load_pulse(name)

        if self.case == "register_dwell":
            cache[0] = self.acadia.seconds_to_cycles(self.register_dwell)
        elif self.case.startswith("test_"):
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

            for label, cap in zip(self._labels, captures):
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
        for label in labels:
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
