"""Constant-time conditional stores must select the real DMA word and envelope.

No board or saved experiment is needed. Two renamed pulse sets on different
channels test phase/amplitude, different lengths, cache-selected predicates,
explicit predicate overrides, and a later unconditional register reload.
"""
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from acadia_qmsmt import IOConfig, QMsmtRuntime
from acadia_qmsmt.sequence_viz import plotting, tracing


class ConstantWordRuntime(QMsmtRuntime):
    stimulus: IOConfig
    pulse_names: tuple

    def main(self):
        io = self.io("stimulus")
        fs = io.channel.interface_sample_frequency
        spc = int(io._samples_per_cycle)
        arrays = [np.full(cycles * spc, amplitude, complex)
                  for cycles, amplitude in ((6, .1 + .03j), (11, -.07 + .15j), (3, .05 - .08j))]
        pulses = [{"name": name, "data": array} for name, array in zip(self.pulse_names, arrays)]
        memories = [io.get_waveform_memory(pulse) for pulse in pulses]
        cache = self.acadia.CacheArray(shape=(1,), dtype=np.dtype("<i4"))

        def sequence(a):
            predicate, selected = a.sequencer().Register(), a.sequencer().Register()
            self.predicate_register = str(predicate)
            predicate.load(cache[0])
            with a.sequencer().test(predicate >= 0):
                selected.load(a.waveform_dma_command(memories[0]))
                a.sequencer().store(src=a.waveform_dma_command(memories[1]), dest=selected,
                                    condition=predicate != 0)
                with a.channel_synchronizer():
                    a.schedule_direct(io.channel, selected)
            selected.load(a.waveform_dma_command(memories[2]))
            with a.channel_synchronizer():
                a.schedule_direct(io.channel, selected)

        self.acadia.compile(sequence)
        self.acadia.attach()
        for pulse in pulses:
            io.load_pulse(pulse)
        for value in (0, 1):
            cache[0] = value
            self.acadia.run()


generic = Path(__file__).with_name("validation_board_config.yaml")
original = yaml.safe_load(generic.read_text())
for alias, names in (("stimulus0", ("first", "second", "last")),
                     ("stimulus2", ("zero_choice", "nonzero_choice", "reload"))):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(original))
        runtime = ConstantWordRuntime(stimulus=original[alias], pulse_names=names)
        trace = tracing.trace_runtime(runtime, envelopes=True, capture_points=True, max_points=2)
        for point in (0, 1):
            trace.select_point(point)
            commands = [c for p in trace.placements for c in p.commands if not c.is_padding]
            assert [c.pulse for c in commands] == [names[point], names[2]], commands
            assert [c.length for c in commands] == [[6, 11][point], 3], commands
            for mode in ("iq", "magnitude"):
                fig, _ = plotting.plot_trace(trace, envelope_mode=mode, envelope_source="memory")
                fig.canvas.draw()
                plt.close(fig)
        trace.register_overrides[runtime.predicate_register] = 0
        trace.relayout()
        commands = [c for p in trace.placements for c in p.commands if not c.is_padding]
        assert commands[0].pulse == names[0]
        trace.register_overrides[runtime.predicate_register] = 1
        trace.relayout()
        commands = [c for p in trace.placements for c in p.commands if not c.is_padding]
        assert commands[0].pulse == names[1]
        print("PASS", alias, names, "conditional words, lengths, I/Q, reload, predicate overrides")
