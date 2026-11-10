"""A stored length-minus-one of zero is ONE clock, not an absent wait.

Compare constant, fixed-cache and pointer-selected waits through the public
Acadia API. The same flag must survive all three ways of supplying a length.
"""
from pathlib import Path
import numpy as np
import yaml
from acadia_qmsmt import IOConfig, QMsmtRuntime
from acadia_qmsmt.sequence_viz import tracing


class WaitRuntime(QMsmtRuntime):
    stimulus: IOConfig
    pointer_selected: bool

    def main(self):
        io = self.io("stimulus")
        cache = self.acadia.CacheArray(shape=(3,), dtype=np.dtype("<i4"))

        def sequence(a):
            register = a.sequencer().Register()
            pointer = a.sequencer().DSP()
            pointer.load(a._firmware.sequencer_bus_decoder["cache"].address().value() + cache.index)
            for j in range(3):
                if self.pointer_selected:
                    register.load(a.sequencer().bus_read(pointer))
                    pointer.pulse_cep()
                else:
                    register.load(cache[j])
                with a.channel_synchronizer():
                    a.dwell(io.channel, register, length_is_minus_one=True)
                with a.channel_synchronizer():
                    a.dwell(io.channel, (j + 1) / a.sequencer_clock_frequency(), length_is_minus_one=True)

        self.acadia.compile(sequence)
        self.acadia.attach()
        for j, value in enumerate((0, 1, 2)):
            cache[j] = value
        self.acadia.run()


config = yaml.safe_load(Path(__file__).with_name("validation_board_config.yaml").read_text())
for pointer_selected in (False, True):
    runtime = WaitRuntime(stimulus=config["stimulus0"], pointer_selected=pointer_selected)
    trace = tracing.trace_runtime(runtime, capture_points=True)
    commands = [c for p in trace.placements for c in p.commands if not c.is_padding]
    assert [c.length for c in commands] == [1, 2, 2, 3, 3, 4], commands
    assert not trace.length_underflows, trace.length_underflows
    print("PASS stored length-minus-one: zero/positive, constant and",
          "pointer-selected" if pointer_selected else "fixed-cache", "waits")
