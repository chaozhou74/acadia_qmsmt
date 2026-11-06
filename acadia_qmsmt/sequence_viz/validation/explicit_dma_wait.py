"""A non-blocking trigger followed by an explicit DMA-running wait honors its mask.

The two channels finish at different times. A third channel marks the release;
waiting for only the shorter channel must not wait for the longer one. The test
varies channels and pulse lengths, with three loop passes each. No runtime names
or experimental timing constants are encoded in the viewer.
"""
from copy import deepcopy
from pathlib import Path

import yaml

from acadia_qmsmt import IOConfig, QMsmtRuntime
from acadia_qmsmt.sequence_viz import trace_runtime


class ExplicitWait(QMsmtRuntime):
    first: IOConfig
    second: IOConfig
    marker: IOConfig
    wait_second: bool

    def main(self):
        first, second, marker = self.io("first"), self.io("second"), self.io("marker")
        def sequence(a):
            with a.sequencer().loop(3):
                with a.channel_synchronizer(block=False):
                    first.schedule_pulse("test_pulse")
                    second.schedule_pulse("test_pulse")
                channel = second.channel if self.wait_second else first.channel
                mask = 1 << channel.num()
                address = a._firmware.sequencer_bus_decoder["dma_running"].address().value()
                running = a.sequencer().bus_read(address, latency=a._bus_latency("dma_running"))
                with a.sequencer().repeat_until(running & mask == 0):
                    pass
                with a.channel_synchronizer():
                    marker.schedule_pulse("test_pulse")
        self.acadia.compile(sequence)
        self.acadia.attach()
        for io in (first,second,marker):
            io.load_pulse("test_pulse")
        self.acadia.run()


config = yaml.safe_load(Path(__file__).with_name("validation_board_config.yaml").read_text())
for pair in ((0,1,2),(2,3,0)):
    for difference in (200e-9, 600e-9):
        ios = [deepcopy(config[f"stimulus{i}"]) for i in pair]
        for io, flat in zip(ios, (200e-9,200e-9+difference,100e-9)):
            io["pulses"]["test_pulse"].update(ramp=0.0,flat=flat,use_stretch=False)
        traces = [trace_runtime(ExplicitWait(first=ios[0],second=ios[1],marker=ios[2],
                                           wait_second=choice)) for choice in (False,True)]
        starts = []
        for trace in traces:
            pulses = [c for block in trace.placements for c in block.commands
                      if c.io_name == "marker" and c.pulse == "test_pulse"]
            assert len(pulses) == 3, len(pulses)
            starts.append([c.start for c in pulses])
        delta = (starts[1][0] - starts[0][0])*traces[0].ns_per_cycle
        assert abs(delta-difference*1e9) < 0.01, (pair,difference,delta)
        print(f"PASS channels={pair} long-minus-short release={delta:.1f} ns; "
              "6/6 marker pulses across two masks and three loop passes")
