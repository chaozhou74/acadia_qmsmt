"""One reused register must read the cache load in force at each block.

This fails when the viewer applies the final cache source to every earlier
packed DMA command or dwell. Pulse names and two DACs are deliberately generic.
"""
from pathlib import Path
import numpy as np
import yaml
from acadia_qmsmt import QMsmtRuntime, IOConfig
from acadia_qmsmt.sequence_viz import tracing


class ReloadRuntime(QMsmtRuntime):
    first: IOConfig
    second: IOConfig

    def main(self):
        first, second = self.io("first"), self.io("second")
        memories = [first.get_waveform_memory(dict(name=f"pulse_{j}", data=np.ones(n*4, complex)*.1))
                    for j, n in enumerate((6, 11, 3))]
        cache = self.acadia.CacheArray(shape=(6,), dtype=np.dtype('<i4'))
        def sequence(a):
            word, length = a.sequencer().Register(), a.sequencer().Register()
            for j in range(3):
                word.load(cache[2*j]); length.load(cache[2*j+1])
                with a.channel_synchronizer():
                    a.schedule_direct(first.channel, word)
                    second.dwell(length)
        self.acadia.compile(sequence)
        self.acadia.attach()
        for j, n in enumerate((6, 11, 3)):
            cache[2*j] = self.acadia.waveform_dma_command(memories[j])
            cache[2*j+1] = n + 2
        self.acadia.run()


config = yaml.safe_load(Path(__file__).with_name('validation_board_config.yaml').read_text())
rt = ReloadRuntime(first=config['stimulus0'], second=config['stimulus2'])
trace = tracing.trace_runtime(rt, capture_points=True)
for channel, lengths in ((str(rt.io('first').channel), [6, 11, 3]),
                         (str(rt.io('second').channel), [8, 13, 5])):
    commands = [c for p in trace.placements for c in p.commands if c.channel == channel and not c.is_padding]
    assert [c.length for c in commands] == lengths, commands
plays = [c.pulse for p in trace.placements for c in p.commands if c.pulse]
assert plays == ['pulse_0', 'pulse_1', 'pulse_2'], plays
print('PASS per-block fixed-cache reloads: DMA identity/length and dwell duration')
