"""Decode changing cache-row predicates/words and loop transitions without runtime-name rules.

Two different channels and renamed pulses, variable counts and opposite arm orders.
No board or saved dataset is required.
"""
from pathlib import Path
import numpy as np
import yaml
from acadia_qmsmt import QMsmtRuntime,IOConfig
from acadia_qmsmt.sequence_viz import tracing

class RowStream(QMsmtRuntime):
    stimulus: IOConfig
    names: tuple
    def main(self):
        io=self.io('stimulus')
        waves=[dict(name=name,data=np.full(4*n,.08+.03j))
               for name,n in zip(self.names,(6,11,3))]
        memories=[io.get_waveform_memory(wave) for wave in waves]
        table=self.acadia.CacheArray(shape=(12,),dtype=np.dtype('<i4'))
        control=self.acadia.CacheArray(shape=(1,),dtype=np.dtype('<i4'))
        def sequence(a):
            pointer=a.sequencer().DSP()
            word,length,selector,end=[a.sequencer().Register() for _ in range(4)]
            base=a._firmware.sequencer_bus_decoder['cache'].address().value()
            pointer.load(base+table.index);pointer.configure(mode='P+1',dsp_cep='reset')
            end.load(control[0])
            with a.sequencer().test(pointer!=end):
                with a.sequencer().repeat_until(pointer==end):
                    for reg in (word,length,selector):
                        reg.load(a.sequencer().bus_read(pointer,latency=a._bus_latency('cache')))
                        pointer.pulse_cep()
                    with a.sequencer().test(selector==0):
                        with a.channel_synchronizer():a.schedule_direct(io.channel,word)
                    # Triggerless tests still cost cycles on their skipped path.
                    with a.sequencer().test(selector!=0):
                        with a.sequencer().test(selector!=1):pass
                    with a.sequencer().test(selector!=0):
                        with a.channel_synchronizer():a.schedule_direct(io.channel,word)
        self.acadia.compile(sequence);self.acadia.attach()
        for wave in waves:io.load_pulse(wave)
        base=self.acadia._firmware.sequencer_bus_decoder['cache'].address().value()
        for count in (2,4):
            for j in range(count):
                table[3*j]=int(self.acadia.waveform_dma_command(memories[j%3]))
                table[3*j+1]=(6,11,3)[j%3]
                table[3*j+2]=j%2
            control[0]=base+table.index+3*count
            self.acadia.run()

original=yaml.safe_load(Path(__file__).with_name('validation_board_config.yaml').read_text())
for channel,names in (('stimulus0',('one','two','three')),('stimulus2',('left','right','return'))):
    rt=RowStream(stimulus=original[channel],names=names)
    trace=tracing.trace_runtime(rt,envelopes=True,capture_points=True,max_points=2)
    for point,count in ((0,2),(1,4),(0,2),(1,4)):
        trace.select_point(point)
        commands=[cmd for pl in trace.placements for cmd in pl.commands if not cmd.is_padding]
        assert [cmd.pulse for cmd in commands]==[names[j%3] for j in range(count)],commands
        assert [cmd.length for cmd in commands]==[(6,11,3)[j%3] for j in range(count)]
        rows=trace.control_flow_summary()
        assert all(row['source']=='resolved' for row in rows if row['kind']=='test' and row['executions'])
        assert not trace.length_underflows
    print('PASS',channel,names,'row predicates, words, variable counts and point switching',flush=True)
