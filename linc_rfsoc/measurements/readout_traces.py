from dataclasses import dataclass
from acadia.runtime import Runtime

from auto_config import AutoConfigMixin
from auto_config import FILE as config_helper_file

@dataclass
class ReadoutTracesRuntime(AutoConfigMixin, Runtime):
    """
    A :class:`Runtime` subclass for sending a signal out of a DAC and capturing
    it on an ADC.
    """

    q_stimulus: dict
    ro_stimulus: dict
    ro_capture: dict

    iterations: int
    plot: bool = True
    figsize: tuple[int] = None
    
    FILE = __file__
        
    def main(self):     
        from acadia import Acadia, DataManager
        import numpy as np
        import logging
        
        logger = logging.getLogger("acadia")

        channel_configs = {"q_stimulus": self.q_stimulus,
                           "ro_stimulus": self.ro_stimulus,
                           "ro_capture": self.ro_capture
                           }

        # Create an acadia object and grab a couple of its channels
        acadia = Acadia()
        self.auto_config_channels(acadia, **channel_configs)

        # Allocate the waveform memories that we'll need
        self.auto_config_waveform_mems(acadia, **channel_configs)
        pi_pulse = self.channel_waveforms["q_stimulus"]["pi_pulse"]
        ro_drive = self.channel_waveforms["ro_stimulus"]["ro_drive"]
        ro_demod = self.channel_waveforms["ro_capture"]["ro_demod"]
        
        # Create a blank waveform between qubit drive and readout drive
        q_blank_wf = self.blank_waveform_generator(acadia, "q_stimulus")(40e-9)

        # Create blank waveform for delay between capture and stimulus
        capture_delay = self.ro_capture["capture_delay"]
        if capture_delay < 0: # capture will be advanced by -capture_delay compare to stimulus
            blank_wf = self.blank_waveform_generator(acadia, "ro_stimulus")(-capture_delay)
        elif capture_delay > 0: # capture will be delayed by capture_delay compare to stimulus
            blank_wf = self.blank_waveform_generator(acadia, "ro_capture")(capture_delay)
        
        
        # Create the record groups for saving captured data
        self.data.add_group(f"traces_g", uniform=True)
        self.data.add_group(f"traces_e", uniform=True)

        # Create a sequence for the sequencer to generate the pulse and capture it
        def sequence(a: Acadia):
            capture_stream = acadia.configure_dsp(self.channel_objs["ro_capture"], ro_demod._decimation) 
        
            with a.channel_synchronizer():
                a.schedule_waveform(pi_pulse)
                a.schedule_waveform(q_blank_wf)
                a.barrier()
                if capture_delay != 0:     
                    a.schedule_waveform(blank_wf)   
                a.schedule_waveform(ro_drive)
                a.stream(capture_stream, ro_demod)

        # Compile the sequence
        acadia.compile(sequence)
                
        # Attach to the hardware
        acadia.attach()

        # Configure channel analog parameters
        self.auto_config_ncos(acadia, **channel_configs)

        # Assemble and load the program
        acadia.assemble()
        acadia.load()

        # set waveform for ro drive
        ro_drive.set(**self.ro_stimulus["signal"])

        # average iterations for preparing qubit in g and e
        amp0 = self.q_stimulus["signal"]["scale"]
        for i in range(self.iterations):
            for state_ in ["g", "e"]:
                # set the pulse amplitude
                if state_ == "g":
                    self.q_stimulus["signal"]["scale"] = 0
                else:
                    self.q_stimulus["signal"]["scale"] = amp0

                pi_pulse.set(**self.q_stimulus["signal"])
                
                # capture data and put in the corresponding group
                acadia.run()
                self.data[f"traces_{state_}"].write(ro_demod.array)
            
            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return
        
        self.final_serve()

    def initialize(self):
        # Set the matplotlib backend to one which we can actually update
        self.figsize = (4, 7) if self.figsize is None else self.figsize

        if self.plot:
            from acadia.processing import DynamicLine
            import matplotlib.pyplot as plt

            self.fig, self.axs = plt.subplots(2, 1, figsize=self.figsize)
            self.fig.tight_layout()
            self.fig.subplots_adjust(left=0.25, bottom=0.25)

            self.lines_re = []
            self.lines_im = []
            for j, state_ in enumerate(["g", "e"]):
                self.lines_re.append(DynamicLine(self.axs[j], ".-"))
                self.lines_im.append(DynamicLine(self.axs[j], ".-"))
                self.axs[j].set_xlabel("Time [s]")
                self.axs[j].set_ylabel("Amp [arb. V]")
                self.axs[j].set_title(f"prepare {state_}")
                self.axs[j].grid()

            self.time_axis = None

        from tqdm.notebook import tqdm
        self.progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.previous_completed_iterations = 0

    def update(self):
        import numpy as np

        # First make sure that we actually have new data to process
        if f"traces_e" not in self.data or len(self.data[f"traces_e"]) == 0:
            return

        # Update the progress bar based on the number of iterations that have been completed
        completed_iterations = len(self.data[f"traces_e"])
        self.progress_bar.update(completed_iterations - self.previous_completed_iterations)
        self.previous_completed_iterations = completed_iterations        

        # We'll make an x-axis for the plot, but we only need to make it once
        if self.time_axis is None:
            # Last index is for quadrature
            samples_per_trace = np.array(self.data["traces_g"].records()).shape[-2]
            capture_time = self.ro_capture["waveforms"]["ro_demod"]["length"] # todo: simplify this, too long.....
            self.time_axis = np.linspace(0, capture_time, samples_per_trace, endpoint=False)

        # Sum the traces from each iteration
        # Each trace has the shape (samples, 2) where the number of samples is determined
        # at runtime from the specified waveform length in seconds. 
        # Because the record group is uniform, when we get the records from the
        # group, they are stacked into a single array of shape (iterations, samples, 2)
        self.trace_summed = np.zeros((2, len(self.time_axis), 2), dtype=np.int64)
        for j, state_ in enumerate(["g", "e"]):
            self.trace_summed[j] = np.sum(self.data[f"traces_{state_}"].records(), axis=0)
        
        if self.plot:
            for i in range(2):
                self.lines_re[i].update(self.time_axis, self.trace_summed[i, :,0])
                self.lines_im[i].update(self.time_axis, self.trace_summed[i, :,1])
                self.axs[i].relim()
                self.axs[i].autoscale(tight=True)
                self.fig.tight_layout()
                self.fig.canvas.draw_idle()

        # Save the data
        self.data.save(self.local_directory)    

    def finalize(self):
        super().finalize()
        self.progress_bar.close()
        if self.plot:
            self.savefig(self.fig)  


    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from linc_rfsoc.measurements import load_config


    from IPython.core.getipython import get_ipython
    get_ipython().run_line_magic("matplotlib", "widget")

    config_dict = load_config() # todo: make the yaml file take complex values

    plot = True
    iterations = 5000


    rt = ReadoutTracesRuntime(**config_dict, plot=plot, iterations=iterations)
    rt.deploy("10.66.3.198", "readout_traces", files=[rt.FILE, config_helper_file])
    rt.display() 

    # some ad hoc processing
    rt._event_loop.join()
    rt.fig

    all_traces = np.concatenate([np.array(rt.data[f"traces_{s}"].records()).astype(float) for s in ["g", "e"]])
    all_traces = all_traces.view(complex).squeeze()
    all_pts = np.mean(all_traces, axis=1)
    fig, ax = plt.subplots(1, 1)
    ax.hist2d(all_pts.real, all_pts.imag, cmap="hot", bins=101)
    ax.set_aspect(1)

    
    # from linc_rfsoc.analysis.generate_readout_kernel import ReadoutKernelGenerator
    # rk = ReadoutKernelGenerator(all_traces, (70 + 73j, 20), (16 + 105j, 20))

    # print(rk.save_kernel(r"../dev_codes//", "test_kernel"))

    # rk.plot_kernel()
    # kernel=load_kernel(r"../dev_codes//"+"readoutkernel_241105_113318.npy")
    

    