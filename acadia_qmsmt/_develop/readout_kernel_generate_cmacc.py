from dataclasses import dataclass
from acadia.runtime import Runtime


@dataclass
class ReadoutTestRuntime_test(Runtime):
    """
    A :class:`Runtime` subclass for sending a signal out of a DAC and capturing
    it on an ADC.
    """

    stimulus: dict
    capture: dict
    phases: tuple 
    iterations: int
    capture_delay: float
    plot: bool = True
    figsize: tuple[int] = None
    
    FILE = __file__
        
    def main(self):     
        from acadia import Acadia, DataManager
        import numpy as np
        import logging
        
        logger = logging.getLogger("acadia")
        # Create an acadia object and grab a couple of its channels
        acadia = Acadia()
        stimulus_channel = acadia.channel(self.stimulus["channel"])
        capture_channel = acadia.channel(self.capture["channel"])
        capture_delay = self.capture_delay

        # Create the waveforms that we'll need 
        stimulus_waveform = acadia.create_waveform_memory(stimulus_channel, **self.stimulus["waveform"])
        capture_waveform = acadia.create_waveform_memory(capture_channel, **self.capture["waveform"]) 
        
        # Create blank waveform for delay between capture and stimulus
        if capture_delay < 0: # capture will be advanced by -capture_delay compare to stimulus
            blank_wf = acadia.create_waveform_memory(stimulus_channel, length=-capture_delay, blank=True)
        elif capture_delay > 0: # capture will be delayed by capture_delay compare to stimulus
            blank_wf = acadia.create_waveform_memory(capture_channel, length=capture_delay, blank=True,
                                              decimation=self.capture["waveform"]["decimation"], 
                                              region=self.capture["waveform"]["region"])

        # Create a record group for saving captured data
        for i in range(len(self.phases)):
            self.data.add_group(f"traces_phi{i}", uniform=True)
        
        kernel_wf = self.capture["kernel_wf"]
        if type(kernel_wf) == float: # constant value kernel
            kernel_wf = np.float64(kernel_wf)
            kernel_cmacc = None
        elif type(kernel_wf) == np.ndarray:
            kernel_cmacc = kernel_wf

        # Create a sequence for the sequencer to generate the pulse and capture it
        def sequence(a: Acadia):
            capture_stream, kernel = acadia.configure_cmacc(capture_channel, kernel=kernel_cmacc, 
                                                            reset_fifo=True, write_mode="upper", last_only=False)
            # capture_stream, kernel = acadia.configure_cmacc(capture_channel, kernel=kernel_cmacc, 
                                                            # reset_fifo=True, write_mode="input", last_only=False)

            acadia.cmacc_load(capture_stream, 0)

            with a.channel_synchronizer():
                if capture_delay != 0:     
                    a.schedule_waveform(blank_wf)   
                a.schedule_waveform(stimulus_waveform)
                a.stream(capture_stream, capture_waveform)

            return kernel

        # Compile the sequence
        kernel = acadia.compile(sequence)
                
        # Attach to the hardware
        acadia.attach()


        # Configure channel analog parameters
        acadia.align_tile_latencies()
        stimulus_channel.set(nco_update_event_source="sysref", **self.stimulus["datapath"])
        capture_channel.set(nco_update_event_source="sysref", **self.capture["datapath"])
        acadia.reset_nco_phase(stimulus_channel)
        acadia.reset_nco_phase(capture_channel)
        acadia.update_nco_phase(stimulus_channel, 2)
        acadia.update_nco_phase(capture_channel, 0)
        acadia.update_ncos_synchronized()

        logger.info(f"kernel waveform {kernel}, {kernel.array.shape}")
        # Set the amplitude of the boxcar integration kernel
        kernel.set(kernel_wf)

        # Assemble and load the program
        acadia.assemble()
        acadia.load()

        amp0 = self.stimulus["signal"]["scale"]
        for i in range(self.iterations):
            for phi_idx, phi in enumerate(self.phases):
                # set the pulse phase
                self.stimulus["signal"]["scale"] = amp0 * np.exp(1j*phi)

                stimulus_waveform.set(**self.stimulus["signal"])
                
                # capture data and put in the corresponding group
                acadia.run()
                self.data[f"traces_phi{phi_idx}"].write(capture_waveform.array)
            
            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return
        
        self.final_serve()

    def initialize(self):
        # Set the matplotlib backend to one which we can actually update
        self.phase_num = len(self.phases)
        self.figsize = (4, 2*self.phase_num+1) if self.figsize is None else self.figsize

        if self.plot:
            from acadia.processing import DynamicLine
            import matplotlib.pyplot as plt

            self.fig, self.axs = plt.subplots(self.phase_num, 1, figsize=self.figsize)
            self.fig.tight_layout()
            self.fig.subplots_adjust(left=0.25, bottom=0.25)

            if self.phase_num == 1:
                self.axs = [self.axs]

            self.lines_re = []
            self.lines_im = []
            for j, phi in enumerate(self.phases):
                self.lines_re.append(DynamicLine(self.axs[j], ".-"))
                self.lines_im.append(DynamicLine(self.axs[j], ".-"))
                self.axs[j].set_xlabel("Time [s]")
                self.axs[j].set_ylabel("Amp [arb. V]")
                self.axs[j].set_title(f"Phase: {np.round(phi/np.pi, 5)}$\pi$")
                self.axs[j].grid()

            self.time_axis = None

        from tqdm.notebook import tqdm
        self.progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.previous_completed_iterations = 0

    def update(self):
        import numpy as np
        import matplotlib.pyplot as plt

        # First make sure that we actually have new data to process
        if f"traces_phi{self.phase_num-1}" not in self.data or len(self.data[f"traces_phi{self.phase_num-1}"]) == 0:
            return

        # Update the progress bar based on the number of iterations that have been completed
        completed_iterations = len(self.data[f"traces_phi{self.phase_num-1}"])
        self.progress_bar.update(completed_iterations - self.previous_completed_iterations)
        self.previous_completed_iterations = completed_iterations        

        # We'll make an x-axis for the plot, but we only need to make it once
        if self.time_axis is None:
            # Last index is for quadrature
            samples_per_trace = np.array(self.data["traces_phi0"].records()).shape[-2]
            capture_time = self.capture["waveform"]["length"]
            self.time_axis = np.linspace(0, capture_time, samples_per_trace, endpoint=False)

        self.trace_summed = np.zeros((self.phase_num, len(self.time_axis), 2), dtype=np.int64)
        for i in range(self.phase_num):
            self.trace_summed[i] = np.sum(self.data[f"traces_phi{i}"].records(), axis=0)
        
        if self.plot:
            for i in range(self.phase_num):
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
    
    from IPython.core.getipython import get_ipython
    get_ipython().run_line_magic("matplotlib", "widget")

    stimulus: dict = {
        "channel": "DAC2",

        "datapath": {
            "vop": 30000,
            "mix_reconstruction": True,
            "nco_frequency": 9.03e9
        },

        "waveform": {
            "length": 0., 
            "fixed_length": 1200e-9 
        },
        
        "signal": {
            "data": ("scipy", "hann"),
            "scale": 0.5
        }
    }

    capture: dict = {
        "channel": "ADC0",

        "datapath": {
            "nco_frequency": -9.03e9
        },

        "waveform": {
            "length": 800*1.25e-9,
            "decimation": 4,
            "region": "plddr"
        },


        # "kernel_wf": np.repeat([0.1, 0], 100) # full length of the capture waveform
        "kernel_wf": np.repeat([0, 0.1, 0.1, 0], 50) # full length of the capture waveform
        # "kernel_wf": np.repeat([0, 0.1, 0.1, 0], 50)[::4] # quoter length of the capture waveform
        # "kernel_wf": np.repeat([ 0, 0.1, 0.2, 0.3], 4)
        # "kernel_wf":np.array([0.1j]*200)
        # "kernel_wf": 0.1
    }

    plot = True
    iterations = 10000
    # phases = (0, np.pi/2, np.pi)
    phases = np.array([0])


    rt = ReadoutTestRuntime_test(stimulus, capture, phases, 
                                 plot=plot, iterations=iterations, capture_delay=280e-9)
    
    rt.deploy("10.66.3.198", "readout_kernel_generate_cmacc", files=[rt.FILE])    
    rt.display()

    # some ad hoc processing
    rt._event_loop.join()

    phi0_data = np.array(rt.data[f"traces_phi0"].records()).astype(float).view(complex).squeeze()
    phi0_trace = np.mean(phi0_data, axis=0)

    plt.figure()
    plt.title("received data")
    plt.plot(phi0_trace.real)
    plt.plot(phi0_trace.imag)
    plt.grid()

    plt.figure()
    plt.title("data derivative (inferred kernel)")
    plt.plot((phi0_trace[1:] - phi0_trace[:-1]).real)
    plt.plot((phi0_trace[1:] - phi0_trace[:-1]).imag)


    plt.figure()
    plt.title("uploaded kernel")
    plt.plot(np.real(rt.capture["kernel_wf"]))
    plt.plot(np.imag(rt.capture["kernel_wf"]))

    