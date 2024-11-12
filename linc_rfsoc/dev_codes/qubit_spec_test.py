from dataclasses import dataclass
from acadia.runtime import Runtime
from acadia.system import Waveform


@dataclass
class QubitSpecRuntime(Runtime):
    """
    A :class:`Runtime` subclass for sending a signal out of a DAC and capturing
    it on an ADC.
    """

    qubit_frequencies: list 
    qubit_stimulus: dict
    readout_stimulus: dict
    readout_capture: dict

    iterations: int
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
        qubit_channel = acadia.channel(self.qubit_stimulus["channel"])
        readout_stimulus_channel = acadia.channel(self.readout_stimulus["channel"])
        readout_capture_channel = acadia.channel(self.readout_capture["channel"])


        # Create the waveforms that we'll need 
        qubit_stimulus_waveform = acadia.create_waveform(qubit_channel, **self.qubit_stimulus["waveform"])
        readout_stimulus_waveform = acadia.create_waveform(readout_stimulus_channel, **self.readout_stimulus["waveform"])
        readout_capture_waveform = acadia.create_waveform(readout_capture_channel, **self.readout_capture["waveform"]) 
        
        blank_wf = acadia.create_waveform(readout_stimulus_channel, length=200e-9, blank=True) 
        #todo: add wait length to input

        # assemble channel objects
        channels_ = [qubit_channel, readout_stimulus_channel, readout_capture_channel]
        channel_configs = [self.qubit_stimulus, self.readout_stimulus, self.readout_capture]
        channel_wfs = [qubit_stimulus_waveform, readout_stimulus_waveform, readout_capture_waveform]
        
        # Create a record group for saving captured data
        self.data.add_group(f"traces", uniform=True)
        
        # todo: this could be simplified
        kernel_wf = self.readout_capture["kernel_wf"]
        if type(kernel_wf) == float: # constant value kernel
            kernel_wf = np.float64(kernel_wf)
            kernel_cmacc = None
        elif type(kernel_wf) == np.ndarray:
            kernel_cmacc = kernel_wf


        # Create a sequence for the sequencer to generate the pulse and capture it
        def sequence(a: Acadia):
            capture_stream, kernel = acadia.configure_cmacc(readout_capture_channel, kernel=kernel_cmacc, reset_fifo=True)

            acadia.cmacc_load(capture_stream, 0)

            with a.channel_synchronizer():
                a.schedule_waveform(qubit_stimulus_waveform)
                a.barrier()
                # a.schedule_waveform(blank_wf)
                # a.barrier()
                a.schedule_waveform(readout_stimulus_waveform)
                a.stream(capture_stream, readout_capture_waveform)


            return kernel

        # Compile the sequence
        kernel = acadia.compile(sequence)
                
        # Attach to the hardware
        acadia.attach()


        # Configure channel analog parameters
        # todo: this should be written as a generic function that automatically resets all channels used
        acadia.align_tile_latencies()
        for ch, config, wf in zip(channels_, channel_configs, channel_wfs):
            if "signal" in config: # is DAC channel
                wf.set(**config["signal"])
            ch.set(nco_update_event_source="sysref", **config["datapath"])
            acadia.reset_nco_phase(ch)
            acadia.update_nco_phase(ch, 0)
        acadia.update_ncos_synchronized()


        # Set the amplitude of the boxcar integration kernel
        kernel.set(kernel_wf)

        # Assemble and load the program
        acadia.assemble()
        acadia.load()

        for i in range(self.iterations):
            for freq_idx, freq in enumerate(self.qubit_frequencies):
                # set the qubit nco freq
                acadia.update_nco_frequency(qubit_channel, freq)
                acadia.update_ncos_synchronized()

                # capture data and put in the corresponding group
                acadia.run()
                self.data[f"traces"].write(readout_capture_waveform.array)
            
            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return
        
        self.final_serve()

    def initialize(self):
        # Set the matplotlib backend to one which we can actually update
        self.figsize = (4, 3) if self.figsize is None else self.figsize

        if self.plot:
            from acadia.processing import DynamicLine
            import matplotlib.pyplot as plt

            self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize)
            self.fig.tight_layout()
            self.fig.subplots_adjust(left=0.25, bottom=0.25)

            self.line_re = DynamicLine(self.ax, ".-")
            self.line_im = DynamicLine(self.ax, ".-")

        from tqdm.notebook import tqdm
        self.progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.previous_completed_iterations = 0

    def update(self):
        import numpy as np
        import matplotlib.pyplot as plt

        n_freqs = len(self.qubit_frequencies)

        # First make sure that we actually have new data to process
        if f"traces" not in self.data or len(self.data[f"traces"]) < n_freqs:
            return

        # Update the progress bar based on the number of iterations
        completed_iterations = len(self.data["traces"]) // n_freqs
        self.progress_bar.update(completed_iterations - self.previous_completed_iterations)
        self.previous_completed_iterations = completed_iterations        


        valid_traces = completed_iterations*n_freqs
        data = self.data["traces"].records()[:valid_traces, ...].squeeze()

        # Get the collection of data and reshape it so that the axes index as: 
        # (iteration, frequency, sample quadrature)
        data_reshaped = data.reshape(completed_iterations, n_freqs, 2)
        data_sum = np.sum(data_reshaped, axis=0)

        if self.plot:
            # self.hist.update(data)
            self.line_re.update(self.qubit_frequencies, data_sum[:,0])
            self.line_im.update(self.qubit_frequencies, data_sum[:,1])
            self.ax.relim()
            self.ax.autoscale(tight=True)
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
    from linc_rfsoc.analysis.generate_readout_kernel import ReadoutKernelGenerator, load_kernel
    
    from IPython.core.getipython import get_ipython
    get_ipython().run_line_magic("matplotlib", "widget")

    qubit_stimulus: dict = {
        "channel": "DAC1",

        "datapath": {
            "vop": 30000,
            "mix_reconstruction": False,
            "nco_frequency": 40e6
        },

        "waveform": {
            "length": 600e-9*1,  # this only works when length = 200e-9
            "fixed_length": 0.0
        },
        
        "signal": {
            "data": ("scipy", "hann"),
            "scale": 0.8
        }
    }

    readout_stimulus: dict = {
        "channel": "DAC4",

        "datapath": {
            "vop": 30000,
            "mix_reconstruction": False,
            "nco_frequency": 40e6
            # "nco_frequency": 8.23e9
        },

        "waveform": {
            "length": 200e-9*1, 
            "fixed_length": 2e-6
        },
        
        "signal": {
            "data": ("scipy", "hann"),
            "scale": 0.8
        }
    }

    readout_capture: dict = {
        "channel": "ADC0",

        "datapath": {
            "nco_frequency": -9.03002e9
        },

        "waveform": {
            "length": 2000*1.25e-9,
            "decimation": 0,
            "region": "plddr"
        },

        "kernel_wf": 0.1
    }

    plot = True
    iterations = 500
    qubit_freqs = np.array([50e6])


    rt = QubitSpecRuntime(qubit_freqs, qubit_stimulus, readout_stimulus, readout_capture, plot=plot, iterations=iterations)
    rt.deploy("10.66.3.198", "qubit_spec_test", files=[rt.FILE])    
    rt.display()

    # some ad hoc processing
    # rt._event_loop.join()
    # rt.fig

    # data = rt.data["traces"].records().astype(float).view(complex).squeeze()
    # data = data.reshape(-1, len(qubit_freqs))
    # data_avg = np.mean(data, axis=0)
    # fig, axs = plt.subplots(2, 1, figsize=(6,6))
    # axs[0].plot(qubit_freqs, data_avg.real, ".-")
    # axs[1].plot(qubit_freqs, data_avg.imag, ".-")

    