from dataclasses import dataclass
from acadia.runtime import Runtime
from acadia.system import Waveform


@dataclass
class ReadoutTestRuntime_cmacc(Runtime):
    """
    A :class:`Runtime` subclass for sending a signal out of a DAC and capturing
    it on an ADC.
    """

    stimulus: dict
    capture: dict
    phases: tuple 
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
        stimulus_channel = acadia.channel(self.stimulus["channel"])
        capture_channel = acadia.channel(self.capture["channel"])

        # Create the waveforms that we'll need 
        stimulus_waveform = acadia.create_waveform(stimulus_channel, **self.stimulus["waveform"])
        capture_waveform = acadia.create_waveform(capture_channel, **self.capture["waveform"]) 
        
        # Create a record group for saving captured data
        self.data.add_group(f"traces", uniform=True)
        
        kernel_wf = self.capture["kernel_wf"]
        if type(kernel_wf) == float: # constant value kernel
            kernel_wf = np.float64(kernel_wf)
            kernel_cmacc = None
        elif type(kernel_wf) == np.ndarray:
            kernel_cmacc = kernel_wf

        # Create a sequence for the sequencer to generate the pulse and capture it
        def sequence(a: Acadia):
            capture_stream, kernel = acadia.configure_cmacc(capture_channel, kernel=kernel_cmacc, reset_fifo=True)

            acadia.cmacc_load(capture_stream, 0)

            with a.channel_synchronizer():
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
                self.data[f"traces"].write(capture_waveform.array)
            
            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return
        
        self.final_serve()

    def initialize(self):
        # Set the matplotlib backend to one which we can actually update
        self.figsize = (4, 3) if self.figsize is None else self.figsize

        if self.plot:
            from acadia.processing import DynamicReadoutHistogram
            import matplotlib.pyplot as plt

            self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize)
            self.fig.tight_layout()
            self.fig.subplots_adjust(left=0.25, bottom=0.25)

            self.hist = DynamicReadoutHistogram(self.ax)

        from tqdm.notebook import tqdm
        self.progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.previous_completed_iterations = 0

    def update(self):
        import numpy as np
        import matplotlib.pyplot as plt

        # First make sure that we actually have new data to process
        if f"traces" not in self.data or len(self.data[f"traces"]) == 0:
            return

        # Update the progress bar based on the number of iterations that have been completed
        completed_iterations = len(self.data[f"traces"])
        self.progress_bar.update(completed_iterations - self.previous_completed_iterations)
        self.previous_completed_iterations = completed_iterations        


 

        data = self.data["traces"].records().squeeze()

        if self.plot:
            # self.hist.update(data)
            self.ax.hist2d(data[:, 0], data[:, 1], bins=51)
            self.ax.relim()
            self.ax.autoscale(tight=True)
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            self.ax.set_aspect(1)

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
    from linc_rfsoc.analysis.generate_readout_kernel import KernelGeneratorBase, load_kernel
    
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
            "length": 200e-9, 
            "fixed_length": 200e-9 
        },
        
        "signal": {
            "data": ("scipy", "hann"),
            "scale": 0.05
        }
    }

    capture: dict = {
        "channel": "ADC0",

        "datapath": {
            "nco_frequency": -9.03e9
        },

        "waveform": {
            "length": 800*1.25e-9,
            "decimation": 0,
            "region": "plddr"
        },

        # "kernel_wf": np.repeat(load_kernel("test_kernel.npy"), 4)
        # "kernel_wf": load_kernel("test_kernel.npy")
        # "kernel_wf": np.concatenate([np.array([0.1+0j]*200), np.array([0.0+0j]*200)])
        # "kernel_wf":np.array([0.1+0j]*200)
        "kernel_wf": 0.1
    }

    plot = True
    iterations = 1000
    # phases = (0, np.pi/2, np.pi)
    phases = np.array([0, np.pi])


    rt = ReadoutTestRuntime_cmacc(stimulus, capture, phases, plot=plot, iterations=iterations)
    rt.deploy("10.66.3.198", "readout_kernel_test_use", files=[rt.FILE])    
    rt.display()

    # some ad hoc processing
    rt._event_loop.join()
    rt.fig

    data = rt.data["traces"].records().squeeze()
    fig, ax = plt.subplots(1, 1)
    ax.plot(data[:, 0], data[:, 1], "o")
    ax.set_aspect(1)

    #
    # rk = ReatoutKernelGenerator(all_data, (-0.2 + 1.72j, 0.5), (0.3  -1.73j, 0.5))
    # rk.save_kernel(r"../dev_codes//")
    # kernel=load_kernel(r"../dev_codes//"+"readoutkernel_241105_113318.npy")
    #

    