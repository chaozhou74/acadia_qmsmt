from dataclasses import dataclass
from acadia.runtime import Runtime
from acadia import DataManager

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
    generate_kernel: bool = False

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
        self.obtain_channels(acadia, **channel_configs)

        # Allocate the waveform memories that we'll need
        q_rotation = self.allocate_waveform_mem(acadia, "q_stimulus", "q_rotation")
        ro_drive = self.allocate_waveform_mem(acadia, "ro_stimulus", "ro_drive")
        ro_demod = self.allocate_waveform_mem(acadia, "ro_capture", "ro_demod")
        
        # Create a blank waveform between qubit drive and readout drive
        q_blank_wf = self.blank_waveform_generator(acadia, "q_stimulus")(40e-9)

        # Create blank waveform for delay between capture and stimulus
        capture_delay = self.ro_capture["capture_delay"]
        if capture_delay < 0: # stimulus will be delayed by -capture_delay compare to capture
            blank_wf = self.blank_waveform_generator(acadia, "ro_stimulus")(-capture_delay)
        elif capture_delay > 0: # capture will be delayed by capture_delay compare to stimulus
            blank_wf = self.blank_waveform_generator(acadia, "ro_capture", "ro_demod")(capture_delay)
        
        # Create the record groups for saving captured data
        self.data.add_group("traces_g", uniform=True)
        self.data.add_group("traces_e", uniform=True)
        self.data.add_group("t_data", uniform=False)

        # Create a sequence for the sequencer to generate the pulse and capture it
        def sequence(a: Acadia):
            capture_stream = acadia.configure_dsp(self.channel_objs["ro_capture"], 
                                                  self.ro_capture["waveforms"]["ro_demod"]["decimation"])
        
            with a.channel_synchronizer():
                a.schedule_waveform(q_rotation)
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
        ro_drive.set(**self.ro_stimulus["signals"]["readout"])

        # average iterations for preparing qubit in g and e
        pi_signal = self.q_stimulus["signals"]["pi_pulse"]
        pi_amp = pi_signal["scale"]
        for i in range(self.iterations):
            for state_ in ["g", "e"]:
                # set the pulse amplitude
                pi_signal["scale"] = 0 if state_ == "g" else pi_amp
                q_rotation.set(**pi_signal) # takes around 1ms

                # capture data and put in the corresponding group
                acadia.run()
                self.data[f"traces_{state_}"].write(ro_demod.array)

            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return
        
        capture_time = self.ro_capture["waveforms"]["ro_demod"]["length"] 
        t_data = np.linspace(0, capture_time, len(ro_demod.array), endpoint=False)
        self.data["t_data"].write(t_data)

        self.final_serve()

    def initialize(self):
        # Set the matplotlib backend to one which we can actually update
        self.figsize = (4, 7) if self.figsize is None else self.figsize

        if self.plot:
            from acadia.processing import DynamicLine
            import matplotlib.pyplot as plt

            self.fig, self.axs = plt.subplots(2, 1, figsize=self.figsize)

            self.lines_re = []
            self.lines_im = []
            for j, state_ in enumerate(["g", "e"]):
                self.lines_re.append(DynamicLine(self.axs[j], ".-"))
                self.lines_im.append(DynamicLine(self.axs[j], ".-"))
                self.axs[j].set_xlabel("Time [s]")
                self.axs[j].set_ylabel("Accumulated Amp [arb. V]")
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

        if self.plot:
            # We'll make an x-axis for the plot, but we only need to make it once
            if self.time_axis is None:
                # Last index is for quadrature
                samples_per_trace = np.array(self.data["traces_g"].records()).shape[-2]
                capture_time = self.ro_capture["waveforms"]["ro_demod"]["length"] 
                self.time_axis = np.linspace(0, capture_time, samples_per_trace, endpoint=False)

            # Sum the traces from each iteration
            # The records have shape (iterations, samples, 2)
            self.trace_summed = np.zeros((2, len(self.time_axis), 2), dtype=np.int64)
            for j, state_ in enumerate(["g", "e"]):
                self.trace_summed[j] = np.sum(self.data[f"traces_{state_}"].records(), axis=0)
        
            for i in range(2):
                self.lines_re[i].update(self.time_axis, self.trace_summed[i, :,0])
                self.lines_im[i].update(self.time_axis, self.trace_summed[i, :,1])
                self.fig.tight_layout()
                # self.fig.canvas.draw_idle()

        # Save the data
        self.data.save(self.local_directory)

    def finalize(self):
        super().finalize()
        self.progress_bar.close()
        if self.plot:
            self.savefig(self.fig)
        
        if self.generate_kernel:
            self.post_processing()


    def post_processing(self):
        # generate readout kernel based on the acquired traces
        from linc_rfsoc.measurements import CONFIG_FILE_PATH
        from linc_rfsoc.analysis.generate_readout_kernel import KernelFromPreparedTraces
        from linc_rfsoc.helpers.plot_utils import add_button

        t_data, g_traces, e_traces = self.parse_data(self.data)
        kernel_gen = KernelFromPreparedTraces(g_traces, e_traces, norm_factor=1, plot=True,
                                              decimation_used=self.ro_capture["waveforms"]["ro_demod"]["decimation"])
        update_kernel = lambda _: kernel_gen.update_kernel(CONFIG_FILE_PATH, "ro_capture.kernel_wf",
                                                           self.local_directory)
        # make a kernel update button and maintain a reference to it for keeping it alive
        self._update_button = add_button(kernel_gen.fig_kernel_gen, update_kernel, label="Update Kernel")


    @staticmethod
    def parse_data(data_manager: DataManager):
        """Parse the acquired data in data_manager to an easy-to-process format

        :param data_manager: data manager object for data acquired using this runtime
        """
        dm = data_manager
        g_traces = np.array(dm["traces_g"].records()).astype(float).view(complex).squeeze()
        e_traces = np.array(dm["traces_e"].records()).astype(float).view(complex).squeeze()
        t_data = np.array(dm["t_data"].records()).astype(float).squeeze()

        return t_data, g_traces, e_traces



    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from linc_rfsoc.measurements import load_config, CONFIG_FILE_PATH

    from IPython.core.getipython import get_ipython
    get_ipython().run_line_magic("matplotlib", "widget")

    config_dict = load_config()

    plot = True
    iterations = 5000


    rt = ReadoutTracesRuntime(**config_dict, plot=plot, iterations=iterations, generate_kernel=True)
    rt.deploy("10.66.3.198", "readout_traces", files=[rt.FILE, config_helper_file], event_loop_period=0.5)
    rt.display()

    # # some ad hoc processing
    # rt._event_loop.join() # this will stop live plotting from working

    # # generate kernel using the acquired data
    # t_data, g_traces, e_traces = rt.parse_data(rt.data)
    # from linc_rfsoc.analysis.generate_readout_kernel import KernelFromPreparedTraces
    # kernel_gen = KernelFromPreparedTraces(g_traces, e_traces, norm_factor=1, plot=True,
    #                                        decimation_used=rt.ro_capture["waveforms"]["ro_demod"]["decimation"])
    # kernel_gen.update_kernel(CONFIG_FILE_PATH, "ro_capture.kernel_wf", r"../dev_codes", "test_kernel")


