from dataclasses import dataclass
from typing import Union
import numpy as np
from acadia.runtime import Runtime

from auto_config import AutoConfigMixin
from auto_config import FILE as config_helper_file


@dataclass
class QubitSpecRuntime(AutoConfigMixin, Runtime):
    """
    A :class:`Runtime` subclass for sending a signal out of a DAC and capturing
    it on an ADC.
    """

    q_stimulus: dict
    ro_stimulus: dict
    ro_capture: dict

    qubit_frequencies: Union[list, np.ndarray]

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
        q_rotation = self.channel_waveforms["q_stimulus"]["q_rotation"]
        ro_drive = self.channel_waveforms["ro_stimulus"]["ro_drive"]
        ro_demod = self.channel_waveforms["ro_capture"]["ro_demod"]

        # Create a blank waveform between qubit drive and readout drive
        q_blank_wf = self.blank_waveform_generator(acadia, "q_stimulus")(40e-9)

        # Create blank waveform for delay between capture and stimulus
        capture_delay = self.ro_capture["capture_delay"]
        if capture_delay < 0:  # capture will be advanced by -capture_delay compare to stimulus
            blank_wf = self.blank_waveform_generator(acadia, "ro_stimulus")(-capture_delay)
        elif capture_delay > 0:  # capture will be delayed by capture_delay compare to stimulus
            blank_wf = self.blank_waveform_generator(acadia, "ro_capture")(capture_delay)

        kernel_wf = self.ro_capture.get("kernel_wf", 0.1)
        if type(kernel_wf) == float:  # constant value kernel
            kernel_wf = np.float64(kernel_wf)
            kernel_cmacc = None
        elif type(kernel_wf) == np.ndarray:
            kernel_cmacc = kernel_wf

        kernel_offset = self.ro_capture.get("kernel_offset", 0)

        # Create the record groups for saving captured data
        self.data.add_group(f"spec", uniform=True)

        # Create a sequence for the sequencer to generate the pulse and capture it
        def sequence(a: Acadia):
            capture_stream, kernel = acadia.configure_cmacc(self.channel_objs["ro_capture"], kernel=kernel_cmacc,
                                                            reset_fifo=True)

            acadia.cmacc_load(capture_stream, kernel_offset)

            with a.channel_synchronizer():
                a.schedule_waveform(q_rotation)
                a.schedule_waveform(q_blank_wf)
                a.barrier()
                if capture_delay != 0:
                    a.schedule_waveform(blank_wf)
                a.schedule_waveform(ro_drive)
                a.stream(capture_stream, ro_demod)

            return kernel

        # Compile the sequence
        kernel = acadia.compile(sequence)

        # Attach to the hardware
        acadia.attach()

        # Configure channel analog parameters
        self.auto_config_ncos(acadia, **channel_configs)

        # Set the amplitude of the boxcar integration kernel
        kernel.set(kernel_wf)

        # Assemble and load the program
        acadia.assemble()
        acadia.load()

        # set waveform for ro drive
        ro_drive.set(**self.ro_stimulus["signals"]["readout"])

        for i in range(self.iterations):
            for freq_idx, freq in enumerate(self.qubit_frequencies):
                # set the qubit nco freq
                acadia.update_nco_frequency(self.channel_objs["q_stimulus"], freq)
                acadia.update_ncos_synchronized()

                # capture data and put in the corresponding group
                acadia.run()
                self.data[f"spec"].write(ro_demod.array)

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
        if "spec" not in self.data or len(self.data["spec"]) < n_freqs:
            return

        # Update the progress bar based on the number of iterations
        completed_iterations = len(self.data["spec"]) // n_freqs
        self.progress_bar.update(completed_iterations - self.previous_completed_iterations)
        self.previous_completed_iterations = completed_iterations

        valid_traces = completed_iterations * n_freqs
        data = self.data["spec"].records()[:valid_traces, ...].squeeze()

        # Get the collection of data and reshape it so that the axes index as: 
        # (iteration, frequency, sample quadrature)
        data_reshaped = data.reshape(completed_iterations, n_freqs, 2)
        data_sum = np.sum(data_reshaped, axis=0)

        if self.plot:
            # self.hist.update(data)
            self.line_re.update(self.qubit_frequencies, data_sum[:, 0])
            self.line_im.update(self.qubit_frequencies, data_sum[:, 1])
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
    from linc_rfsoc.measurements import load_config
    from linc_rfsoc.analysis.generate_readout_kernel import ReadoutKernelGenerator, load_kernel

    from IPython.core.getipython import get_ipython

    get_ipython().run_line_magic("matplotlib", "widget")

    config_dict = load_config()

    plot = True
    iterations = 500
    qubit_freqs = np.linspace(-20e6, 20e6, 101) + 8.23e9

    rt = QubitSpecRuntime(**config_dict, qubit_frequencies=qubit_freqs, plot=plot, iterations=iterations)
    rt.deploy("10.66.3.198", "qubit_spec", files=[rt.FILE, config_helper_file])
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
