from dataclasses import dataclass
from typing import Union
import numpy as np
from numpy.typing import NDArray
from acadia.runtime import Runtime
from acadia import DataManager

from auto_config import AutoConfigMixin

@dataclass
class QubitSpecRuntime(AutoConfigMixin, Runtime):
    """
    A :class:`Runtime` subclass for qubit spectroscopy
    """

    q_stimulus: dict
    ro_stimulus: dict
    ro_capture: dict

    qubit_frequencies: Union[list, np.ndarray]

    iterations: int
    plot: bool = True
    figsize: tuple[int] = None

    def __post_init__(self):
        self.FILES = [__file__, super().FILE]

    def main(self):
        from acadia import Acadia, DataManager
        import logging

        logger = logging.getLogger("acadia")

        channel_configs = {"q_stimulus": self.q_stimulus,
                           "ro_stimulus": self.ro_stimulus,
                           "ro_capture": self.ro_capture
                           }

        # Create an acadia object and grab a couple of its channels
        acadia = Acadia()
        channel_objs = self.obtain_channels(acadia, **channel_configs)

        # Allocate the waveform memories that we'll need
        q_rotation = self.allocate_waveform_mem(acadia, "q_stimulus", "q_rotation")
        ro_drive = self.allocate_waveform_mem(acadia, "ro_stimulus", "ro_drive")
        ro_pts = self.allocate_waveform_mem(acadia, "ro_capture", "ro_pts")

        # Create a blank waveform between qubit drive and readout drive
        q_blank_wf = self.blank_waveform_generator(acadia, "q_stimulus")(40e-9)

        # Create blank waveform for delay between capture and stimulus
        capture_delay = self.ro_capture["capture_delay"]
        if capture_delay < 0: # stimulus will be delayed by -capture_delay compare to capture
            blank_wf = self.blank_waveform_generator(acadia, "ro_stimulus")(-capture_delay)
        elif capture_delay > 0: # capture will be delayed by capture_delay compare to stimulus
            blank_wf = self.blank_waveform_generator(acadia, "ro_capture", "ro_pts")(capture_delay)

        # get the kernel and IQ offsets fom the config dict
        kernel_wf = self.ro_capture.get("kernel_wf", [0.1])
        cmacc_offset = self.ro_capture.get("cmacc_offset", 0)

        # Create the record groups for saving captured data
        self.data.add_group(f"iq_pts", uniform=True)
        self.data.add_group(f"freqs", uniform=False)
        self.data[f"freqs"].write(self.qubit_frequencies)

        # Create a sequence for the sequencer to generate the pulse and capture it
        def sequence(a: Acadia):
            capture_stream, kernel = a.configure_cmacc(channel_objs["ro_capture"], kernel=kernel_wf)

            a.cmacc_load(capture_stream, cmacc_offset)

            with a.channel_synchronizer():
                a.schedule_waveform(q_rotation)
                a.schedule_waveform(q_blank_wf)
                a.barrier()
                if capture_delay != 0:
                    a.schedule_waveform(blank_wf)
                a.schedule_waveform(ro_drive)
                a.stream(capture_stream, ro_pts)

            return kernel

        # Compile the sequence
        kernel = acadia.compile(sequence)
        # Attach to the hardware
        acadia.attach()
        # Configure channel analog parameters
        self.auto_config_ncos(acadia, **channel_configs)
        # Assemble and load the program
        acadia.assemble()
        acadia.load()

        # Set the amplitude of the boxcar integration kernel
        kernel.set(kernel_wf)
        # set waveform for ro drive
        ro_drive.set(**self.ro_stimulus["signals"]["readout"])
        q_rotation.set(**self.q_stimulus["signals"]["pi_pulse"])

        for i in range(self.iterations):
            for freq_idx, freq in enumerate(self.qubit_frequencies):
                # set the qubit nco freq
                acadia.update_nco_frequency(self.channel_objs["q_stimulus"], freq)
                acadia.update_ncos_synchronized()

                # capture data and put in the corresponding group
                acadia.run()
                self.data[f"iq_pts"].write(ro_pts.array)

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

            self.line_re = DynamicLine(self.ax, ".-", label="I")
            self.line_im = DynamicLine(self.ax, ".-", label="I")
            self.ax.set_xlabel("q drive frequency")
            self.ax.grid()

        from tqdm.notebook import tqdm
        self.progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.previous_completed_iterations = 0

    def update(self):
        import numpy as np

        n_freqs = len(self.qubit_frequencies)

        # First make sure that we actually have new data to process
        if "iq_pts" not in self.data or len(self.data["iq_pts"]) < n_freqs:
            return

        # Update the progress bar based on the number of iterations
        completed_iterations = len(self.data["iq_pts"]) // n_freqs
        self.progress_bar.update(completed_iterations - self.previous_completed_iterations)
        self.previous_completed_iterations = completed_iterations

        if self.plot:
            valid_traces = completed_iterations * n_freqs
            data = self.data["iq_pts"].records()[:valid_traces, ...].squeeze()

            # Get the collection of data and reshape it so that the axes index as: 
            # (iteration, frequency, sample quadrature)
            data_reshaped = data.reshape(completed_iterations, n_freqs, 2)
            data_summed = np.sum(data_reshaped, axis=0)

            self.line_re.update(self.qubit_frequencies, data_summed[:, 0])
            self.line_im.update(self.qubit_frequencies, data_summed[:, 1])
            self.fig.tight_layout()
            # self.fig.canvas.draw_idle()

        # Save the data
        self.data.save(self.local_directory)

    def finalize(self):
        super().finalize()
        self.progress_bar.close()
        if self.plot:
            self.savefig(self.fig)
        self.post_process()

    def post_process(self) -> float:
        """ Fit the spectroscopy to a Lorentzian function to extract the qubit on-resonance frequency.

        """
        from acadia_qmsmt.analysis import population_in_quadrant
        from acadia_qmsmt.analysis.fitting import rotate_iq
        from acadia_qmsmt.analysis.fitting.lorentzian import Lorentzian
        from acadia_qmsmt.measurements import CONFIG_FILE_PATH
        from acadia_qmsmt.helpers.plot_utils import add_button
        from acadia_qmsmt.helpers.yaml_editor import update_yaml

        freqs, iq_pts = self.parse_data(self.data)
        try:
            e_quadrant = self.ro_capture.get("state_quadrants", None)[1]
            data_to_fit = population_in_quadrant(iq_pts, e_quadrant, axis=0)
            data_type = "e population"
        except TypeError:
            iq_pts = np.mean(iq_pts, axis=0)
            data_to_fit = rotate_iq(iq_pts).real
            data_type = "rotated I component"

        fit = Lorentzian(freqs, data_to_fit)
        f0 = fit.ufloat_results["x0"]
        fit_fig, fit_ax = fit.plot()
        fit_ax.set_xlabel("q drive frequency")
        fit_ax.set_ylabel(data_type)

        def _update_q_freq(event):
            new_param_dict = {"q_stimulus.nco_config.nco_frequency": np.round(f0.n)}
            update_yaml(CONFIG_FILE_PATH, new_param_dict, verbose=True)

        # add a button for updating the qubit freq in the yaml file
        self._update_button = add_button(fit_fig, _update_q_freq, label="Update Qubit Freq")

        return f0


    @staticmethod
    def parse_data(data_manager: DataManager) -> NDArray[complex]:
        """Parse the acquired data in data_manager to a ready-to-process format

        :param data_manager: data manager object for data acquired using this runtime
        """
        dm = data_manager
        freqs = dm["freqs"].records()[0].astype(float).squeeze()
        iq_pts = dm["iq_pts"].records().astype(float).view(complex).squeeze()
        iq_pts = np.reshape(iq_pts, (-1, len(freqs)))
        return freqs, iq_pts




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from acadia_qmsmt.measurements import load_config

    from IPython.core.getipython import get_ipython
    get_ipython().run_line_magic("matplotlib", "widget")

    config_dict = load_config()

    plot = True
    iterations = 50
    qubit_freqs = np.linspace(-20e6, 20e6, 101) + 8.23e9

    rt = QubitSpecRuntime(**config_dict, qubit_frequencies=qubit_freqs, plot=plot, iterations=iterations)
    rt.deploy("10.66.3.198", "qubit_spec", files=rt.FILES)
    rt.display()

    # some ad-hoc processing
    # rt._event_loop.join()
    # rt.fig
