from dataclasses import dataclass
from typing import Union, Tuple, Literal
import numpy as np
from numpy.typing import NDArray
from acadia.runtime import Runtime
from acadia import DataManager

from auto_config import AutoConfigMixin
from auto_config import FILE as config_helper_file

@dataclass
class QubitPwrRabiRuntime(AutoConfigMixin, Runtime):
    """
    A :class:`Runtime` subclass for qubit powerRabi
    """

    q_stimulus: dict
    ro_stimulus: dict
    ro_capture: dict

    qubit_amp_scales: Union[list, np.ndarray]

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
            blank_wf = self.blank_waveform_generator(acadia, "ro_capture", "ro_demod")(capture_delay)

        # get the kernel and IQ offsets fom the config dict
        kernel_wf = self.ro_capture.get("kernel_wf", [0.1])
        cmacc_offset = self.ro_capture.get("cmacc_offset", 0)

        # Create the record groups for saving captured data
        self.data.add_group(f"iq_pts", uniform=True)
        self.data.add_group(f"amp_scales", uniform=False)
        self.data[f"amp_scales"].write(self.qubit_amp_scales)

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

        pi_signal = self.q_stimulus["signals"]["pi_pulse"]
        amp0 = pi_signal["scale"]
        for i in range(self.iterations):
            for amp_idx, amp_scale in enumerate(self.qubit_amp_scales):
                # set the pulse amp
                pi_signal["scale"] = amp0 * amp_scale
                q_rotation.set(**pi_signal)

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

            self.line_re = DynamicLine(self.ax, ".-")
            self.line_im = DynamicLine(self.ax, ".-")
            self.ax.set_xlabel("q drive relative scale")
            self.ax.grid()

        from tqdm.notebook import tqdm
        self.progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.previous_completed_iterations = 0

    def update(self):
        import numpy as np

        n_amps = len(self.qubit_amp_scales)

        # First make sure that we actually have new data to process
        if f"iq_pts" not in self.data or len(self.data[f"iq_pts"]) < n_amps:
            return

        # Update the progress bar based on the number of iterations
        completed_iterations = len(self.data["iq_pts"]) // n_amps
        self.progress_bar.update(completed_iterations - self.previous_completed_iterations)
        self.previous_completed_iterations = completed_iterations

        if self.plot:
            valid_traces = completed_iterations * n_amps
            data = self.data["iq_pts"].records()[:valid_traces, ...].squeeze()

            # Get the collection of data and reshape it so that the axes index as: 
            # (iteration, amps, sample quadrature)
            data_reshaped = data.reshape(completed_iterations, n_amps, 2)
            data_summed = np.sum(data_reshaped, axis=0)

            self.line_re.update(self.qubit_amp_scales, data_summed[:,0])
            self.line_im.update(self.qubit_amp_scales, data_summed[:,1])
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
        """ Fit the Rabi result to a sinusoidal to extract the qubit pi-pulse amplitude.

        """
        #todo: this is getting really bloated...
        from linc_rfsoc.analysis import population_in_quadrant
        from linc_rfsoc.analysis.fitting import rotate_iq
        from linc_rfsoc.analysis.fitting.cosine import Cosine
        from linc_rfsoc.measurements import CONFIG_FILE_PATH
        from linc_rfsoc.helpers.plot_utils import add_button
        from linc_rfsoc.helpers.yaml_editor import update_yaml

        amps, iq_pts = self.parse_data(self.data)
        try:
            e_quadrant = self.ro_capture.get("state_quadrants", None)[1]
            data_to_fit = population_in_quadrant(iq_pts, e_quadrant, axis=0)
            data_type = "e population"
        except TypeError:
            iq_pts = np.mean(iq_pts, axis=0)
            data_to_fit = rotate_iq(iq_pts).real
            data_type = "rotated I component"

        fit = Cosine(amps, data_to_fit)
        scale2 = abs(1/fit.ufloat_results["f"]/2)
        fit_fig, fit_ax = fit.plot()
        fit_ax.set_xlabel("q drive amplitude factor")
        fit_ax.set_ylabel(data_type)

        def _update_q_freq(event):
            #todo: currently the amp is loaded as a complex number, but this is a good example
            # that shows using amp and phase might be more convenient
            amp0 = abs(self.q_stimulus["signals"]["pi_pulse"]["scale"]) 
            new_param_dict = {"q_stimulus.signals.pi_pulse.scale": np.round(amp0*scale2.n, 4)}
            update_yaml(CONFIG_FILE_PATH, new_param_dict, verbose=True)

        # add a button for updating the qubit freq in the yaml file
        self._update_button = add_button(fit_fig, _update_q_freq, label="Update Qubit Amp")

        return scale2


    @staticmethod
    def parse_data(data_manager: DataManager) -> NDArray[complex]:
        """Parse the acquired data in data_manager to a ready-to-process format

        :param data_manager: data manager object for data acquired using this runtime
        """
        dm = data_manager
        amps = dm["amp_scales"].records()[0].astype(float).squeeze()
        iq_pts = dm["iq_pts"].records().astype(float).view(complex).squeeze()
        iq_pts = np.reshape(iq_pts, (-1, len(amps)))
        return amps, iq_pts




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from linc_rfsoc.measurements import load_config

    from IPython.core.getipython import get_ipython
    get_ipython().run_line_magic("matplotlib", "widget")

    config_dict = load_config()

    plot = True
    iterations = 200
    qubit_amp_scales = np.linspace(-1.5, 1.5, 61) # not the scale in "signal", is multiply factor on that


    rt = QubitPwrRabiRuntime(**config_dict, qubit_amp_scales=qubit_amp_scales, plot=plot, iterations=iterations)
    rt.deploy("10.66.3.198", "qubit_power_rabi", files=[rt.FILE, config_helper_file])    
    rt.display()

    # some ad-hoc processing
    # rt._event_loop.join()
    # rt.fig
