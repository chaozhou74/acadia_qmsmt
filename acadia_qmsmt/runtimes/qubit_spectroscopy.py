from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime
from acadia.sample_arithmetic import sample_to_complex
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig
from acadia.runtime import annotate_method

import logging
logger = logging.getLogger("acadia")

class QubitSpectroscopyRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` subclass for qubit spectroscopy
    """
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    qubit_frequencies: Union[list, np.ndarray]

    iterations: int
    run_delay: int

    saturation_pulse_config: dict = None
    readout_pulse_name: str = "readout"
    capture_memory_name: str = "readout_accumulated"
    capture_window_name: str = "boxcar"

    figsize: tuple[int] = None
    yaml_path: str = None

    def main(self):
        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")

        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)
        qubit = Qubit(qubit_stimulus_io)

        self.data.add_group(f"points", uniform=True)
        def sequence(a: Acadia):

            with a.channel_synchronizer():
                qubit_stimulus_io.schedule_pulse(self.saturation_pulse_config)
                a.barrier()
                readout_resonator.measure(self.readout_pulse_name, self.capture_memory_name, self.capture_window_name)

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_pulse(self.readout_pulse_name)
        qubit_stimulus_io.load_pulse(self.saturation_pulse_config)
        

        for i in range(self.iterations):
            for frequency in self.qubit_frequencies:
                qubit.set_frequency(frequency)

                # capture data and put in the corresponding group
                self.acadia.run(minimum_delay=self.run_delay)

                wf = readout_capture_io.get_waveform_memory(self.capture_memory_name)
                self.data[f"points"].write(wf.array)

            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return

        self.final_serve()


    def initialize(self):
        pass

    def update(self):
        # save current data
        self.data.save(self.local_directory)

    def finalize(self):
        super().finalize()
        from acadia_qmsmt.plotting import save_registered_plots
        save_registered_plots(self)



    @annotate_method(is_data_processor=True)
    def process_current_data(self, thresholded:bool=False):
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes, find_iq_rotation, rotate_iq
        data_spec = reshape_iq_data_by_axes(self.data["points"].records(), self.qubit_frequencies)
        if data_spec is None:
            return
        else:
            completed_iterations = len(data_spec)
        self.data_iq = data_spec.astype(float).view(complex).squeeze()
        self.avg_iq = np.mean(self.data_iq, axis=0)
        self.shots = (1-np.sign(self.data_iq.real))/2
        self.avg_shots = np.mean(self.shots, axis=0)

        # for non thresholded data, find the best rotation angle that
        # puts all the information on the I quadrature 
        rot_angle = find_iq_rotation(self.avg_iq)
        self.data_iq_rot = rotate_iq(self.data_iq, rot_angle)
        self.avg_iq_rot = rotate_iq(self.avg_iq, rot_angle)

        if thresholded:
            self.data_to_fit = self.avg_shots
        else:
            from acadia_qmsmt.analysis import rotate_iq
            self.data_to_fit = self.avg_iq_rot.real
        self.thresholded = thresholded

        try:
            from acadia_qmsmt.analysis.fitting import Lorentzian
            self.fit = Lorentzian(self.qubit_frequencies, self.data_to_fit)
            self.fitted_f0 = self.fit.ufloat_results["x0"]
        except Exception as e:
            logger.error(f"Error fitting: {e}", exc_info=True)
            self.fit = None

        return completed_iterations
    

    @annotate_method(plot_name="1 qubit spectrocopy", axs_shape=(1,1))
    def plot_data(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        axs.plot(self.qubit_frequencies, self.data_to_fit, "o")
        if self.fit is not None:
            self.fit.plot_fitted(axs, oversample=5, label=f"{self.fitted_f0:.5g}")

        axs.set_xlabel("Frequency [Hz]")
        if self.thresholded:
            axs.set_ylabel("e pop")
            axs.set_ylim(-0.02, 1.02)
        else:
            axs.set_ylabel("re(data) after rotation")

        axs.legend()
        axs.grid(True)
        return fig, axs

    @annotate_method(plot_name="2 bin averaged", axs_shape=(1,1))
    def plot_bin_avg(self, axs=None, n_avg=1):
        from acadia_qmsmt.plotting import prepare_plot_axes, plot_binaveraged
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)
        if self.thresholded:
            axs.set_ylabel("e pop")
            fig, axs = plot_binaveraged(self.qubit_frequencies, self.shots, axs, n_avg=n_avg, vmin=0, v_max=1)
        else:
            axs.set_ylabel("re(data) after rotation")
            fig, axs = plot_binaveraged(self.qubit_frequencies, self.data_iq_rot.real, axs, n_avg=n_avg)
        axs.set_ylabel("Probe freq [Hz]")
        return fig, axs


    @annotate_method(button_name="update frequency")
    def update_freq(self):
        if self.fit is not None:
            self.update_io_yaml_field("qubit_stimulus", "channel_config.nco_frequency", np.round(self.fitted_f0.n))



