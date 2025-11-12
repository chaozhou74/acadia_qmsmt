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

    saturation_pulse_name: str = "saturation"
    readout_pulse_name: str = "readout"
    capture_memory_name: str = "readout_accumulated"
    capture_window_name: str = None

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
                qubit_stimulus_io.schedule_pulse(self.saturation_pulse_name)
                a.barrier()
                readout_resonator.measure(self.readout_pulse_name, self.capture_memory_name, self.capture_window_name)

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_pulse(self.readout_pulse_name)
        qubit_stimulus_io.load_pulse(self.saturation_pulse_name)
        

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
    def process_current_data(self, readout_classifier: str = None):
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes
        
        data_spec = reshape_iq_data_by_axes(self.data["points"].records(), self.qubit_frequencies)
        if data_spec is None:
            return
        
        completed_iterations = data_spec.shape[0]

        readout_resonator = MeasurableResonator(self.io("readout_stimulus"), self.io("readout_capture"))

        self.data_complex = data_spec.astype(float).view(complex).reshape(completed_iterations, len(self.qubit_frequencies))
        self.shots = readout_resonator.classify_measurement(self.data_complex, readout_classifier)
        self.avg_shots = np.mean(self.shots, axis=0)

        try:
            from acadia_qmsmt.analysis.fitting import Lorentzian
            self.fit = Lorentzian(self.qubit_frequencies, self.avg_shots)
            self.fitted_f0 = self.fit.ufloat_results["x0"]
        except Exception as e:
            logger.error(f"Error fitting: {e}", exc_info=True)
            self.fit = None

        return completed_iterations
    

    @annotate_method(plot_name="1 qubit spectroscopy", axs_shape=(1,1))
    def plot_data(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        axs.plot(self.qubit_frequencies, self.avg_shots, "o")
        if self.fit is not None:
            self.fit.plot_fitted(axs, oversample=5, label=f"{self.fitted_f0:.5g}")

        axs.set_xlabel("Probe Frequency [Hz]")
        axs.set_ylabel("Average Measurement")
        axs.set_ylim(-0.02, 1.02)

        axs.legend()
        axs.grid(True)
        return fig, axs

    @annotate_method(plot_name="2 bin averaged", axs_shape=(1,1))
    def plot_bin_avg(self, axs=None, n_avg=1):
        from acadia_qmsmt.plotting import plot_binaveraged
        fig, axs = plot_binaveraged(self.qubit_frequencies, self.shots, axs, n_avg=n_avg, vmin=0, v_max=1, figsize=self.figsize)
        axs.set_ylabel("Average Measurement")
        axs.set_xlabel("Probe Frequency [Hz]")
        return fig, axs


    @annotate_method(button_name="update frequency")
    def update_freq(self):
        if self.fit is not None:
            self.update_io_yaml_field("qubit_stimulus", "channel_config.nco_frequency", np.round(self.fitted_f0.n))



