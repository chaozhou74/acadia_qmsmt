from typing import Union, Annotated

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime, WaveformMemory
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig
from acadia.runtime import annotate_method


class QubitEFPulseAmplitudeCalibrationRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` for calibrating the amplitudes of pulses for qubit drives.plot_pcolormesh_fft
    """
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    # Note that these amplitudes override the ``scale`` parameter in the configuration
    qubit_amplitudes: Union[list, np.ndarray]

    iterations: int
    run_delay: int

    qubit_ge_pulse_name: str = "R_x_180"
    qubit_ef_pulse_name: str = "R_x_180_ef"
    readout_pulse_name: str = "readout"
    capture_memory_name: str = "readout_accumulated"
    capture_window_name: str = "matched"

    figsize: tuple[int] = None
    yaml_path: str = None

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")

        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)

        self.data.add_group(f"points", uniform=True)
        

        def sequence(a: Acadia):
            with a.channel_synchronizer():
                qubit_stimulus_io.schedule_pulse(self.qubit_ge_pulse_name)
                qubit_stimulus_io.schedule_pulse(self.qubit_ef_pulse_name)
                qubit_stimulus_io.schedule_pulse(self.qubit_ge_pulse_name)
                a.barrier()
                readout_resonator.measure(self.readout_pulse_name, self.capture_memory_name, self.capture_window_name)

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_pulse(self.readout_pulse_name)
        qubit_stimulus_io.load_pulse(self.qubit_ge_pulse_name)


        for i in range(self.iterations):
            for amplitude in self.qubit_amplitudes:
                qubit_stimulus_io.load_pulse(self.qubit_ef_pulse_name, scale=amplitude)
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
    def process_current_data(self, readout_classifier: Annotated[str, "IOConfig", "readout_capture.classifiers"] = None):
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes
        data = reshape_iq_data_by_axes(self.data["points"].records(), self.qubit_amplitudes, to_complex=True)
        if data is None:
            return

        completed_iterations = len(data)
        readout_resonator = MeasurableResonator(self.io("readout_stimulus"), self.io("readout_capture"))

        self.data_complex = data
        self.shots = readout_resonator.classify_measurement(self.data_complex, readout_classifier)
        self.avg_shots = np.mean(self.shots, axis=0)



        from acadia_qmsmt.analysis.fitting import Cosine
        self.fit = Cosine(self.qubit_amplitudes, self.avg_shots)
        self.fitted_pi_amp = abs(1/self.fit.ufloat_results["f"]/2)
        return completed_iterations
    

    @annotate_method(plot_name="1 qubit power rabi", axs_shape=(1,1))
    def plot_data(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        axs.plot(self.qubit_amplitudes, self.avg_shots, "o")
        self.fit.plot_fitted(axs, oversample=5, label=f"pi_amp: {self.fitted_pi_amp:.5g}")

        axs.set_xlabel("Drive Amplitude [DAC]")
        axs.set_ylabel("Average Measurement")
        axs.legend()
        return fig, axs


    @annotate_method(plot_name="2 qubit histogram at 0 and pi", axs_shape=(1, 2))
    def plot_data_hist2d(self, axs=None, bins=50, log_scale:bool=False):
        from acadia_qmsmt.plotting import prepare_plot_axes, plot_multiple_hist2d
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,2))

        data_g = self.data_complex[:, np.argmin(abs(self.qubit_amplitudes))]
        closet_pi_amp_idx = np.argmin(abs(self.qubit_amplitudes-self.fitted_pi_amp))
        data_e = self.data_complex[:, closet_pi_amp_idx]

        plot_multiple_hist2d(data_g, data_e, plot_ax=axs, bins=bins, log_scale=log_scale)
        axs[0].set_title("amp 0")
        axs[1].set_title(f"amp {np.round(self.qubit_amplitudes[closet_pi_amp_idx],9)}")
        return fig, axs




    @annotate_method(button_name="update pipulse amp")
    def update_amp(self):
        self.update_io_yaml_field("qubit_stimulus", f"pulses.{self.qubit_ef_pulse_name}.scale", np.round(self.fitted_pi_amp.n, 5))

