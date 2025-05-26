from typing import Union, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime, WaveformMemory
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig
from acadia.runtime import annotate_method


class QubitRabiWithPrepareRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` for calibrating the amplitudes of pulses for qubit drives.
    """
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    # Note that these amplitudes override the ``scale`` parameter in the configuration
    qubit_amplitudes: Union[list, np.ndarray]

    iterations: int
    run_delay: int

    qubit_pulse_name: str = None
    qubit_pulse_waveform_name: str = None
    readout_window_name: str = None
    readout_stimulus_waveform_name: str = None
    fit_quadratic: bool = False
    plot: bool = True
    figsize: tuple[int] = None

    prepare_quadrant: Literal[1,2,3,4] = 1
    prepare_pulse_name: str = None
    prepare_pulse_waveform_name: str = None
    yaml_path: str = None

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")

        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)
        qubit = Qubit(qubit_stimulus_io)
        qubit_blank_wf = qubit_stimulus_io.blank_waveform_generator()(3e-6) # for readout to empty # todo: should be an inpit

        self.data.add_group(f"points", uniform=True)

        def sequence(a: Acadia):
            
            qubit.prepare(self.prepare_quadrant, readout_resonator, self.prepare_pulse_name,
                          "readout",
                          "readout_accumulated", 
                          self.readout_window_name)

            readout_resonator.prepare_cmacc(self.readout_window_name)

            with a.channel_synchronizer():
                qubit.pulse(qubit_blank_wf)
                qubit.pulse(self.qubit_pulse_name)
                a.barrier()
                readout_resonator.measure("readout", "readout_accumulated")

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_waveform("readout", self.readout_stimulus_waveform_name)

        # Precompute the envelope so that we're not recalculating it every time, only scaling it
        qubit_pulse_samples = qubit_stimulus_io.compute_waveform(self.qubit_pulse_name, self.qubit_pulse_waveform_name)
        
        # Precompute the envelope for qubit prepare pulse and load it
        qubit_stimulus_io.load_waveform(self.prepare_pulse_name, self.prepare_pulse_waveform_name)
        

        for i in range(self.iterations):
            for amplitude in self.qubit_amplitudes:
                qubit_stimulus_io.load_waveform(self.qubit_pulse_name, qubit_pulse_samples, scale=amplitude)

                self.acadia.run(minimum_delay=self.run_delay)
                wf = readout_capture_io.get_waveform_memory("readout_accumulated")
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
        if self.plot:
            from acadia_qmsmt.plotting import save_registered_plots
            save_registered_plots(self)



    @annotate_method(is_data_processor=True)
    def process_current_data(self, thresholded:bool=True):
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes
        data_spec = reshape_iq_data_by_axes(self.data["points"].records(), self.qubit_amplitudes)
        if data_spec is None:
            return
        else:
            completed_iterations = len(data_spec)
        self.data_iq = data_spec.astype(float).view(complex).squeeze()
        self.avg_iq = np.mean(self.data_iq, axis=0)

        if thresholded:
            self.data_to_fit = np.mean((1-np.sign(self.data_iq.real))/2, axis=0)
        else:
            from acadia_qmsmt.analysis import rotate_iq
            self.data_to_fit = rotate_iq(self.avg_iq).real
        self.thresholded = thresholded

        from acadia_qmsmt.analysis.fitting import Cosine
        self.fit = Cosine(self.qubit_amplitudes, self.data_to_fit)
        self.fitted_pi_amp = abs(1/self.fit.ufloat_results["f"]/2)
        return completed_iterations
    

    @annotate_method(plot_name="1 qubit power rabi", axs_shape=(1,1))
    def plot_data(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        axs.plot(self.qubit_amplitudes, self.data_to_fit, ".")
        self.fit.plot_fitted(axs, oversample=5, label=f"pi_amp: {self.fitted_pi_amp:.5g}")

        axs.set_xlabel("Drive Amplitude [DAC]")
        if self.thresholded:
            axs.set_ylabel("e pop")
            axs.set_ylim(-0.02, 1.02)
        else:
            axs.set_ylabel("re(data) after rotation")

        axs.legend()
        return fig, axs


    @annotate_method(plot_name="2 qubit histogram at 0 and pi", axs_shape=(1, 2))
    def plot_data_hist2d(self, axs=None, bins=50, log_scale:bool=False):
        from acadia_qmsmt.plotting import prepare_plot_axes, plot_multiple_hist2d
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,2))

        data_g = self.data_iq[:, np.argmin(abs(self.qubit_amplitudes))]
        closet_pi_amp_idx = np.argmin(abs(self.qubit_amplitudes-self.fitted_pi_amp))
        data_e = self.data_iq[:, closet_pi_amp_idx]

        plot_multiple_hist2d(data_g, data_e, plot_ax=axs, bins=bins, log_scale=log_scale)
        axs[0].set_title("amp 0")
        axs[1].set_title(f"amp {np.round(self.qubit_amplitudes[closet_pi_amp_idx],9)}")
        return fig, axs




    @annotate_method(button_name="update pipulse amp")
    def update_amp(self):
        self.update_io_yaml_field("qubit_stimulus", f"waveforms.{self.qubit_pulse_waveform_name}.scale", np.round(self.fitted_pi_amp.n, 5))

