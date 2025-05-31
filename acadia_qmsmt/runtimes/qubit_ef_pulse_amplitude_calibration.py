from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime, WaveformMemory
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig
from acadia.runtime import annotate_method

def sine(pulse_amp, oscillation_amp, oscillation_freq, offset):
    return oscillation_amp * np.cos(2 * np.pi * pulse_amp * oscillation_freq) + offset

def quadratic(pulse_amp, a, amp_0, offset):
    return a*(pulse_amp - amp_0)**2 + offset

class QubitEFPulseAmplitudeCalibrationRuntime(QMsmtRuntime):
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

    qubit_ge_pulse_mem: str = None
    qubit_ge_pulse_wf: str = None

    qubit_ef_pulse_mem: str = None
    qubit_ef_pulse_wf: str = None

    readout_window_name: str = None
    readout_stimulus_waveform_name: str = None
    fit_quadratic: bool = False
    plot: bool = True
    figsize: tuple[int] = None
    yaml_path: str = None

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")

        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)
        qubit = Qubit(qubit_stimulus_io)

        self.data.add_group(f"points", uniform=True)
        
        # reconstruct a ef pipulse flat-top pulse as a fully defined waveform based on the yaml
        ef_memory_full, ef_waveform_data = qubit_stimulus_io.generate_full_waveform_from_windowed(self.qubit_ef_pulse_mem,
                                                                                    self.qubit_ef_pulse_wf)

        def sequence(a: Acadia):
            readout_resonator.prepare_cmacc(self.readout_window_name)

            with a.channel_synchronizer():
                qubit.pulse(self.qubit_ge_pulse_mem)
                qubit.pulse(ef_memory_full)
                qubit.pulse(self.qubit_ge_pulse_mem)
                a.barrier()
                readout_resonator.measure("readout", "readout_accumulated")

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_waveform("readout", self.readout_stimulus_waveform_name)
        qubit_stimulus_io.load_waveform(self.qubit_ge_pulse_mem, self.qubit_ge_pulse_wf)

        ef_detune = qubit_stimulus_io.get_config("waveforms", self.qubit_ef_pulse_wf, "detune")


        for i in range(self.iterations):
            for amplitude in self.qubit_amplitudes:
                qubit_stimulus_io.load_waveform(ef_memory_full, 
                                                ef_waveform_data, scale=amplitude, detune=ef_detune)

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
        data = reshape_iq_data_by_axes(self.data["points"].records(), self.qubit_amplitudes)
        if data is None:
            return
        else:
            completed_iterations = len(data)
        self.data_iq = data.astype(float).view(complex).squeeze()
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

        axs.plot(self.qubit_amplitudes, self.data_to_fit, "o")
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
        self.update_io_yaml_field("qubit_stimulus", f"waveforms.{self.qubit_ef_pulse_wf}.scale", np.round(self.fitted_pi_amp.n, 5))





# --------------------------------------- in jpynb plotting ------------------------
    # def initialize(self):
        
    #     if self.plot:
    #         from acadia.processing import DynamicLine
    #         import matplotlib.pyplot as plt
    #         from IPython.display import display
    #         from ipywidgets import Label

    #         self.figsize = (4, 3) if self.figsize is None else self.figsize
    #         self.fig, ax = plt.subplots(1, 1, figsize=self.figsize)
    #         self.fig.tight_layout()
    #         self.fig.subplots_adjust(left=0.25, bottom=0.25)

    #         self.line_pop = DynamicLine(ax, ".", color="red")
    #         self.line_fit = DynamicLine(ax, "--", color="red")
    #         ax.set_xlabel("Pulse Amplitude [arb.]")
    #         ax.set_ylabel("Population [FS]")
    #         ax.set_xlim(self.qubit_amplitudes[0], self.qubit_amplitudes[-1])
    #         ax.set_ylim(-1.1, 1.1)
    #         ax.grid()

    #         self.amplitude_label = Label(style={"description_width": "initial"})
    #         display(self.amplitude_label)

    #         from tqdm.notebook import tqdm

    #     else:
    #         from tqdm import tqdm

    #     self.fit = None
    #     self.iterations_progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
    #     self.iterations_previous = 0

    # def update(self):
    #     # First make sure that we actually have new data to process
    #     if "points" not in self.data or len(self.data["points"]) < len(self.qubit_amplitudes):
    #         return

    #     # Update the progress bar based on the number of iterations
    #     completed_iterations = len(self.data["points"]) // len(self.qubit_amplitudes)
    #     if completed_iterations == 0:
    #         return

    #     self.iterations_progress_bar.update(completed_iterations - self.iterations_previous)

    #     valid_points = completed_iterations*len(self.qubit_amplitudes)
    #     data = self.data["points"].records()[:valid_points, ...]
    #     data = data.reshape(completed_iterations, len(self.qubit_amplitudes), 2)

    #     # Threshold the data according to the I quadrature
    #     shots = np.sign(data[:,:,0], dtype=np.int32)
    #     self.avg = np.mean(shots, axis=0)
        

    #     # Fit the data to a sine
    #     try:
    #         amin = np.argmin(self.avg)
    #         amax = np.argmax(self.avg)
    #         osc_period = 2*abs(self.qubit_amplitudes[amin]-self.qubit_amplitudes[amax])
    #         p0 = (abs(amin-amax)/2, 1/osc_period, (amin+amax)/2)
    #         self.fit, pcov = curve_fit(sine, self.qubit_amplitudes, self.avg, p0=p0)
            
    #     except:
    #         pass
        
    #     if self.plot:
    #         self.line_pop.update(self.qubit_amplitudes, self.avg, rescale_axis=False)
    #         if self.fit is not None:
    #             self.amplitude_label.value = f"Pi pulse amplitude: {round(0.5/self.fit[1], 6)}"
    #             self.line_fit.update(self.qubit_amplitudes, sine(self.qubit_amplitudes, *self.fit), rescale_axis=False)
    #         self.fig.canvas.draw_idle() 

    #     self.data.save(self.local_directory)
    #     self.iterations_previous = completed_iterations

    # def finalize(self):
    #     super().finalize()
    #     self.iterations_progress_bar.close()
    #     if self.plot:
    #         self.savefig(self.fig)

