from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime
from acadia.sample_arithmetic import sample_to_complex
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig
from acadia.runtime import annotate_method

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

    saturation_pulse_fixed_length: float = 1e-3 - 100e-9
    saturation_pulse_ramp_time: float = 100e-9
    saturation_pulse_amplitude: float = 0.1
    readout_window_name: str = None
    readout_stimulus_waveform_name: str = None
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

        # Create the qubit saturation waveform manually, as this is not a waveform
        # that the user will likely need to keep in the configuration file after this
        # measurement
        saturation_waveform = self.acadia.create_waveform_memory(
            qubit._stimulus.channel, 
            length=self.saturation_pulse_ramp_time, 
            fixed_length=self.saturation_pulse_fixed_length)

        self.data.add_group(f"points", uniform=True)

        def sequence(a: Acadia):
            readout_resonator.prepare_cmacc(self.readout_window_name)

            with a.channel_synchronizer():
                a.schedule_waveform(saturation_waveform)
                a.barrier()
                readout_resonator.measure("readout", "readout_accumulated")

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_waveform("readout", self.readout_stimulus_waveform_name)
        qubit_stimulus_io.load_waveform(saturation_waveform, {"data": "hann"}, self.saturation_pulse_amplitude)

        for i in range(self.iterations):
            for frequency in self.qubit_frequencies:
                qubit.set_frequency(frequency)

                # capture data and put in the corresponding group
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
    def process_current_data(self, thresholded:bool=False):
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes
        data_spec = reshape_iq_data_by_axes(self.data["points"].records(), self.qubit_frequencies)
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

        from acadia_qmsmt.analysis.fitting import Lorentzian
        self.fit = Lorentzian(self.qubit_frequencies, self.data_to_fit)
        self.fitted_f0 = self.fit.ufloat_results["x0"]

        return completed_iterations
    

    @annotate_method(plot_name="qubit spectrocopy", axs_shape=(1,1))
    def plot_data(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        axs.plot(self.qubit_frequencies, self.data_to_fit, ".")
        self.fit.plot_fitted(axs, oversample=5, label=f"{self.fitted_f0:.5g}")

        axs.set_xlabel("Frequency [Hz]")
        if self.thresholded:
            axs.set_ylabel("e pop")
        else:
            axs.set_ylabel("re(data) after rotation")

        axs.legend()
        return fig, axs


    @annotate_method(button_name="update frequency")
    def update_freq(self):
        self.update_io_yaml_field("qubit_stimulus", "channel_config.nco_frequency", np.round(self.fitted_f0.n))



    # --------------------------------------- in jpynb plotting ------------------------
    # def initialize(self):
        
    #     if self.plot:
    #         from acadia.processing import DynamicLine
    #         import matplotlib.pyplot as plt
    #         from IPython.display import display
    #         from ipywidgets import FloatText, HBox, Button

    #         self.figsize = (4, 3) if self.figsize is None else self.figsize
    #         self.fig, axs = plt.subplots(2, 1, figsize=self.figsize)
    #         self.fig.tight_layout()
    #         self.fig.subplots_adjust(left=0.25, bottom=0.25)

    #         ax = axs[0]
    #         self.line_mag = DynamicLine(ax, ".-", label="Mag", color="blue")
    #         ax.set_xlabel("Qubit Frequency [Hz]")
    #         ax.set_ylabel("Magnitude [arb.]", color="blue")
    #         ax.tick_params(axis='y', labelcolor="blue")
    #         ax.grid()

    #         ax_phase = axs[1]
    #         # ax_phase = ax.twinx()
    #         self.line_phase = DynamicLine(ax_phase, ".-", label="Phase", color="red")
    #         ax_phase.set_ylabel("Phase [rad.]", color="red")
    #         ax_phase.tick_params(axis='y', labelcolor="red")

    #         from tqdm.notebook import tqdm

    #     else:
    #         from tqdm import tqdm

    #     self.iterations_progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
    #     self.iterations_previous = 0

    #     self.data_summed = None
    #     self.data_complex = None
    #     self.data_mags = None
    #     self.data_phases =  None

    # def update(self):
    #     # First make sure that we actually have new data to process
    #     if "points" not in self.data or len(self.data["points"]) < len(self.qubit_frequencies):
    #         return

    #     # Update the progress bar based on the number of iterations
    #     completed_iterations = len(self.data["points"]) // len(self.qubit_frequencies)
    #     if completed_iterations == 0:
    #         return

    #     self.iterations_progress_bar.update(completed_iterations - self.iterations_previous)

    #     self.process_data()
    #     self.update_plot()               
    #     self.data.save(self.local_directory)

    #     self.iterations_previous = completed_iterations

    # def finalize(self):
    #     super().finalize()
    #     self.iterations_progress_bar.close()
    #     if self.plot:
    #         self.savefig(self.fig)
    #     self.fit_lorentzian()

    # def process_data(self):
    #     completed_iterations = len(self.data["points"]) // len(self.qubit_frequencies)
    #     valid_traces = completed_iterations*len(self.qubit_frequencies)
    #     data = self.data["points"].records()[:valid_traces, ...]

    #     # Get the collection of data and reshape it so that the axes index as: 
    #     # (iteration, frequency, sample time, sample quadrature)
    #     samples_per_trace = data.shape[-2]
    #     data_reshaped = data.reshape(-1, len(self.qubit_frequencies), samples_per_trace, 2)
        
    #     # Slice the data so that we have an array containing only the traces
    #     # we didn't have the last time update() was called
    #     self.new_data = data_reshaped[self.iterations_previous:, :, :, :]

    #     # Sum the new data and then add it to the aggregated array of trace data
    #     new_data_summed = np.sum(self.new_data, axis=(0,2), keepdims=False)
    #     if self.data_summed is None:
    #         self.data_summed = new_data_summed
    #     else:
    #         self.data_summed += new_data_summed

    #     # Convert the summed sample data to a complex number and choose 
    #     # the scale so that we turn the sum into a mean
    #     self.data_complex = sample_to_complex(self.data_summed, scale=float(completed_iterations))
    #     self.data_mags = np.abs(self.data_complex)
    #     self.data_phases = np.angle(self.data_complex)

    # def update_plot(self):
    #     if self.plot:
    #         self.line_mag.update(self.qubit_frequencies, self.data_mags)
    #         self.line_phase.update(self.qubit_frequencies, self.data_phases)
    #         self.fig.canvas.draw_idle() 

    # def fit_lorentzian(self) -> float:
    #     from acadia_qmsmt.analysis import rotate_iq
    #     from acadia_qmsmt.analysis.fitting.lorentzian import Lorentzian
    #     from acadia_qmsmt.helpers.plot_utils import add_button

    #     # data_complex has a small absolute value that will case ill-conditioned covariance matrix
    #     iq_pts = self.data_summed[:, 0] + 1j * self.data_summed[:, 1]
    #     data_to_fit = rotate_iq(iq_pts).real

    #     fit = Lorentzian(self.qubit_frequencies, data_to_fit)
    #     f0 = fit.ufloat_results["x0"]

    #     if self.plot:
    #         fit_fig, fit_ax = fit.plot()
    #         fit_ax.set_xlabel("Qubit Frequency [Hz]")
    #         fit_ax.set_ylabel("rotated I component")

    #         def _update_freq(event):
    #             self.update_ioconfig("qubit_stimulus", "channel_config.nco_frequency", np.round(f0.n))

    #         # add a button for updating the qubit freq in the yaml file
    #         self._update_button = add_button(fit_fig, _update_freq, label="Update Frequency")

    #     return f0

