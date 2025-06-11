from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime
from acadia.sample_arithmetic import sample_to_complex
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, IOConfig, Qubit
from acadia.runtime import annotate_method

class ResonatorSpectroscopyPrepQubitRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` subclass for readout spectroscopy
    """
    # The name of the sections in the yaml file for the required channels
    stimulus: IOConfig
    capture: IOConfig
    qubit_stimulus: IOConfig

    frequencies: Union[list, np.ndarray]
    qubit_prep_pulse_mem: str

    iterations: int
    run_delay: int

    capture_window_name: str = None
    stimulus_waveform_name: str = None
    electrical_delay: float = 0
    plot: bool = True
    figsize: tuple[int] = None
    yaml_path: str = None

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

        stimulus_io = self.io("stimulus")
        capture_io = self.io("capture")
        qubit_stimulus_io = self.io("qubit_stimulus")

        resonator = MeasurableResonator(stimulus_io, capture_io)
        qubit = Qubit(qubit_stimulus_io)

        # Create the record group for saving captured data
        self.data.add_group(f"points", uniform=True)

        # Create a sequence for the sequencer to generate the pulse and capture it
        def sequence(a: Acadia):
            with a.channel_synchronizer():
                qubit.pulse("rotation")
                a.barrier()
                # Measure the resonator by driving the "readout" waveform on the stimulus IO
                # and capture into the "readout_accumulated" waveform on the capture IO
                resonator.measure("readout", "readout_accumulated", self.capture_window_name)

        # Compile the sequence
        self.acadia.compile(sequence)
        # Attach to the hardware
        self.acadia.attach()
        # Configure channel analog parameters
        self.configure_channels()
        # Assemble and load the program
        self.acadia.assemble()
        self.acadia.load()

        # Load the window memory with the data from the config file
        resonator.load_windows()
        # Load the stimulus waveform named "readout" with the specified signal
        stimulus_io.load_waveform("readout", self.stimulus_waveform_name)
        qubit_stimulus_io.load_waveform("rotation", self.qubit_prep_pulse_mem)

        for i in range(self.iterations):
            for frequency in self.frequencies:
                resonator.set_frequency(frequency)

                # capture data and put in the corresponding group
                self.acadia.run(minimum_delay=self.run_delay)

                wf = capture_io.get_waveform_memory("readout_accumulated")
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
    def process_current_data(self, e_delay:float=76E-9, auto_edelay:bool=False):
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes
        data_spec = reshape_iq_data_by_axes(self.data["points"].records(), self.frequencies)
        if data_spec is None:
            return
        else:
            completed_iterations = len(data_spec)
        self.data_iq = data_spec.astype(float).view(complex).squeeze()
        self.avg_iq = np.mean(self.data_iq, axis=0)

        if auto_edelay:
            from numpy import polyfit
            k, b = polyfit(self.frequencies, np.unwrap(np.angle(self.avg_iq)), deg=1)
            e_delay = -k / np.pi / 2
        self.avg_iq_corrected = self.avg_iq * np.exp(1j * self.frequencies * e_delay * np.pi * 2)

        self.e_delay_applied = e_delay

        return completed_iterations
    

    @annotate_method(plot_name="mag_phase_vs_dac", axs_shape=(2,1))
    def plot_data(self, axs=None, apply_e_delay:bool=True, unwrap_phase:bool=True):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(2,1), figsize=self.figsize)

        data = self.avg_iq_corrected if apply_e_delay else self.avg_iq
        axs[0].plot(self.frequencies, np.abs(data), "o")

        phases = np.angle(data, deg=True)
        if unwrap_phase:
            phases = np.unwrap(phases, period=360)
        e_delay_label = None if not apply_e_delay else f"edelay: {self.e_delay_applied}"
        axs[1].plot(self.frequencies, phases, ".-", label=e_delay_label)

        axs[1].set_xlabel("Frequency [Hz]")
        axs[1].set_ylabel("Phase (deg)")
        axs[0].set_ylabel("Mag (a.u.)")

        for ax in axs:
            ax.legend()
            ax.grid(True)

        fig.tight_layout()
        return fig, axs







    # # --------------- live plot in jpynb ---------------------
    # def initialize(self):
        
    #     if self.plot:
    #         from acadia.processing import DynamicLine
    #         import matplotlib.pyplot as plt
    #         from IPython.display import display
    #         from ipywidgets import FloatText, HBox, Button

    #         self.figsize = (4, 3) if self.figsize is None else self.figsize
    #         self.fig, ax = plt.subplots(1, 1, figsize=self.figsize)
    #         self.fig.tight_layout()
    #         self.fig.subplots_adjust(left=0.25, bottom=0.25)

    #         self.line_mag = DynamicLine(ax, ".-", label="Mag", color="blue")
    #         ax.set_xlabel("Frequency [Hz]")
    #         ax.set_ylabel("Magnitude [arb.]", color="blue")
    #         ax.tick_params(axis='y', labelcolor="blue")
    #         ax.grid()

    #         ax_phase = ax.twinx()
    #         self.line_phase = DynamicLine(ax_phase, ".-", label="Phase", color="red")
    #         ax_phase.set_ylabel("Phase [rad.]", color="red")
    #         ax_phase.tick_params(axis='y', labelcolor="red")

    #         # Create interactive widgets for controlling the electrical delay
    #         self._electrical_delay_input = FloatText(value=self.electrical_delay, 
    #                                                     description="Electrical delay: ",
    #                                                     style={"description_width": "initial"})
    #         self._electrical_delay_update_button = Button(description="Set")
    #         self._electrical_delay_update_button.on_click(self.update_electrical_delay)
    #         self._electrical_delay_auto_button = Button(description="Auto")
    #         self._electrical_delay_auto_button.on_click(self.auto_electrical_delay)

    #         display(HBox([self._electrical_delay_input, 
    #                     self._electrical_delay_update_button, 
    #                     self._electrical_delay_auto_button]))

    #         from tqdm.notebook import tqdm

    #     else:
    #         from tqdm import tqdm

    #     self.iterations_progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
    #     self.iterations_previous = 0

    #     self.electrical_delay_phases = 2*np.pi*self.frequencies*self.electrical_delay
    #     self.data_summed = None
    #     self.data_complex = None
    #     self.data_mags = None
    #     self.data_phases =  None

    # def update(self):
    #     # First make sure that we actually have new data to process
    #     if "points" not in self.data or len(self.data["points"]) < len(self.frequencies):
    #         return

    #     # Update the progress bar based on the number of iterations
    #     completed_iterations = len(self.data["points"]) // len(self.frequencies)
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
    #     completed_iterations = len(self.data["points"]) // len(self.frequencies)
    #     valid_traces = completed_iterations*len(self.frequencies)
    #     data = self.data["points"].records()[:valid_traces, ...]

    #     # Get the collection of data and reshape it so that the axes index as: 
    #     # (iteration, frequency, sample time, sample quadrature)
    #     samples_per_trace = data.shape[-2]
    #     data_reshaped = data.reshape(-1, len(self.frequencies), samples_per_trace, 2)
        
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
    #     # Simultaneously, choose the scale so that the result is independent
    #     # of amplitude
    #     if self.stimulus_waveform_name is None:
    #         waveform_config = list(self.io("stimulus")._config["waveforms"].values())[0]
    #     else:
    #         waveform_config = self.io("stimulus")._config["waveforms"][self.stimulus_waveform_name]
        
    #     waveform_scale = waveform_config["scale"] if "scale" in waveform_config else 1.0
    #     total_scale = completed_iterations / waveform_scale 
    #     self.data_complex = sample_to_complex(self.data_summed, scale=total_scale)
    #     self.data_mags = np.abs(self.data_complex)
    #     self.data_phases = np.angle(self.data_complex)

    # def update_plot(self):
    #     if self.plot:
    #         display_phases = np.unwrap(self.data_phases - self.electrical_delay_phases)
    #         self.line_mag.update(self.frequencies, self.data_mags)
    #         self.line_phase.update(self.frequencies, display_phases)
    #         self.fig.canvas.draw_idle() 

    # def update_electrical_delay(self, delay_value: float = None, lock: bool = True):
    #     from ipywidgets import Button
    #     if delay_value is None or isinstance(delay_value, Button):
    #         if not self.plot:
    #             raise ValueError("Must provide electrical delay value when it cannot"
    #                             " be retrieved from the display input.")
    #         delay_value = self._electrical_delay_input.value
        

    #     if lock and not self._update_lock.acquire(timeout=5):
    #         return False

    #     self.electrical_delay = delay_value
    #     self.electrical_delay_phases = 2*np.pi*self.frequencies*self.electrical_delay
        
    #     if self.plot:
    #         self._electrical_delay_input.value = self.electrical_delay
    #         self.update_plot()

    #     if lock:
    #         self._update_lock.release()

    #     return True

    # def auto_electrical_delay(self, lock: bool = True) -> float:
    #     if lock and not self._update_lock.acquire(timeout=5):
    #         return None

    #     def model(freqs, delay, phi0):
    #         return 2*np.pi*freqs*delay + phi0
    
    #     popt,pcov = curve_fit(model, self.frequencies, np.unwrap(self.data_phases))
    #     self.update_electrical_delay(popt[0], lock=False)

    #     if lock:
    #         self._update_lock.release()

    #     return popt[0]

    # def fit_lorentzian(self) -> float:
    #     from acadia_qmsmt.analysis import population_in_quadrant
    #     from acadia_qmsmt.analysis.fitting import rotate_iq
    #     from acadia_qmsmt.analysis.fitting.lorentzian import Lorentzian
    #     from acadia_qmsmt.helpers.plot_utils import add_button
    #     from acadia_qmsmt.helpers.yaml_editor import update_yaml

    #     fit = Lorentzian(self.frequencies, np.abs(self.data_complex))
    #     f0 = fit.ufloat_results["x0"]

    #     if self.plot:
    #         fit_fig, fit_ax = fit.plot()
    #         fit_ax.set_xlabel("Frequency [Hz]")
    #         fit_ax.set_ylabel(data_type)

    #         def _update_freq(event):
    #             self.update_ioconfig("stimulus", "channel_config.nco_frequency", np.round(f0.n))

    #         # add a button for updating the qubit freq in the yaml file
    #         self._update_button = add_button(fit_fig, _update_freq, label="Update Frequency")

    #     return f0
