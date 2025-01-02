from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime, Waveform
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, IOConfig

class ResonatorSpecRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` subclass for readout spectroscopy
    """
    # The name of the sections in the yaml file for the required channels
    stimulus: IOConfig
    capture: IOConfig

    frequencies: Union[list, np.ndarray]

    iterations: int
    plot: bool = True
    figsize: tuple[int] = None

    cmacc_kernel: str = "boxcar"
    stimulus_signal_name: str = "smoothed_boxcar"
    electrical_delay: float = 0

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

        stimulus_io = self.io("stimulus")
        capture_io = self.io("capture")

        resonator = MeasurableResonator(stimulus_io, capture_io)

        # Create the record group for saving captured data
        self.data.add_group(f"iq_pts", uniform=True)

        # Create a sequence for the sequencer to generate the pulse and capture it
        def sequence(a: Acadia):
            resonator.prepare_cmacc(self.cmacc_kernel)

            with a.channel_synchronizer():
                # Measure the resonator by driving the "readout" waveform on the stimulus IO
                # and capture into the "readout_accumulated" waveform on the capture IO
                resonator.measure("readout", "readout_accumulated")

        # Compile the sequence
        self.acadia.compile(sequence)
        # Attach to the hardware
        self.acadia.attach()
        # Configure channel analog parameters
        self.configure_channels()
        # Assemble and load the program
        self.acadia.assemble()
        self.acadia.load()

        # Load the kernel memory with the data from the config file
        resonator.load_kernels()
        # Load the stimulus waveform named "readout" with the specified signal
        stimulus_io.load_memory(readout=self.stimulus_signal_name)

        for i in range(self.iterations):
            for frequency in self.frequencies:
                resonator.set_frequency(frequency)

                # capture data and put in the corresponding group
                self.acadia.run()

                wf = capture_io.get_waveform("readout_accumulated")
                self.data[f"iq_pts"].write(wf.array)

            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return

        self.final_serve()

    def initialize(self):
        
        if self.plot:
            from acadia.processing import DynamicLine
            import matplotlib.pyplot as plt
            from IPython.display import display
            from ipywidgets import FloatText, HBox, Button

            self.figsize = (4, 3) if self.figsize is None else self.figsize
            self.fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            self.fig.tight_layout()
            self.fig.subplots_adjust(left=0.25, bottom=0.25)

            self.line_mag = DynamicLine(ax, ".-", label="Mag", color="blue")
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Magnitude [arb.]", color="blue")
            ax.tick_params(axis='y', labelcolor="blue")
            ax.grid()

            ax_phase = ax.twinx()
            self.line_phase = DynamicLine(ax_phase, ".-", label="Phase", color="red")
            ax_phase.set_ylabel("Phase [rad.]", color="red")
            ax_phase.tick_params(axis='y', labelcolor="red")

            # Create interactive widgets for controlling the electrical delay
            self._electrical_delay_input = FloatText(value=self.electrical_delay, 
                                                        description="Electrical delay: ",
                                                        style={"description_width": "initial"})
            self._electrical_delay_update_button = Button(description="Set")
            self._electrical_delay_update_button.on_click(self.update_electrical_delay)
            self._electrical_delay_auto_button = Button(description="Auto")
            self._electrical_delay_auto_button.on_click(self.auto_electrical_delay)

            display(HBox([self._electrical_delay_input, 
                        self._electrical_delay_update_button, 
                        self._electrical_delay_auto_button]))

            from tqdm.notebook import tqdm

        else:
            from tqdm import tqdm

        self.iterations_progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.iterations_previous = 0
        self.frequencies_progress_bar = tqdm(desc="Frequency Sweep Points", dynamic_ncols=True, total=len(self.frequencies)*self.iterations)
        self.frequencies_previous = 0

        self.electrical_delay_phases = 2*np.pi*self.frequencies*self.electrical_delay
        self.data_summed = None
        self.data_complex = None
        self.data_mags = None
        self.data_phases =  None

    def update(self):
        # First make sure that we actually have new data to process
        if "iq_pts" not in self.data or len(self.data["iq_pts"]) < len(self.frequencies):
            return

        # Update the progress bar based on the number of iterations
        completed_iterations = len(self.data["iq_pts"]) // len(self.frequencies)
        self.iterations_progress_bar.update(completed_iterations - self.iterations_previous)

        completed_frequencies = len(self.data["iq_pts"])
        self.frequencies_progress_bar.update(completed_frequencies - self.frequencies_previous)

        # Only continue processing data if we have at least one complete iteration
        if completed_iterations != 0:
            valid_traces = completed_iterations*len(self.frequencies)
            self.process_data(valid_traces)
            self.update_plot()               
            self.data.save(self.local_directory)

        self.iterations_previous = completed_iterations
        self.frequencies_previous = completed_frequencies

    def finalize(self):
        super().finalize()
        self.iterations_progress_bar.close()
        self.frequencies_progress_bar.close()
        if self.plot:
            self.savefig(self.fig)
        self.fit_lorentzian()

    def process_data(self, valid_traces):
        data = self.data["iq_pts"].records()[:valid_traces, ...]

        # Get the collection of data and reshape it so that the axes index as: 
        # (iteration, frequency, sample time, sample quadrature)
        samples_per_trace = data.shape[-2]
        data_reshaped = data.reshape(-1, len(self.frequencies), samples_per_trace, 2)
        
        # Slice the data so that we have an array containing only the traces
        # we didn't have the last time update() was called
        self.new_data = data_reshaped[self.iterations_previous:, :, :, :]

        # Sum the new data and then add it to the aggregated array of trace data
        new_data_summed = np.sum(self.new_data, axis=(0,2), keepdims=False)
        if self.data_summed is None:
            self.data_summed = new_data_summed
        else:
            self.data_summed += new_data_summed

        # Convert the summed sample data to a complex number and choose 
        # the scale so that we turn the sum into a mean
        # Simultaneously, choose the scale so that the result is independent
        # of amplitude
        signal_config = self._ios["stimulus"]._config["signals"][self.stimulus_signal_name]
        signal_scale = signal_config["scale"] if "scale" in signal_config else 1.0
        total_scale = (valid_traces // len(self.frequencies)) / signal_scale 
        self.data_complex = Waveform.sample_to_complex(self.data_summed, scale=total_scale)
        self.data_mags = np.abs(self.data_complex)
        self.data_phases = np.angle(self.data_complex)

    def update_plot(self):
        if self.plot:
            display_phases = np.unwrap(self.data_phases - self.electrical_delay_phases)
            self.line_mag.update(self.frequencies, self.data_mags)
            self.line_phase.update(self.frequencies, display_phases)
            self.fig.canvas.draw_idle() 

    def update_electrical_delay(self, delay_value: float = None, lock: bool = True):
        from ipywidgets import Button
        if delay_value is None or isinstance(delay_value, Button):
            if not self.plot:
                raise ValueError("Must provide electrical delay value when it cannot"
                                " be retrieved from the display input.")
            delay_value = self._electrical_delay_input.value
        

        if lock and not self._update_lock.acquire(timeout=5):
            return False

        self.electrical_delay = delay_value
        self.electrical_delay_phases = 2*np.pi*self.frequencies*self.electrical_delay
        
        if self.plot:
            self._electrical_delay_input.value = self.electrical_delay
            self.update_plot()

        if lock:
            self._update_lock.release()

        return True

    def auto_electrical_delay(self, lock: bool = True) -> float:
        if lock and not self._update_lock.acquire(timeout=5):
            return None

        def model(freqs, delay, phi0):
            return 2*np.pi*freqs*delay + phi0
    
        popt,pcov = curve_fit(model, self.frequencies, np.unwrap(self.data_phases))
        self.update_electrical_delay(popt[0], lock=False)

        if lock:
            self._update_lock.release()

        return popt[0]

    def fit_lorentzian(self) -> float:
        from acadia_qmsmt.analysis import population_in_quadrant
        from acadia_qmsmt.analysis.fitting import rotate_iq
        from acadia_qmsmt.analysis.fitting.lorentzian import Lorentzian
        from acadia_qmsmt.helpers.plot_utils import add_button
        from acadia_qmsmt.helpers.yaml_editor import update_yaml

        fit = Lorentzian(self.frequencies, np.abs(self.data_complex))
        f0 = fit.ufloat_results["x0"]

        if self.plot:
            fit_fig, fit_ax = fit.plot()
            fit_ax.set_xlabel("Frequency [Hz]")
            fit_ax.set_ylabel(data_type)

            def _update_freq(event):
                self.update_ioconfig("stimulus", "channel_config.nco_frequency", np.round(f0.n))

            # add a button for updating the qubit freq in the yaml file
            self._update_button = add_button(fit_fig, _update_freq, label="Update Frequency")

        return f0

if __name__ == "__main__":
    freqs = np.linspace(-20e6, 20e6, 101) + 8.23e9
    rt = ResonatorSpecRuntime(stimulus="stimulus", capture="capture", frequencies=freqs, plot=True, iterations=50)
    rt.deploy("10.66.3.198")
    rt.display()
