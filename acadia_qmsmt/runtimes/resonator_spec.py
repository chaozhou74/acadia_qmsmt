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
        # Set the matplotlib backend to one which we can actually update
        self.figsize = (4, 3) if self.figsize is None else self.figsize

        if self.plot:
            from acadia.processing import DynamicLine
            import matplotlib.pyplot as plt
            from IPython.display import display
            from ipywidgets import Label

            self.fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            self.fig.tight_layout()
            self.fig.subplots_adjust(left=0.25, bottom=0.25)

            self.line_mag = DynamicLine(ax, ".-", label="Mag", color="blue")
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Magnitude [arb.]")
            ax.tick_params(axis='y', labelcolor="blue")
            ax.grid()

            ax_phase = ax.twinx()
            self.line_phase = DynamicLine(ax_phase, ".-", label="Phase", color="red")
            ax_phase.set_ylabel("Phase [rad.]")
            ax_phase.tick_params(axis='y', labelcolor="red")

            # Create a label for displaying the electrical delay
            self._delay_label = Label("Electrical delay: ")
            display(self._delay_label)

            from tqdm.notebook import tqdm

        else:
            from tqdm import tqdm

        self.iterations_progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.iterations_previous = 0
        self.frequencies_progress_bar = tqdm(desc="Frequency Sweep Points", dynamic_ncols=True, total=len(self.frequencies)*self.iterations)
        self.frequencies_previous = 0

        import numpy as np
        self.data_summed = None
        self.electrical_delay_phases = np.exp(2*np.pi*1j*self.frequencies*self.electrical_delay)

    def update(self):
        import numpy as np

        n_freqs = len(self.frequencies)

        # First make sure that we actually have new data to process
        if "iq_pts" not in self.data or len(self.data["iq_pts"]) < n_freqs:
            return

        # Update the progress bar based on the number of iterations
        completed_iterations = len(self.data["iq_pts"]) // len(self.frequencies)
        self.iterations_progress_bar.update(completed_iterations - self.iterations_previous)

        completed_frequencies = len(self.data["iq_pts"])
        self.frequencies_progress_bar.update(completed_frequencies - self.frequencies_previous)

        # Only continue processing data if we have at least one complete iteration
        if completed_iterations != 0:
        
            valid_traces = completed_iterations*len(self.frequencies)
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
            self.data_complex = Waveform.sample_to_complex(self.data_summed, scale=completed_iterations/signal_scale)

            # Apply the electrical delay
            self.data_complex *= self.electrical_delay_phases

            # We now have a 1D array of the amplitudes as a function of frequency,
            # so we can do whatever processing we want
            mags = np.abs(self.data_complex)
            phases = np.unwrap(np.angle(self.data_complex))

            # Update the fit
            def model(freqs, delay, phi0):
                return 2*np.pi*freqs*delay + phi0
        
            popt,pcov = curve_fit(model, self.frequencies, phases)
            self.fit_electrical_delay = popt[0]
            self.fit_electrical_delay_error = pcov[0,0]

            # Update the plot itself
            if self.plot:
                self.line_mag.update(self.frequencies, mags)
                self.line_phase.update(self.frequencies, phases)
                self.fig.canvas.draw_idle()
                self._delay_label.value = f"Fit electrical delay = {round(self.fit_electrical_delay*1e9,1)} ns +/- {round(self.fit_electrical_delay_error*1e12)} ps"
                

            # Save the data
            self.data.save(self.local_directory)

        self.iterations_previous = completed_iterations
        self.frequencies_previous = completed_frequencies

    def finalize(self):
        super().finalize()
        self.iterations_progress_bar.close()
        self.frequencies_progress_bar.close()
        if self.plot:
            self.savefig(self.fig)
        self.post_process()

    def post_process(self) -> float:
        """ Fit the spectroscopy to a Lorentzian function to extract the resonator frequency.

        """
        from acadia_qmsmt.analysis import population_in_quadrant
        from acadia_qmsmt.analysis.fitting import rotate_iq
        from acadia_qmsmt.analysis.fitting.lorentzian import Lorentzian
        from acadia_qmsmt.helpers.plot_utils import add_button
        from acadia_qmsmt.helpers.yaml_editor import update_yaml

        # Parse the data from the records for fitting
        iq_pts = self.data["iq_pts"].records().astype(float).view(complex).squeeze()
        iq_pts = np.reshape(iq_pts, (-1, len(self.frequencies)))

        fit = Lorentzian(self.frequencies, iq_pts)
        f0 = fit.ufloat_results["x0"]
        fit_fig, fit_ax = fit.plot()
        fit_ax.set_xlabel("Frequency [Hz]")
        fit_ax.set_ylabel(data_type)

        def _update_freq(event):
            self.update_ioconfig("stimulus", "channel_config.nco_frequency", np.round(f0.n))

        # add a button for updating the qubit freq in the yaml file
        self._update_button = add_button(fit_fig, _update_freq, label="Update Frequency")

        return f0

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from acadia_qmsmt.measurements import load_config

    from IPython.core.getipython import get_ipython
    get_ipython().run_line_magic("matplotlib", "widget")

    config_dict = load_config()

    plot = True
    iterations = 50
    freqs = np.linspace(-20e6, 20e6, 101) + 8.23e9

    rt = QubitSpecRuntime(**config_dict, frequencies=freqs, plot=plot, iterations=iterations)
    rt.deploy("10.66.3.198", "qubit_spec", files=rt.FILES)
    rt.display()

    # some ad-hoc processing
    # rt._event_loop.join()
    # rt.fig
