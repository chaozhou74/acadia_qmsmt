from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime
from acadia.runtime import annotate_method
from acadia.sample_arithmetic import sample_to_complex
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, IOConfig

class ResonatorSpectroscopyRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` subclass for readout spectroscopy
    """
    # The name of the sections in the yaml file for the required channels
    stimulus: IOConfig
    capture: IOConfig

    frequencies: Union[list, np.ndarray]

    iterations: int
    run_delay: int

    stimulus_pulse_name: str = "readout"
    capture_memory_name: str = "readout_accumulated"
    capture_window_name: str = None


    plot: bool = True
    figsize: tuple[int] = None
    yaml_path: str = None

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

        stimulus_io = self.io("stimulus")
        capture_io = self.io("capture")

        resonator = MeasurableResonator(stimulus_io, capture_io)

        # Create the record group for saving captured data
        self.data.add_group(f"points", uniform=True)
        # Create a sequence for the sequencer to generate the pulse and capture it
        def sequence(a: Acadia):

            with a.channel_synchronizer():
                # Measure the resonator by driving the "readout" waveform on the stimulus IO
                # and capture into the "readout_accumulated" waveform on the capture IO
                resonator.measure(self.stimulus_pulse_name, self.capture_memory_name, self.capture_window_name)

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
        # Load the stimulus pulse named "readout" with the specified signal
        stimulus_io.load_pulse(self.stimulus_pulse_name)   # since the readout pulse memory only has one set of sample in it, we don't have to specify the samples here, it will just use that one

        for i in range(self.iterations):
            for j, frequency in enumerate(self.frequencies):
                resonator.set_frequency(frequency)

                # capture data and put in the corresponding group
                self.acadia.run(minimum_delay=self.run_delay)

                wf = capture_io.get_waveform_memory(self.capture_memory_name)
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

        from acadia_qmsmt.analysis.fitting.lorentzian import Lorentzian
        self.fit = Lorentzian(self.frequencies, np.abs(self.avg_iq))
        self.fitted_f0 = self.fit.ufloat_results["x0"]
        self.e_delay_applied = e_delay

        return completed_iterations
    

    @annotate_method(plot_name="mag_phase_vs_dac", axs_shape=(2,1))
    def plot_data(self, axs=None, apply_e_delay:bool=True, unwrap_phase:bool=True):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(2,1), figsize=self.figsize)

        data = self.avg_iq_corrected if apply_e_delay else self.avg_iq
        axs[0].plot(self.frequencies, np.abs(data), "o")
        
        # axs[0].plot(self.frequencies, self.fit.eval(), "-", label=f"{self.fitted_f0}")
        self.fit.plot_fitted(axs[0], oversample=1, label=f"{self.fitted_f0}")

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

    @annotate_method(button_name="update frequency")
    def update_freq(self):
        self.update_io_yaml_field("stimulus", "channel_config.nco_frequency", np.round(self.fitted_f0.n))
        self.update_io_yaml_field("capture", "channel_config.nco_frequency", np.round(self.fitted_f0.n))


