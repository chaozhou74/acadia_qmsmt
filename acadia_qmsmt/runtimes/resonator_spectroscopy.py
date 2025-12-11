from typing import Union, Literal

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
        from acadia_qmsmt.plotting import save_registered_plots
        save_registered_plots(self)


    @annotate_method(is_data_processor=True)
    def process_current_data(self, 
                            electrical_delay: Union[float, str] = "auto", 
                            fit_type: Literal["magnitude", "phase"] = "phase"):
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes
        
        data = reshape_iq_data_by_axes(self.data["points"].records(), self.frequencies)
        if data is None:
            return 0

        completed_iterations = len(data)

        self.data_iq = data.astype(float).view(complex).reshape(completed_iterations, len(self.frequencies))
        self.avg_iq = np.mean(self.data_iq, axis=0)
        self.fit_type = fit_type


        if electrical_delay == "auto":
            # find edelay
            phase_data = np.unwrap(np.angle(self.avg_iq))
            k_fit_idx =  np.max([len(self.frequencies)//10, 4])
            k0, _ = np.polyfit(self.frequencies[:k_fit_idx], phase_data[:k_fit_idx], deg=1)
            k1, _ = np.polyfit(self.frequencies[-k_fit_idx:], phase_data[-k_fit_idx:], deg=1)
            electrical_delay = -(k0 + k1)/2 / np.pi / 2
        self.electrical_delay_applied = electrical_delay
        self.avg_iq_corrected = self.avg_iq * np.exp(1j * self.frequencies * electrical_delay * np.pi * 2)
        self.phase_corrected = np.unwrap(np.angle(self.avg_iq_corrected))

        self.fit = None
        try:
            # Do fits in units of Hz for numerical stability
            if fit_type == "magnitude":
                from acadia_qmsmt.analysis.fitting import Lorentzian
                self.fit = Lorentzian(self.frequencies*1e-9, np.abs(self.avg_iq))
                self.fitted_f0 = self.fit.ufloat_results["x0"]*1e9
                
            elif fit_type == "phase":
                from acadia_qmsmt.analysis.fitting import Arctan
                self.fit = Arctan(self.frequencies*1e-9, self.phase_corrected/np.pi*180)
                self.fitted_f0 = self.fit.ufloat_results["x0"]*1e9
        except:
            pass

        return completed_iterations
    

    @annotate_method(plot_name="magnitude_phase_vs_frequency", axs_shape=(2,1))
    def plot_data(self, axs=None, unwrap_phase: bool = True):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(2,1), figsize=self.figsize)

        axs[0].plot(self.frequencies/1e9, np.abs(self.avg_iq_corrected), "o")
        phases = self.phase_corrected if unwrap_phase else self.phase_corrected % (2*np.pi)
        axs[1].plot(self.frequencies/1e9, phases/np.pi*180, ".-")

        if self.fit is not None:
            plot_fit_ax = axs[0] if self.fit_type == "magnitude" else axs[1]
            self.fit.plot_fitted(plot_fit_ax, label=f"{self.fitted_f0}")
            plot_fit_ax.legend()


        axs[1].set_xlabel("Frequency [GHz]")
        axs[1].set_ylabel("Phase (deg)")
        axs[0].set_ylabel("Magnitude (a.u.)")
        title = f"electrical delay: {self.electrical_delay_applied:.6g} s"
        if self.fit is not None:
            title += f", f0: {self.fitted_f0/1e9:.6g} GHz"
        axs[0].set_title(title)

        for ax in axs:
            ax.grid(True)

        fig.tight_layout()
        return fig, axs

    @annotate_method(button_name="update frequency")
    def update_freq(self):
        self.update_io_yaml_field("stimulus", "channel_config.nco_frequency", np.round(self.fitted_f0.n))
        self.update_io_yaml_field("capture", "channel_config.nco_frequency", np.round(self.fitted_f0.n))


