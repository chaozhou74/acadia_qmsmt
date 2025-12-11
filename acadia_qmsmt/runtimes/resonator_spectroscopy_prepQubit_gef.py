from typing import Union, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime
from acadia.sample_arithmetic import sample_to_complex
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, IOConfig, Qubit
from acadia.runtime import annotate_method

class ResonatorSpectroscopyPrepQubitGEFRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` subclass for readout spectroscopy
    """
    # The name of the sections in the yaml file for the required channels
    stimulus: IOConfig
    capture: IOConfig
    qubit_stimulus: IOConfig

    frequencies: Union[list, np.ndarray]

    iterations: int
    run_delay: int

    qubit_ge_pulse_name: str = "R_x_180"
    qubit_gf_pulse_name: str = "R_x_180_gf"
    readout_pulse_name: str = "readout"
    capture_memory_name: str = "readout_accumulated"
    capture_window_name: str = None

    figsize: tuple[int] = None
    yaml_path: str = None

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

        stimulus_io = self.io("stimulus")
        capture_io = self.io("capture")
        qubit_stimulus_io = self.io("qubit_stimulus")

        resonator = MeasurableResonator(stimulus_io, capture_io)

        # Create the record group for saving captured data
        self.data.add_group(f"points", uniform=True)

        # Create a sequence for the sequencer to generate the pulse and capture it
        prep_selection_cache = self.acadia.CacheArray(shape=(1,), dtype=np.dtype("<i4"))

        def sequence(a: Acadia):
            prep_selection_reg = a.sequencer().Register()
            prep_selection_reg.load(prep_selection_cache[0])
            
            with a.sequencer().test(prep_selection_reg == 0):
                with a.channel_synchronizer():
                    resonator.measure(self.readout_pulse_name, self.capture_memory_name, self.capture_window_name)

            with a.sequencer().test(prep_selection_reg == 1):
                with a.channel_synchronizer():
                    qubit_stimulus_io.schedule_pulse(self.qubit_ge_pulse_name)
                    a.barrier()
                    resonator.measure(self.readout_pulse_name, self.capture_memory_name, self.capture_window_name)
            
            with a.sequencer().test(prep_selection_reg == 2):
                with a.channel_synchronizer():
                    qubit_stimulus_io.schedule_pulse(self.qubit_gf_pulse_name)
                    a.barrier()
                    resonator.measure(self.readout_pulse_name, self.capture_memory_name, self.capture_window_name)

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
        stimulus_io.load_pulse(self.readout_pulse_name)
        qubit_stimulus_io.load_pulse(self.qubit_ge_pulse_name)
        qubit_stimulus_io.load_pulse(self.qubit_gf_pulse_name)

        for i in range(self.iterations):
            for frequency in self.frequencies:
                resonator.set_frequency(frequency)
                for prep in [0,1,2]:
                    prep_selection_cache[0] = prep
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
    def process_current_data(self, e_delay:Union[float, str]="auto", fit_type:Literal["mag", "phase"]="phase"):
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes
        data = reshape_iq_data_by_axes(self.data["points"].records(), self.frequencies, [0, 1, 2])
        if data is None:
            return
        else:
            completed_iterations = len(data)
        self.data_iq = data.astype(float).view(complex).squeeze()
        self.avg_iq = np.mean(self.data_iq, axis=0)
        self.fit_type = fit_type

        if e_delay == "auto":
            # find edelay
            phase_data = np.unwrap(np.angle(self.avg_iq[:, 0]))
            k_fit_idx =  np.max([len(self.frequencies)//10, 4])
            k0, _ = np.polyfit(self.frequencies[:k_fit_idx], phase_data[:k_fit_idx], deg=1)
            k1, _ = np.polyfit(self.frequencies[-k_fit_idx:], phase_data[-k_fit_idx:], deg=1)
            e_delay = -(k0 + k1)/2 / np.pi / 2
        self.e_delay_applied = e_delay
        self.avg_iq_corrected = self.avg_iq * np.exp(1j * self.frequencies[:, np.newaxis] * e_delay * np.pi * 2)
        self.phase_corrected = np.array([np.unwrap(np.angle(iq)) for iq in self.avg_iq_corrected.T]).T

        if fit_type == "mag":
            from acadia_qmsmt.analysis.fitting import Lorentzian
            self.fit_g = Lorentzian(self.frequencies, np.abs(self.avg_iq[:, 0]))
            self.fit_e = Lorentzian(self.frequencies, np.abs(self.avg_iq[:, 1]))
            self.fit_f = Lorentzian(self.frequencies, np.abs(self.avg_iq[:, 2]))
            
        elif fit_type == "phase":
            from acadia_qmsmt.analysis.fitting import Arctan
            self.fit_g = Arctan(self.frequencies, self.phase_corrected[:, 0]/np.pi*180)
            self.fit_e = Arctan(self.frequencies, self.phase_corrected[:, 1]/np.pi*180)
            self.fit_f = Arctan(self.frequencies, self.phase_corrected[:, 2]/np.pi*180)

        self.fitted_f0_g = self.fit_g.ufloat_results["x0"]
        self.fitted_f0_e = self.fit_e.ufloat_results["x0"]
        self.fitted_f0_f = self.fit_f.ufloat_results["x0"]

        return completed_iterations
    

    @annotate_method(plot_name="mag_phase_vs_dac", axs_shape=(2,1))
    def plot_data(self, axs=None, unwrap_phase:bool=True):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(2,1), figsize=self.figsize)

        for i in range(3):
            axs[0].plot(self.frequencies, np.abs(self.avg_iq_corrected[:, i]), "o")
            phases = self.phase_corrected # if unwrap_phase else self.phase_corrected % (2*np.pi)
            axs[1].plot(self.frequencies, phases[:, i]/np.pi*180, ".-")

        plot_fit_ax = axs[0] if self.fit_type == "mag" else axs[1]
        self.fit_g.plot_fitted(plot_fit_ax, label=f"prep g f0: {self.fitted_f0_g}")
        self.fit_e.plot_fitted(plot_fit_ax, label=f"prep e f0: {self.fitted_f0_e}")
        self.fit_f.plot_fitted(plot_fit_ax, label=f"prep f f0: {self.fitted_f0_f}")
        plot_fit_ax.legend()

        axs[1].set_xlabel("Frequency [Hz]")
        axs[1].set_ylabel("Phase (deg)")
        axs[0].set_ylabel("Mag (a.u.)")
        axs[0].set_title(f"edelay applied: {self.e_delay_applied:.6g}, "
                         f"chi: {(self.fitted_f0_e - self.fitted_f0_g)/1e6:.6g} MHz")
        for ax in axs:
            ax.grid(True)

        fig.tight_layout()
        return fig, axs


