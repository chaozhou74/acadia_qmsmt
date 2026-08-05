from typing import Union, Literal

import numpy as np

from acadia import Acadia, DataManager
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
                            fit_type: Literal["magnitude", "phase"] = "phase") -> int:
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes
        
        data = reshape_iq_data_by_axes(self.data["points"].records(), self.frequencies)
        if data is None:
            return 0

        completed_iterations = len(data)

        self.data_iq = data.astype(float).view(complex).reshape(completed_iterations, len(self.frequencies))
        self.avg_iq = np.mean(self.data_iq, axis=0)
        self.fit_type = fit_type

        # --- 1. Electrical Delay Correction ---
        if electrical_delay == "auto":
            # Unwrap first to find linear delay
            phase_data = np.unwrap(np.angle(self.avg_iq))
            k_fit_idx =  np.max([len(self.frequencies)//10, 4])
            k0, _ = np.polyfit(self.frequencies[:k_fit_idx], phase_data[:k_fit_idx], deg=1)
            k1, _ = np.polyfit(self.frequencies[-k_fit_idx:], phase_data[-k_fit_idx:], deg=1)
            electrical_delay = -(k0 + k1)/2 / np.pi / 2
            
        self.electrical_delay_applied = electrical_delay
        self.avg_iq_corrected = self.avg_iq * np.exp(1j * self.frequencies * electrical_delay * np.pi * 2)
        
        # Keep a reference to the smooth, unwrapped phase (radians) for standard fitting
        self.phase_corrected = np.unwrap(np.angle(self.avg_iq_corrected))

        # --- 2. Fitting ---
        self.fit = None
        try:
            # Do fits in units of Hz for numerical stability
            if fit_type == "magnitude":
                from acadia_qmsmt.analysis.fitting import Lorentzian
                self.fit = Lorentzian(self.frequencies*1e-9, np.abs(self.avg_iq))
                self.fitted_f0 = self.fit.ufloat_results["x0"]*1e9
                
            elif fit_type == "phase":
                from acadia_qmsmt.analysis.fitting import Arctan
                # Fit to unwrapped degrees
                self.fit = Arctan(self.frequencies*1e-9, self.phase_corrected/np.pi*180)
                self.fitted_f0 = self.fit.ufloat_results["x0"]*1e9

        except:
            pass

        return completed_iterations
    

    @annotate_method(plot_name="magnitude_phase_vs_frequency", axs_shape=(2,1))
    def plot_data(self, axs=None, unwrap_phase: bool = True):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(2,1), figsize=self.figsize)

        # Plot Magnitude
        axs[0].errorbar(self.frequencies/1e9, np.abs(self.avg_iq_corrected), yerr=np.std(np.abs(self.data_iq), axis=0)/len(self.data_iq), fmt="o")
        
        # Plot Phase
        # Calculate degrees first
        phases_deg = self.phase_corrected / np.pi * 180
        
        if unwrap_phase:
            plot_phases = phases_deg
        else:
            plot_phases = (phases_deg + 180) % 360 - 180
            
        axs[1].errorbar(self.frequencies/1e9, plot_phases, yerr=np.std(self.phase_corrected/np.pi*180, axis=0)/len(self.phase_corrected), fmt="o")

        # Plot Fit Overlays
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
    def update_freq(self,ensure_proper_NCO_rounding: bool = True,ro_stim_samp_freq: int = 6.4e9, ro_capture_samp_freq: int = 2.4e9):

        if ensure_proper_NCO_rounding:
            # TODO: Don't hard-code sampling rates, get them from YAML config
            filtered_freq = self.align_ncos(np.round(self.fitted_f0.n), int(ro_stim_samp_freq), int(ro_capture_samp_freq))
        else:
            filtered_freq = {"f_in_modified": np.round(self.fitted_f0.n)}

        self.update_io_yaml_field("stimulus", "channel_config.nco_frequency", filtered_freq["f_in_modified"])
        self.update_io_yaml_field("capture", "channel_config.nco_frequency", filtered_freq["f_in_modified"])



    def align_ncos(self, f_in, s1, s2):
        import math

        from fractions import Fraction

        # 1. Cast input to Fraction to prevent floating-point drift
        f_in_frac = Fraction(str(f_in)) if isinstance(f_in, float) else Fraction(f_in)
        
        # 2. Find the LCM of the two sampling rates
        L = math.lcm(s1, s2)
        
        # 3. K must be a multiple of L. 
        # We find the integer multiplier 'm' that brings K closest to (f_in * 2^48)
        target_m = (f_in_frac * (1 << 48)) / L
        m = round(target_m)
        
        # K is our perfectly aligned, shared constant
        K = m * L
        
        # 4. The ideal modified input frequency
        f_in_mod_frac = Fraction(K, 1 << 48)
        f_in_mod = float(f_in_mod_frac)
        
        # 5. Compute the ideal double words (guaranteed exact integers, no rounding needed)
        w1_double = K // s1
        w2_double = K // s2
        
        # 6. Apply masking and Nyquist logic
        mask = (1 << 48) - 1
        w1 = w1_double & mask
        w2 = w2_double & mask
        
        nz1 = w1_double >> 48
        nz2 = w2_double >> 48
        
        # 7. Reverse flow computation to prove f_out_1 == f_out_2
        f_out_1 = float((Fraction(w1, 1 << 48) + nz1) * s1)
        f_out_2 = float((Fraction(w2, 1 << 48) + nz2) * s2)
        
        return {
            "f_in_original": f_in,
            "f_in_modified": f_in_mod,
            "delta_required": float(f_in_mod_frac - f_in_frac),
            "nco_1": {
                "word": w1,
                "nyquist_zone": nz1,
                "f_out": f_out_1
            },
            "nco_2": {
                "word": w2,
                "nyquist_zone": nz2,
                "f_out": f_out_2
            }
        }