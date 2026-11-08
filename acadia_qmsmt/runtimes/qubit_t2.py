from typing import Union, Annotated, Literal
from functools import partial
import logging

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime, WaveformMemory
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig
from acadia.runtime import annotate_method

logger = logging.getLogger("acadia")

# second pi/2 pulse phases (rad) used by `do_quadrature_Ramsey`: in-phase and quadrature
INTERLEAVED_PHASES = (0.0, np.pi / 2)

def dual_sine_decay(t, A, tau, B, f1, f2):
    return A * np.exp(-t/tau) * np.cos(2*np.pi*f1*t) * np.cos(2*np.pi*f2*t)  + B

class QubitCoherenceRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` subclass for measuring the T2 of a qubit.

    When `do_quadrature_Ramsey` is True, every delay is measured twice back to back, with the second
    pi/2 pulse at phase 0 and at pi/2 (`initial_qubit_phase` is ignored). The two quadratures give the
    sign of the detuning, see `plot_oscillation_with_sign`.
    """
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    delay_times: Union[list, np.ndarray]
    virtual_detuning: float = 0
    do_echo:bool = False


    iterations: int
    run_delay: int
    
    qubit_pi_pulse_name: str = "R_x_180"

    initial_qubit_phase: float = 0.0
    do_quadrature_Ramsey: bool = False

    readout_pulse_name: str = "readout"
    capture_memory_name: str = "readout_accumulated"
    capture_window_name: str = None


    figsize: tuple[int] = None
    yaml_path: str = None

    def main(self):
        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")

        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)
        qubit = Qubit(qubit_stimulus_io)

        self.data.add_group(f"points", uniform=True)

        # Create an array in the cache that we can use to pass the 
        # delay value to the sequencer so that we don't have to reassemble every time
        cache = self.acadia.CacheArray(shape=(1,), dtype=np.dtype("<i4"))



        qubit_pi_over_2_pulse_name = qubit.make_rotation_pulse(90, 0, self.qubit_pi_pulse_name)
        second_pi_over_2_pulse_name = qubit_stimulus_io.duplicate_pulse(qubit_pi_over_2_pulse_name)

        # phases of the second pi/2 pulse, one shot per phase for every delay (the last axis of the data)
        if self.do_quadrature_Ramsey:
            if self.initial_qubit_phase != 0:
                logger.warning("do_quadrature_Ramsey is enabled: every delay is measured with the second "
                               "pi/2 pulse at phase 0 and pi/2. The input initial_qubit_phase is ignored.")
            qubit_phases = INTERLEAVED_PHASES
        else:
            qubit_phases = (self.initial_qubit_phase,)

        def sequence(a: Acadia):
            # Initialize a Register for the dwell time
            counter = a.sequencer().Register()

            # Load the counter with the value we put into the cache
            counter.load(cache[0])

            with a.channel_synchronizer():
                qubit_stimulus_io.schedule_pulse(qubit_pi_over_2_pulse_name)
            
            with a.channel_synchronizer():
                qubit_stimulus_io.dwell(counter)
                
            if self.do_echo:
                with a.channel_synchronizer():
                   qubit_stimulus_io.schedule_pulse(self.qubit_pi_pulse_name)

            with a.channel_synchronizer():
                qubit_stimulus_io.dwell(counter)

            with a.channel_synchronizer():
                qubit_stimulus_io.schedule_pulse(second_pi_over_2_pulse_name)
                a.barrier()
                readout_resonator.measure(self.readout_pulse_name, self.capture_memory_name, self.capture_window_name)


        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_pulse(self.readout_pulse_name)
        qubit_stimulus_io.load_pulse(qubit_pi_over_2_pulse_name)
        

        if self.do_echo:
            qubit_stimulus_io.load_pulse(self.qubit_pi_pulse_name, phase=np.pi/2)
        
        # get the original scale and detune for applying virtual phase shift
        second_pulse_scale = qubit_stimulus_io.get_config("pulses", second_pi_over_2_pulse_name, "scale")

        # get the original detune on the pulse to make it effectively CW later
        pulse_detune = self._ios["qubit_stimulus"].get_config("pulses", self.qubit_pi_pulse_name).get("detune", 0)

        # Determine how many cycles each delay interval should be
        dsp_count_values = self.acadia.seconds_to_cycles(self.delay_times/2)

        for i in range(self.iterations):
            for idx_delay,delay in enumerate(dsp_count_values):
                # Update the delay amount in the cache
                cache[0] = delay

                for qubit_phase in qubit_phases:
                    # Shift the phase of the second pulse according to the virtual detuning and the pulse detune
                    scale = second_pulse_scale * np.exp(2 * np.pi * 1j *
                                                        (self.virtual_detuning + pulse_detune) * self.delay_times[idx_delay] + 1j*qubit_phase)
                    qubit_stimulus_io.load_pulse(second_pi_over_2_pulse_name, scale=scale)

                    # Capture data and put in the corresponding group
                    self.acadia.run(minimum_delay=self.run_delay)
                    wf = readout_capture_io.get_waveform_memory(self.capture_memory_name)
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
    def process_current_data(self, readout_classifier: Annotated[str, "IOConfig", "readout_capture.classifiers"] = None,
            fit_function: Literal["ExpCosine", "StretchedExpCosine"] = "ExpCosine"):
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes
        from acadia_qmsmt.analysis.fitting import ExpCosine, StretchedExpCosine

        from acadia_qmsmt.utils.fourier_transform import fft_mag

        fit_classes = {
            "ExpCosine": ExpCosine,
            "StretchedExpCosine": StretchedExpCosine,
        }
        FitClass = fit_classes[fit_function]

        # with do_quadrature_Ramsey each delay has two shots (phase 0, pi/2) on the last axis
        axes = (self.delay_times, len(INTERLEAVED_PHASES)) if self.do_quadrature_Ramsey else (self.delay_times,)
        data = reshape_iq_data_by_axes(self.data["points"].records(), *axes, to_complex=True)
        if data is None:
            return
        completed_iterations = len(data)
        readout_resonator = MeasurableResonator(self.io("readout_stimulus"), self.io("readout_capture"))
        self.data_complex = data
        # classify both quadratures together so they share the same classifier
        shots = readout_resonator.classify_measurement(self.data_complex, readout_classifier)
        if self.do_quadrature_Ramsey:
            # `self.shots` is always the phase 0 data, so all the plots except "oscillation with sign" are unchanged
            self.shots, self.shots_pi2 = shots[..., 0], shots[..., 1]
        else:
            self.shots, self.shots_pi2 = shots, None
        self.avg_shots = np.mean(self.shots, axis=0)
        self.sigma_shots = np.std(self.shots, axis=0) / np.sqrt(completed_iterations)
        self.fit = FitClass(self.delay_times * 1e6, self.avg_shots, sigma=self.sigma_shots)
        self.fitted_t2_us = self.fit.ufloat_results["tau"]
        self.fitted_detune_MHz = self.fit.ufloat_results["f"]
        self.fitted_beta = self.fit.ufloat_results.get("beta", None)

        self.fft = fft_mag(self.delay_times*1e6, self.avg_shots - np.mean(self.avg_shots))

        self.sign_fit = None
        if self.do_quadrature_Ramsey:
            self._fit_oscillation_with_sign(FitClass, completed_iterations)
        return completed_iterations

    def _fit_oscillation_with_sign(self, FitClass, completed_iterations):
        """
        Combine the phase 0 and phase pi/2 Ramsey fringes into a phase that grows linearly with the delay.

        With the qubit starting in g and the second pi/2 pulse at phase phi, the excited population is
        (1 + A exp(-t/T2) cos(2 pi f_delta t - phi)) / 2, with f_delta = f_qubit - f_drive. So the phase 0 fringe
        is the cosine and the phase pi/2 fringe is the sine of 2 pi f_delta t, whose unwrapped angle is a straight
        line with slope 2 pi f_delta. Unlike the cosine fit, this keeps the sign of f_delta.

        Sets `sign_fit` (Linear fit of the unwrapped phase in rad vs time in us, None if unavailable),
        `sign_times_us`, `sign_phase`, `sign_phase_sigma` and `f_delta_MHz`.
        """
        from acadia_qmsmt.analysis.fitting import Linear

        times_us = self.delay_times * 1e6
        avg_pi2 = np.mean(self.shots_pi2, axis=0)
        sigma_pi2 = np.std(self.shots_pi2, axis=0) / np.sqrt(completed_iterations)
        self.fit_pi2 = FitClass(times_us, avg_pi2, sigma=sigma_pi2)

        # Both quadratures oscillate around the same offset, take it from the two cosine fits
        offset = np.mean([self.fit.ufloat_results["of"].n, self.fit_pi2.ufloat_results["of"].n])
        if not np.isfinite(offset):
            logger.warning("Cosine fits failed, can't compute the oscillation with sign.")
            return

        cos_part = self.avg_shots - offset
        sin_part = avg_pi2 - offset
        order = np.argsort(times_us)  # unwrap needs increasing delay
        amplitude_sq = np.maximum(cos_part**2 + sin_part**2, 1e-12)
        # first order error propagation of arctan2(sin_part, cos_part). Points with a small amplitude get a
        # large error instead of being removed. Capped at pi (phase completely undetermined) so that
        # the sigma floor in the fitter isn't dominated by these points.
        sigma_phase = np.sqrt((cos_part * sigma_pi2)**2 + (sin_part * self.sigma_shots)**2) / amplitude_sq
        sigma_phase = np.minimum(sigma_phase, np.pi)

        self.sign_times_us = times_us[order]
        self.sign_phase = np.unwrap(np.arctan2(sin_part, cos_part)[order])
        self.sign_phase_sigma = sigma_phase[order]
        # error_model="wls": the phase can be in [-1, 1], which "auto" would take for shot probabilities
        self.sign_fit = Linear(self.sign_times_us, self.sign_phase, sigma=self.sign_phase_sigma, error_model="wls")
        self.f_delta_MHz = self.sign_fit.ufloat_results["m"] / (2 * np.pi)

    @annotate_method(plot_name="1 qubit T2", axs_shape=(1,1))
    def plot_data(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        label = f"T2 (us): {self.fitted_t2_us:.4g}\nDetune(MHz): {self.fitted_detune_MHz:.6g}"
        if self.fitted_beta is not None:
            label += f"\nbeta: {self.fitted_beta:.3g}"

        self.fit.plot(axs, oversample=5, result_kwargs={"label": label})
        axs.set_xlabel("Time [us]")
        axs.set_ylabel("Average Measurement")
        if np.ptp(self.avg_shots) > 0.5 and np.ptp(self.avg_shots) < 1.0:
            axs.set_ylim(-0.02, 1.02)

        title = f"T2{'E' if self.do_echo else 'R'}: {self.fit.ufloat_results['tau']:.5g}"
        if self.fitted_beta is not None:
            title += f", β: {self.fitted_beta:.3g}"
        axs.set_title(title)

        axs.legend()
        return fig, axs
    
    @annotate_method(plot_name="2 bin averaged T2", axs_shape=(1,1))
    def plot_bin_avg(self, axs=None, n_avg=1):
        from acadia_qmsmt.plotting import prepare_plot_axes, plot_binaveraged
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        fig, axs = plot_binaveraged(self.delay_times*1e6, self.shots, axs, n_avg=n_avg, vmin=0, v_max=1)
        axs.set_ylabel("Time [us]")
        return fig, axs

    @annotate_method(plot_name="3: T2 fft", axs_shape=(1,1))
    def plot1_T2_fft(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, ax = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)
        ax.plot(self.fft[0][2:], self.fft[1][2:])
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("FFT Magnitude")
        ax.set_yscale("log")
        ax.grid(True)

        return fig, ax

    @annotate_method(plot_name="4: oscillation with sign", axs_shape=(1,1))
    def plot_oscillation_with_sign(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        if not self.do_quadrature_Ramsey or getattr(self, "sign_fit", None) is None:
            axs.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=axs.transAxes)
        else:
            label = (f"f_delta (MHz): {self.f_delta_MHz:.6g}\n"
                     f"offset (rad): {self.sign_fit.ufloat_results['b']:.3g}")
            self.sign_fit.plot(axs, oversample=2, result_kwargs={"label": label})
            axs.set_xlabel("Time [us]")
            axs.set_ylabel("arctan(osc. with pi/2 / osc. without) [rad]")
        axs.set_title("oscillation with sign")
        return fig, axs


    @annotate_method(button_name="update qubit nco frequency (if Ramsey)")
    def update_frequencies(self):

        if self.do_echo:
            logger.warning("The echo refocuses the detuning, there is no frequency to update from.")
            return

        if self.do_quadrature_Ramsey and getattr(self, "sign_fit", None) is not None: # do signed fit
            f_delta = self.f_delta_MHz.n * 1e6
            if not np.isfinite(f_delta):
                logger.warning("The fit of the oscillation with sign failed, not updating the NCO frequency.")
                return

            nco_freq = self._ios["qubit_stimulus"].get_config("channel_config", "nco_frequency")
            pulse_detune = self._ios["qubit_stimulus"].get_config("pulses", self.qubit_pi_pulse_name).get("detune", 0)
            detuning = self.virtual_detuning + f_delta  # f_qubit - (f_nco + f_pulse_detune)
            self.update_io_yaml_field("qubit_stimulus", f"channel_config.nco_frequency", np.round(nco_freq + detuning))
            self.qubit_freq = np.round(nco_freq + pulse_detune + detuning)

        else: # don't do signed fit, because signed fit is not enabled
            if self.fit is not None:
                nco_freq = self._ios["qubit_stimulus"].get_config("channel_config", "nco_frequency")
                # we assume the virtual detuning is large enough that the apparent detuning always has the same sign
                # as the virtual detuning
                detuning = self.virtual_detuning - abs(self.fit.ufloat_results['f'].n*1e6) * np.sign(self.virtual_detuning)
                self.update_io_yaml_field("qubit_stimulus", f"channel_config.nco_frequency", 
                                          np.round(nco_freq + detuning)
                )
                pulse_detune = self._ios["qubit_stimulus"].get_config("pulses", self.qubit_pi_pulse_name).get("detune", 0)
                qubit_freq0 = nco_freq + pulse_detune
                self.qubit_freq = np.round(qubit_freq0 + detuning)
