from typing import Union, Annotated
from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime, WaveformMemory
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig
from acadia.runtime import annotate_method

def dual_sine_decay(t, A, tau, B, f1, f2):
    return A * np.exp(-t/tau) * np.cos(2*np.pi*f1*t) * np.cos(2*np.pi*f2*t)  + B

class QubitCoherenceRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` subclass for measuring the T2 of a qubit.
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

    readout_pulse_name: str = "readout"
    capture_memory_name: str = "readout_accumulated"
    capture_window_name: str = None


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

        # Create an array in the cache that we can use to pass the 
        # delay value to the sequencer so that we don't have to reassemble every time
        cache = self.acadia.CacheArray(shape=(1,), dtype=np.dtype("<i4"))



        qubit_pi_over_2_pulse_name = qubit.make_rotation_pulse(90, 0, self.qubit_pi_pulse_name)
        second_pi_over_2_pulse_name = qubit_stimulus_io.duplicate_pulse(qubit_pi_over_2_pulse_name)



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
            qubit_stimulus_io.load_pulse(self.qubit_pi_pulse_name)
        
        # get the original scale and detune for applying virtual phase shift
        second_pulse_scale = qubit_stimulus_io.get_config("pulses", second_pi_over_2_pulse_name, "scale")

        # get the origianl detune on the pulse to make it effectively CW later
        pulse_detune = self._ios["qubit_stimulus"].get_config("pulses", self.qubit_pi_pulse_name).get("detune", 0)

        # Determine how many cycles each delay interval should be
        dsp_count_values = self.acadia.seconds_to_cycles(self.delay_times/2)

        for i in range(self.iterations):
            for idx_delay,delay in enumerate(dsp_count_values):
                # Update the delay amount in the cache
                cache[0] = delay

                # Shift the phase of the second pulse according to the virtual detuning and the pulse detune
                scale = second_pulse_scale * np.exp(2 * np.pi * 1j *
                                                    (self.virtual_detuning + pulse_detune) * self.delay_times[idx_delay])
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
    def process_current_data(self, readout_classifier: Annotated[str, "IOConfig", "readout_capture.classifiers"] = None):
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes
        from acadia_qmsmt.analysis.fitting import ExpCosine

        data = reshape_iq_data_by_axes(self.data["points"].records(), self.delay_times, to_complex=True)
        if data is None:
            return

        completed_iterations = len(data)
        readout_resonator = MeasurableResonator(self.io("readout_stimulus"), self.io("readout_capture"))

        self.data_complex = data
        self.shots = readout_resonator.classify_measurement(self.data_complex, readout_classifier)
        self.avg_shots = np.mean(self.shots, axis=0)
        self.sigma_shots = np.std(self.shots, axis=0) / np.sqrt(completed_iterations)
        
        self.fit = ExpCosine(self.delay_times * 1e6, self.avg_shots, sigma=self.sigma_shots)
        self.fitted_t2_us = self.fit.ufloat_results["tau"]
        self.fitted_detune_MHz = self.fit.ufloat_results["f"]
        return completed_iterations
    

    @annotate_method(plot_name="1 qubit T2", axs_shape=(1,1))
    def plot_data(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        self.fit.plot(axs, oversample=5, 
                            result_kwargs={"label":f"T2 (us): {self.fitted_t2_us:.4g}\nDetune(MHz): {self.fitted_detune_MHz:.6g}"})
        # self.fit.plot_fitted(axs, oversample=5, label=f"T2 (us): {self.fitted_t2_us:.4g}\nDetune(MHz): {self.fitted_detune_MHz:.4g}")

        axs.set_xlabel("Time [us]")
        axs.set_ylabel("Average Measurement")
        if np.ptp(self.avg_shots) > 0.5 and np.ptp(self.avg_shots) < 1.0:
            axs.set_ylim(-0.02, 1.02)

        axs.set_title(f"T2{'E' if self.do_echo else 'R'}: {self.fit.ufloat_results['tau']:.5g}")

        axs.legend()
        return fig, axs
    
    @annotate_method(plot_name="2 bin averaged T2", axs_shape=(1,1))
    def plot_bin_avg(self, axs=None, n_avg=1):
        from acadia_qmsmt.plotting import prepare_plot_axes, plot_binaveraged
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        fig, axs = plot_binaveraged(self.delay_times*1e6, self.shots, axs, n_avg=n_avg, vmin=0, v_max=1)
        axs.set_ylabel("Time [us]")
        return fig, axs


    @annotate_method(button_name="update qubit nco frequency (if Ramsey)")
    def update_frequencies(self):

        if not self.do_echo:
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
