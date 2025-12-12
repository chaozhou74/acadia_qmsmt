from typing import Union, Annotated

import numpy as np
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig, QubitQmCooler
from acadia.runtime import annotate_method


class QMT2Runtime(QMsmtRuntime):
    """
    A :class:`Runtime` subclass for sweeping the rabi time
    """
    qubit_stimulus: IOConfig
    bs_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    delay_times: Union[list, np.ndarray]
    soft_detune: float = 0
    do_echo:bool = False

    iterations: int
    run_delay: int


    qubit_pi_pulse_name: str = "R_x_180"
    bs_pulse_name: str = "swap"
    readout_pulse_name: str = "readout"
    capture_memory_name: str = "readout_accumulated"
    capture_window_name: str = "matched"

    cool_swap_pulse_name: str = "swap"
    cool_qm_rounds: int = 0

    plot: bool = True
    figsize: tuple[int] = None
    yaml_path: str = None

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")
        bs_stimulus_io = self.io("bs_stimulus")

        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)
        qubit = Qubit(qubit_stimulus_io)
        cooler = QubitQmCooler(qubit, readout_resonator, bs_stimulus_io)

        if self.cool_qm_rounds>0:
            self.data.add_group(f"prep", uniform=True)

        self.data.add_group(f"points", uniform=True)

        # Create an array in the cache that we can use to pass the
        # delay value to the sequencer so that we don't have to reassemble every time
        cache = self.acadia.CacheArray(shape=(1,), dtype=np.dtype("<i4"))
        prep_capture_mem = readout_capture_io.get_waveform_memory(self.capture_memory_name).duplicate()

        qubit_pi_over_2_initial = qubit.make_rotation_pulse(90, 0, self.qubit_pi_pulse_name)
        qubit_pi_over_2_final = qubit.make_rotation_pulse(90, 0, self.qubit_pi_pulse_name)


        def sequence(a: Acadia):
            # Initialize a DSP to act as a counter
            counter_1 = a.sequencer().DSP()
            counter_2 = a.sequencer().DSP()

            # Load the counter with the value we put into the cache
            counter_1.load(cache[0])
            counter_2.load(cache[0])

            if self.cool_qm_rounds>0:
                cooler.cool(1, self.qubit_pi_pulse_name,
                    self.readout_pulse_name, self.capture_memory_name, self.capture_window_name,
                    self.cool_swap_pulse_name, self.cool_qm_rounds) # efficient use of register

            with a.channel_synchronizer(block=True):
                qubit.schedule_pulse(qubit_pi_over_2_initial)
                a.barrier()
                bs_stimulus_io.schedule_pulse(self.bs_pulse_name)

            # Start the counter for the 1st half wait and wait until it reaches zero
            # counter_1.start_count(inc=int(np.int32(-1).astype(np.uint32)))
            # with a.sequencer().repeat_until(counter_1 == 0):
            #     pass

            with a.channel_synchronizer():
                qubit_stimulus_io.dwell(counter_1)

            if self.do_echo: 
                with a.channel_synchronizer(block=True):
                    bs_stimulus_io.schedule_pulse(self.bs_pulse_name)
                    a.barrier()
                    qubit.schedule_pulse(self.qubit_pi_pulse_name)
                    a.barrier()
                    bs_stimulus_io.schedule_pulse(self.bs_pulse_name)

            with a.channel_synchronizer():
                qubit_stimulus_io.dwell(counter_2)

            with a.channel_synchronizer():
                bs_stimulus_io.schedule_pulse(self.bs_pulse_name)
                a.barrier()
                qubit.schedule_pulse(qubit_pi_over_2_final)
                
            with a.channel_synchronizer():
                readout_resonator.measure(self.readout_pulse_name, self.capture_memory_name, self.capture_window_name)

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_pulse(self.readout_pulse_name)
        
        qubit_stimulus_io.load_pulse(qubit_pi_over_2_initial)
        bs_stimulus_io.load_pulse(self.bs_pulse_name)

        if self.do_echo:
            qubit_stimulus_io.load_pulse(self.qubit_pi_pulse_name)


        # Determine how many cycles each delay interval should be 
        dsp_count_values = self.acadia.seconds_to_cycles(self.delay_times/2)

        qubit_pi_over_2_scale = qubit_stimulus_io.get_config("pulses", qubit_pi_over_2_final,"scale")
        # qubit_stimulus_io.load_pulse(qubit_pi_over_2_final)

        configure_streams = False
        for i in range(self.iterations):
            for j, delay in enumerate(dsp_count_values):
                cache[0] = delay # sweep delay
                final_pulse_scale = qubit_pi_over_2_scale * np.exp(1j * self.delay_times[j] * self.soft_detune * 2 * np.pi)
                qubit_stimulus_io.load_pulse(qubit_pi_over_2_final, scale=final_pulse_scale)
                # capture data and put in the corresponding group
                self.acadia.run(minimum_delay=self.run_delay, configure_streams=configure_streams)
                wf = readout_capture_io.get_waveform_memory(self.capture_memory_name)
                self.data[f"points"].write(wf.array)
                # configure_streams = False

                if self.cool_qm_rounds>0:
                    wf = readout_capture_io.get_waveform_memory(prep_capture_mem)
                    self.data[f"prep"].write(wf.array)

            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return

        self.final_serve()


    def initialize(self):
        pass

    def update(self):
        # get current completed data
        # self.process_current_data()

        # save current data
        self.data.save(self.local_directory)

    def finalize(self):
        super().finalize()
        from acadia_qmsmt.plotting import save_registered_plots
        if self.plot:
            save_registered_plots(self)


    @annotate_method(is_data_processor=True)
    def process_current_data(self, readout_classifier: Annotated[str, "IOConfig", "readout_capture.classifiers"]=None):
        # First make sure that we actually have new data to process
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes
        from acadia_qmsmt.utils.fourier_transform import fft_mag
        
        data = reshape_iq_data_by_axes(self.data["points"].records(), self.delay_times, to_complex=True)
        if data is None:
            return

        completed_iterations = len(data)
        readout_resonator = MeasurableResonator(self.io("readout_stimulus"), self.io("readout_capture"))

        self.data_complex = data
        self.shots = readout_resonator.classify_measurement(self.data_complex, readout_classifier)
        self.avg_shots = np.mean(self.shots, axis=0)
        self.sigma_shots = np.std(self.shots, axis=0) / np.sqrt(completed_iterations)


        if self.cool_qm_rounds > 0:
            # merge all sweep axes
            self.prep_data = reshape_iq_data_by_axes(self.data["prep"].records())

        from acadia_qmsmt.analysis.fitting import ExpCosine
        self.delay_times_us = self.delay_times * 1e6
        self.fit = ExpCosine(self.delay_times_us, self.avg_shots, self.sigma_shots)
        self.fitted_t2_us = self.fit.ufloat_results["tau"]
        self.fitted_detune_MHz = self.fit.ufloat_results["f"]

        self.fft_freq, self.fft_data = fft_mag(self.delay_times*1e6, self.avg_shots)
        return completed_iterations



    @annotate_method(plot_name="1: T2 thresholded", axs_shape=(1,1))
    def plot1_T2(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        self.fit.plot(axs, oversample=5, 
                            result_kwargs={"label":f"T2 (us): {self.fitted_t2_us:.4g}\nDetune(MHz): {self.fitted_detune_MHz:.4g}"})

        axs.set_xlabel("Time [us]")
        axs.set_ylabel("e pop")
        axs.set_ylim(-0.02, 1.02)
        axs.set_title(f"T2{'E' if self.do_echo else 'R'}: {self.fit.ufloat_results['tau']:.5g}")

        axs.legend()
        return fig, axs

    @annotate_method(plot_name="2: T2 fft", axs_shape=(1,1))
    def plot1_T2_fft(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)
        axs.plot(self.fft_freq, self.fft_data)
        axs.set_yscale("log")
        axs.grid(True)
        axs.set_xlabel("Frequency (MHz)")
        return fig, axs

    @annotate_method(plot_name="3: T2_bin_averaged", axs_shape=(1,1))
    def plot2_traces_raw(self, axs=None, n_avg:int=1):
        from acadia_qmsmt.plotting import plot_binaveraged
        fig, ax = plot_binaveraged(self.delay_times_us, self.shots, axs, n_avg, self.figsize)
        ax.set_ylabel("delay times [us]")
        fig.tight_layout()
        return fig, axs
