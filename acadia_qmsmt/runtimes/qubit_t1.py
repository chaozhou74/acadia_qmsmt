from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime, WaveformMemory
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig
from acadia.runtime import annotate_method

def decay(t, A, tau, B):
    return A*np.exp(-t/tau) + B

class QubitRelaxationRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` subclass for measuring the relaxation time (T1) of a qubit.
    """
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    delay_times: Union[list, np.ndarray]

    iterations: int
    run_delay: int

    qubit_pulse_name: str = "R_x_180"
    readout_pulse_name: str = "readout"
    capture_memory_name: str = "readout_accumulated"
    capture_window_name: str = "boxcar"


    plot: bool = True
    figsize: tuple[int] = None
    yaml_path: str = None

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")

        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)

        self.data.add_group(f"points", uniform=True)

        # Create an array in the cache that we can use to pass the 
        # delay value to the sequencer so that we don't have to reassemble every time
        cache = self.acadia.CacheArray(shape=(1,), dtype=np.dtype("<i4"))

        def sequence(a: Acadia):
            # Initialize a DSP to act as a counter
            counter = a.sequencer().Register()
            # Load the counter with the value we put into the cache
            counter.load(cache[0])

            with a.channel_synchronizer():
                qubit_stimulus_io.schedule_pulse(self.qubit_pulse_name)
                qubit_stimulus_io.dwell(counter)
                # a.barrier() # doesn't work as of June 26th, 2025 - JWOG
            with a.channel_synchronizer():
                readout_resonator.measure(self.readout_pulse_name, self.capture_memory_name, self.capture_window_name)


        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_pulse(self.readout_pulse_name)
        qubit_stimulus_io.load_pulse(self.qubit_pulse_name)

        # Determine how many cycles each delay interval should be
        dsp_count_values = self.acadia.delay_times_to_counter_values(self.delay_times)

        for i in range(self.iterations):
            for delay in dsp_count_values:
                cache[0] = delay

                # capture data and put in the corresponding group
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
        if self.plot:
            from acadia_qmsmt.plotting import save_registered_plots
            save_registered_plots(self)



    @annotate_method(is_data_processor=True)
    def process_current_data(self, thresholded:bool=True):
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes
        data = reshape_iq_data_by_axes(self.data["points"].records(), self.delay_times)
        if data is None:
            return
        else:
            completed_iterations = len(data)
        self.data_iq = data.astype(float).view(complex).squeeze()
        self.avg_iq = np.mean(self.data_iq, axis=0)
        self.shots = (1-np.sign(self.data_iq.real))/2
        

        if thresholded:
            self.data_to_fit = np.mean(self.shots, axis=0)
            self.data_sigma = np.std(self.shots, axis=0)/np.sqrt(completed_iterations)
        else:
            from acadia_qmsmt.analysis import rotate_iq, find_iq_rotation
            rot_angle = find_iq_rotation(self.avg_iq)
            self.data_to_fit = rotate_iq(self.avg_iq, rot_angle).real
            self.data_sigma = np.std(rotate_iq(self.data_iq, rot_angle).real, axis=0)/np.sqrt(completed_iterations)
        self.thresholded = thresholded

        from acadia_qmsmt.analysis.fitting import Exponential
        self.delay_times_us = self.delay_times * 1e6
        self.fit = Exponential(self.delay_times_us, self.data_to_fit, sigma=self.data_sigma)
        self.fitted_t1_us = self.fit.ufloat_results["tau"]
        return completed_iterations
    

    @annotate_method(plot_name="1 qubit T1", axs_shape=(1,1))
    def plot_data(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        self.fit.plot(axs, oversample=5,
                            result_kwargs={"label": f"T1 (us): {self.fitted_t1_us:.4g}"})
        axs.set_title(f"T1: {self.fit.ufloat_results['tau']:.5g}")

        axs.set_xlabel("Time [us]")
        if self.thresholded:
            axs.set_ylabel("e pop")
            axs.set_ylim(-0.02, 1.02)
        else:
            axs.set_ylabel("re(data) after rotation")

        axs.legend()
        return fig, axs
    
    @annotate_method(plot_name="2 bin averaged T1", axs_shape=(1,1))
    def plot_bin_avg(self, axs=None, n_avg=1):
        from acadia_qmsmt.plotting import prepare_plot_axes, plot_binaveraged
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        fig, axs = plot_binaveraged(self.delay_times_us, self.shots, axs, n_avg=n_avg, vmin=0, v_max=1)
        axs.set_ylabel("Time [us]")
        return fig, axs

