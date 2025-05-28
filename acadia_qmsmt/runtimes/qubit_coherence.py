from typing import Union
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
    A :class:`Runtime` subclass for measuring the relaxation time (T1) of a qubit.
    """
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    delay_times: Union[list, np.ndarray]

    iterations: int
    run_delay: int

    qubit_first_pulse_name: str = None
    qubit_first_pulse_waveform_name: str = None
    qubit_second_pulse_name: str = None
    qubit_second_pulse_waveform_name: str = None
    virtual_detuning: float = 0
    readout_window_name: str = None
    readout_stimulus_waveform_name: str = None
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
        qubit = Qubit(qubit_stimulus_io)

        self.data.add_group(f"points", uniform=True)

        # Create an array in the cache that we can use to pass the 
        # delay value to the sequencer so that we don't have to reassemble every time
        cache = self.acadia.CacheArray(shape=(1,), dtype=np.dtype("<i4"))

        def sequence(a: Acadia):
            # Initialize a DSP to act as a counter
            counter = a.sequencer().DSP()

            # Load the counter with the value we put into the cache
            counter.load(cache[0])

            readout_resonator.prepare_cmacc(self.readout_window_name)

            with a.channel_synchronizer(block=False):
                qubit.pulse(self.qubit_first_pulse_name)
                
            # Start the counter and wait until it reaches zero
            counter.start_count(inc=int(np.int32(-1).astype(np.uint32)))
            with a.sequencer().repeat_until(counter == 0):
                pass

            with a.channel_synchronizer():
                qubit.pulse(self.qubit_second_pulse_name)
                a.barrier()
                readout_resonator.measure("readout", "readout_accumulated")

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_waveform("readout", self.readout_stimulus_waveform_name)
        qubit_stimulus_io.load_waveform(self.qubit_first_pulse_name, self.qubit_first_pulse_waveform_name)
        second_pulse_scale = qubit_stimulus_io.get_config("waveforms", self.qubit_second_pulse_waveform_name, "scale")
        
        # Determine how many cycles each delay interval should be
        first_pulse_memory = qubit_stimulus_io.get_waveform_memory(self.qubit_first_pulse_name)
        dsp_count_values = self.acadia.delay_times_to_counter_values(self.delay_times, first_pulse_memory)

        for i in range(self.iterations):
            for idx_delay,delay in enumerate(dsp_count_values):
                # Update the delay amount in the cache
                cache[0] = delay

                # Shift the phase of the second pulse according to the virtual detuning
                scale = second_pulse_scale * np.exp(2 * np.pi * 1j * self.virtual_detuning * self.delay_times[idx_delay])
                qubit_stimulus_io.load_waveform(self.qubit_second_pulse_name, self.qubit_second_pulse_waveform_name, scale=scale)

                # Capture data and put in the corresponding group
                self.acadia.run(minimum_delay=self.run_delay)
                wf = readout_capture_io.get_waveform_memory("readout_accumulated")
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

        from acadia_qmsmt.analysis.fitting import ExpCosine
        self.delay_times_us = self.delay_times * 1e6
        self.fit = ExpCosine(self.delay_times_us, self.data_to_fit, sigma=self.data_sigma)
        self.fitted_t2_us = self.fit.ufloat_results["tau"]
        self.fitted_detune_MHz = self.fit.ufloat_results["f"]
        return completed_iterations
    

    @annotate_method(plot_name="1 qubit T2", axs_shape=(1,1))
    def plot_data(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        self.fit.plot(axs, oversample=5, 
                            result_kwargs={"label":f"T2 (us): {self.fitted_t2_us:.4g}\nDetune(MHz): {self.fitted_detune_MHz:.4g}"})
        # self.fit.plot_fitted(axs, oversample=5, label=f"T2 (us): {self.fitted_t2_us:.4g}\nDetune(MHz): {self.fitted_detune_MHz:.4g}")

        axs.set_xlabel("Time [us]")
        if self.thresholded:
            axs.set_ylabel("e pop")
            axs.set_ylim(-0.02, 1.02)
        else:
            axs.set_ylabel("re(data) after rotation")

        axs.set_title(f"T2: {self.fit.ufloat_results['tau']:.5g}")

        axs.legend()
        return fig, axs
    
    @annotate_method(plot_name="2 bin averaged T2", axs_shape=(1,1))
    def plot_bin_avg(self, axs=None, n_avg=1):
        from acadia_qmsmt.plotting import prepare_plot_axes, plot_binaveraged
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        fig, axs = plot_binaveraged(self.delay_times_us, self.shots, axs, n_avg=n_avg, vmin=0, v_max=1)
        axs.set_ylabel("Time [us]")
        return fig, axs



# # --------------------------------------- in jpynb plotting ------------------------
#     def initialize(self):
        
#         if self.plot:
#             from acadia.processing import DynamicLine
#             import matplotlib.pyplot as plt
#             from IPython.display import display
#             from ipywidgets import Label, Layout, Box, Checkbox

#             self.figsize = (4, 3) if self.figsize is None else self.figsize
#             self.fig, ax = plt.subplots(1, 1, figsize=self.figsize)
#             self.fig.tight_layout()
#             self.fig.subplots_adjust(left=0.25, bottom=0.25)

#             self.line_pop = DynamicLine(ax, ".", color="red")
#             self.line_fit = DynamicLine(ax, "--", color="red")
#             ax.set_xlabel("Delay Time [s]")
#             ax.set_ylabel("Measurement Polarization")
#             ax.set_xlim(self.delay_times[0], self.delay_times[-1])
#             ax.set_ylim(-1.1, 1.1)
#             ax.grid()

#             self.dual_frequency_checkbox = Checkbox(value=False, description="Dual-frequency fit", style={"description_width": "initial"})
#             def update_dual_frequency_fit(x):
#                 self.dual_frequency_fit = self.dual_frequency_checkbox.value
#             self.dual_frequency_checkbox.observe(update_dual_frequency_fit)

#             self.decay_label = Label(style={"description_width": "initial"})
#             self.freq_label = Label(style={"description_width": "initial"})

#             label_layout = Layout(display="flex", flex_flow="column", align_items="stretch")
#             label_box = Box(children=[self.dual_frequency_checkbox, self.decay_label, self.freq_label], layout=label_layout)
#             display(label_box)

#             from tqdm.notebook import tqdm

#         else:
#             from tqdm import tqdm

#         self.fit = None
#         self.dual_frequency_fit = False
#         self.iterations_progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
#         self.iterations_previous = 0

#     def update(self):
#         # First make sure that we actually have new data to process
#         if "points" not in self.data or len(self.data["points"]) < len(self.delay_times):
#             return

#         # Update the progress bar based on the number of iterations
#         completed_iterations = len(self.data["points"]) // len(self.delay_times)
#         if completed_iterations == 0:
#             return

#         self.iterations_progress_bar.update(completed_iterations - self.iterations_previous)

#         valid_points = completed_iterations*len(self.delay_times)
#         data = self.data["points"].records()[:valid_points, ...]
#         data = data.reshape(completed_iterations, len(self.delay_times), 2)

#         # Threshold the data according to the I quadrature
#         shots = np.sign(data[:,:,0], dtype=np.int32)
#         self.avg = np.mean(shots, axis=0)

#         try:
#             amax = np.argmax(self.avg)
#             amin = np.argmin(self.avg)
#             p0 = (-(abs(np.min(self.avg)) - 0.5), 
#                     self.delay_times[len(self.delay_times) // 3], 
#                     0, 
#                     0.5/(self.delay_times[amax] - self.delay_times[amin]))

#             if self.dual_frequency_fit:
#                 fit_func = dual_sine_decay
#                 p0 = (*p0, 0)
#             else:
#                 fit_func = partial(dual_sine_decay, f2=0)

#             self.fit, pcov = curve_fit(fit_func, self.delay_times, self.avg, p0=p0)
#         except:
#             pass

#         if self.plot:
#             if self.fit is not None:
#                 self.decay_label.value = f"Decay time: {round(self.fit[1]*1e6, 3)} us"
#                 if self.dual_frequency_fit:
#                     self.freq_label.value = f"Oscillation frequencies: {round(self.fit[3]*1e-3, 4)} and {round(self.fit[4]*1e-3, 4)} kHz"
#                     self.line_fit.update(self.delay_times, dual_sine_decay(self.delay_times, *self.fit), rescale_axis=False)
#                 else:
#                     self.freq_label.value = f"Oscillation frequency: {round(self.fit[3]*1e-3, 4)} kHz"
#                     self.line_fit.update(self.delay_times, dual_sine_decay(self.delay_times, *self.fit, f2=0), rescale_axis=False)
            
#             self.line_pop.update(self.delay_times, self.avg, rescale_axis=False)
#             self.fig.canvas.draw_idle() 
        
#         self.data.save(self.local_directory)
#         self.iterations_previous = completed_iterations

#     def finalize(self):
#         super().finalize()
#         self.iterations_progress_bar.close()
#         if self.plot:
#             self.savefig(self.fig)

