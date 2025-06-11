from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime, WaveformMemory
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig
from acadia.runtime import annotate_method

def flopping(pulse_amp, oscillation_amp, oscillation_freq):
    return oscillation_amp * np.cos(2 * np.pi * pulse_amp * oscillation_freq)

class QubitAnharmonicityRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` for measuring the anharmonicity of a qubit.
    """
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    frequencies: Union[list, np.ndarray]

    iterations: int
    run_delay: int

    qubit_pulse_name: str = None
    qubit_pulse_waveform_name: str = None
    ef_pulse_length: float = 2e-6
    ef_pulse_scale: complex = 0.1
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

        ef_pulse = self.acadia.create_waveform_memory(qubit_stimulus_io.channel, length=self.ef_pulse_length)

        self.data.add_group(f"points", uniform=True)

        def sequence(a: Acadia):

            with a.channel_synchronizer():
                qubit.pulse(self.qubit_pulse_name)
                qubit.pulse(ef_pulse)
                qubit.pulse(self.qubit_pulse_name)
                a.barrier()
                readout_resonator.measure("readout", "readout_accumulated", self.readout_window_name)

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_waveform("readout", self.readout_stimulus_waveform_name)
        qubit_stimulus_io.load_waveform(self.qubit_pulse_name, self.qubit_pulse_waveform_name)
        
        from scipy.signal.windows import hann
        envelope = qubit_stimulus_io.compute_waveform(ef_pulse, {"data": "hann"})
        sample_times = np.arange(ef_pulse.size, dtype=np.float64) / qubit_stimulus_io.interface_sample_frequency

        for i in range(self.iterations):
            for frequency in self.frequencies:
                # load the modulated pulse into the waveform
                modulated_pulse = envelope * np.exp(2 * np.pi * 1j * frequency * sample_times)
                ef_pulse.load(modulated_pulse, scale=self.ef_pulse_scale)
                
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
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes, find_iq_rotation, rotate_iq
        data_spec = reshape_iq_data_by_axes(self.data["points"].records(), self.frequencies)
        if data_spec is None:
            return
        else:
            completed_iterations = len(data_spec)
        self.data_iq = data_spec.astype(float).view(complex).squeeze()
        self.avg_iq = np.mean(self.data_iq, axis=0)
        self.shots = (1-np.sign(self.data_iq.real))/2
        self.avg_shots = np.mean(self.shots, axis=0)

        # for non thresholded data, find the best rotation angle that
        # puts all the information on the I quadrature 
        rot_angle = find_iq_rotation(self.avg_iq)
        self.data_iq_rot = rotate_iq(self.data_iq, rot_angle)
        self.avg_iq_rot = rotate_iq(self.avg_iq, rot_angle)

        if thresholded:
            self.data_to_fit = self.avg_shots
        else:
            from acadia_qmsmt.analysis import rotate_iq
            self.data_to_fit = self.avg_iq_rot.real
        self.thresholded = thresholded

        from acadia_qmsmt.analysis.fitting import Lorentzian
        self.fit = Lorentzian(self.frequencies, self.data_to_fit)
        self.fitted_f0_MHz = self.fit.ufloat_results["x0"]/1e6
        self.qubit_nco = self._ios["qubit_stimulus"].get_config("channel_config", "nco_frequency")
        return completed_iterations
    

    @annotate_method(plot_name="1 qubit spectrocopy", axs_shape=(1,1))
    def plot_data(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        self.fit.plot(axs, oversample=5, 
                      result_kwargs=dict(label=f"{self.fitted_f0_MHz:.5g} MHz"))

        axs.set_xlabel("Detuning [Hz]")
        if self.thresholded:
            axs.set_ylabel("e pop")
            axs.set_ylim(-0.02, 1.02)
        else:
            axs.set_ylabel("re(data) after rotation")

        axs.legend()
        axs.grid(True)
        axs.set_title(f"NCO: {self.qubit_nco/1e9} GHz, fitted_deune: {self.fitted_f0_MHz} MHz")
        return fig, axs

    @annotate_method(plot_name="2 bin averaged", axs_shape=(1,1))
    def plot_bin_avg(self, axs=None, n_avg=1):
        from acadia_qmsmt.plotting import prepare_plot_axes, plot_binaveraged
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)
        if self.thresholded:
            axs.set_ylabel("e pop")
            fig, axs = plot_binaveraged(self.frequencies, self.shots, axs, n_avg=n_avg, vmin=0, v_max=1)
        else:
            axs.set_ylabel("re(data) after rotation")
            fig, axs = plot_binaveraged(self.frequencies, self.data_iq_rot.real, axs, n_avg=n_avg)
        axs.set_ylabel("Probe freq [Hz]")
        return fig, axs




    @annotate_method(button_name="update ef frequency")
    def update_freq(self, target_waveform=["R_x_180_ef", "R_x_180_selective_ef"]):
        fitted_f0 = np.round(self.fitted_f0_MHz.n*1e6)
        for wf in target_waveform:
            self.update_io_yaml_field("qubit_stimulus", f"waveforms.{wf}.detune", fitted_f0)














    # def initialize(self):
        
    #     if self.plot:
    #         from acadia.processing import DynamicLine
    #         import matplotlib.pyplot as plt
    #         from IPython.display import display
    #         from ipywidgets import Label

    #         self.figsize = (4, 3) if self.figsize is None else self.figsize
    #         self.fig, ax = plt.subplots(1, 1, figsize=self.figsize)
    #         self.fig.tight_layout()
    #         self.fig.subplots_adjust(left=0.25, bottom=0.25)

    #         self.line_pop = DynamicLine(ax, ".", color="red")
    #         self.line_fit = DynamicLine(ax, "--", color="red")
    #         ax.set_xlabel("Pulse Detuning [Hz]")
    #         ax.set_ylabel("Population [FS]")
    #         ax.set_xlim(self.frequencies[0], self.frequencies[-1])
    #         ax.set_ylim(-1.1, 1.1)
    #         ax.grid()

    #         self.amplitude_label = Label(style={"description_width": "initial"})
    #         display(self.amplitude_label)

    #         from tqdm.notebook import tqdm

    #     else:
    #         from tqdm import tqdm

    #     self.fit = None
    #     self.iterations_progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
    #     self.iterations_previous = 0

    # def update(self):
    #     # First make sure that we actually have new data to process
    #     if "points" not in self.data or len(self.data["points"]) < len(self.frequencies):
    #         return

    #     # Update the progress bar based on the number of iterations
    #     completed_iterations = len(self.data["points"]) // len(self.frequencies)
    #     if completed_iterations == 0:
    #         return

    #     self.iterations_progress_bar.update(completed_iterations - self.iterations_previous)

    #     valid_points = completed_iterations*len(self.frequencies)
    #     data = self.data["points"].records()[:valid_points, ...]
    #     data = data.reshape(completed_iterations, len(self.frequencies), 2)

    #     # Threshold the data according to the I quadrature
    #     shots = np.sign(data[:,:,0], dtype=np.int32)
    #     self.avg = np.mean(shots, axis=0)
        

    #     # Fit the data to a sine
    #     # try:
    #     #     amin = np.argmin(self.avg)
    #     #     amax = np.argmax(self.avg)
    #     #     osc_period = 2*abs(self.qubit_amplitudes[amin]-self.qubit_amplitudes[amax])
    #     #     p0 = (abs(amin-amax)/2, 1/osc_period)
    #     #     self.fit, pcov = curve_fit(flopping, self.qubit_amplitudes, self.avg, p0=p0)
            
    #     # except:
    #     #     pass
        
    #     if self.plot:
    #         self.line_pop.update(self.frequencies, self.avg, rescale_axis=False)
    #         # if self.fit is not None:
    #         #     self.amplitude_label.value = f"Pi pulse amplitude: {round(0.5/self.fit[1], 6)}"
    #         #     self.line_fit.update(self.qubit_amplitudes, flopping(self.qubit_amplitudes, *self.fit), rescale_axis=False)
    #         self.fig.canvas.draw_idle() 

    #     self.data.save(self.local_directory)
    #     self.iterations_previous = completed_iterations

    # def finalize(self):
    #     super().finalize()
    #     self.iterations_progress_bar.close()
    #     if self.plot:
    #         self.savefig(self.fig)

