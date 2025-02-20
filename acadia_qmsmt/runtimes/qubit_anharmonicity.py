from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime, WaveformMemory
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig

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
            readout_resonator.prepare_cmacc(self.readout_window_name)

            with a.channel_synchronizer():
                qubit.pulse(self.qubit_pulse_name)
                qubit.pulse(ef_pulse)
                qubit.pulse(self.qubit_pulse_name)
                a.barrier()
                readout_resonator.measure("readout", "readout_accumulated")

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
        
        if self.plot:
            from acadia.processing import DynamicLine
            import matplotlib.pyplot as plt
            from IPython.display import display
            from ipywidgets import Label

            self.figsize = (4, 3) if self.figsize is None else self.figsize
            self.fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            self.fig.tight_layout()
            self.fig.subplots_adjust(left=0.25, bottom=0.25)

            self.line_pop = DynamicLine(ax, ".", color="red")
            self.line_fit = DynamicLine(ax, "--", color="red")
            ax.set_xlabel("Pulse Amplitude [arb.]")
            ax.set_ylabel("Population [FS]")
            ax.set_xlim(self.frequencies[0], self.frequencies[-1])
            ax.set_ylim(-1.1, 1.1)
            ax.grid()

            self.amplitude_label = Label(style={"description_width": "initial"})
            display(self.amplitude_label)

            from tqdm.notebook import tqdm

        else:
            from tqdm import tqdm

        self.fit = None
        self.iterations_progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.iterations_previous = 0

    def update(self):
        # First make sure that we actually have new data to process
        if "points" not in self.data or len(self.data["points"]) < len(self.frequencies):
            return

        # Update the progress bar based on the number of iterations
        completed_iterations = len(self.data["points"]) // len(self.frequencies)
        if completed_iterations == 0:
            return

        self.iterations_progress_bar.update(completed_iterations - self.iterations_previous)

        valid_points = completed_iterations*len(self.frequencies)
        data = self.data["points"].records()[:valid_points, ...]
        data = data.reshape(completed_iterations, len(self.frequencies), 2)

        # Threshold the data according to the I quadrature
        shots = np.sign(data[:,:,0], dtype=np.int32)
        self.avg = np.mean(shots, axis=0)
        

        # Fit the data to a sine
        # try:
        #     amin = np.argmin(self.avg)
        #     amax = np.argmax(self.avg)
        #     osc_period = 2*abs(self.qubit_amplitudes[amin]-self.qubit_amplitudes[amax])
        #     p0 = (abs(amin-amax)/2, 1/osc_period)
        #     self.fit, pcov = curve_fit(flopping, self.qubit_amplitudes, self.avg, p0=p0)
            
        # except:
        #     pass
        
        if self.plot:
            self.line_pop.update(self.frequencies, self.avg, rescale_axis=False)
            # if self.fit is not None:
            #     self.amplitude_label.value = f"Pi pulse amplitude: {round(0.5/self.fit[1], 6)}"
            #     self.line_fit.update(self.qubit_amplitudes, flopping(self.qubit_amplitudes, *self.fit), rescale_axis=False)
            self.fig.canvas.draw_idle() 

        self.data.save(self.local_directory)
        self.iterations_previous = completed_iterations

    def finalize(self):
        super().finalize()
        self.iterations_progress_bar.close()
        if self.plot:
            self.savefig(self.fig)

