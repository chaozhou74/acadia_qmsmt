from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime, WaveformMemory
from acadia.utils import sys_nanosleep
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig

def sinc(f, amp, f0, width, offset):
    return amp*np.sinc((f - f0)/width) + offset

class CavitySpectroscopyRuntime(QMsmtRuntime):

    cavity_stimulus: IOConfig
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    cavity_frequencies: Union[list, np.ndarray]

    iterations: int
    cavity_pulse_fixed_length: float = 1e-6
    cavity_pulse_amplitude: float = 0.1
    qubit_pulse_name: str = None
    qubit_pulse_waveform_name: str = None
    qubit_saturation_pulse: dict[str,float] = None
    readout_window_name: str = None
    readout_stimulus_waveform_name: str = None
    plot: bool = True
    figsize: tuple[int] = None

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

        cavity_stimulus_io = self.io("cavity_stimulus")
        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")

        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)
        qubit = Qubit(qubit_stimulus_io)

        cavity_waveform = self.acadia.create_waveform_memory(
            cavity_stimulus_io.channel, 
            length=100e-9,
            fixed_length=self.cavity_pulse_fixed_length)

        if self.qubit_saturation_pulse is not None:
            qubit_pulse = self.acadia.create_waveform_memory(
                qubit_stimulus_io.channel,
                length=self.qubit_saturation_pulse.get("length", 0.0),
                fixed_length=self.qubit_saturation_pulse.get("fixed_length", 0.0)
            )
        else:
            qubit_pulse = self.qubit_pulse_name

        self.data.add_group(f"points", uniform=True)

        def sequence(a: Acadia):
            readout_resonator.prepare_cmacc(self.readout_window_name)

            with a.channel_synchronizer():
                a.schedule_waveform(cavity_waveform)
                a.barrier()
                qubit.pulse(qubit_pulse)
                a.barrier()
                readout_resonator.measure("readout", "readout_accumulated")

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_waveform("readout", self.readout_stimulus_waveform_name)
        if self.qubit_saturation_pulse is None:
            qubit_stimulus_io.load_waveform(self.qubit_pulse_name, self.qubit_pulse_waveform_name)
        else:
            qubit_pulse.set("hann", scale=self.qubit_saturation_pulse.get("scale", 1.0))
        cavity_waveform.set("hann", scale=self.cavity_pulse_amplitude)

        for i in range(self.iterations):
            for frequency in self.cavity_frequencies:
                cavity_stimulus_io.set_nco_frequency(frequency)
                self.acadia.update_ncos_synchronized()

                self.acadia.run()
                wf = readout_capture_io.get_waveform_memory("readout_accumulated")
                self.data[f"points"].write(wf.array)
                sys_nanosleep(5000000)

            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return

        self.final_serve()

    def initialize(self):
        
        if self.plot:
            from acadia.processing import DynamicLine
            import matplotlib.pyplot as plt
            from IPython.display import display
            from ipywidgets import Label, Layout, Box

            self.figsize = (4, 3) if self.figsize is None else self.figsize
            self.fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            self.fig.tight_layout()
            self.fig.subplots_adjust(left=0.25, bottom=0.25)

            self.line_pop = DynamicLine(ax, ".", label="Mag", color="red")
            self.line_fit = DynamicLine(ax, "--", label="Mag", color="red")
            ax.set_xlabel("Cavity Frequency [Hz]")
            ax.set_ylabel("Qubit Population [FS]", color="red")
            ax.tick_params(axis='y', labelcolor="blue")
            ax.set_xlim(self.cavity_frequencies[0], self.cavity_frequencies[-1])
            ax.set_ylim(-1.1, 1.1)
            ax.grid()

            self.frequency_label = Label(style={"description_width": "initial"})
            label_layout = Layout(display="flex", flex_flow="column", align_items="stretch")
            label_box = Box(children=[self.frequency_label], layout=label_layout)
            display(label_box)

            from tqdm.notebook import tqdm

        else:
            from tqdm import tqdm

        self.iterations_progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.iterations_previous = 0

    def update(self):
        # First make sure that we actually have new data to process
        if "points" not in self.data or len(self.data["points"]) < len(self.cavity_frequencies):
            return

        # Update the progress bar based on the number of iterations
        completed_iterations = len(self.data["points"]) // len(self.cavity_frequencies)
        self.iterations_progress_bar.update(completed_iterations - self.iterations_previous)

        if completed_iterations == 0:
            return

        valid_points = completed_iterations*len(self.cavity_frequencies)
        data = self.data["points"].records()[:valid_points, ...]
        data = data.reshape(completed_iterations, len(self.cavity_frequencies), 2)

        # Threshold the data according to the I quadrature
        shots = np.sign(data[:,:,0], dtype=np.int32)
        avg = np.mean(shots, axis=0)
        self.line_pop.update(self.cavity_frequencies, avg, rescale_axis=False)

        try:
            amax = np.argmax(avg)
            amin = np.argmax(avg)

            # Do the fitting in units of GHz to improve numerical stability
            p0 = (abs(avg[amax] - avg[amin]),  # amp
                1e-9*self.cavity_frequencies[amax], # f0
                1e-9/self.cavity_pulse_fixed_length, # width
                (avg[0] + avg[-1])/2) # offset
            params, pcov, info, mesg, ier = curve_fit(sinc, self.cavity_frequencies*1e-9, avg, p0=p0, full_output=True)
            self.fit = {"params": params, "pcov": pcov, "info": info, "mesg": mesg, "ier": ier}
            self.line_fit.update(self.cavity_frequencies, sinc(self.cavity_frequencies*1e-9, *params), rescale_axis=False)
            self.frequency_label.value = f"Cavity frequency: {round(params[1], 8):2.8f} GHz"
        except:
            pass

        self.fig.canvas.draw_idle() 

        self.data.save(self.local_directory)

        self.iterations_previous = completed_iterations

    def finalize(self):
        super().finalize()
        self.iterations_progress_bar.close()
        if self.plot:
            self.savefig(self.fig)
