from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig

def decay(t, A, tau, B):
    return A*np.exp(-t/tau) + B

class CavityRelaxationRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` subclass for measuring the relaxation time (T1) of a 
    cavity dispersively coupled to a qubit.
    """
    cavity_stimulus: IOConfig
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    delay_times: Union[list, np.ndarray]

    iterations: int
    run_delay: int

    cavity_pulse: dict[str,float] = {"length": 100e-9}
    qubit_pulse_name: str = None
    qubit_pulse_waveform_name: str = None
    readout_window_name: str = None
    readout_stimulus_waveform_name: str = None
    plot: bool = True
    figsize: tuple[int] = None
    yaml_path: str = None

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

        cavity_stimulus_io = self.io("cavity_stimulus")
        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")

        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)
        qubit = Qubit(qubit_stimulus_io)

        cavity_pulse = self.acadia.create_waveform_memory(
            cavity_stimulus_io.channel, 
            length=self.cavity_pulse.get("length", 0.0)
        )

        self.data.add_group(f"points", uniform=True)

        # Create an array in the cache that we can use to pass the 
        # delay value to the sequencer so that we don't have to reassemble every time
        cache = self.acadia.CacheArray(shape=(1,), dtype=np.dtype("<i4"))

        def sequence(a: Acadia):
            # Initialize a DSP to act as a counter
            counter = a.sequencer().DSP()

            # Load the counter with the value we put into the cache
            counter.load(cache[0])

            with a.channel_synchronizer(block=False):
                a.schedule_waveform(cavity_pulse, stretch_length=self.cavity_pulse.get("stretch_length", 0.0))
                
            # Start the counter and wait until it reaches zero
            counter.start_count(inc=int(np.int32(-1).astype(np.uint32)))
            with a.sequencer().repeat_until(counter == 0):
                pass

            with a.channel_synchronizer():
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
        cavity_stimulus_io.load_waveform(cavity_pulse, {"data": "hann"}, scale=self.cavity_pulse.get("scale", 0.9999))

        # Determine how many cycles each delay interval should be
        dsp_count_values = self.acadia.delay_times_to_counter_values(self.delay_times, cavity_pulse)

        for i in range(self.iterations):
            for delay in dsp_count_values:
                cache[0] = delay

                # capture data and put in the corresponding group
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
            ax.set_xlabel("Delay Time [s]")
            ax.set_ylabel("Measurement Polarization")
            ax.set_xlim(self.delay_times[0], self.delay_times[-1])
            ax.set_ylim(5e-3, 1)
            ax.set_yscale("log")
            ax.grid()

            self.decay_label = Label(style={"description_width": "initial"})
            display(self.decay_label)

            from tqdm.notebook import tqdm

        else:
            from tqdm import tqdm

        self.fit = None
        self.iterations_progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.iterations_previous = 0

    def update(self):
        # First make sure that we actually have new data to process
        if "points" not in self.data or len(self.data["points"]) < len(self.delay_times):
            return

        # Update the progress bar based on the number of iterations
        completed_iterations = len(self.data["points"]) // len(self.delay_times)
        if completed_iterations == 0:
            return

        self.iterations_progress_bar.update(completed_iterations - self.iterations_previous)

        valid_points = completed_iterations*len(self.delay_times)
        data = self.data["points"].records()[:valid_points, ...]
        data = data.reshape(completed_iterations, len(self.delay_times), 2)

        # Threshold the data according to the I quadrature
        shots = np.sign(data[:,:,0], dtype=np.int32)
        self.avg = np.mean(shots, axis=0)

        # Convert the average into a population
        self.pop = (1-self.avg)/2
        try:
            log_pop = np.log(np.clip(self.pop, 1e-12, 1))

            # Because the probability of 0 photons will increase over time, this is actually 1 - exp decay
            p0 = (abs(np.min(log_pop)) - abs(np.max(log_pop)), self.delay_times[len(self.delay_times) // 2], log_pop[-1])
            self.fit, pcov = curve_fit(decay, self.delay_times, log_pop, p0=p0)
        except:
            pass
        
        if self.plot:
            self.line_pop.update(self.delay_times, self.pop, rescale_axis=False)
            if self.fit is not None:
                self.line_fit.update(self.delay_times, np.exp(decay(self.delay_times, *self.fit)), rescale_axis=False)
                self.decay_label.value = f"Decay time: {round(self.fit[1]*1e6, 3)} us"
            self.fig.canvas.draw_idle() 

        self.data.save(self.local_directory)
        self.iterations_previous = completed_iterations

    def finalize(self):
        super().finalize()
        self.iterations_progress_bar.close()
        if self.plot:
            self.savefig(self.fig)

