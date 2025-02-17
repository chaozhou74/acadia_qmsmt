"""
We generally don't use a length Rabi experiment to tune up a π-pulse; this is just to show 
how a pulse length sweep could be done with Acadia. 

In the current implementation, no other pulses can follow the pulse whose length we are 
sweeping on the qubit channel. This is because the qubit pulse memory has a fixed length-
we only update the data in that memory while adjusting the waiting time between the qubit 
pulse and the measurement.

This method lets us sweep the pulse length to arbitrary values, since the pulse shape is 
first defined as a continuous function and then sampled at discrete time steps. However, 
if the pulse length is not an integer multiple of the clock cycle, there may be an extra 
delay of up to a few nanoseconds between the qubit pulse and the measurement.

"""

from typing import Union

import numpy as np

from acadia import Acadia, DataManager
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig



class QubitLengthRabiRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` subclass for sweeping the rabi time
    """
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    flat_length_list: Union[list, np.ndarray]

    iterations: int
    run_delay: int

    qubit_pulse_name: str = None
    qubit_pulse_waveform_name: str = None
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

        self.data.add_group(f"points", uniform=True)

        # Create an array in the cache that we can use to pass the
        # delay value to the sequencer so that we don't have to reassemble every time
        cache = self.acadia.CacheArray(shape=(1,), dtype=np.dtype("<i4"))


        # generate memory and waveform data for qubit pulse with each specific flat length
        from acadia.pulse_shaping import prepare_flattop_length_sweep

        qubit_mem_cgf = qubit_stimulus_io.get_config("memories", self.qubit_pulse_name)
        ramp_time = qubit_mem_cgf["length"]
        qubit_wf_cfg = qubit_stimulus_io.get_config("waveforms", self.qubit_pulse_waveform_name)
        ramp_func = qubit_wf_cfg["data"]
        qubit_long_waveform_mem, wf_datas = prepare_flattop_length_sweep(self.acadia, qubit_stimulus_io.channel,
                                                                         self.flat_length_list, ramp_func, **qubit_mem_cgf)


        def sequence(a: Acadia):
            # Initialize a DSP to act as a counter
            counter = a.sequencer().DSP()

            # Load the counter with the value we put into the cache
            counter.load(cache[0])

            readout_resonator.prepare_cmacc(self.readout_window_name)

            with a.channel_synchronizer(block=False):
                a.schedule_waveform(qubit_long_waveform_mem)

            # Start the counter and wait until it reaches zero
            counter.start_count(inc=int(np.int32(-1).astype(np.uint32)))
            with a.sequencer().repeat_until(counter == 0):
                pass

            with a.channel_synchronizer():
                readout_resonator.measure("readout", "readout_accumulated")

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_waveform("readout", self.readout_stimulus_waveform_name)


        # Determine how many cycles each delay interval should be
        seq_clock_freq = self.acadia.sequencer_clock_frequency()
        rounded_delay_times = np.ceil((self.flat_length_list + ramp_time) * seq_clock_freq) / seq_clock_freq
        extra_delays = rounded_delay_times - (self.flat_length_list + ramp_time)
        if any(rounded_delay_times - (self.flat_length_list + ramp_time)) != 0:
            logger.warning(f"The delay time between the qubit pulse and measurement has been rounded, "
                           f"resulting in max additional delay of {max(extra_delays)}.")
        dsp_count_values = self.acadia.delay_times_to_counter_values(rounded_delay_times, qubit_long_waveform_mem)

        for i in range(self.iterations):
            for wf_idx, delay in enumerate(dsp_count_values):
                cache[0] = delay
                qubit_long_waveform_mem.set(wf_datas[wf_idx], scale=qubit_wf_cfg["scale"])
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

            self.line_pop = DynamicLine(ax, ".", color="C0")
            self.line_fit = DynamicLine(ax, "--", color="C1")
            ax.set_xlabel("Pulse length [s.]")
            ax.set_ylabel("Polarization [FS]")
            ax.set_xlim(self.flat_length_list[0], self.flat_length_list[-1])
            ax.set_ylim(-1.1, 1.1)
            ax.grid()

            self.length_label = Label(style={"description_width": "initial"})
            display(self.length_label)

            from tqdm.notebook import tqdm

        else:
            from tqdm import tqdm

        self.fit = None
        self.iterations_progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.iterations_previous = 0

    def update(self):
        # First make sure that we actually have new data to process
        if "points" not in self.data or len(self.data["points"]) < len(self.flat_length_list):
            return

        # Update the progress bar based on the number of iterations
        completed_iterations = len(self.data["points"]) // len(self.flat_length_list)
        if completed_iterations == 0:
            return

        self.iterations_progress_bar.update(completed_iterations - self.iterations_previous)

        valid_points = completed_iterations * len(self.flat_length_list)
        data = self.data["points"].records()[:valid_points, ...]
        data = data.reshape(completed_iterations, len(self.flat_length_list), 2)

        # Threshold the data according to the I quadrature
        shots = np.sign(data[:, :, 0], dtype=np.int32)
        self.avg = np.mean(shots, axis=0)

        # Fit the data to a sine
        try:
            from acadia_qmsmt.analysis.fitting.exp_cosine import ExpCosine
            self.fit = ExpCosine(self.flat_length_list, self.avg)

        except:
            pass

        if self.plot:
            self.line_pop.update(self.flat_length_list, self.avg, rescale_axis=False)
            if self.fit is not None:
                t_pi = (np.pi - self.fit.ufloat_results['phi']) / (self.fit.ufloat_results['f'] * 2 * np.pi)
                t_pi = t_pi % (1 / self.fit.ufloat_results['f'])
                self.length_label.value = (f"Pi pulse flat-part length: {t_pi*1e9} ns, "
                                           f"decay constant: {self.fit.ufloat_results['tau']*1e6} us")
                self.line_fit.update(self.flat_length_list, self.fit.result.eval(coordinates=self.flat_length_list),
                                     rescale_axis=False)
            self.fig.canvas.draw_idle()

        self.data.save(self.local_directory)
        self.iterations_previous = completed_iterations

    def finalize(self):
        super().finalize()
        self.iterations_progress_bar.close()
        # if self.plot:
        #     self.savefig(self.fig)
