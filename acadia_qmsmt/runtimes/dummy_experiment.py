from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime, WaveformMemory
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig

# can define various little helper functions if you'd like
def sine(pulse_amp, oscillation_amp, oscillation_freq, offset):
    return oscillation_amp * np.cos(2 * np.pi * pulse_amp * oscillation_freq) + offset

def quadratic(pulse_amp, a, amp_0, offset):
    return a*(pulse_amp - amp_0)**2 + offset

class MyDummyExpRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` that isn't used for anything, but is *heavily* commented to explain the general structure. 

    This is meant to act as a bit of a rosetta stone between the fpga_lib/GUI way of doing things, and the Acadia way of doing things.
    If you see references to "the GUI", this refers to the fpga GUI made by Phil Reinhold. -- commenting by JWOG with guidance from Chao
    """
    # The first part of the class is creating various objects to be used in the experiment, analagous to the FloatParameter, BoolParameter, etc. of fpga_lib

    # NOTE: the syntax that we are using here is that of "type hints" in python, where you say thing1: thing_type, this is purely for documentation purposes
    # NOTE: all variables introduced in this manner that are not set to anything will be need to be passed when calling the class

    # Here we specify the inputs and outputs of the experiment. Think of the "waves" tab in the GUI. IOConfig is a special type that will come up later
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    
    qubit_amplitudes: Union[list, np.ndarray] # Note that these amplitudes override the ``scale`` parameter in the configuration

    iterations: int # "iterations" is the the number of shots to run, there's no more "blocks", so think of avgs_per_block = 1, and iterations = n_blocks
    run_delay: int # this is the post_delay, in ns, as ns is the default unit of time

    qubit_pulse_name: str = None
    qubit_pulse_waveform_name: str = None
    readout_window_name: str = None
    readout_stimulus_waveform_name: str = None
    fit_quadratic: bool = False
    plot: bool = True
    figsize: tuple[int] = None

    def main(self):
        """
        This is the meat of the experiment, similar to 'sequence' of fpga_lib/GUI, but with more things going on as support apparatus (at least for now, all of this is in the foreground)

        The general flow will be (roughly):
        1- some setup code initializing various waveforms (for each of the IOConfig objects in the class) and other objects (analagous to qubit, readout = ancilla, ancilla_ro of fpga_lib/GUI)
        2- the sequence method, which defines the inner-most sequence on the board that we'll loop over
        3- several function calls that compile the sequence and load it onto the board (analagous to make_tables and various behind-the-scenes things)
        4- a bit more setup, specifically (or often) related to loading waveforms into memory. This is distinct from part 1 because in part 3 we specify exactly what board we're using
        5- standard outer 'sweeper' for-loops that change some things (like delay time, drive freq, pulse amplitude, etc.) and then calls acadia.run() to run the aforementioned sequence

        """

        # Section 1 - some setup code
        # @Chao I'd like more details about the finer details of the logger and how to use it
        import logging
        logger = logging.getLogger("acadia")

        # initialize each of the IOConfig objects by passing the names of each of the variables to self.io(), this will load a bunch of stuff from config.yaml into a dictionary
        # It won't actually set anything now, but it will gather a bunch of information (like what DAC/ADC to use, etc.)
        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")

        # creates Qubit and Readout objects, combining information about multiple channels into one object for later use
        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)
        qubit = Qubit(qubit_stimulus_io)

        # Need to create a place for the acquired data to go. By convention, if we are only looking at thresholded values, we'll use "points", if we are taking entire trajectories we'll use "traces"
        
        self.data.add_group(f"points", uniform=True) # the 'uniform' boolean is about whether you'll be saving data of the same shape every single time (almost always true)

        # Section 2 - the sequence
        # Here, we define the skeleton structure of the pulse sequence we'll be running, handling any sweeps with an outer loop. This is therefore the inner-most part of the for loop later on
        def sequence(a: Acadia):
            readout_resonator.prepare_cmacc(self.readout_window_name) # cmacc = "Capture, Multiply, ACCumulate", so this does various setup things to prepare for doing readout
            # note that prepare_cmacc has argument 'output_last_only' that determines whether you just keep the accumulated IQ point, or the whole accumulation (and therefore trajectory)

            # channel_synchronizer() and barrier() work hand in hand, it sync's everything between barrier() calls
            
            # Example:
            # qubit pulse 1
            # qubit pulse 2
            # cavity pulse
            # barrier()
            # readout
            # This will play simultaneously the cavity pulse and qubit pulses 1 & 2 chained together, then wait for that to finish, then play readout


            with a.channel_synchronizer():
                qubit.pulse(self.qubit_pulse_name) # play a pulse on the qubit
                a.barrier() # 'barrier' is the new 'sync', but note that that the synchronizer will automatically cascade pulses played on the same channel, even without barrier() calls
                readout_resonator.measure("readout", "readout_accumulated") # 'capture the readout, multiply by envelope, accumulates, shifts (to put blobs in quadrants), and saves value

        # section 3 - function calls equivalent to "make_tables"
        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        # Section 4 - more setup (why can't be done earlier?)
        readout_resonator.load_windows() # recall that 'window' is the new 'envelope'
        readout_stimulus_io.load_waveform("readout", self.readout_stimulus_waveform_name)

        qubit_pulse_samples = qubit_stimulus_io.compute_waveform(self.qubit_pulse_name, self.qubit_pulse_waveform_name) # Precompute the envelope so that we're not recalculating it every time, only scaling it

        # Section 5 - outer 'sweeper' for-loop(s)
        for i in range(self.iterations): # always sweep iterations on the outer loop!
            for amplitude in self.qubit_amplitudes:
                # update whatever values you're sweeping (pulse amplitude, delay time, frequency, etc.)
                qubit_stimulus_io.load_waveform(self.qubit_pulse_name, qubit_pulse_samples, scale=amplitude)

                # run and get results, write the results to data
                self.acadia.run(minimum_delay=self.run_delay)
                wf = readout_capture_io.get_waveform_memory("readout_accumulated")
                self.data[f"points"].write(wf.array)

            # Rhis allows for aborting part-way through experiment
            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return

        self.final_serve() # If you're all done with the experiment, do various disconnect/wrapping up things in the background

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
            ax.set_xlim(self.qubit_amplitudes[0], self.qubit_amplitudes[-1])
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
        if "points" not in self.data or len(self.data["points"]) < len(self.qubit_amplitudes):
            return

        # Update the progress bar based on the number of iterations
        completed_iterations = len(self.data["points"]) // len(self.qubit_amplitudes)
        if completed_iterations == 0:
            return

        self.iterations_progress_bar.update(completed_iterations - self.iterations_previous)

        valid_points = completed_iterations*len(self.qubit_amplitudes)
        data = self.data["points"].records()[:valid_points, ...]
        data = data.reshape(completed_iterations, len(self.qubit_amplitudes), 2)

        # Threshold the data according to the I quadrature
        shots = np.sign(data[:,:,0], dtype=np.int32)
        self.avg = np.mean(shots, axis=0)
        

        # Fit the data to a sine
        try:
            amin = np.argmin(self.avg)
            amax = np.argmax(self.avg)
            osc_period = 2*abs(self.qubit_amplitudes[amin]-self.qubit_amplitudes[amax])
            p0 = (abs(amin-amax)/2, 1/osc_period, (amin+amax)/2)
            self.fit, pcov = curve_fit(sine, self.qubit_amplitudes, self.avg, p0=p0)
            
        except:
            pass
        
        if self.plot:
            self.line_pop.update(self.qubit_amplitudes, self.avg, rescale_axis=False)
            if self.fit is not None:
                self.amplitude_label.value = f"Pi pulse amplitude: {round(0.5/self.fit[1], 6)}"
                self.line_fit.update(self.qubit_amplitudes, sine(self.qubit_amplitudes, *self.fit), rescale_axis=False)
            self.fig.canvas.draw_idle() 

        self.data.save(self.local_directory)
        self.iterations_previous = completed_iterations

    def finalize(self):
        super().finalize()
        self.iterations_progress_bar.close()
        if self.plot:
            self.savefig(self.fig)

