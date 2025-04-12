import os
import time
from itertools import product
from typing import Union, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime, WaveformMemory
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig

class ReadoutHistogramRuntime(QMsmtRuntime):
    """
    Saturates the qubit and creates a histogram of measurement outcomes
    """
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    iterations: int
    run_delay: int

    hardware_accumulator: bool = True
    saturation_pulse_fixed_length: float = 1e-3 - 100e-9
    saturation_pulse_ramp_time: float = 100e-9
    saturation_pulse_amplitude: float = 0.1
    readout_window_name: str = None
    readout_stimulus_waveform_name: str = None
    plot: bool = True
    figsize: tuple[int] = (4,4)
    yaml_path: str = None
    histogram_scale: Literal["log","linear"] = "linear"
    histogram_bins_I: int = 50
    histogram_bins_Q: int = 50
    histogram_colormap: str = "hot"

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")

        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)
        qubit = Qubit(qubit_stimulus_io)

        # Create the qubit saturation waveform manually, as this is not a waveform
        # that the user will likely need to keep in the configuration file after this
        # measurement
        saturation_waveform = self.acadia.create_waveform_memory(
            qubit._stimulus.channel, 
            length=self.saturation_pulse_ramp_time, 
            fixed_length=self.saturation_pulse_fixed_length)

        self.data.add_group(f"points", uniform=True)
        readout_name = f"readout_{'accumulated' if self.hardware_accumulator else 'trace'}"

        def sequence(a: Acadia):
            readout_resonator.prepare_cmacc(
                self.readout_window_name, 
                output_type=("upper" if self.hardware_accumulator else "input"), 
                output_last_only=self.hardware_accumulator)

            with a.channel_synchronizer():
                a.schedule_waveform(saturation_waveform)
                a.barrier()
                readout_resonator.measure("readout", readout_name)

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_waveform("readout", self.readout_stimulus_waveform_name)
        qubit_stimulus_io.load_waveform(saturation_waveform, {"data": "hann"}, self.saturation_pulse_amplitude)

        for i in range(self.iterations):
            self.acadia.run(minimum_delay=self.run_delay)
            wf = readout_capture_io.get_waveform_memory(readout_name)
            self.data[f"points"].write(wf.array)

            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return

        self.final_serve()

    def initialize(self):
        
        if self.plot:
            from acadia.processing import DynamicLine
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
            from matplotlib.scale import get_scale_names
            from IPython.display import display
            from ipywidgets import Box, Layout, Dropdown, Checkbox, FloatText, Label, HTML, Button, Text

            self.fig, self.ax_histogram = plt.subplots(figsize=self.figsize)
            self.fig.tight_layout()
            self.ax_histogram.axvline(0, color="white")
            self.ax_histogram.axhline(0, color="white")
            self.ax_histogram.set_xlabel("I [arb.]")
            self.ax_histogram.set_ylabel("Q [arb.]")
            self.histogram_plot = None

            from tqdm.notebook import tqdm

        else:
            from tqdm import tqdm

        self.iterations_progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.iterations_previous = 0

    def update(self):
        # First make sure that we actually have new data to process
        if "points" not in self.data or len(self.data["points"]) < 2:
            return

        # Update the progress bar based on the number of iterations
        self.iterations_progress_bar.update(len(self.data["points"]) - self.iterations_previous)

        # Sum over the second dimension
        # For IQ points this will work fine since they're arrays with a single entry, and it
        # allows us to interchangeably process full traces or just IQ points
        # Use index 0 for the second dimension, since iq points are arrays with a single entry
        points = np.sum(self.data["points"].records(), axis=1, keepdims=False)

        # Given all the points we collected, compute bin edges so that the histogram view 
        # will contain all the points regardless of which sequence we choose to view
        
        all_I_values = np.reshape(points[:, 0], -1)
        self.histogram_I_edges = np.histogram_bin_edges(all_I_values, bins=self.histogram_bins_I)
        all_Q_values = np.reshape(points[:, 1], -1)
        self.histogram_Q_edges = np.histogram_bin_edges(all_Q_values, bins=self.histogram_bins_Q)
        
        self.histogram, _, _ = np.histogram2d(
            points[:, 0], 
            points[:, 1], 
            bins=[self.histogram_I_edges, self.histogram_Q_edges],
            density=True)

        self.update_plot(lock=False) # The lock is already acquired by the event loop         
        self.iterations_previous = len(self.data["points"])

    def finalize(self):
        super().finalize()
        self.iterations_progress_bar.close()
        # if self.plot:
        #     self.savefig(self.fig, close_canvas=False)

    def update_plot(self, lock: bool = True, process: bool = True):
        """
        Update the view of the plot. We can choose to optionally acquire the update 
        lock so that we can call this in response to UI events, outside of the 
        runtime event loop
        """
        if self.plot:
            from matplotlib.colors import Normalize, LogNorm

            if lock and not self._update_lock.acquire(timeout=5):
                raise ValueError("Failed to acquire update lock when updating plot")

            # Create the histogram view
            if self.histogram_plot is not None:
                self.histogram_plot.remove()
            
            if self.histogram_scale == "log":
                histogram_flat = self.histogram.reshape(-1)
                nonzero_indices = np.nonzero(histogram_flat)
                vmin = np.min(histogram_flat[nonzero_indices])
                norm = LogNorm(vmin=vmin, vmax=np.max(self.histogram), clip=True)
            else:
                norm = Normalize(vmin=np.min(self.histogram), vmax=np.max(self.histogram), clip=True)
                
            self.histogram_plot = self.ax_histogram.pcolormesh(
                self.histogram_I_edges, 
                self.histogram_Q_edges, 
                self.histogram.T,
                cmap=self.histogram_colormap,
                norm=norm)
            self.ax_histogram.set_xlim(self.histogram_I_edges[0], self.histogram_I_edges[-1])
            self.ax_histogram.set_ylim(self.histogram_Q_edges[0], self.histogram_Q_edges[-1])

            self.fig.canvas.draw_idle() 

            if lock:
                self._update_lock.release()
