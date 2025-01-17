import os
import time
from itertools import product
from typing import Union, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime, WaveformMemory
from acadia.utils import clock_monotonic_ns, sys_nanosleep
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig

class ReadoutHistogramRuntime(QMsmtRuntime):
    """
    Saturates the qubit and creates a histogram of measurement outcomes
    """
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    iterations: int
    num_clusters: int = 2
    saturation_pulse_fixed_length: float = 1e-3 - 100e-9
    saturation_pulse_ramp_time: float = 100e-9
    saturation_pulse_amplitude: float = 0.1
    readout_window_name: str = None
    readout_stimulus_waveform_name: str = None
    plot: bool = True
    figsize: tuple[int] = (4,4)
    histogram_scale: Literal["log","linear"] = "linear"
    histogram_bins_I: int = 50
    histogram_bins_Q: int = 50
    histogram_colormap: str = "hot"
    histogram_circle_alpha: float = 1.0
    histogram_circle_facecolor: str = "white"
    histogram_circle_fill: bool = False
    histogram_circle_edgecolor: str = "white"
    histogram_circle_linewidth: float = 1.0

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

        self.data.add_group(f"pts", uniform=True)

        def sequence(a: Acadia):
            readout_resonator.prepare_cmacc(self.readout_window_name)

            with a.channel_synchronizer():
                a.schedule_waveform(saturation_waveform)
                a.barrier()
                readout_resonator.measure("readout", "readout_accumulated")

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_waveform("readout", self.readout_stimulus_waveform_name)
        saturation_waveform.set("hann", self.saturation_pulse_amplitude)

        for i in range(self.iterations):
            sys_nanosleep(1000000)
            self.acadia.run()
            wf = readout_capture_io.get_waveform_memory("readout_accumulated")
            self.data[f"pts"].write(wf.array)

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
        if "pts" not in self.data or len(self.data["pts"]) < 2:
            return

        # Update the progress bar based on the number of iterations
        self.iterations_progress_bar.update(len(self.data["pts"]) - self.iterations_previous)
        self.update_plot(lock=False) # The lock is already acquired by the event loop         
        self.iterations_previous = len(self.data["pts"])

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

            # Use index 0 for the second dimension, since iq points are arrays with a single entry
            points = self.data["pts"].records()[:, 0, :]

            # Given all the points we collected, compute bin edges so that the histogram view 
            # will contain all the points regardless of which sequence we choose to view
            
            all_I_values = np.reshape(points[:, 0], -1)
            histogram_I_edges = np.histogram_bin_edges(all_I_values, bins=self.histogram_bins_I)
            all_Q_values = np.reshape(points[:, 1], -1)
            histogram_Q_edges = np.histogram_bin_edges(all_Q_values, bins=self.histogram_bins_Q)
            
            histogram, _, _ = np.histogram2d(
                points[:, 0], 
                points[:, 1], 
                bins=[histogram_I_edges, histogram_Q_edges],
                density=True)
            
            if self.histogram_scale == "log":
                histogram_flat = histogram.reshape(-1)
                nonzero_indices = np.nonzero(histogram_flat)
                vmin = np.min(histogram_flat[nonzero_indices])
                norm = LogNorm(vmin=vmin, vmax=np.max(histogram), clip=True)
            else:
                norm = Normalize(vmin=np.min(histogram), vmax=np.max(histogram), clip=True)
            
            self.histogram_plot = self.ax_histogram.pcolormesh(
                histogram_I_edges, 
                histogram_Q_edges, 
                histogram.T,
                cmap=self.histogram_colormap,
                norm=norm)
            self.ax_histogram.set_xlim(histogram_I_edges[0], histogram_I_edges[-1])
            self.ax_histogram.set_ylim(histogram_Q_edges[0], histogram_Q_edges[-1])

            self.fig.canvas.draw_idle() 

            if lock:
                self._update_lock.release()
