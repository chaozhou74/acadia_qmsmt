import os
import time
from itertools import product
from typing import Union
from pathlib import Path
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime, WaveformMemory
from acadia.utils import clock_monotonic_ns
from acadia.runtime import annotate_method
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig

import logging

logger = logging.getLogger("acadia")

class ReadoutWindowCalibrationRuntime(QMsmtRuntime):
    """
    Capture readout traces with qubit prepared in g and e.

    g state preparation is done by just relaxing.
    e state preparation is done using the pi_pulse parameters in q_stimulus["waveforms"]["q_rotation"] and
    q_stimulus["signals"]["pi_pulse"]

    postprocess is able to generate readout kernel even when the state preparation is imperfect.
    """
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    readout_stimulus_memory: str = "readout"
    readout_stimulus_waveform: str = "readout"
    qubit_pulse_name: str = None
    qubit_pulse_waveform_name: str = None

    readout_capture_memory: str = "readout_trace"

    iterations: int
    plot: bool = True
    figsize: tuple[int] = None
    generate_kernel: bool = False
    
    yaml_path: str = None

    def main(self):
        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")

        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)
        qubit = Qubit(qubit_stimulus_io)


        # Create the record groups for saving captured data
        self.data.add_group("traces_g", uniform=True)
        self.data.add_group("traces_e", uniform=True)
        self.data.add_group("t_data", uniform=False)

        # Core FPGA (PL) sequence
        def sequence(a: Acadia):

            with a.channel_synchronizer():
                qubit.pulse(self.qubit_pulse_name)
                a.barrier()
                readout_resonator.measure_trace(self.readout_stimulus_memory, self.readout_capture_memory)

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()

        readout_stimulus_io.load_waveform(self.readout_stimulus_memory, self.readout_stimulus_waveform)
        t_data = None
        # Core python loop that will be running on the ARM (PS)
        for i in range(self.iterations):
            for state_ in ["g", "e"]:
                # set the pulse amplitude
                amp_ = 0 if state_ == "g" else qubit_stimulus_io.get_config("waveforms", self.qubit_pulse_waveform_name, "scale")

                qubit_stimulus_io.load_waveform(self.qubit_pulse_name, self.qubit_pulse_waveform_name, scale=amp_)

                # capture data and put in the corresponding group
                self.acadia.run()
                wf = readout_capture_io.get_waveform_memory(self.readout_capture_memory)
                self.data[f"traces_{state_}"].write(wf.array)
                
                # calculate t_list based on capture time and length of capture waveform
                if t_data is None:                    
                    capture_time = readout_capture_io.get_config("memories", self.readout_capture_memory,"length")
                    t_data = np.linspace(0, capture_time, len(wf.array), endpoint=False)
                    self.data["t_data"].write(t_data)

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
    def process_current_data(self, g_center:complex= None, g_radius:float=None,
                             e_center: complex = None, e_radius: float = None,
                             i_threshold=None, q_threshold=None, bins: int = 50):
        from acadia_qmsmt.analysis.generate_readout_kernel import KernelFromGETraces

        # gather time and IQ trace data
        self.t_data = np.array(self.data["t_data"].records()).astype(float).squeeze()
        self.g_traces = np.array(self.data["traces_g"].records()).astype(float).view(complex).squeeze()
        self.e_traces = np.array(self.data["traces_e"].records()).astype(float).view(complex).squeeze()

        if self.e_traces is None:
            return
        else:
            completed_iterations = len(self.g_traces)

        try:
            decimation_used = self.io("readout_capture").get_config("memories", self.readout_capture_memory, "decimation")
            self.kernel_gen = KernelFromGETraces(self.g_traces, self.e_traces, (g_center, g_radius), (e_center, e_radius),
                                            i_threshold, q_threshold,
                                            bins=bins, plot=False, decimation_used=decimation_used)
        except Exception as e:
            logger.error(f"Error calculating kernel: {e}")

        return  completed_iterations


    @annotate_method(plot_name="kernel generation")
    def plot_kernel_generation(self, fig=None, log_scale:bool=False):
        if fig is None:
            from matplotlib.pyplot import figure
            fig = figure(figsize=self.figsize)
        # adjust aspect ratio to look good in the gui
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
        ax_left = fig.add_subplot(gs[0, 0])
        ax_middle = fig.add_subplot(gs[0, 1])
        ax_right = fig.add_subplot(gs[1, :])
        axs = [ax_left, ax_middle, ax_right]
        self.kernel_gen.plot_kernel_generation(axs, log_scale=log_scale)

        return fig, axs

    @annotate_method(plot_name="result kernel", axs_shape=(1,1))
    def plot_calculated_kernel(self, axs=None, plot_uploaded:bool=False):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1, 1), figsize=self.figsize)
        self.kernel_gen.plot_kernel(axs, plot_uploaded=plot_uploaded)

        return fig, axs

    @annotate_method(plot_name="raw data", axs_shape=(2,2))
    def plot_raw_data(self, axs=None, log_scale:bool=False):
        from acadia_qmsmt.plotting import prepare_plot_axes, plot_multiple_hist2d
        fig, axs = prepare_plot_axes(axs, axs_shape=(2, 2), figsize=self.figsize)

        # IQ points for g/e preparation
        g_pts_raw, e_pts_raw = self.kernel_gen.g_pts_raw, self.kernel_gen.e_pts_raw
        plot_multiple_hist2d(g_pts_raw, e_pts_raw, plot_ax=axs[:, 0],
                             bins=self.kernel_gen.bins, log_scale=log_scale)

        # averaged raw traces for g/e preparation
        for i, traces in enumerate([self.g_traces, self.e_traces]):
            prep = ["g", "e"][i]
            axs[i, 1].plot(self.t_data, np.mean(traces.real, axis=0), label=f"prep {prep}, re")
            axs[i, 1].plot(self.t_data, np.mean(traces.imag, axis=0), label=f"prep {prep}, im")
            axs[i, 1].legend()
            axs[i, 1].grid(True)
            axs[i, 0].set_title(f"prep {prep}")
        return fig, axs

    @annotate_method(button_name="update window")
    def update_window(self, window_name:str = "matched"):
        # can't use self.yaml_path here! because the reloaded runtime will not find the right path
        yaml_path = self.io("readout_capture")._config["__yaml_path__"]
        kernel_dir = Path(yaml_path).parent/"readout_kernels"
        # save kernel
        kernel_path = self.kernel_gen.save_kernel(kernel_dir,
                          "readout_capture_"+window_name+datetime.now().strftime("_%y%m%d_%H%M%S"))
        # update yaml
        self.update_io_yaml_field("readout_capture", f"windows.{window_name}.data", kernel_path)
        self.update_io_yaml_field("readout_capture", f"windows.{window_name}.offset", self.kernel_gen.cmacc_offset)



