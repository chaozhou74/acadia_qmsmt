import os
import time
from itertools import product
from typing import Union, Tuple, List, Literal
from pathlib import Path
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.ndimage import label

from acadia import Acadia, DataManager, Runtime, WaveformMemory
from acadia.utils import clock_monotonic_ns
from acadia.runtime import annotate_method
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig

import logging

StateCircleType = Tuple[complex, np.floating]  # (I_center + 1j * Q_center, circle_radius)
ComplexDataPointsType = Union[List[complex], np.ndarray[complex]]
ComplexDataTracesType = Union[List, np.ndarray]

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


    qubit_pulse_name: str = "R_x_180"
    readout_pulse_name: str = "readout"
    capture_memory_name: str = "readout_trace"
    capture_window_name: str = None

    iterations: int
    run_delay:int = 200e3 #ns

    figsize: tuple[int] = None
    
    yaml_path: str = None

    def main(self):
        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")

        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)


        # Create the record groups for saving captured data
        self.data.add_group("traces_g", uniform=True)
        self.data.add_group("traces_e", uniform=True)
        self.data.add_group("t_data", uniform=False)

        # Core FPGA (PL) sequence
        def sequence(a: Acadia):

            with a.channel_synchronizer():
                qubit_stimulus_io.schedule_pulse(self.qubit_pulse_name)
                a.barrier()
                readout_resonator.measure_trace(self.readout_pulse_name, self.capture_memory_name)

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()

        readout_stimulus_io.load_pulse(self.readout_pulse_name)
        t_data = None

        # Core python loop that will be running on the ARM (PS)
        for i in range(self.iterations):
            for state_ in ["g", "e"]:

                if state_ == "g":
                    qubit_stimulus_io.load_pulse(self.qubit_pulse_name, scale=0.)
                else:
                    qubit_stimulus_io.load_pulse(self.qubit_pulse_name) # !!Note: scale not provided will use the pulse in yaml file; scale=None will set pulse scale to 1

                # capture data and put in the corresponding group
                self.acadia.run(minimum_delay=self.run_delay)
                wf = readout_capture_io.get_waveform_memory(self.capture_memory_name)
                self.data[f"traces_{state_}"].write(wf.array)
                
                # calculate t_list based on capture time and length of capture waveform
                if t_data is None:                    
                    capture_time = readout_capture_io.get_config("memories", self.capture_memory_name,"length")
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
        from acadia_qmsmt.plotting import save_registered_plots
        save_registered_plots(self)

    @annotate_method(is_data_processor=True)
    def process_current_data(self, 
                                g_center: complex = None, 
                                g_radius: float = None,
                                e_center: complex = None, 
                                e_radius: float = None,
                                i_threshold = None, 
                                q_threshold = None, 
                                bins: int = 50):

        # gather time and IQ trace data
        self.t_data = np.array(self.data["t_data"].records()).astype(float).squeeze()
        self.g_traces_raw = np.array(self.data["traces_g"].records()).astype(float).view(complex).squeeze()
        self.e_traces_raw = np.array(self.data["traces_e"].records()).astype(float).view(complex).squeeze()

        if self.e_traces_raw is None:
            return

        # Save these, we'll use them in plotting
        self.g_pts_raw = np.sum(self.g_traces_raw, axis=1)
        self.e_pts_raw = np.sum(self.e_traces_raw, axis=1)
        self.bins = bins
        self.mask_with_pre_kernel = False
        
        completed_iterations = len(self.g_traces_raw)
        decimation_used = self.io("readout_capture").get_config("memories", self.capture_memory_name, "decimation")

        try:
            self.generate_kernel(self.g_traces_raw, 
                                self.e_traces_raw, 
                                self.g_pts_raw, 
                                self.e_pts_raw,
                                (g_center, g_radius), 
                                (e_center, e_radius),
                                i_threshold, 
                                q_threshold,
                                bins=bins, 
                                mask_with_pre_kernel=self.mask_with_pre_kernel,
                                decimation_used=decimation_used)
        except Exception as e:
            logger.error(f"Error calculating kernel: {e}")

        self.g_trace_pwr = np.mean(np.abs(self.g_traces_raw)**2, axis=0)
        self.e_trace_pwr = np.mean(np.abs(self.e_traces_raw)**2, axis=0)

        return completed_iterations


    @annotate_method(plot_name="1. kernel generation")
    def plot_kernel_generation(self, fig=None, log_scale: bool = True, bins: int = None):
        from matplotlib.colors import LogNorm
        from matplotlib.patches import Circle

        if fig is None:
            from matplotlib.pyplot import figure
            fig = figure(figsize=self.figsize)

        # adjust aspect ratio to look good in the gui
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
        ax_left = fig.add_subplot(gs[0, 0])
        ax_middle = fig.add_subplot(gs[0, 1])
        ax_right = fig.add_subplot(gs[1, :])
        axs = [ax_left, ax_middle, ax_right]
        
        norm = LogNorm() if log_scale else None
        bins = self.bins if bins is None else bins
        state_label_color = "grey" if log_scale else "w"
        
        if not self.mask_with_pre_kernel:
            axs[0].set_title("raw IQ points and g/e selection")
        else:
            axs[0].set_title("IQ points after applying direct subtraction kernel")
        axs[0].hist2d(self.all_pts_for_sel.real, self.all_pts_for_sel.imag, bins=bins, cmap="hot", norm=norm)
        for circ, state in zip([self.g_circle, self.e_circle], ["g", "e"]):
            circ_i, circ_q, circ_r = circ[0].real, circ[0].imag, circ[1]
            axs[0].add_patch(Circle((circ_i, circ_q), circ_r, edgecolor=state_label_color, facecolor="none"))
            axs[0].text(circ_i + circ_r*0.8, circ_q + circ_r*0.8, state, fontsize=12, va='center', color=state_label_color)
        axs[0].set_aspect('equal')

        axs[1].set_title("average of selected IQ traces")
        trace_colors = [(0.27, 0.51, 0.71), (1.0, 0.65, 0.47), (0.18, 0.31, 0.56), (0.94, 0.5, 0.5)]
        for i, trace in enumerate([self.g_trace_avg, self.e_trace_avg]):
            axs[1].plot(trace.real, ".-", color=trace_colors[2 * i], label=f"{['g', 'e'][i]} trace, re")
            axs[1].plot(trace.imag, ".-", color=trace_colors[2 * i + 1], label=f"{['g', 'e'][i]} trace, im")
            axs[1].set_xlabel("pts")
            axs[1].legend()
            axs[1].grid(True)

        axs[2].set_title("IQ points after applying kernel")
        new_iq_pts_all = np.concatenate([self.g_pts_new, self.e_pts_new])

        if hasattr(self, "cmacc_offset"):
            x0, y0 = -self.cmacc_offset[0], -self.cmacc_offset[1]      

            hist_range = [
                [np.min(new_iq_pts_all.real), np.max(new_iq_pts_all.real)],
                [min(np.min(new_iq_pts_all.imag), y0), max(np.max(new_iq_pts_all.imag), y0)]
            ]

            hist, xedges, yedges, _ = axs[2].hist2d(new_iq_pts_all.real, new_iq_pts_all.imag,
                                                    bins=bins, cmap="hot", norm=norm, range=hist_range)

            axs[2].axvline(x=x0, color=state_label_color, linestyle='-', linewidth=0.5)
            axs[2].axhline(y=y0, color=state_label_color, linestyle='-', linewidth=0.5)
            text_d = (xedges[-1]-xedges[0]) * 0.03
            for i, quadrant in enumerate(self.state_quadrants):
                sign_x = -1 if quadrant in [2,3] else 1
                sign_y = -1 if quadrant in [3,4] else 1
            #     axs[2].text(x0 + text_d * sign_x, y0 + text_d * sign_y, QubitStateLabels[i], fontsize=12,
            #                 va='center', ha='center', color=state_label_color)
        else:
            hist, xedges, yedges, _ = axs[2].hist2d(new_iq_pts_all.real, new_iq_pts_all.imag,
                                                    bins=bins, cmap="hot", norm=norm)

        axs[2].set_aspect('equal')
        fig.tight_layout()

        return fig, axs

    @annotate_method(plot_name="3. result kernel", axs_shape=(1,1))
    def plot_calculated_kernel(self, axs=None, plot_uploaded: bool = True):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1, 1), figsize=self.figsize)
        
        if plot_uploaded:
            kernel = self.kernel_upload
        else:
            kernel = self.kernel_trace

        axs.set_title("calculated kernel")
        axs.plot(kernel.real, label="re")
        axs.plot(kernel.imag, label="im")
        axs.set_xlabel("pts")
        fig.tight_layout()
        axs.legend()
        axs.grid()

        return fig, axs

    @annotate_method(plot_name="2. raw data", axs_shape=(2,2))
    def plot_raw_data(self, axs=None, log_scale: bool = False):
        from acadia_qmsmt.plotting import prepare_plot_axes, plot_multiple_hist2d
        fig, axs = prepare_plot_axes(axs, axs_shape=(2, 2), figsize=self.figsize)

        # IQ points for g/e preparation
        plot_multiple_hist2d(self.g_pts_raw, self.e_pts_raw, plot_ax=axs[:, 0],
                             bins=self.bins, log_scale=log_scale)

        # averaged raw traces for g/e preparation
        for i, traces in enumerate([self.g_traces_raw, self.e_traces_raw]):
            prep = ["g", "e"][i]
            axs[i, 1].plot(self.t_data, np.mean(traces.real, axis=0), label=f"prep {prep}, re")
            axs[i, 1].plot(self.t_data, np.mean(traces.imag, axis=0), label=f"prep {prep}, im")
            axs[i, 1].legend()
            axs[i, 1].grid(True)
            axs[i, 0].set_title(f"prep {prep}")
        return fig, axs

    
    @annotate_method(plot_name = "4. capture power")
    def plot_capture_amplitude_post_selected(self, axs=None, plot_decay_fit:bool=True, decay_start_time_us:float = None, fit_type:Literal['<power>', '<voltage>^2'] = '<voltage>^2'):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, figsize=self.figsize)
        if fit_type=='<power>': 
            axs.plot(self.t_data*1e6, self.g_trace_pwr, label="g trace")
            axs.plot(self.t_data*1e6, self.e_trace_pwr, label="e trace")
        elif fit_type=='<voltage>^2': 
            axs.plot(self.t_data*1e6, np.abs(self.g_trace_avg)**2, label="g trace")
            axs.plot(self.t_data*1e6, np.abs(self.e_trace_avg)**2, label="e trace")
            
        axs.set_xlabel("Time (us)")
        axs.grid()
        axs.legend()
        axs.set_title("Capture power post selected")

        if plot_decay_fit:
            from acadia_qmsmt.analysis.fitting import Exponential
            decay_start_time = decay_start_time_us*1e-6 if decay_start_time_us is not None else 1.7e-6
            decay_start_idx = np.argmin(np.abs(self.t_data - decay_start_time))

            t_data_decay_us = self.t_data[decay_start_idx:] * 1e6
            if fit_type=='<power>': 
                g_data_decay_us = self.g_trace_pwr[decay_start_idx:]
                e_data_decay_us = self.e_trace_pwr[decay_start_idx:]
            elif fit_type=='<voltage>^2': 
                g_data_decay_us = np.abs(self.g_trace_avg[decay_start_idx:])**2
                e_data_decay_us = np.abs(self.e_trace_avg[decay_start_idx:])**2
            else: 
                raise NameError('Please use <power> or <voltage>^2 as `fit_type`.')

            self.g_fit = Exponential(t_data_decay_us, g_data_decay_us)
            self.e_fit = Exponential(t_data_decay_us, e_data_decay_us)

            self.fitted_g_t1_us = self.g_fit.ufloat_results["tau"]
            self.fitted_e_t1_us = self.e_fit.ufloat_results["tau"]

            self.g_fit.plot_fitted(axs, oversample=5,
                            **{"label": f"|g> $\kappa$/2$\pi$ (MHz): {1/2/np.pi/self.fitted_g_t1_us:.4g}"})
            self.e_fit.plot_fitted(axs, oversample=5,
                            **{"label": f"|e> $\kappa$/2$\pi$ (MHz): {1/2/np.pi/self.fitted_e_t1_us:.4g}"})

        return fig, axs

    @annotate_method(button_name="update window")
    def update_window(self, window_name: str = "matched", biased_g_offset: int = -1000):
        # can't use self.yaml_path here! because the reloaded runtime will not find the right path
        yaml_path = self.io("readout_capture")._config["__yaml_path__"]
        kernel_dir = os.path.join(os.path.dirname(yaml_path), "readout_kernels")
        if not os.path.exists(kernel_dir):
            os.mkdir(kernel_dir)
        
        # save kernel
        filename_prefix = self.io("readout_capture")._config["__yaml_key__"]
        filename_time = datetime.now().strftime("%y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{window_name}_{filename_time}.npy"
        full_filepath = os.path.join(kernel_dir, filename)
        np.save(full_filepath, self.kernel_upload)

        # update yaml
        offset = self.cmacc_offset
        self.update_io_yaml_field("readout_capture", f"windows.{window_name}.data", full_filepath)
        self.update_io_yaml_field("readout_capture", f"windows.{window_name}.offset", offset)
        self.update_io_yaml_field("readout_capture", f"windows.{window_name}_biased_g.data", full_filepath)
        self.update_io_yaml_field("readout_capture", f"windows.{window_name}_biased_g.offset", (offset[0]+biased_g_offset, offset[1]))

    def interpolate_kernel(self, kernel: np.ndarray, decimation_used: int):
        """
        Determine if any additional decimation was used to sample traces for the kernel, and if so,
        interpolate it for the CMACC (which always operates at a decimation of 4).
        """
        if decimation_used == 4:
            kernel_array = kernel

        elif decimation_used == 1:
            # decimate the kernel trace by averaging every 4 adjacent points
            length = len(kernel) - (len(kernel) % 4)
            reshaped = kernel[:length].reshape(-1, 4)
            kernel_array = reshaped.mean(axis=1)

        elif (decimation_used > 0) and (decimation_used % 4 == 0):
            # stretch the kernel trace via interpolation
            scale = int(decimation_used // 4)
            original_len = len(kernel)
            interp_func = interp1d(np.arange(0, original_len), kernel, kind="cubic")
            x_fine = np.linspace(0, original_len-1, (original_len-1) * scale+1)
            kernel_interp = interp_func(x_fine)
            # the points at the beginning will just be repeated
            kernel_array = np.concatenate(([kernel[0]] * (scale-1), kernel_interp))

        else:
            raise ValueError("Invalid trace decimation, must be 1 or a positive multiple of 4 ")

        return kernel_array / np.max(abs(kernel_array))

    def generate_kernel(self, 
                        g_traces_raw,
                        e_traces_raw,
                        g_pts_raw,
                        e_pts_raw,
                        g_circle: StateCircleType = None, 
                        e_circle: StateCircleType = None,
                        i_threshold: int = None, 
                        q_threshold: int = None,
                        bins: int = 50, 
                        sigma_factor: float = 2.5, 
                        average_radius: bool = True,
                        norm_factor: float = 1, 
                        decimation_used: int = 4,
                        mask_with_pre_kernel: bool = False,
                        debug: bool = False):
        """
        Generate readout kernel based on the complex IQ traces from g/e state preparation.

        Note this does not simply take the difference of the two acquired traces. Instead, it identifies the g and e
        circles and only uses the traces that fall into those circles.

        When not provided, the g/e state locations are identified using the `find_state_circles` function, which
        should hopefully be quite robust even when the state preparation fidelity is low.

        :param g_traces: complex IQ traces from g state preparation, should have the shape of (iterations, time_points)
        :param e_traces: complex IQ traces from e state preparation, should have the shape of (iterations, time_points)

        :param g_circle: (I_center + 1j * Q_center, circle_radius). When None, will perform automatic searching
        :param e_circle: (I_center + 1j * Q_center, circle_radius). When None, will perform automatic searching

        :param i_threshold: The I position of the separation line. Default to center of the g and e blobs.
        :param q_threshold: The Q position of the separation line. Default to 3-sigma away from the center of the g
            blob.
        :param bins: number of bins in the 2D histogram for automatic searching of state circle
        :param sigma_factor: The sigma factor for the radius of the state circle (e.g., 2 for 2-sigma).
        :param average_radius: When True, average the radius of all state circles.
        :param norm_factor: normalization factor for the generated kernel.
        :param decimation_used: Decimation factor used when getting the `traces` data. This will only be used for
            generating the kernel to be uploaded to cmacc. See `KernelGeneratorBase.decimation_used`
        :param debug:
        :param mask_with_pre_kernel: If True, apply the direct subtraction kernel to the raw traces before
            identifying the g/e circles. This is useful when the directly integrated g/e points are not well separated,
            for example, when the traces has extra carrier frequencies that was not fully demodulated out.
        """

        all_traces = np.concatenate([g_traces_raw, e_traces_raw])

        # apply direct subtraction kernel to the raw traces?
        if mask_with_pre_kernel:
            kernel_trace_mid = np.conjugate(np.mean(g_traces_raw, axis=0) - np.mean(e_traces_raw, axis=0))
            kernel_trace_mid /= np.max(abs(kernel_trace_mid)) / norm_factor  # normalize to 1

            # calculate expected IQ points when simple subtraction kernel is applied
            g_pts_for_sel = np.sum(g_traces_raw * kernel_trace_mid, axis=1)
            e_pts_for_sel = np.sum(e_traces_raw * kernel_trace_mid, axis=1)
        else:
            g_pts_for_sel, e_pts_for_sel = g_pts_raw, e_pts_raw

        
        # if at least one of the circle parameters is not provided, use search function
        g_center, g_radius = (None, None) if g_circle is None else g_circle
        e_center, e_radius = (None, None) if e_circle is None else e_circle

        # Find state circles
        if any(x is None for x in (g_center, g_radius, e_center, e_radius)):
            g_e_circles = ReadoutWindowCalibrationRuntime.find_state_circles(g_pts_for_sel, e_pts_for_sel, bins=bins,
                                                sigma_factor=sigma_factor, average_radius=average_radius, debug=debug)

        self.g_circle = (g_center if g_center is not None else g_e_circles[0][0],
                         g_radius if g_radius is not None else g_e_circles[0][1])
        self.e_circle = (e_center if e_center is not None else g_e_circles[1][0],
                         e_radius if e_radius is not None else g_e_circles[1][1])


        # generate g/e masks based on the identified circles
        self.all_pts_for_sel = np.concatenate([g_pts_for_sel, e_pts_for_sel])
        g_mask = abs(self.all_pts_for_sel - self.g_circle[0]) < self.g_circle[1]
        e_mask = abs(self.all_pts_for_sel - self.e_circle[0]) < self.e_circle[1]

        # Average the traces within the masked regions and calculate the kernel
        self.g_trace_avg = np.mean(all_traces[g_mask], axis=0)
        self.e_trace_avg = np.mean(all_traces[e_mask], axis=0)
        self.kernel_trace = np.conjugate(self.g_trace_avg - self.e_trace_avg)
        kernel_trace_norm = self.kernel_trace / np.max(abs(self.kernel_trace)) * norm_factor

        # Take the raw traces and integrate them against the kernel to visualize the transformation
        self.g_pts_new = np.sum(g_traces_raw * kernel_trace_norm, axis=1)
        self.e_pts_new =  np.sum(e_traces_raw * kernel_trace_norm, axis=1)

        # Generate CMACC offset

        # post-kernel g and e center locations
        g_center_pk = np.sum(kernel_trace_norm * self.g_trace_avg)
        e_center_pk = np.sum(kernel_trace_norm * self.e_trace_avg)

        # Calculate default Q offset based on center of the new g and e circle
        if i_threshold is None:
            offset_I = -(g_center_pk.real + e_center_pk.real) / 2
        else:
            offset_I = -i_threshold

        # Calculate default Q offset based on the 3-sigma radius of the g states
        if q_threshold is None:
            radius = np.std(self.g_pts_new - g_center_pk) * 3
            # offset_Q = ((g_center_pk.imag > 0) * 2 - 1) * radius - g_center_pk.imag
            offset_Q = radius - g_center_pk.imag
        else:
            offset_Q = -q_threshold

        offset = (int(offset_I), int(offset_Q))
        complex_offset = offset_I + 1j*offset_Q
        self.state_quadrants = (ReadoutWindowCalibrationRuntime.find_quadrant(g_center_pk + complex_offset), 
                                ReadoutWindowCalibrationRuntime.find_quadrant(e_center_pk + complex_offset))
        self.cmacc_offset = offset

        # Finally, interpolate the kernel for use in the CMACC if needed
        self.kernel_upload = self.interpolate_kernel(kernel_trace_norm, decimation_used)
            
    @staticmethod
    def _hist2d_with_indices(iq_pts: ComplexDataPointsType, **kwargs):
        """
        Compute a 2D histogram of IQ data points, as well as the bin indices of where each data point falls in the grid.

        :param iq_pts: List of complex IQ points
        :param kwargs: kwargs for `np.histogram2d`
        :return:
        """
        hist, xedges, yedges = np.histogram2d(iq_pts.real, iq_pts.imag, **kwargs)
        bin_indices = np.digitize(iq_pts.real, xedges[:-1]) - 1, np.digitize(iq_pts.imag, yedges[:-1]) - 1
        return hist, xedges, yedges, bin_indices

    @staticmethod
    def find_most_significant_blob(iq_pts: ComplexDataPointsType,
                               bins: int = 50, sigma_factor: float = 2.5) -> StateCircleType:
        """
        Identify the most significant Gaussian blob in an 1D array of complex IQ points.
        Different blobs are separated by looking at bin connectivity.

        No fitting is performed here, so hopefully this is fast and robust...

        :param iq_pts: 1D array of complex numbers representing IQ points.
        :param bins: Number of bins for the 2D histogram.
        :param sigma_factor: The factor for the radius (e.g., 2 for 2-sigma).
        :return: A tuple (center, radius) for the most significant blob.
        """
        # Create a 2D histogram
        hist, xedges, yedges, bin_indices = ReadoutWindowCalibrationRuntime._hist2d_with_indices(iq_pts, bins=bins)

        # Find the connected components in the histogram
        structure = np.ones((3, 3))  # Connectivity structure for 2D neighbors
        labeled, num_features = label(hist > 0, structure=structure)

        # Find the connected region with the largest total hist count (most significant blob)
        blob_weights = [hist[labeled == i].sum() for i in range(1, num_features + 1)]
        most_significant_blob_idx = np.argmax(blob_weights) + 1

        # Mask for points in the largest blob
        largest_blob_mask = (labeled == most_significant_blob_idx)

        # Extract points in the largest blob by mapping histogram bins back to points
        in_largest_blob = largest_blob_mask[bin_indices]

        blob_points = iq_pts[in_largest_blob]
        blob_pt_bin_indices = bin_indices[0][in_largest_blob], bin_indices[1][in_largest_blob]
        pt_weights = hist[blob_pt_bin_indices]

        # Compute the center and radius
        center = np.average(blob_points, weights=pt_weights)
        distances = np.abs(blob_points - center)
        radius = sigma_factor * np.std(distances)

        return center, radius

    @staticmethod
    def find_state_circles(*state_pts: ComplexDataPointsType, 
                            bins: int = 50,
                            sigma_factor: float = 2.5, 
                            average_radius: bool = True, 
                            debug: bool = False) -> List[StateCircleType]:
        """
        Find the state circle for an arbitrary number of prepared state data points.

        The raw data from each prepared state will be masked based on its prominence comparing to data from other sates
        before looking for the state circle.

        :param state_pts: Variable number of lists of complex IQ data points for each prepared state.
        :param bins: Number of bins for the 2D histogram.
        :param sigma_factor: The sigma factor for the radius (e.g., 2 for 2-sigma).
        :param average_radius: When True, average the radius of all state circles
        :return:
        """
        center_list = []
        r_list = []
        n_states = len(state_pts)

        # make individual histograms for each state using the same grid, for prominence comparison
        hist_list = []
        bin_indices_list = []
        all_pts = np.concatenate(state_pts).flatten()
        compare_hist_bin = 41  # this needs to be relatively small for the data masking to work
        real_bins = np.linspace(all_pts.real.min(), all_pts.real.max(), compare_hist_bin)
        imag_bins = np.linspace(all_pts.imag.min(), all_pts.imag.max(), compare_hist_bin)

        if debug:
            fig, axs = plt.subplots(len(state_pts), 2, figsize=(4, n_states * 3))
            axs = [axs] if n_states == 1 else axs
            fig.suptitle("find_state_circles debug")

        for i, pts in enumerate(state_pts):
            hist, _, _, bin_indices = ReadoutWindowCalibrationRuntime._hist2d_with_indices(pts, bins=[real_bins, imag_bins])
            hist_list.append(hist / len(pts))
            bin_indices_list.append(bin_indices)
            if debug:
                axs[i, 0].hist2d(pts.real, pts.imag, bins=[real_bins, imag_bins], cmap="hot")
                axs[i, 0].set_aspect(1)

        hist_all = np.sum(hist_list, axis=0)

        # find circle for each state
        for i, pts in enumerate(state_pts):
            # make a hist of the current state
            hist, bin_indices = hist_list[i], bin_indices_list[i]

            # mask data of the current state by comparing the hist of current data and the rest data
            hist_diff = 2 * hist - hist_all
            prominent_region = (hist_diff > 0)  # region in the 2D hist where current data is higher than the rest
            data_mask = prominent_region[bin_indices]  # mask data points in that region
            pts_vld = pts[data_mask]

            # find state circle using the valid data
            center, radius = ReadoutWindowCalibrationRuntime.find_most_significant_blob(pts_vld, bins, sigma_factor)
            center_list.append(center)
            r_list.append(radius)

            if debug:
                from matplotlib.patches import Circle
                axs[i, 1].hist2d(pts_vld.real, pts_vld.imag, bins=bins, cmap="hot")
                axs[i, 1].add_patch(Circle((center.real, center.imag), radius, edgecolor="w", facecolor="none"))
                axs[i, 1].set_aspect(1)

        if average_radius:
            r_list = [np.mean(r_list)] * n_states

        state_circles = [(c, r) for c, r in zip(center_list, r_list)]

        return state_circles

    @staticmethod
    def find_quadrant(z: Union[complex, NDArray[complex]]):
        """
        find the quadrant number of a complex number z

        :param z: (Array of) complex number
        :return: quadrant value, [1,2,3,4]
        """
        quadrant = np.array([3, 2, 4, 1])[2 * (z.real > 0) + (z.imag > 0)]
        return quadrant


