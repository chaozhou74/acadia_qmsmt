from datetime import datetime
from typing import Union, List, Tuple, Literal
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.interpolate import interp1d

from acadia_qmsmt.helpers.yaml_editor import update_yaml
from acadia_qmsmt.analysis import StateCircleType, ComplexDataTracesType, QubitStateLabels, find_quadrant, quadrant_signs
from acadia_qmsmt.analysis.state_discrimination import find_state_circles, mask_state_with_circle

class KernelGeneratorBase:
    def __init__(self, traces: Union[List, np.ndarray], g_circle: StateCircleType, e_circle: StateCircleType,
                 norm_factor:float = 1, decimation_used: int = 4, plot=True):
        """
        generate matched kernel based on a given set of IQ traces and specified g/e locations.

        :param traces: raw data complex IQ traces, should have the dimension of (iterations, time_points)
        :param g_circle: (I_center + 1j * Q_center, circle_radius)
        :param e_circle: (I_center + 1j * Q_center, circle_radius)
        :param norm_factor: normalization factor for the generated kernel
        :param decimation_used: Decimation factor used when getting the `traces` data. This will only be used for
            generating the kernel to be uploaded to cmacc.
            Since by default the input data to cmacc will always have a decimation of 4, if the trace data used to
            generate the kernel array had a different decimation factor, the resulting kernel array must be adjusted
            (interpolated or decimated) to match the expected decimation of the incoming data to cmacc.
        :param plot:
        """
        self.all_traces = np.array(traces, dtype=np.complex128)
        self.all_pts = np.sum(traces, axis=1)
        self.g_circle = g_circle
        self.e_circle = e_circle
        self.norm_factor = norm_factor
        self.decimation_used = decimation_used
        self.generate_kernel_trace(g_circle, e_circle)
        self.generate_kernel_for_upload(norm_factor, decimation_used)
        self.generate_cmacc_offset()
        if plot:
            self.plot_kernel_generation()

    def generate_kernel_trace(self, g_circle: StateCircleType, e_circle: StateCircleType):
        """calculate the kernel trace based on the selected data traces

        :param g_circle: (I_center + 1j * Q_center, circle_radius)
        :param e_circle: (I_center + 1j * Q_center, circle_radius)
        """
        self.g_mask = mask_state_with_circle(self.all_pts, g_circle)
        self.e_mask = mask_state_with_circle(self.all_pts, e_circle)
        self.g_trace_avg = np.mean(self.all_traces[self.g_mask], axis=0)
        self.e_trace_avg = np.mean(self.all_traces[self.e_mask], axis=0)

        self.kernel_trace = np.conjugate(self.g_trace_avg - self.e_trace_avg)
        kernel_norm = self.kernel_trace / np.max(abs(self.kernel_trace)) * self.norm_factor # normalize to 1
        self.kernel_trace_norm = kernel_norm

        # calculate expected IQ points when kernel is used
        self.new_iq_pts = np.sum(self.all_traces * kernel_norm, axis=1)

        return self.kernel_trace

    def generate_kernel_for_upload(self, norm_factor: float = 1, decimation_used: int = None):
        """
        Calculate the kernel array for uploading to cmacc based on the decimation used to get the `traces` data

        :return:
        """

        deci = self.decimation_used if decimation_used is None else decimation_used
        scale = deci // 4
        kernel = self.kernel_trace

        if deci == 4:
            kernel_array = kernel

        elif deci == 1:
            # decimate the kernel trace by averaging every 4 adjacent points
            length = len(kernel) - (len(kernel) % 4)
            reshaped = kernel[:length].reshape(-1, 4)
            kernel_array = reshaped.mean(axis=1)

        elif (scale > 0) and (deci % 4 == 0):
            # stretch the kernel trace via interpolation
            scale = int(scale)
            original_len = len(kernel)
            interp_func = interp1d(np.arange(0, original_len), kernel, kind="cubic")
            x_fine = np.linspace(0, original_len-1, (original_len-1) * scale+1)
            kernel_interp = interp_func(x_fine)
            # the points at the beginning will just be repeated
            kernel_array = np.concatenate(([kernel[0]] * (scale-1), kernel_interp))

        else:
            raise ValueError("Invalid trace decimation, must be 1 or a positive multiple of 4 ")

        self.kernel_upload = kernel_array / np.max(abs(kernel_array)) * norm_factor

        return self.kernel_upload

    def plot_kernel(self, plot_uploaded=True):
        if plot_uploaded:
            kernel = self.kernel_upload
        else:
            kernel = self.kernel_trace
        plt.figure()
        plt.title("calculated kernel")
        plt.plot(kernel.real, label="re")
        plt.plot(kernel.imag, label="im")
        plt.xlabel("pts")
        plt.tight_layout()
        plt.legend()

    def save_kernel(self, directory, filename=None):
        """
        Save the kernel array to a .npy file.

        :param directory: Directory where the kernel file will be saved.
        :param name: Name of the kernel file. Defaults to 'kernel_%y%m%d_%H%M%S.npy'.
        :return:
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        save_name = datetime.now().strftime("kernel_%y%m%d_%H%M%S") if filename is None else filename
        save_path = os.path.join(directory, save_name)
        np.save(save_path, self.kernel_upload)
        print(f"kernel saved to: {save_path}.npy")
        return save_path + ".npy"

    def update_kernel(self, yaml_file: str, key_path: str, kernel_dir: str, kernel_name: str = None):
        """
        Update the path to the kernel file in a yaml config file.

        :param yaml_file: Path to the yaml config file
        :param key_path: Dot-separated string representing the hierarchical path to the "kernel_wf" key in the
            nested dict structure of the YAML file (e.g., "ro_capture.kernel_wf")
        :param kernel_dir: Directory where the new kernel file will be saved.
        :param kernel_name: Name of the new kernel file. Default to a name based on  the channel configuration
            path and the current timestamp.
        :return:
        """
        if kernel_name is None:
            kernel_name = key_path + datetime.now().strftime("_%y%m%d_%H%M%S")
        kernel_path = self.save_kernel(kernel_dir, kernel_name)
        update_dict = {key_path: kernel_path}
        update_yaml(yaml_file, update_dict, keep_format=False, verbose=True)

    def plot_kernel_generation(self, log_scale=False):
        """
        Plot the selection of g/e circles for kernel generation, the average of the selected g/e traces, and the
        expected IQ points after applying kernel.

        :param log_scale:
        :return:
        """
        from matplotlib.colors import LogNorm
        norm = LogNorm() if log_scale else None

        fig, axs = plt.subplots(3, 1, figsize=(5, 8))
        axs[0].set_title("raw IQ points and g/e selection")
        axs[0].hist2d(self.all_pts.real, self.all_pts.imag, bins=101, cmap="hot", norm=norm)
        for circ, state in zip([self.g_circle, self.e_circle], ["g", "e"]):
            circ_i, circ_q, circ_r = circ[0].real, circ[0].imag, circ[1]
            axs[0].add_patch(patches.Circle((circ_i, circ_q), circ_r, edgecolor="w", facecolor="none"))
            axs[0].text(circ_i + circ_r * 0.8, circ_q + circ_r * 0.8, state, fontsize=12, va='center', color="w")
        axs[0].set_aspect('equal')

        axs[1].set_title("IQ traces after selection")
        trace_colors = [(0.27, 0.51, 0.71), (1.0, 0.65, 0.47), (0.18, 0.31, 0.56), (0.94, 0.5, 0.5)]
        for i, trace in enumerate([self.g_trace_avg, self.e_trace_avg]):
            axs[1].plot(trace.real, ".-", color=trace_colors[2 * i], label=f"{['g', 'e'][i]} trace, re")
            axs[1].plot(trace.imag, ".-", color=trace_colors[2 * i + 1], label=f"{['g', 'e'][i]} trace, im")
            axs[1].set_xlabel("pts")
            axs[1].legend()

        axs[2].set_title("IQ points after applying kernel")
        hist, xedges, yedges, _ = axs[2].hist2d(self.new_iq_pts.real, self.new_iq_pts.imag,
                                                bins=101, cmap="hot", norm=norm)
        if hasattr(self, "cmacc_offset"):
            x0, y0 = -self.cmacc_offset.real, -self.cmacc_offset.imag
            y0 = np.clip(y0, yedges[0], yedges[-1])
            axs[2].axvline(x=x0, color='w', linestyle='-', linewidth=0.5)
            axs[2].axhline(y=y0, color='w', linestyle='-', linewidth=0.5)
            text_d = (xedges[-1]-xedges[0]) * 0.03
            for i, quadrant in enumerate(self.state_quadrants):
                sign_x, sign_y = quadrant_signs(quadrant)
                axs[2].text(x0 + text_d * sign_x, y0 + text_d * sign_y, QubitStateLabels[i], fontsize=12,
                            va='center', ha='center', color="w")

        axs[2].set_aspect('equal')

        fig.tight_layout()
        # fig.show(block=False)

        self.fig = fig

        return fig, axs

    def generate_cmacc_offset(self, *args, **kwargs):
        pass

    def update_cmacc_offset(self, yaml_file: str, offset_key_path: str, quadrant_key_path: str):
        pass


class KernelFromGETraces(KernelGeneratorBase):
    def __init__(self, g_traces: ComplexDataTracesType, e_traces: ComplexDataTracesType,
                 g_circle: StateCircleType = None, e_circle: StateCircleType = None,
                 i_threshold: int = None, q_threshold: int = None,
                 bins: int = 50, sigma_factor: float = 2.5, average_radius=True,
                 norm_factor:float = 1, decimation_used: int = 4,
                 plot=True, debug=False):
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
        :param plot:
        """
        self.g_traces_raw = g_traces
        self.e_traces_raw = e_traces
        self.g_pts_raw = np.sum(g_traces, axis=1)
        self.e_pts_raw = np.sum(e_traces, axis=1)
        self.i_threshold = i_threshold
        self.q_threshold = q_threshold
        self.all_traces = np.concatenate([g_traces, e_traces])
        if (g_circle and e_circle) is None:  # at least one of the circles is not provided
            g_e_circles = find_state_circles(self.g_pts_raw, self.e_pts_raw, bins=bins,
                                             sigma_factor=sigma_factor, average_radius=average_radius, debug=debug)

        g_circle = g_circle if g_circle is not None else g_e_circles[0]
        e_circle = e_circle if e_circle is not None else g_e_circles[1]

        super().__init__(self.all_traces, g_circle, e_circle, norm_factor, decimation_used, plot)


    def generate_cmacc_offset(self, i_threshold: int = None, q_threshold: int = None) \
            -> Tuple[complex, Tuple[int, int]]:
        """
        Calculate the optimal cmacc offset (the preloaded value) for splitting the post-kernel g and e states into
        different quadrants on the IQ plane.

        :param i_threshold: The I position of the separation line. Default to center of the g and e blobs.
        :param q_threshold: The Q position of the separation line. Default to 3-sigma away from the center of the g
            blob. When f+ states are involved, this might need to be manually assigned.

        :return: (cmacc offset, (quadrant of the g state, quadrant of the e state))
        """
        i_threshold = self.i_threshold if i_threshold is None else i_threshold
        q_threshold = self.q_threshold if q_threshold is None else q_threshold

        # post-kernel g and e locations
        g_center_pk = np.sum(self.kernel_trace_norm * self.g_trace_avg)
        e_center_pk = np.sum(self.kernel_trace_norm * self.e_trace_avg)

        # Calculate default Q offset based on center of the new g and e circle
        if i_threshold is None:
            offset_I = -(g_center_pk.real + e_center_pk.real) / 2
        else:
            offset_I = -i_threshold

        # Calculate default Q offset based on the 3-sigma radius of the g states
        if q_threshold is None:
            radius = np.std(self.new_iq_pts[self.g_mask] - g_center_pk) * 3
            offset_Q = ((g_center_pk.imag > 0) * 2 - 1) * radius - g_center_pk.imag
        else:
            offset_Q = -q_threshold

        # get the quadrant of where the two states falls in
        def _get_shifted_quadrant(center):
            shifted_ = center + offset_I + 1j* offset_Q
            quadrant = find_quadrant(shifted_)
            return quadrant

        # offset = int(offset_I) + 1j * int(offset_Q) # todo: `acadia.cmacc_load` needs to support complex input
        offset = int(offset_I)
        q_g, q_e = _get_shifted_quadrant(g_center_pk), _get_shifted_quadrant(e_center_pk)

        self.cmacc_offset = offset
        self.state_quadrants = (q_g, q_e)

        return offset, (q_g, q_e)

    def update_cmacc_offset(self, yaml_file: str, offset_key_path: str, quadrant_key_path: str):
        """
        Update the cmacc_offset and g_e_quadrant in a yaml config file.

        :param yaml_file: Path to the yaml config file
        :param offset_key_path: Dot-separated string representing the hierarchical path to the "cmacc_offset" key in the
            nested dict structure of the YAML file (e.g., "ro_capture.cmacc_offset")
        :param offset_key_path: key path to the "state_quadrants" key in the YAML file.
        :return:
        """
        update_dict = {offset_key_path: self.cmacc_offset, quadrant_key_path: [*self.state_quadrants]}
        update_yaml(yaml_file, update_dict, keep_format=False, verbose=True)


def load_kernel(kernel_path):
    kernel = np.load(kernel_path)
    return kernel


if __name__ == "__main__":
    from acadia.data import DataManager
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    plt.ion()


    data_dir = "/tmp/241127_133001"
    dm = DataManager()
    dm.load(data_dir)
    g_traces = np.array(dm[f"traces_g"].records()).astype(float).view(complex).squeeze()
    e_traces = np.array(dm[f"traces_e"].records()).astype(float).view(complex).squeeze()
    all_data = np.concatenate([g_traces, e_traces])

    kgen = KernelFromGETraces(np.repeat(g_traces, 2, axis=0), e_traces, plot=True)
    from acadia_qmsmt.helpers.plot_utils import add_button

    save_kernel = lambda _: kgen.save_kernel(r"../dev_codes//")
    button = add_button(kgen.fig, save_kernel, "save kernel")


    # kernel=load_kernel(r"../dev_codes//"+"readoutkernel_241105_113318.npy")
    #
    # pts = rk.e_pts_raw
    # # center, radius = find_most_significant_blob(pts, sigma_factor=3)
    #
    # c_g, c_e = find_state_circles(rk.g_pts_raw, rk.e_pts_raw)
    # center, radius = c_e
    #
    # fig, ax = plt.subplots(1, 1)
    # ax.hist2d(pts.real, pts.imag, bins=51, cmap="hot")
    # ax.add_patch(patches.Circle((center.real, center.imag), radius, edgecolor="w", facecolor="none"))
    # plt.show()
    # ax.set_aspect(1)

    plt.figure()
    plt.hist2d(kgen.new_iq_pts[kgen.g_mask].real, kgen.new_iq_pts[kgen.g_mask].imag, bins=51)
    plt.hist2d(kgen.all_pts[kgen.g_mask].real, kgen.all_pts[kgen.g_mask].imag, bins=51)