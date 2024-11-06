from datetime import datetime
from typing import Union, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def mask_state_with_circle(iq_pts:Union[List[complex], np.ndarray[complex]], state_circle:Tuple[complex, float],
                           plot: Union[bool, int] = True, state_name: str = "", plot_ax=None):
    """
    :param iq_pts: input IQ data points, should be an array of complex numbers
    :param state_circle: (I_center + 1j * Q_center, circle_radius)
    :param plot: if true, plot selected data
    :param state_name: name of the state, will be used in the plotting title.
    :return:
    """
    state_center, state_r = state_circle
    mask = abs(iq_pts-state_center) < state_r

    if plot:
        if plot_ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            ax = plot_ax
        ax.set_title(f'{state_name} state selection')
        ax.hist2d(iq_pts.real.flatten(), iq_pts.imag.flatten(), bins=101)
        theta = np.linspace(0, 2 * np.pi, 201)
        ax.plot(state_center.real + state_r * np.cos(theta), state_center.imag + state_r * np.sin(theta), color='r')
        ax.set_aspect(1)

    return mask

def load_kernel(kernel_path):
    kernel = np.load(kernel_path)
    return kernel

class ReatoutKernelGenerator:
    def __init__(self, traces:Union[List, np.ndarray], g_circle:Tuple[complex, float], e_circle:Tuple[complex, float],
                 plot=True):
        """
        generate matched kernel based on a given set of IQ traces and specified g/e locations.

        :param traces: raw data complex IQ traces, should have the dimension of (iterations, time_points)
        :param g_circle: (I_center + 1j * Q_center, circle_radius)
        :param e_circle: (I_center + 1j * Q_center, circle_radius)
        :param plot:
        """
        # todo: auto-fit for g_circle and e_circle
        self.iq_traces = np.array(traces)
        self.iq_pts = np.mean(traces, axis=1)
        self.g_circle = g_circle
        self.e_circle = e_circle
        self.kernel = np.ones(traces.shape[1])
        self.generate_kernel(plot)

    def generate_kernel(self, plot=True):
        self.g_mask = mask_state_with_circle(self.iq_pts, self.g_circle, plot=False)
        self.e_mask = mask_state_with_circle(self.iq_pts, self.e_circle, plot=False)
        self.g_trace_avg = np.mean(self.iq_traces[self.g_mask], axis=0)
        self.e_trace_avg = np.mean(self.iq_traces[self.e_mask], axis=0)
        kernel = np.conjugate(self.g_trace_avg - self.e_trace_avg)
        self.kernel = kernel/np.max(abs(kernel))

        self.new_iq_pts = np.mean(self.iq_traces * self.kernel, axis=1)

        if plot:
            fig, axs = plt.subplots(3, 1, figsize=(6,10))
            axs[0].set_title("raw IQ points and g/e selection")
            axs[0].hist2d(self.iq_pts.real, self.iq_pts.imag, bins=101, cmap="hot")
            for circ, state in zip([self.g_circle, self.e_circle], ["g", "e"]):
                circ_i, circ_q, circ_r = circ[0].real, circ[0].imag, circ[1]
                axs[0].add_patch(patches.Circle((circ_i, circ_q), circ_r, edgecolor="w", facecolor="none"))
                axs[0].text(circ_i+circ_r, circ_q+circ_r, state, fontsize=12, va='center', color="w")
            axs[0].set_aspect('equal')

            axs[1].set_title("IQ traces after selection")
            for i, trace in enumerate([self.g_trace_avg, self.e_trace_avg]):
                axs[1].plot(trace.real, ".-", color=f"C{i}", label=f"{['g','e'][i]} trace, re")
                axs[1].plot(trace.imag, ".--", color=f"C{i}", label=f"{['g','e'][i]} trace, im")
                axs[1].set_xlabel("pts")
                axs[1].legend()

            axs[2].set_title("IQ points after applying kernel")
            axs[2].hist2d(self.new_iq_pts.real, self.new_iq_pts.imag, bins=101, cmap="hot")
            axs[2].set_aspect('equal')
            plt.show()

        return self.kernel

    def save_kernel(self, save_dir, save_name=None):
        save_name = datetime.now().strftime("readoutkernel_%y%m%d_%H%M%S") if save_name is None else save_name
        np.save(save_dir+save_name, self.kernel)
        return save_dir+save_name



if __name__ == "__main__":
    import matplotlib
    matplotlib.use('tkagg')
    from acadia.data import DataManager

    data_dir = "/tmp/241104_150246"
    dm = DataManager()
    dm.load(data_dir)
    all_data = np.concatenate([np.array(dm[f"traces_phi0"].records()).astype(float),
                               np.array(dm[f"traces_phi1"].records()).astype(float)])
    all_data = all_data.view(complex).squeeze()


    rk = ReatoutKernelGenerator(all_data, (-1.6 + 0j, 0.5), (1.6 + 0j, 0.5))
    # rk.save_kernel(r"../dev_codes//")
    # kernel=load_kernel(r"../dev_codes//"+"readoutkernel_241105_113318.npy")