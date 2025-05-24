import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.colors import LogNorm
from acadia_qmsmt.analysis.unit_converter import t2f



class FFT:
    def __init__(self, t_list: NDArray, data: NDArray, axis: int = -1, zero_padding: int = 1, remove_zero_freq=True):

        """
        Perform FFT on the input data and return the frequency and amplitude spectrum.

        :param t_list: Time values (1D array).
        :param data: Signal array (1D or ND).
        :param axis: Axis along which to apply the FFT.
        :param zero_padding: Padding factor to increase FFT resolution.
        :param remove_zero_freq: When True, the 0 freq point will be removed from fft results
        :return:
            - F_freq: Frequency values.
            - F_data: FFT amplitude (same shape as data, but length modified along `axis`).
        """
        self.data = data
        self.axis = axis
        N = len(t_list)
        T = t_list[1] - t_list[0]
        N_padded = N * zero_padding

        # Pad along specified axis
        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (0, N_padded - N)
        padded_data = np.pad(data, pad_width, mode="constant")

        # FFT along axis
        F_data = np.fft.fft(padded_data, axis=axis)
        F_data = 2.0 / N_padded * np.abs(np.take(F_data, indices=range(N_padded // 2), axis=axis))

        # Frequency axis
        F_freq = np.fft.fftfreq(N_padded, T)[:N_padded // 2]

        start_index = 1 if remove_zero_freq else 0
        self.fft_freqs = F_freq[start_index:]
        self.fft_data = np.take(F_data, indices=range(start_index, F_data.shape[axis]), axis=axis)


    def plot_1d(self, plot_ax: Axis = None, t_unit: str = None):
        """

        :param plot_ax: Optional matplotlib Axes object to plot into.
        :param t_unit: Unit of time. Used to convert freq axis label.
        :return:
        """
        if plot_ax is None:
            fig, plot_ax = plt.subplots()
        else:
            fig = plot_ax.get_figure()

        if self.data.ndim != 1:
            raise ValueError(f"plot_1d can only take 1D data, got data dimension: {self.data.ndim}")
        plot_ax.plot(self.fft_freqs, self.fft_data)
        plot_ax.set_yscale("log")
        plot_ax.set_xlabel(f"FFT Freq ({t2f(t_unit) if t_unit is not None else ' '})")
        plot_ax.set_ylabel("FFT Amplitude")
        plot_ax.grid(True)

        return fig, plot_ax

    def plot_2d(self, x_vals, plot_ax: Axis = None, t_unit: str = None):
        """

        :param x_vals: x value list for plot
        :param plot_ax: Optional matplotlib Axes object to plot into.
        :param t_unit: Unit of time. Used to convert freq axis label.
        :return:
        """
        if plot_ax is None:
            fig, plot_ax = plt.subplots()
        else:
            fig = plot_ax.get_figure()

        if self.data.ndim != 2:
            raise ValueError(f"plot_2d can only take 2D data, got data dimension: {self.data.ndim}")

        if self.axis in (1, -1):
            pcm = plot_ax.pcolormesh(x_vals, self.fft_freqs, self.fft_data.T, cmap="inferno")
        elif self.axis == 0:
            pcm = plot_ax.pcolormesh(x_vals, self.fft_freqs, self.fft_data, cmap="inferno")
        else:
            raise ValueError("for 2D fft, the axis must be 0, 1 or -1")

        plot_ax.set_xlabel(f"FFT Freq ({t2f(t_unit) if t_unit is not None else ''})")
        fig.colorbar(pcm, ax=plot_ax)

        return fig, plot_ax



def fft(t_list: NDArray, data: NDArray, axis: int = -1, t_unit: str = None, plot=False, plot_ax=None,
        zero_padding: int = 1, remove_zero_freq=False):
    """
    Perform FFT on the input data and return the frequency and amplitude spectrum.

    :param t_list: Time values (1D array).
    :param data: Signal array (1D or ND).
    :param t_unit: Unit of time. Used to convert freq axis label.
    :param axis: Axis along which to apply the FFT.
    :param plot: Whether to plot the result.
    :param plot_ax: Optional matplotlib Axes object to plot into.
    :param zero_padding: Padding factor to increase FFT resolution.
    :param remove_zero_freq: When True, the 0 freq point will be removed from fft results
    :return:
        - F_freq: Frequency values.
        - F_data: FFT amplitude (same shape as data, but length modified along `axis`).
    """
    fft_obj = FFT(t_list, data, axis, zero_padding, remove_zero_freq=remove_zero_freq)

    if plot:
        fft_obj.plot_1d(plot_ax, t_unit)

    return fft_obj.fft_freqs, fft_obj.fft_data


def fft_one_freq(t_list: NDArray, data: NDArray, freq: float, zero_padding: int = 1, axis: int = -1) -> NDArray:
    """
    Compute the DFT amplitude at a single frequency along a specified axis.

    :param t_list: 1D time array.
    :param data: N-dimensional signal array.
    :param freq: Frequency of interest.
    :param zero_padding: Padding factor to increase resolution.
    :param axis: Axis along which to compute the DFT.
    :return: DFT amplitude at `freq`, shape matches `data` with FFT axis removed.
    """

    N = data.shape[axis]
    N_padded = N * (1 + zero_padding)
    T = t_list[1] - t_list[0]
    t = np.arange(N_padded) * T

    # Pad along axis
    pad_width = [(0, 0)] * data.ndim
    pad_width[axis] = (0, N_padded - N)
    padded_data = np.pad(data, pad_width, mode="constant")

    # Move axis to end for broadcasting
    padded_data = np.moveaxis(padded_data, axis, -1)

    # Compute DFT component
    kernel = np.exp(-2j * np.pi * freq * t)
    dft = np.sum(padded_data * kernel, axis=-1)

    # Normalize and return magnitude
    dft_mag = np.abs(dft / N * 2)

    return dft_mag

