import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from linc_rfsoc.analysis.unit_converter import t2f


def fft(t_list:NDArray, data:NDArray, t_unit:str=None, plot=True, plot_ax=None, zero_padding:int=1):
    """
    Perform FFT on the input data and return the frequency and amplitude spectrum.

    :param t_list: Time list
    :param data: Signal list
    :param t_unit: Unit of time. If provided, will then determine the freq unit
    :param plot: When True, plot the fft result
    :param plot_ax: Matplotlib Axes object for plotting. If None, a new figure and axes will be created.
    :param zero_padding: Factor for zero-padding the data to increase the FFT resolution. Default is 1 (no zero padding)
    :return:
        - `F_freq`: Frequency values corresponding to the FFT spectrum.
        - `F_data`: Amplitude spectrum of the signal.
    """
    N = len(t_list)
    T = t_list[1] - t_list[0]

    N_padded = N * zero_padding
    data_padded = np.pad(data, (0, (N_padded - N)), 'constant')
    F_data = np.fft.fft(data_padded)
    F_data = 2.0 / N_padded * np.abs(F_data[:N_padded // 2]) * zero_padding
    F_freq = np.fft.fftfreq(N_padded, T)[:N_padded // 2]

    if plot:
        if plot_ax is None:
            fig, plot_ax = plt.subplots()
        plot_ax.plot(F_freq, F_data)
        plot_ax.set_yscale("log")
        if t_unit is not None:
            plot_ax.set_xlabel(f"Freq {t2f(t_unit)}")
        plot_ax.grid(True)

    return F_freq, F_data


def fft_one_freq(t_list:NDArray, data:NDArray, freq:float, zero_padding:int=1):
    """
    Calculate the DFT coefficient at a single frequency.

    :param t_list: Time list.
    :param data: Signal list.
    :param freq: Target frequency.
    :param zero_padding: Zero padding factor. Default is 1 (no zero padding).

    """

    # Pad the signal with zeros
    padded_data = np.pad(data, (0, len(data) * zero_padding), mode='constant')
    # Time vector based on the original time step
    t = np.arange(0, len(padded_data), 1) * (t_list[1] - t_list[0])
    # Calculate the DFT coefficient at the target frequency
    dft_coefficient = np.sum(padded_data * np.exp(-2j * np.pi * freq * t))
    # normalize and add the negative component
    dft_coefficient = np.abs(dft_coefficient / len(data) * 2)

    return dft_coefficient


