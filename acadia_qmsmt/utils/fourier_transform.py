import numpy as np
from numpy.typing import NDArray

def fft_mag(t_list: NDArray, data: NDArray, axis: int = -1, t_unit: str = None, remove_zero_freq: bool = False):
    """
    Calculate FFT magnitude of the input data and return the frequency and root-power spectrum.
    Because the (square root of the) power spectrum is being returned, only the positive-frequency 
    values in both the data and frequency arrays are returned.

    :param t_list: Time values (1D array).
    :param data: Signal array (1D or ND).
    :param t_unit: Unit of time. Used to convert freq axis label.
    :param axis: Axis along which to apply the FFT.
    :param remove_zero_freq: When True, the 0 freq point will be removed from fft results
    :return:
        - F_freq: Frequency values.
        - F_data: FFT magnitude (same shape as data, but length modified along `axis`).
    """
    N = len(t_list)

    # FFT along axis
    F_data = np.fft.fft(data, axis=axis)
    F_data = 2.0 / N * np.abs(np.take(F_data, indices=range(N // 2), axis=axis))

    # Frequency axis
    F_freq = np.fft.fftfreq(N, t_list[1] - t_list[0])[:(N // 2)]

    start_index = 1 if remove_zero_freq else 0
    fft_freqs = F_freq[start_index:]
    fft_data = np.take(F_data, indices=range(start_index, F_data.shape[axis]), axis=axis)

    return fft_freqs, fft_data


def dft_mag(t_list: NDArray, data: NDArray, freq: float, axis: int = -1) -> NDArray:
    """
    Compute the DFT magnitude at a single frequency along a specified axis.

    :param t_list: 1D time array.
    :param data: N-dimensional signal array.
    :param freq: Frequency of interest.
    :param axis: Axis along which to compute the DFT.
    :return: DFT magnitude at `freq`, shape matches `data` with FFT axis removed.
    """

    N = data.shape[axis]
    T = t_list[1] - t_list[0]
    t = np.arange(N) * T

    # Move axis to end for broadcasting
    data = np.moveaxis(data, axis, -1)

    # Compute DFT component
    kernel = np.exp(-2j * np.pi * freq * t)
    dft = np.sum(padded_data * kernel, axis=-1)

    # Normalize and return magnitude
    return np.abs(dft / N * 2)

