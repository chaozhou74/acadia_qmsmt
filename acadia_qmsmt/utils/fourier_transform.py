from typing import Sequence

import numpy as np
from numpy.typing import NDArray

def fft_mag(t_list: NDArray, data: NDArray, axis: int = -1, remove_zero_freq: bool = False):
    """
    Calculate the FFT amplitude spectrum of the input data.

    For real-valued `data`, returns the single-sided spectrum (frequencies >= 0), with a 2/N rescaling that folds in
    the (redundant, conjugate) negative-frequency half. For complex-valued `data`, returns the full double-sided
    spectrum (all frequencies, no rescaling), since positive and negative frequency bins carry independent information
    and cannot be folded together.

    :param t_list: Time values (1D array), assumed uniformly spaced.
    :param data: Signal array (1D or ND), real- or complex-valued.
    :param axis: Axis along which to apply the FFT.
    :param remove_zero_freq: When True, the 0-freq point is removed from the results.
    :return:
        - fft_freqs: Frequency values (positive-only if real input, full range if complex).
        - fft_data: FFT magnitude (same shape as data, but length modified along `axis`).
    """
    N = len(t_list)
    if data.shape[axis] != N:
        raise ValueError(
            f"data.shape[{axis}]={data.shape[axis]} does not match len(t_list)={N}"
        )

    dt = t_list[1] - t_list[0]
    F_data = np.fft.fft(data, axis=axis)

    if np.iscomplexobj(data):
        # Double-sided: sort into ascending frequency order, magnitude only, no rescaling.
        F_freq = np.fft.fftshift(np.fft.fftfreq(N, dt))
        F_data = np.abs(np.fft.fftshift(F_data, axes=axis))
    else:
        # Single-sided: fold negative frequencies into the positive half via 2/N rescaling.
        F_data = 2.0 / N * np.abs(np.take(F_data, indices=range(N // 2), axis=axis))
        F_freq = np.fft.fftfreq(N, dt)[: (N // 2)]

    if remove_zero_freq:
        if np.iscomplexobj(data):
            zero_idx = np.argmin(np.abs(F_freq))
            F_freq = np.delete(F_freq, zero_idx)
            F_data = np.delete(F_data, zero_idx, axis=axis)
        else:
            F_freq = F_freq[1:]
            F_data = np.take(F_data, indices=range(1, F_data.shape[axis]), axis=axis)

    return F_freq, F_data


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
    dft = np.sum(data * kernel, axis=-1)

    # Normalize and return magnitude
    return np.abs(dft / N * 2)



def bandpass_filter_trace(trace, fs:float, passbands:tuple[float, float]|Sequence[tuple[float, float]],
                          rolloff:float=None, zero_pad:bool=True):
    """
    Arbitrary digital bandpass filter for a complex-valued trace via FFT masking.

    trace     : 1D complex (or real) ndarray, time-domain samples
    fs        : sample rate, Hz
    passbands : a single (f_lo, f_hi) tuple, or a list of them, Hz, each in (-fs/2, fs/2).
                Since the trace is complex, pass/reject regions are NOT
                mirrored around DC -- give the exact ranges you want kept.
                E.g. passbands=[(2e6, 8e6)] keeps only +2..+8 MHz and
                rejects -8..-2 MHz (and everything else). f_lo must be < f_hi;
                split a band that wraps past +/-fs/2 into two entries.
    rolloff   : Hz, width of smooth edge transition on each band boundary.
                None/0 -> brick-wall (sharp cutoff, more time-domain ringing).
    zero_pad  : zero-pad to 2x length before FFT (avoids circular wrap-around
                of the filter's impulse response into the trace ends), then
                trim back. Turn off only for genuinely periodic/continuous data.

    returns   : filtered trace, same length as input, complex dtype
    """
    trace = np.asarray(trace)

    # accept a single (f_lo, f_hi) band as shorthand for [(f_lo, f_hi)]
    if len(passbands) and np.isscalar(passbands[0]):
        if len(passbands) != 2:
            raise ValueError("a single passband must be (f_lo, f_hi)")
        passbands = [passbands]

    n = len(trace)
    nfft = 2 * n if zero_pad else n

    x = np.zeros(nfft, dtype=complex)
    x[:n] = trace

    freqs = np.fft.fftfreq(nfft, d=1 / fs)
    X = np.fft.fft(x)

    mask = np.zeros(nfft)
    for f_lo, f_hi in passbands:
        if f_lo >= f_hi:
            raise ValueError(f"band ({f_lo}, {f_hi}): f_lo must be < f_hi")
        if rolloff:
            edge = rolloff / 4  # tanh width scale ~ matches transition to `rolloff`
            band = 0.5 * (np.tanh((freqs - f_lo) / edge) - np.tanh((freqs - f_hi) / edge))
        else:
            band = ((freqs >= f_lo) & (freqs <= f_hi)).astype(float)
        mask = np.maximum(mask, band)  # union of bands

    y = np.fft.ifft(X * mask)
    return y[:n]
