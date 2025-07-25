from typing import Union, Literal, List
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axis import Axis
import numpy as np
from scipy.optimize import curve_fit

from acadia_qmsmt.plotting import prepare_plot_axes
from acadia_qmsmt.analysis import to_complex

"""
Collection of commonly used plotting functions
"""


def plot_binaveraged(axis_vals, raw_data, plot_ax=None, n_avg=1, figsize=None, vmin=None, v_max=None, **kwargs) -> [Figure, Axis]:
    """
    Plot a pcolormesh of bin-averaged raw traces.

    Divides a sequence of repeated 1D traces into bins of size `n_avg`,
    computes the average of each bin, and plots the result as a 2D image with the
    vertical axis given by `axis_vals`.

    :param axis_vals (array-like): Values along the vertical axis (e.g., time, frequency).
    :param raw_data (ndarray): 2D array of shape (n_repeats, n_samples).
    :param plot_axs: Existing maplotlib Axes object to plot into.
    :param n_avg: Number of traces to average per bin (default is 1 = no binning).
    :param figsize: Figure size if a new figure is created.
    :return:
    """
    fig, ax = prepare_plot_axes(plot_ax, axs_shape=(1, 1), figsize=figsize)

    n_avg = int(n_avg)
    n_runs = len(raw_data)
    n_lines = n_runs // n_avg
    data_vld = raw_data[:n_lines * n_avg]
    data_ba = data_vld.reshape(n_lines, n_avg, -1).mean(axis=1)  # bin averaged
    bin_index = np.arange(n_lines)
    cmap = kwargs.pop("cmap", "bwr")
    pcm = ax.pcolormesh(bin_index, axis_vals, data_ba.T, cmap=cmap, vmin=vmin, vmax=v_max, **kwargs)
    fig.colorbar(pcm, ax=ax)
    ax.set_xlabel(f"iterations ({n_avg}x) ")
    if cmap == "bwr":
        ax.set_facecolor("k") # make the masked/nan points be distinct from actual mid value points

    return fig, ax


def plot_multiple_hist2d(*iq_pts: np.ndarray, plot_ax=None, bins=51, log_scale: bool = False,
                             figsize=None, **kwargs):
    """
    Plot 2D histograms of IQ points (complex or real-valued 2D arrays).

    Accepts either:
      - 1D complex arrays: [a + bj, ...]
      - 2D float arrays of shape (N, 2): [[I, Q], ...]

    :param iq_pts: One or more arrays of IQ data.
    :param plot_ax: Optional Matplotlib axes or figure.
    :param bins: Number of bins for histograms.
    :param log_scale: Use logarithmic color scale.
    :param figsize: Figure size if creating new figure.
    :param kwargs: Additional kwargs passed to hist2d.
    :return: (fig, axs) tuple
    """
    from matplotlib.colors import LogNorm
    norm = LogNorm() if log_scale else None

    iq_pts = [to_complex(pts, flatten=True) for pts in iq_pts]
    fig, axs = prepare_plot_axes(plot_ax, axs_shape=(1, len(iq_pts)), figsize=figsize)
    axs = np.atleast_1d(axs)

    all_pts = np.concatenate(iq_pts)
    hist_range = ((all_pts.real.min(), all_pts.real.max()), (all_pts.imag.min(), all_pts.imag.max()))

    for i, pts in enumerate(iq_pts):
        axs[i].hist2d(pts.real, pts.imag, cmap="hot",
                      bins=bins, range=hist_range, norm=norm, **kwargs)
        axs[i].set_aspect('equal', adjustable='box')
        axs[i].autoscale(enable=True, axis='both', tight=True)
    if len(axs) == 1:
        axs = axs[0]
    return fig, axs



def plot_pcolormesh_fft(sweep_freqs, fft_freqs, fft_data, plot_ax=None, figsize=(8, 6),
                        root_quadratic_fit=True, fft_peak_threshold=0.4, 
                        fit_freq_min=None, fit_freq_max=None, freq_scale=None) -> [Figure, Axis]:
    """
    Plot a pcolormesh of FFT data and fit the peak track with a sqrare root quadratic function.

    Written based on John's original general plotter code

    sweep_freqs (1D array): Sweep freqs in Hz
    fft_freqs (1D array): FFT frequencies.
    fft_data (2D array): FFT amplitudes. Shape = (len(sweep_freqs), len(fft_freqs)).
    plot_ax: Optional matplotlib axis to plot into.
    figsize: Figure size.
    root_quadratic_fit: when True, fit the fft peak track with a root quadratic function.
    fft_peak_threshold: Relative threshold (0–1) to mask out low FFT peaks.
    :return:
    """
    fig, ax = prepare_plot_axes(plot_ax, axs_shape=(1, 1), figsize=figsize)
    pcm = ax.pcolormesh(sweep_freqs, fft_freqs, fft_data.T, cmap="inferno", shading="auto")
    fig.colorbar(pcm, ax=ax)

    ax.set_ylabel("FFT Frequency")
    ax.set_title(f"FFT of time signal")
    if not root_quadratic_fit:
        return fig, ax

    fit_freq_min = 0 if fit_freq_min is None else fit_freq_min
    fit_freq_max = np.inf if fit_freq_max is None else fit_freq_max

    freq_mask = np.where((fft_freqs > fit_freq_min) & (fft_freqs < fit_freq_max))[0]

    fft_data = fft_data[:, freq_mask]
    fft_freqs = fft_freqs[freq_mask]

    fft_max_idxes = np.argmax(fft_data, axis=1)
    fft_maxes = np.max(fft_data, axis=1)
    peak_mask = fft_maxes > np.ptp(fft_maxes) * fft_peak_threshold

    x_peak = sweep_freqs[peak_mask]
    freq_peak = fft_freqs[fft_max_idxes][peak_mask]

    ax.plot(x_peak, freq_peak, 'w.', label="Peak Trace", linestyle='')

    # guess freq scale based on the relative values of x and y
    if freq_scale is None:
        freq_scale = (fft_freqs[-1] - fft_freqs[0])/(sweep_freqs[-1]-sweep_freqs[0])
        freq_scale = 10 ** int(np.round(np.log10(freq_scale)/3)*3)

    def _fit_model(f, f0, g):
        return np.sqrt(g**2 + (f-f0)**2 * freq_scale**2)

    if len(x_peak) >= 3:
        p0 = (x_peak[np.argmin(freq_peak)], np.min(freq_peak))
        bounds = ((np.min(x_peak), 0), (np.max(x_peak), np.max(freq_peak)))
        popt, pcov = curve_fit(_fit_model, x_peak, freq_peak, p0=p0, bounds=bounds)


        title_text = f"On-resonance Freq: {popt[0]:.3e}   g: {popt[1] / 2:.3e}"
        ax.set_title(f"FFT of time signal\n{title_text}")

        fine_x = np.linspace(sweep_freqs[0], sweep_freqs[-1], 200)
        ax.plot(fine_x, _fit_model(fine_x, *popt), 'w--', label="Parabolic Fit")
        ax.axvline(popt[0], linestyle='dotted', color='w', label="On-Resonance")
    else:
        ax.set_title("FFT of time signal\n(Not enough peak data to fit parabola)")
    ax.set_ylim(min(fft_freqs), max(fft_freqs))
    fig.tight_layout()

    return fig, ax
