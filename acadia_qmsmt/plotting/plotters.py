from typing import Union, Literal, List
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axis import Axis
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
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


    
    pcm = ax.pcolormesh(sweep_freqs, fft_freqs, fft_data.T, cmap="inferno", shading="auto")
    fig.colorbar(pcm, ax=ax)
    # ax.plot(x_peak, freq_peak, 'w.', label="Peak Trace", linestyle='')

    # guess freq scale based on the relative values of x and y
    if freq_scale is None:
        freq_scale = (fft_freqs[-1] - fft_freqs[0])/(sweep_freqs[-1]-sweep_freqs[0])
        freq_scale = 10 ** int(np.round(np.log10(freq_scale)/3)*3)

    def _fit_model(f, f0, g):
        return np.sqrt(g**2 + (f-f0)**2 * freq_scale**2)

    center_freq = None
    swap_time = None
    if len(x_peak) >= 3:
        p0 = (x_peak[np.argmin(freq_peak)], np.min(freq_peak))
        bounds = ((np.min(x_peak), 0), (np.max(x_peak), np.max(freq_peak)))
        popt, pcov = curve_fit(_fit_model, x_peak, freq_peak, p0=p0, bounds=bounds)


        center_freq = popt[0]
        swap_time = 1/(popt[1]*2) if popt[1] != 0 else np.inf
        title_text = f"On-resonance Freq: {popt[0]:.5g} GHz   g: {popt[1] / 2:.5g} MHz, Tswap: {swap_time*1e3:.3g} ns"
        ax.set_title(f"FFT of time signal\n{title_text}")

        fine_x = np.linspace(sweep_freqs[0], sweep_freqs[-1], 200)
        ax.plot(fine_x, _fit_model(fine_x, *popt), 'w--', label="Parabolic Fit", linewidth=1)
        ax.axvline(popt[0], linestyle='dotted', color='w', label="On-Resonance")
    else:
        ax.set_title("FFT of time signal\n(Not enough peak data to fit parabola)")
    ax.set_ylim(min(fft_freqs), max(fft_freqs))
    fig.tight_layout()

    params={
        "center_freq": center_freq,
        "swap_time": swap_time,
    }


    return fig, ax, params
    return fig, ax


def plot_histogram(val_dict, err_dict = None, plot_axs=None, add_labels=True):
    """ Plot a histogram of values for a given set of measured value dict, with optional error bars.
    """
    fig, axs = prepare_plot_axes(plot_axs)

    bar_xaxis = list(val_dict.keys())
    vals =  list(val_dict.values())
    errs = [err_dict[axis] for axis in bar_xaxis] if err_dict is not None else None

    bars = axs.bar(bar_xaxis, vals, yerr=errs)
    if add_labels:
        for bar in bars:
            height = np.round(bar.get_height(), 4)
            va = 'bottom' if height > 0 else 'top'
            axs.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va=va)
        
    axs.grid(1)
    return fig, axs


# ----------------- for density matrix visualizaiton ----------------------------
def cmap2d_hsv(w):
    cmap = plt.get_cmap("hsv")
    rgba = np.array(cmap((np.angle(w) / (2 * np.pi) + 0.5) % 1.0))
    rgba[..., -1] = np.abs(w)
    return rgba


def cmap2d_balanced(z):
    """
    Map complex z -> RGBA; complex z should have amplitude between 0 and1
    - RGB from custom balanced cyclic phase colormap
    - alpha from |z| 
    """
    
    # some anchor colors picked by gpt... suppose to be perceptually uniform to human eye
    anchors = np.array([
        [0.30, 0.35, 0.80],  # indigo / deep blue
        [0.20, 0.65, 0.75],  # teal-cyan
        [0.85, 0.70, 0.20],  # amber / goldenrod (not neon yellow)
        [0.75, 0.30, 0.60],  # magenta/rose
        [0.30, 0.35, 0.80],  # wrap back to deep blue
    ])
    
    # Build a continuous colormap through these anchors
    x = np.linspace(0, 1, len(anchors))
    cdict = {
        "red": [(x[i], anchors[i, 0], anchors[i, 0]) for i in range(len(anchors))],
        "green": [(x[i], anchors[i, 1], anchors[i, 1]) for i in range(len(anchors))],
        "blue": [(x[i], anchors[i, 2], anchors[i, 2]) for i in range(len(anchors))],
    }
    cmap = LinearSegmentedColormap("phase_balanced_cyclic", cdict)

    z = np.asarray(z)
    phase = np.angle(z)               
    phase01 = (phase + np.pi) / (2*np.pi)

    rgb = np.array(cmap(phase01))[..., :3]
    alpha = np.clip(np.abs(z), 0.0, 1.0)[..., None]

    rgba = np.concatenate([rgb, alpha], axis=-1)
    return rgba

def plot_density_matrix(rho:np.ndarray, plot_ax=None, cmap_2d:callable=None, max_amp=1, add_cbar=True):
    """
    Plot a density matrix `rho` on a 2D grid. Each matrix element is drawn as a square
    and colored by `cmap_2d`.

    :param rho: Density matrix to visualize. Expected shape (dim, dim), complex.
    :param plot_ax: Axes to draw on. If None, a new figure/axes will be created.
    :param cmap_2d: Function that maps a complex number (or array of complex numbers) to RGBA.
    :param max_amp: Maximum amplitude of elements in the density matrix, for normalzing the max amplitude before color mapping
    :param add_cbar:  when True, make a coloar bar based on `cmap`

    """
    from acadia_qmsmt.analysis.tomography import _digits_in_base
    rho = np.array(rho)
    dim = rho.shape[0]
    n_qubits = int(np.sqrt(dim))
    basis_labels = ["".join(row.astype(str)) for row in _digits_in_base(2, n_qubits)]
    
    cmap_2d = cmap2d_balanced if cmap_2d is None else cmap_2d
    
    fig, plot_ax = prepare_plot_axes(plot_ax)
    
    plot_ax.fill(np.array([0, dim, dim, 0]), np.array([0, 0, dim, dim]), color='white')
    
    def blob(x, y, z, ax):
        hs = 1 / 2
        xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
        ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
        ax.fill(xcorners, ycorners, color=cmap_2d(z / max_amp))
    
    def make_phase_cmap_from_cmap2d(cmap2d_func):
        """
        Take a complex->RGBA mapper (like cmap2d_hsv) and build
        a phase-only ListedColormap over [-pi, pi].
        """
        phases = np.linspace(-np.pi, np.pi, 256)
        rgba = cmap2d_func(np.exp(1j * phases))
        # rgba is (256,4). ListedColormap accepts RGBA directly.
        return mcolors.ListedColormap(rgba, name="test")
    
    for x in range(dim):
        for y in range(dim):
            _x = x + 1
            _y = y + 1
            blob(_x - 0.5, dim - _y + 0.5, rho[y, x], ax=plot_ax)
            text = f"${abs(rho[y, x]):.2f} \\angle{int(np.angle(rho[y, x], deg=True))}^\circ$"
            plot_ax.text(_x - 0.5, dim - _y + 0.5, text,
                         ha='center', va='center')
    
    # Frame
    plot_ax.set_aspect("equal")
    plot_ax.set_frame_on(False)
    
    # Grid ticks
    plot_ax.set_xlim(0, dim)
    plot_ax.set_ylim(0, dim)
    plot_ax.xaxis.set_major_locator(plt.IndexLocator(1, 0))
    plot_ax.yaxis.set_major_locator(plt.IndexLocator(1, 0))
    
    # x/y axis
    plot_ax.set_xticks(0.5 + np.arange(dim))
    plot_ax.set_xticklabels(basis_labels)
    plot_ax.xaxis.tick_top()
    plot_ax.set_yticks(0.5 + np.arange(dim))
    plot_ax.set_yticklabels(basis_labels[::-1])
    
    # color map
    if add_cbar:
        # build a dummy ScalarMappable: phase ∈ [-π, π] -> hsv colormap
        phase_cmap = make_phase_cmap_from_cmap2d(cmap_2d)
        norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=phase_cmap)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, ax=plot_ax, pad=0.02)
        cbar.set_label("phase [rad]")
        cbar.set_ticks([-np.pi, 0, np.pi])
        cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
    
    fig.tight_layout()
    
    return fig, plot_ax
