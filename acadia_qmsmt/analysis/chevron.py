from matplotlib.axes import Axes
import numpy as np
from scipy.optimize import curve_fit

from acadia_qmsmt.plotting import prepare_plot_axes
from acadia_qmsmt.utils.fourier_transform import fft_mag

class Chevron:
    def __init__(self, sweep_freqs_Hz:np.ndarray, t_list_sec:np.ndarray, data:np.ndarray):
        self.sweep_freqs_Hz = sweep_freqs_Hz
        self.t_list_sec = t_list_sec
        self.data = data
        self.fft_freqs, self.fft_data = fft_mag(self.t_list_sec, self.data, axis=1, remove_zero_freq=True)

        self.fitted_f0 = None
        self.fitted_g = None
        self.fitted_g0 = None
        self.fitted_t0 = None
        self.best_swap_freq = None
        self.best_swap_time = None

        try:
            self.fit_fft()
            self.fit_center_time_linecut()
        except Exception as e:
            pass

    def plot_chevron(self, ax:Axes=None, figsize=None):
        fig, axs = prepare_plot_axes(ax, axs_shape=(1,1), figsize=figsize)
        pcm = axs.pcolormesh(self.sweep_freqs_Hz/1e9, self.t_list_sec*1e6, self.data.T, vmin=0, vmax=1, cmap="bwr")
        fig.colorbar(pcm, ax=axs, label="epop")
        axs.set_xlabel("BS freq (GHz)")
        axs.set_ylabel("Time (us)")
        axs.set_xlim(min(self.sweep_freqs_Hz)/1e9, max(self.sweep_freqs_Hz)/1e9)
        axs.set_ylim(min(self.t_list_sec)*1e6, max(self.t_list_sec)*1e6)
        if (self.best_swap_freq is not None) and (self.best_swap_time is not None):
            self.best_swap_str = f"Best SWAP freq: {self.best_swap_freq/1e9:.6g} GHz, time: {self.best_swap_time*1e9:.3g} ns"
        else:
            self.best_swap_str = ''
        if self.fitted_t0 is not None and self.fitted_f0 is not None:

            axs.plot(
                self.fitted_f0/1e9, self.fitted_t0*1e6,
                marker='x',                # Marker shape: 'x'
                color='black',             # Marker color (the main 'x')
                markersize=14,             # Size of the marker
                markeredgewidth=4,         
                linestyle=''               # Crucial: set to '' so no line connects the point
            )
            axs.plot(
            self.fitted_f0/1e9, self.fitted_t0*1e6,
            marker='x',                # Marker shape: 'x'
            color='yellow',             # Marker color (the main 'x')
            markersize=10,             # Size of the marker
            markeredgewidth=3,         
            linestyle=''               # Crucial: set to '' so no line connects the point
        )
        fig.tight_layout()
        return fig, axs


    def fit_fft(self, peak_threshold=0.4, fft_freq_min=None, fft_freq_max=None, 
                sweep_freq_min=None, sweep_freq_max=None, freq_scale:int=1):
        fft_freq_min = min(self.fft_freqs) if fft_freq_min is None else fft_freq_min
        fft_freq_max = max(self.fft_freqs) if fft_freq_max is None else fft_freq_max
        sweep_freq_min = min(self.sweep_freqs_Hz) if sweep_freq_min is None else sweep_freq_min
        sweep_freq_max = max(self.sweep_freqs_Hz) if sweep_freq_max is None else sweep_freq_max

        fft_freq_mask = np.where((self.fft_freqs > fft_freq_min) & (self.fft_freqs < fft_freq_max))[0]
        fft_data = self.fft_data[:, fft_freq_mask]
        fft_freqs = self.fft_freqs[fft_freq_mask]


        fft_max_idxes = np.argmax(fft_data, axis=1)
        fft_maxes = np.max(fft_data, axis=1)
        peak_mask = fft_maxes > np.ptp(fft_maxes) * peak_threshold
        sweep_freq_mask = (self.sweep_freqs_Hz >= sweep_freq_min) & (self.sweep_freqs_Hz <= sweep_freq_max)
        peak_mask = peak_mask & sweep_freq_mask

        self.sweep_freq_fit = self.sweep_freqs_Hz[peak_mask]
        self.fft_freq_fit = fft_freqs[fft_max_idxes][peak_mask]

        def _fit_model(f, f0, g):
            # apply freq scale for non-1st order processes
            return np.sqrt(g ** 2 + (f - f0) ** 2 * freq_scale ** 2) # we enforce proper unit at the input of the class
        
        p0 = (self.sweep_freq_fit[np.argmin(self.fft_freq_fit)], np.min(self.fft_freq_fit))
        bounds = ((np.min(self.sweep_freq_fit), 0), (np.max(self.sweep_freq_fit), np.max(self.fft_freq_fit)))
        popt, pcov = curve_fit(_fit_model, self.sweep_freq_fit, self.fft_freq_fit, p0=p0, bounds=bounds)
        self.fitted_f0 = popt[0]
        self.fitted_g = abs(popt[1] / 2)
        self.fit_freq_scale = freq_scale
        self.fit_fft_freq_min = fft_freq_min
        self.fit_fft_freq_max = fft_freq_max
        self.fit_sweep_freq_min = sweep_freq_min
        self.fit_sweep_freq_max = sweep_freq_max

        self.best_swap_freq = self.fitted_f0

        return self.fitted_f0, self.fitted_g


    def fit_center_time_linecut(self):
        center_freq_idx = np.argmin(np.abs(self.sweep_freqs_Hz - self.fitted_f0))
        time_linecut = self.data[center_freq_idx, :]

        if time_linecut[0] < 0.5:# determine the sign of oscillation based on 1st data point
            def _fit_model(t, A, g0, t0):
                return A * (1 + np.cos(2 * np.pi * 2 * g0 * (t - t0)))
        else:
            def _fit_model(t, A, g0, t0):
                return A * (1 - np.cos(2 * np.pi * 2 * g0 * (t - t0)))

        p0 = (np.max(time_linecut), self.fitted_g, 1. / (4 * self.fitted_g))
        bounds = ((0, 0, 0), (1.0, np.inf, np.inf))
        popt, pcov = curve_fit(_fit_model, self.t_list_sec, time_linecut, p0=p0, bounds=bounds)
        self.fitted_g0 = abs(popt[1])
        self.fitted_t0 = popt[2]
        self.best_swap_time = round(self.fitted_t0 * 1e9 / 5) * 5e-9
        return self.fitted_g0, self.fitted_t0


    def plot_fft(self, ax:Axes=None, figsize=None):
        fig, ax = prepare_plot_axes(ax, axs_shape=(1, 1), figsize=figsize)
        pcm = ax.pcolormesh(self.sweep_freqs_Hz/1e9, self.fft_freqs/1e6, self.fft_data.T, cmap="inferno", shading="auto")
        fig.colorbar(pcm, ax=ax)
        ax.set_title("FFT of time signal")

        if self.fitted_f0 is not None:
            def _fit_model(f, f0, g):
                return np.sqrt(g ** 2 + (f - f0) ** 2 * self.fit_freq_scale**2) # we enforce proper unit at the input of the class

            ax.plot(self.sweep_freq_fit/1e9, self.fft_freq_fit/1e6, 'w.', linestyle='')
            title_text = f"f0: {self.fitted_f0/1e9:.5g} [GHz],   g: {self.fitted_g/1e6:.4g} [MHz],  Drive order: {self.fit_freq_scale}"
            ax.set_title(f"FFT of time signal\n{title_text}")
            fine_x = np.linspace(min(self.sweep_freqs_Hz), max(self.sweep_freqs_Hz), len(self.sweep_freqs_Hz)*10)
            ax.plot(fine_x/1e9, _fit_model(fine_x, self.fitted_f0, self.fitted_g*2)/1e6, 'w--', label="Fit")
            ax.axvline(self.fitted_f0/1e9, linestyle='dotted', color='w', label="On-Resonance")
            ax.set_ylim(self.fit_fft_freq_min/1e6, self.fit_fft_freq_max/1e6)
            ax.set_xlim(self.fit_sweep_freq_min/1e9, self.fit_sweep_freq_max/1e9)
        
        ax.set_ylabel("FFT Frequency (MHz)")
        ax.set_xlabel("Sweep Frequency (GHz)")
        fig.tight_layout()

        return fig, ax


